from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import zipfile

from diffusers import AutoencoderKL
from loguru import logger
import requests
from safetensors.torch import save_file
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from transformers import SiglipImageProcessor, SiglipVisionModel
import typer

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def download_vitonhd(
    output_dir: str = typer.Option("./data/vitonhd", help="The directory where the dataset will be extracted."),
):
    """Downloads the VITON-HD dataset from the official link and extracts specific folders to the specified directory."""
    vitonhd_link = "https://www.dropbox.com/scl/fi/xu08cx3fxmiwpg32yotd7/zalando-hd-resized.zip?rlkey=ks83mdv2pvmrdl2oo2bmmn69w&e=2&dl=1"
    zip_filename = "zalando-hd-resized.zip"

    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Specific folders to extract (other folders are not necessary for VTOFF)
    allowed_folders = {"test/cloth/", "test/image/", "train/cloth/", "train/image/"}

    try:
        # Download the dataset
        logger.info("Downloading VITON-HD dataset from the official source (may take ~5 minutes)...")
        response = requests.get(vitonhd_link, stream=True)
        response.raise_for_status()  # Raise an error for HTTP codes other than 200

        # Save the zip file
        zip_path = output_path / zip_filename
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Downloaded dataset to {zip_path}")

        # Extract only specific folders
        logger.info("Extracting specific folders from dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file in zip_ref.namelist():
                # Check if the file belongs to one of the allowed folders
                if any(file.startswith(folder) for folder in allowed_folders):
                    zip_ref.extract(file, output_path)
        logger.info(f"Dataset extracted to {output_path}")

        # Optionally, delete the zip file after extraction
        zip_path.unlink()
        logger.info("Temporary zip file removed.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred during download: {e}")
    except zipfile.BadZipFile:
        logger.error("Error: The downloaded file is not a valid zip archive.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


@app.command()
def clean_vitonhd(
    data_dir: str = typer.Option("./data/vitonhd", help="The directory where VITON-HD dataset is extracted."),
):
    """Removes duplicate and leaked image files from the specified VITON-HD dataset directories.

    We computed phash of images using `imagehash` library https://github.com/JohannesBuchner/imagehash and detected same
    images. See `tryoffdiff/notebooks/vitonhd_duplicates.ipynb` for details.
    """

    with open("./references/vitonhd_duplicate_filenames.json") as f:
        files_to_remove = json.load(f)

    # Iterate over each folder and remove files
    for folder, file_types in files_to_remove.items():
        filenames = file_types["duplicates"] + file_types.get("leaked", [])
        for filename in filenames:
            for subfolder in ["cloth", "image"]:
                file_path = os.path.join(data_dir, folder, subfolder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed: {file_path}")
                else:
                    logger.info(f"File not found, skipped: {file_path}")


def load_vae_model(model_name: str, device: str) -> AutoencoderKL:
    """Load and return the VAE model specified by `model_name`."""
    model_paths: dict[str, str] = {
        "sd14": "CompVis/stable-diffusion-v1-4",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    }
    vae_path = model_paths.get(model_name)
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(device)
    vae.eval()
    logger.info(f"Loaded VAE model from: {vae_path}")
    return vae


class PadToSquare:
    def __call__(self, img):
        _, h, w = img.shape  # Get the original dimensions
        max_side = max(h, w)
        pad_h = (max_side - h) // 2
        pad_w = (max_side - w) // 2
        padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
        return transforms.functional.pad(img, padding, padding_mode="edge")


def create_transform(model_name: str) -> transforms.Compose:
    """Define and return appropriate transforms based on `model_name`."""
    base_transform = transforms.Compose(
        [
            PadToSquare(),  # Custom transform to pad the image to a square
            transforms.Resize((512, 512)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    name_to_transform = {
        "sd14": base_transform,
        "sdxl": transforms.Compose(
            [
                PadToSquare(),  # Custom transform to pad the image to a square
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
        "sd3": base_transform,
        "siglip": base_transform,
    }
    return name_to_transform.get(model_name)


@app.command()
def vae_encode_vitonhd(
    data_dir: str = typer.Option(..., help="Path to the VITONHD dataset directory."),
    model_name: str = typer.Option("sd14", help="Abbreviation of the VAE model: ['sd14', 'sdxl', 'sd3']."),
    batch_size: int = typer.Option(default=16),
):
    """Encode images using VAE for the VITON-HD dataset and save them using safetensors.

    This concatenates VAE encodings of both clothing and model images into a single file.

    Examples:
        $ python tryoffdiff/dataset.py vae-encode-vitonhd \
        --data-dir "<path-to-data-dir>" \
        --model-name "sd3" \
        --batch-size 4
    """

    class CustomVitonDataset(Dataset):
        def __init__(self, root_dir: Path, transform: transforms.Compose):
            self.cloth_dir = root_dir / "cloth"
            self.cond_dir = root_dir / "image"
            self.image_filenames = [i.name for i in os.scandir(self.cloth_dir) if i.is_file()]
            self.transform = transform

        def __len__(self) -> int:
            return len(self.image_filenames)

        def __getitem__(self, idx: int):
            image_filename = self.image_filenames[idx]
            cond_path = self.cond_dir / image_filename
            cloth_path = self.cloth_dir / image_filename

            cond_img = read_image(str(cond_path))
            cloth_img = read_image(str(cloth_path))

            return self.transform(cloth_img), self.transform(cond_img), image_filename

    device = "cuda"
    vae_scale_factor = 0.18215
    vae = load_vae_model(model_name, device)

    for split in ["train", "test"]:
        input_dir = Path(data_dir) / split
        output_dir = Path(data_dir).parent / f"vitonhd-enc-{model_name}" / split
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = CustomVitonDataset(root_dir=input_dir, transform=create_transform(model_name))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        total_batches = len(dataloader)

        with (
            tqdm(total=total_batches, desc=f"Encoding {split}", unit="batch") as pbar,
            ThreadPoolExecutor(max_workers=4) as executor,
        ):
            futures = []
            for batch in dataloader:
                x, cond, img_names = batch
                x, cond = x.to(device), cond.to(device)

                with torch.no_grad():
                    latent_x = vae.encode(x).latent_dist.sample() * vae_scale_factor

                    if model_name == "sd3":
                        latent_cond = vae.encode(cond).latent_dist.sample() * vae_scale_factor
                        # Concatenate along width dimension: (batch_size, 16, height, width*2)
                        latents = torch.cat((latent_x, latent_cond), dim=3)
                    else:
                        latents = latent_x

                latents_cpu = latents.cpu()

                for latent, img_name in zip(latents_cpu, img_names, strict=True):
                    save_path = output_dir / f"{img_name}.safetensors"
                    futures.append(executor.submit(save_file, {"vae_enc": latent}, save_path))

                pbar.update(1)

            for future in tqdm(futures, desc=f"Saving {split}", unit="batch"):
                future.result()

    logger.success(f"Dataset saved at: {output_dir.parent}.")


@app.command()
def siglip_encode_vitonhd(
    data_dir: str = typer.Option(..., help="Path to the VITONHD dataset directory."),
    batch_size: int = typer.Option(default=64),
):
    """Encode images using SigLIP-B/16-512 for the VITON-HD dataset and save them using safetensors.

    Examples:
        $ python tryoffdiff/dataset.py siglip-encode-vitonhd \
         --data-dir "<path-to-data-dir>"
    """

    class CustomVitonDataset(Dataset):
        def __init__(self, root_dir: Path, transform: transforms.Compose):
            self.cond_dir = root_dir / "image"
            self.image_filenames = [i.name for i in os.scandir(self.cond_dir) if i.is_file()]
            self.transform = transform

        def __len__(self) -> int:
            return len(self.image_filenames)

        def __getitem__(self, idx: int):
            image_filename = self.image_filenames[idx]
            cond_path = self.cond_dir / image_filename
            cond_img = read_image(str(cond_path))
            return self.transform(cond_img), image_filename

    device = "cuda"
    image_processor = SiglipImageProcessor.from_pretrained(
        "google/siglip-base-patch16-512", do_resize=False, do_rescale=False, do_normalize=False
    )
    image_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512", device_map=device)
    image_encoder.eval()

    for split in ["train", "test"]:
        input_dir = Path(data_dir) / split
        output_dir = Path(data_dir).parent / f"vitonhd-enc-siglip/{split}"
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = CustomVitonDataset(root_dir=input_dir, transform=create_transform("siglip"))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        total_batches = len(dataloader)
        with (
            tqdm(total=total_batches, desc=f"Encoding {split}", unit="batch") as pbar,
            ThreadPoolExecutor(max_workers=4) as executor,
        ):
            futures = []
            for batch in dataloader:
                cond, img_names = batch

                with torch.no_grad():
                    inputs = image_processor(images=cond, return_tensors="pt")
                    inputs = {k: v.to(image_encoder.device) for k, v in inputs.items()}
                    outputs = image_encoder(**inputs)
                    image_feats = outputs.last_hidden_state  # [batch_size, 1024, 768]
                image_feats = image_feats.cpu()

                for latent, img_name in zip(image_feats, img_names, strict=True):
                    save_path = output_dir / f"{img_name}.safetensors"
                    futures.append(executor.submit(save_file, {"siglip_enc": latent}, save_path))

                pbar.update(1)

            for future in tqdm(futures, desc=f"Saving {split}", unit="batch"):
                future.result()

    logger.success(f"Dataset saved at: {output_dir.parent}.")


if __name__ == "__main__":
    app()
