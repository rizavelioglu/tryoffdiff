from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
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
    """Downloads the VITON-HD dataset and extracts specific folders to the specified directory."""
    urls = [
        # Official link
        "https://www.dropbox.com/scl/fi/xu08cx3fxmiwpg32yotd7/zalando-hd-resized.zip?rlkey=ks83mdv2pvmrdl2oo2bmmn69w&e=2&dl=1",
        # Unofficial link
        "https://www.kaggle.com/api/v1/datasets/download/marquis03/high-resolution-viton-zalando-dataset",
    ]
    zip_filename = "zalando-hd-resized.zip"

    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Specific folders to extract (other folders are not necessary for VTOFF)
    allowed_folders = {"test/cloth/", "test/image/", "train/cloth/", "train/image/"}

    def is_valid_zip(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                return zf.testzip() is None
        except Exception:
            return False

    for idx, url in enumerate(urls):
        try:
            logger.info(f"Attempting to download VITON-HD dataset from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            zip_path = output_path / zip_filename
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info(f"Downloaded dataset to {zip_path}")

            if not is_valid_zip(zip_path):
                logger.error(f"Downloaded file from {url} is not a valid zip archive.")
                zip_path.unlink(missing_ok=True)
                if idx == len(urls) - 1:
                    raise zipfile.BadZipFile("All download attempts failed.")
                continue

            logger.info("Extracting specific folders from dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for file in zip_ref.namelist():
                    if any(file.startswith(folder) for folder in allowed_folders):
                        zip_ref.extract(file, output_path)
            logger.info(f"Dataset extracted to {output_path}")

            zip_path.unlink()
            logger.info("Temporary zip file removed.")
            break  # Success, exit loop

        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred during download from {url}: {e}")
            if idx == len(urls) - 1:
                logger.error("All download attempts failed.")
        except zipfile.BadZipFile:
            logger.error(f"Error: The downloaded file from {url} is not a valid zip archive.")
            if idx == len(urls) - 1:
                logger.error("All download attempts failed.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            if idx == len(urls) - 1:
                logger.error("All download attempts failed.")


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


class CustomDataset(Dataset):
    def __init__(self, root_dir: Path, transform: transforms.Compose, folder: str):
        """Initialize the dataset.

        Args:
            root_dir: Path to the dataset split directory (e.g., train or test)
            transform: Transforms to apply to the images
            folder: Which folder to load images from ('cloth' or 'image')
        """
        self.img_dir = root_dir / folder
        self.image_filenames = [i.name for i in os.scandir(self.img_dir) if i.is_file()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        image_filename = self.image_filenames[idx]
        img = read_image(str(self.img_dir / image_filename))
        return self.transform(img), image_filename


@app.command()
def vae_encode_dataset(
    data_dir: str = typer.Option(..., help="Path to the dataset directory."),
    model_name: str = typer.Option("sd14", help="Abbreviation of the VAE model: ['sd14', 'sdxl', 'sd3']."),
    batch_size: int = typer.Option(default=8),
    data_name: str = typer.Option(..., help="Name of the dataset: ['vitonhd', 'dresscode']."),
):
    """Encode images using VAE for the specified dataset and save them using safetensors.

    Examples:
        Encoding the Dresscode dataset using the SD3-VAE model:
        $ python tryoffdiff/dataset.py vae-encode-dataset \
        --data-dir ".../dresscode/" \
        --model-name "sd3" \
        --batch-size 32 \
        --data-name "dresscode"

        Encoding the VITON-HD dataset using the SD1.4-VAE model:
        $ python tryoffdiff/dataset.py vae-encode-dataset \
        --data-dir ".../vitonhd/" \
        --model-name "sd14" \
        --batch-size 32 \
        --data-name "vitonhd"
    """

    device = "cuda"
    vae = load_vae_model(model_name, device)

    for split in ["train", "test"]:
        input_dir = Path(data_dir) / split
        output_dir = Path(data_dir).parent / f"{data_name}-enc-{model_name}" / split
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = CustomDataset(root_dir=input_dir, transform=create_transform(model_name), folder="cloth")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        with (
            tqdm(total=len(dataloader), desc=f"Encoding {split}", unit="batch") as pbar,
            ThreadPoolExecutor(max_workers=4) as executor,
        ):
            futures = []
            for batch in dataloader:
                x, img_names = batch
                x = x.to(device)

                with torch.no_grad():
                    latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
                    latents = latents.cpu()

                for latent, img_name in zip(latents, img_names, strict=True):
                    save_path = output_dir / f"{img_name}.safetensors"
                    futures.append(executor.submit(save_file, {"vae_enc": latent}, save_path))

                pbar.update(1)

            for future in tqdm(futures, desc=f"Saving {split}", unit="batch"):
                future.result()

    logger.success(f"Dataset saved at: {output_dir.parent}.")


@app.command()
def siglip_encode_images(
    data_dir: str = typer.Option(..., help="Path to the dataset directory."),
    batch_size: int = typer.Option(default=128),
    data_name: str = typer.Option(..., help="Name of the dataset: ['vitonhd', 'dresscode']."),
):
    """Encode images using SigLIP-B/16-512 for the defined dataset and save them using safetensors.

    Examples:
        $ python tryoffdiff_private/dataset.py siglip-encode-images \
         --data-dir "<path-to-data-dir>" \
         --data-name "vitonhd" \
         --batch-size 128
    """
    ckpt = "google/siglip-base-patch16-512"
    image_processor = SiglipImageProcessor.from_pretrained(ckpt, do_resize=False, do_rescale=False, do_normalize=False)
    image_encoder = SiglipVisionModel.from_pretrained(ckpt, device_map="cuda").eval()

    for split in ["train", "test"]:
        input_dir = Path(data_dir) / split
        output_dir = Path(data_dir).parent / f"{data_name}-enc-siglip2" / split
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = CustomDataset(root_dir=input_dir, transform=create_transform("siglip"), folder="image")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        with (
            tqdm(total=len(dataloader), desc=f"Encoding {split}", unit="batch") as pbar,
            ThreadPoolExecutor(max_workers=4) as executor,
        ):
            futures = []
            for batch in dataloader:
                cond, img_names = batch

                # Process batch on GPU
                with torch.no_grad():
                    inputs = image_processor(images=cond, return_tensors="pt")
                    inputs = {k: v.to(image_encoder.device) for k, v in inputs.items()}
                    outputs = image_encoder(**inputs)
                    image_feats = outputs.last_hidden_state.cpu()

                for latent, img_name in zip(image_feats, img_names, strict=True):
                    save_path = output_dir / f"{img_name}.safetensors"
                    futures.append(executor.submit(save_file, {"siglip_enc": latent}, save_path))

                pbar.update(1)

            for future in tqdm(futures, desc=f"Saving {split}", unit="batch"):
                future.result()

    logger.success(f"Dataset saved at: {output_dir.parent}.")


@app.command()
def restructure_dresscode_to_vitonhd(
    zip_dir: str = typer.Option(..., help="Path to the DressCode dataset zip file."),
    output_dir: str = typer.Option("./data/dresscode", help="The directory where the dataset will be saved."),
):
    """Extracts only images/ subfolders and txt files from DressCode.zip, then organizes paired images into VITON-HD-like structure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    main_folders = ["dresses", "lower_body", "upper_body"]
    images_subfolder = "images/"
    pair_files = ["train_pairs.txt", "test_pairs_paired.txt"]
    files_to_extract = []

    # Step 1: Extract only images/ folders and pair files
    with zipfile.ZipFile(zip_dir, "r") as zip_ref:
        for member in zip_ref.namelist():
            # Extract pair files
            if any(member == pf for pf in pair_files):
                files_to_extract.append(member)
            # Extract images/ subfolders from each main folder
            for folder in main_folders:
                prefix = f"{folder}/{images_subfolder}"
                if member.startswith(prefix) and not member.endswith("/"):
                    files_to_extract.append(member)

        for file in files_to_extract:
            zip_ref.extract(file, output_path)

    # Step 2: Organize paired images into VITON-HD-like structure
    split_root = output_path.parent / (output_path.name + "-split")
    for split in ["train", "test"]:
        for subfolder in ["cloth", "image"]:
            (split_root / split / subfolder).mkdir(parents=True, exist_ok=True)

    # Read pairs
    category_mapping = {"0": "upper_body", "1": "lower_body", "2": "dresses"}

    def read_pairs(file_path):
        with open(file_path) as f:
            return [line.strip().split("\t") for line in f.readlines()]

    train_pairs = read_pairs(output_path / "train_pairs.txt")
    test_pairs = read_pairs(output_path / "test_pairs_paired.txt")

    def copy_images(pairs, dest_dir):
        for model_img, garment_img, c in pairs:
            category_folder = category_mapping.get(c)
            for idx, img in enumerate([model_img, garment_img]):
                src_path = output_path / category_folder / "images" / img
                dst_folder = "image" if idx == 0 else "cloth"
                dst_path = split_root / dest_dir / dst_folder / img
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {src_path} -> {dst_path}")

    copy_images(train_pairs, "train")
    copy_images(test_pairs, "test")

    logger.success(f"DressCode dataset restructured at: {split_root}")


if __name__ == "__main__":
    app()
