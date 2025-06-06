import os
from pathlib import Path

from loguru import logger
from safetensors import safe_open
from torch.utils.data import Dataset
from torchvision.io import read_image

from tryoffdiff.dataset import create_transform


class VitonVAESigLIPDataset(Dataset):
    """
    Dataset for VITON-HD images that provides both VAE and SigLIP encodings.

    Supports multiple VAE model types (e.g., "sd14", "sdxl").
        "sd14" VAE encodings: [batch_size, 4, 64, 64]
        "sdxl" VAE encodings: [batch_size, 4, 128, 128]
        SigLIP encodings: [batch_size, 1024, 768]
    """

    def __init__(self, root_dir: Path, inference: bool = False):
        self.root_dir = root_dir
        self.inference = inference
        self.vae_model_type = Path(root_dir).parent.stem.split("-enc-")[1]
        self.enc_filenames = [i.path for i in os.scandir(self.root_dir) if i.is_file()]

    def __len__(self) -> int:
        return len(self.enc_filenames)

    def __getitem__(self, idx: int):
        # Adjust encoding path based on specified VAE model type
        enc_path = self.enc_filenames[idx].replace(f"-enc-{self.vae_model_type}", "-enc-siglip")

        # Load SigLIP encodings
        with safe_open(enc_path, framework="pt") as f:
            siglip_encodings = f.get_tensor("siglip_enc")

        # Return SigLIP encodings with image name during inference
        if self.inference:
            img_name = Path(enc_path).stem
            return siglip_encodings, img_name

        # Otherwise, load VAE encodings and return both encodings
        with safe_open(self.enc_filenames[idx], framework="pt") as f:
            vae_encodings = f.get_tensor("vae_enc")

        return vae_encodings, siglip_encodings


class DressCodeDataset(Dataset):
    def __init__(
        self, root_dir: Path, inference: bool = False, category: str | None = None, img_encoder: str = "siglip"
    ):
        self.root_dir = root_dir
        self.inference = inference
        self.category = category
        self.data_dir = self.root_dir.parents[1] / "dresscode"
        self.img_dir = self.data_dir / f"{'test' if self.inference else 'train'}/image"

        self.pairs = self._load_pairs()
        self.transform = create_transform(model_name=img_encoder)

    def _load_pairs(self):
        """Load image pairs from the respective file based on category."""
        category_suffix = f"_{self.category}" if self.category in {"lowerbody", "upperbody", "dresses"} else ""
        filename = f"{'test_pairs_paired' if self.inference else 'train_pairs'}{category_suffix}.txt"
        file_path = self.data_dir / filename
        logger.info(f"Loading pairs from {file_path}")

        # Each line reads: | reference_img - garment_img - class_label(if present) |
        with open(file_path) as f:
            return [line.strip().split("\t") for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """Fetch reference image, garment encoding (if training), and class label (if available)."""
        pair_data = self.pairs[idx]
        ref_img_name, garment_img_name = pair_data[:2]
        cls = int(pair_data[2]) if len(pair_data) == 3 else None  # Check if class label is available

        ref_img = read_image(self.img_dir / ref_img_name)
        ref_img = self.transform(ref_img)

        if self.inference:
            return (ref_img, ref_img_name, cls) if cls is not None else (ref_img, ref_img_name)

        # Load garment VAE encodings
        enc_filename = self.root_dir / f"{garment_img_name}.safetensors"
        with safe_open(enc_filename, framework="pt") as f:
            vae_encodings = f.get_tensor("vae_enc")

        return (ref_img, vae_encodings, cls) if cls is not None else (ref_img, vae_encodings)
