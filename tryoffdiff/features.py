import os
from pathlib import Path

from safetensors import safe_open
from torch.utils.data import Dataset


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
