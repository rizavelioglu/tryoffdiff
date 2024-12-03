from enum import Enum, unique
from typing import Any

from diffusers import UNet2DConditionModel
import torch
from torch import nn


class TryOffDiff(nn.Module):
    """A model using adapters to align image feature dimensions with text feature dimensions for conditioning the UNet in a diffusion process."""

    def __init__(self):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

        # Projection layers
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.proj = nn.Linear(1024, 77)
        self.norm = nn.LayerNorm(768)

    def project_embeddings(self, x):
        """
        Align image features to match the expected input size for the UNet's cross-attention layers.

        Args:
            x (torch.Tensor): Input features of shape [batch_size, 1024, 768].

        Returns:
            torch.Tensor: Processed features of shape [batch_size, 77, 768].
        """
        x = self.transformer(x)  # Add contextual information: [batch_size, 1024, 768]
        x = self.proj(x.permute(0, 2, 1))  # Project and reshape: [batch_size, 768, 77]
        x = self.norm(x.permute(0, 2, 1))  # Normalize and return: [batch_size, 77, 768]
        return x

    def forward(self, noisy_latents, t, cond_emb):
        """
        Forward pass for predicting noise given latents, time step, and conditioning embeddings.

        Args:
            noisy_latents (torch.Tensor): Noisy latent vectors of shape [batch_size, C, H, W].
            t (torch.Tensor): Time step for diffusion process.
            cond_emb (torch.Tensor): Conditioning embeddings of shape [batch_size, 1024, 768].

        Returns:
            torch.Tensor: Predicted noise for denoising.
        """
        cond_emb = self.project_embeddings(cond_emb)
        noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=cond_emb).sample
        return noise_pred


@unique
class ModelName(Enum):
    TryOffDiff = TryOffDiff


def create_model(model_name: str, **kwargs: Any) -> Any:
    model_class = ModelName[model_name].value
    return model_class(**kwargs)
