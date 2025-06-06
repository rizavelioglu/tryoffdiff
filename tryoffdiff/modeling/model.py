from enum import Enum, unique
from typing import Any

from diffusers import UNet2DConditionModel
import torch
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel


class TryOffDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.proj = nn.Linear(1024, 77)
        self.norm = nn.LayerNorm(768)

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
        cond_emb = self.transformer(cond_emb)  # Add contextual information: [batch_size, 1024, 768]
        cond_emb = self.proj(cond_emb.permute(0, 2, 1))  # Project and reshape: [batch_size, 768, 77]
        cond_emb = self.norm(cond_emb.permute(0, 2, 1))  # Normalize: [batch_size, 77, 768]
        return self.unet(noisy_latents, t, encoder_hidden_states=cond_emb).sample


class TryOffDiffv2Base(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize image encoder
        self.image_processor = SiglipImageProcessor.from_pretrained(
            "google/siglip-base-patch16-512", do_resize=False, do_rescale=False, do_normalize=False
        )
        self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512", device_map="cuda")
        self.image_encoder.requires_grad_(False)

        # Initialize UNet & Adapter layers
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.proj = nn.Linear(1024, 77)
        self.norm = nn.LayerNorm(768)

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.proj.weight)

    @torch.no_grad()
    def get_cond_emb(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.image_encoder.device)
        return self.image_encoder(**inputs).last_hidden_state  # [batch_size, 1024, 768]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")


class TryOffDiffv2Single(TryOffDiffv2Base):
    def forward(self, noisy_latents, t, cond_img):
        cond_emb = self.get_cond_emb(cond_img)
        cond_emb = self.proj(cond_emb.transpose(1, 2))
        cond_emb = self.norm(cond_emb.transpose(1, 2))
        return self.unet(noisy_latents, t, encoder_hidden_states=cond_emb).sample


class TryOffDiffv2Multi(TryOffDiffv2Base):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=768,
            class_embed_type=None,
            num_class_embeds=3,
        )
        # Load the pretrained weights into the custom model, skipping incompatible keys
        pretrained_state_dict = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        ).state_dict()
        self.unet.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, noisy_latents, t, cond_img, class_labels):
        cond_emb = self.get_cond_emb(cond_img)
        cond_emb = self.proj(cond_emb.transpose(1, 2))
        cond_emb = self.norm(cond_emb.transpose(1, 2))
        return self.unet(noisy_latents, t, encoder_hidden_states=cond_emb, class_labels=class_labels).sample


@unique
class ModelName(Enum):
    TryOffDiff = TryOffDiff
    TryOffDiffv2Single = TryOffDiffv2Single
    TryOffDiffv2Multi = TryOffDiffv2Multi


def create_model(model_name: str, **kwargs: Any) -> Any:
    model_class = ModelName[model_name].value
    return model_class(**kwargs)
