import os

from diffusers import AutoencoderKL, EulerDiscreteScheduler, PNDMScheduler
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import typer

from tryoffdiff.config import InferenceConfig, dataclass_cli
from tryoffdiff.features import DressCodeDataset, VitonVAESigLIPDataset
from tryoffdiff.modeling.model import create_model

app = typer.Typer(pretty_exceptions_enable=False)


@app.command("tryoffdiff")
@dataclass_cli
@torch.no_grad()
def inference_tryoffdiff(config: InferenceConfig):
    """Run inference on TryOffDiff model."""
    os.makedirs(config.output_dir, exist_ok=True)

    # faster inference on some devices
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load models
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").eval().cuda()
    net = create_model(config.model_class)
    net.load_state_dict(torch.load(config.model_path, weights_only=False))
    net.eval().to(config.device)

    # Enable faster inference
    net = torch.compile(net)

    # Set up scheduler
    scheduler = PNDMScheduler.from_pretrained(config.scheduler_dir)

    # Prepare dataloader
    val_set = VitonVAESigLIPDataset(root_dir=config.val_img_dir, inference=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Prepare output transform
    output_transform = transforms.Normalize(mean=[-1], std=[2])

    # Set up generator
    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    # Create output directory
    output_path = (
        config.output_dir / "vitonhd-test" / f"seed{config.seed}-n{config.num_inference_steps}-g{config.guidance_scale}"
    )
    os.makedirs(output_path, exist_ok=True)

    # Start inference
    for cond, img_name in tqdm(val_loader, desc="Processing batches"):
        cond = cond.to(config.device)
        batch_size = cond.size(0)  # Adjust batch size for the last batch
        scheduler.set_timesteps(config.num_inference_steps)  # Reset scheduler for each batch

        # Initialize noise for this batch
        x = torch.randn(batch_size, 4, 64, 64, generator=generator, device=config.device)
        uncond = torch.zeros_like(cond) if config.guidance_scale else None

        # Denoising Loop
        for t in scheduler.timesteps:
            if config.guidance_scale:
                noise_pred = net(torch.cat([x] * 2), t, torch.cat([uncond, cond]))
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = net(x, t, cond)
            x = scheduler.step(noise_pred, t, x).prev_sample

        # Decode images 1-by-1, otherwise consumes lots of memory
        images = [vae.decode(1 / 0.18215 * im.unsqueeze(0)).sample.cpu() for im in x]
        images = output_transform(images)

        for img, name in zip(images, img_name, strict=True):
            grid = torchvision.utils.make_grid(img, nrow=1, normalize=True, scale_each=True)
            torchvision.utils.save_image(grid, output_path / name.replace("jpg", "png"))


@app.command("tryoffdiffv2")
@dataclass_cli
@torch.no_grad()
def inference_tryoffdiffv2(config: InferenceConfig):
    """Run inference on TryOffDiffv2 models."""
    os.makedirs(config.output_dir, exist_ok=True)

    # faster inference on some devices
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.cache_size_limit = 128
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load model
    net = create_model(config.model_class)
    state_dict = torch.load(config.model_path, weights_only=False)
    # Check if model was compiled during training
    if any("_orig_mod" in key for key in state_dict):
        net = torch.compile(net)
        net.load_state_dict(state_dict, strict=False)
    else:  # If not, load the model as is and compile it
        net.load_state_dict(state_dict)
        net = torch.compile(net)
    net.eval().to(config.device)

    # Load VAE or latent upscaler
    if config.upscale_latents:
        typer.echo("[WARNING] Model predictions will be upscaled by 2!")
        from diffusers import StableDiffusionLatentUpscalePipeline

        class LatentUpscalePipeline(StableDiffusionLatentUpscalePipeline):
            def encode_prompt(
                self,
                prompt,
                device,
                do_classifier_free_guidance,
                negative_prompt=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
            ):
                return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

        upscaler = LatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float32
        ).to("cuda")
        # Precompute positive embeddings
        prompt = "studio photo of a garment, high resolution, clean neutral background, photorealistic"
        text_inputs = upscaler.tokenizer(
            prompt, padding="max_length", max_length=25, truncation=True, return_tensors="pt"
        ).to("cuda")
        text_encoder_out = upscaler.text_encoder(text_inputs.input_ids, output_hidden_states=True)
        prompt_embeds = text_encoder_out.hidden_states[-1]
        pooled_prompt_embeds = text_encoder_out.pooler_output

        # Precompute negative embeddings (e.g., empty string for unconditional)
        negative_prompt = "blurry, low resolution, noisy, grainy, distorted, artifacts, overexposed, underexposed"
        uncond_inputs = upscaler.tokenizer(
            negative_prompt, padding="max_length", max_length=25, truncation=True, return_tensors="pt"
        ).to("cuda")
        uncond_encoder_out = upscaler.text_encoder(uncond_inputs.input_ids, output_hidden_states=True)
        negative_prompt_embeds = uncond_encoder_out.hidden_states[-1]
        negative_pooled_prompt_embeds = uncond_encoder_out.pooler_output
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().cuda()

    # Set up scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(config.scheduler_dir, use_karras_sigmas=True)
    scheduler.is_scale_input_called = True  # suppress warning

    # Prepare dataloader
    if config.dataset_type not in ["dresscode", "dc-upperbody", "dc-lowerbody", "dc-dresses"]:
        raise ValueError("Provide one of [dresscode, dc-upperbody, dc-lowerbody, dc-dresses]")
    category = None if config.dataset_type == "dresscode" else config.dataset_type.split("-")[1]
    dataset = DressCodeDataset(root_dir=config.val_img_dir, category=category, inference=True)
    val_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Prepare output transform
    output_transform = transforms.Normalize(mean=[-1], std=[2])

    # Set up generator
    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    # Create output directory
    output_path = (
        config.output_dir
        / config.dataset_type
        / f"e{config.model_filename.split('_')[-1].split('.')[0]}-seed{config.seed}-n{config.num_inference_steps}-g{config.guidance_scale}"
    )
    os.makedirs(output_path, exist_ok=True)

    # Start inference
    for batch in tqdm(val_loader, desc="Processing batches"):
        # Handle different return types based on dataset
        if config.dataset_type == "dresscode":
            cond, img_name, label = batch
            label = label.to(config.device)
        else:
            cond, img_name = batch
            label = None
        cond = cond.to(config.device)
        batch_size = cond.size(0)  # Adjust batch size for the last batch
        scheduler.set_timesteps(config.num_inference_steps)

        # Initialize noise for this batch
        x = torch.randn(batch_size, 4, 64, 64, generator=generator, device=config.device)
        uncond = torch.zeros_like(cond) if config.guidance_scale else None

        # Denoising Loop
        for t in scheduler.timesteps.to(config.device):
            # Apply CFG
            if config.guidance_scale:
                if config.dataset_type == "dresscode":
                    noise_pred = net(torch.cat([x] * 2), t, torch.cat([uncond, cond]), torch.cat([label, label]))
                else:
                    noise_pred = net(torch.cat([x] * 2), t, torch.cat([uncond, cond]))

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                if config.dataset_type == "dresscode":
                    noise_pred = net(x, t, cond, label)
                else:
                    noise_pred = net(x, t, cond)

            x = scheduler.step(noise_pred, t, x).prev_sample

        if config.upscale_latents:
            for img, name in zip(x, img_name, strict=False):
                image = upscaler(
                    image=img,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=20,
                    guidance_scale=1.5,
                    generator=generator,
                ).images[0]
                image.save(output_path / name.replace("jpg", "png"))
        else:
            # Decode images 1-by-1, otherwise consumes lots of memory
            images = [vae.decode(1 / vae.config.scaling_factor * im.unsqueeze(0)).sample.cpu() for im in x]
            images = output_transform(images)

            for img, name in zip(images, img_name, strict=True):
                grid = torchvision.utils.make_grid(img, nrow=1, normalize=True, scale_each=True)
                torchvision.utils.save_image(grid, output_path / name.replace("jpg", "png"))


if __name__ == "__main__":
    app()
