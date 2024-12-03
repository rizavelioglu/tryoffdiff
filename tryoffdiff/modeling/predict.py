import os

from diffusers import AutoencoderKL, PNDMScheduler
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import typer

from tryoffdiff.config import InferenceConfig, dataclass_cli
from tryoffdiff.features import VitonVAESigLIPDataset
from tryoffdiff.modeling.model import create_model
from tryoffdiff.plots import should_visualize, visualize_results

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

    # Set up scheduler
    scheduler = PNDMScheduler.from_pretrained(config.scheduler_dir)
    scheduler.set_timesteps(config.num_inference_steps)

    # Prepare dataloader
    val_set = VitonVAESigLIPDataset(root_dir=config.val_img_dir, inference=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Prepare output transform
    output_transform = transforms.Normalize(mean=[-1], std=[2])

    # Set up generator
    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    if config.process_all_samples:
        output_path = (
            config.output_dir
            / "vitonhd-test"
            / f"seed{config.seed}-n{config.num_inference_steps}-g{config.guidance_scale}"
        )
        os.makedirs(output_path, exist_ok=True)

        # Enable faster inference
        net = torch.compile(net)

        for cond, img_name in tqdm(val_loader, desc="Processing batches"):
            cond = cond.to(config.device)
            batch_size = cond.size(0)  # Adjust batch size for the last batch
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

    else:
        # Process only the first batch
        cond = next(iter(val_loader))[0].to(config.device)
        uncond = torch.zeros_like(cond) if config.guidance_scale else None

        # Initialize noise
        x = torch.randn(config.batch_size, 4, 64, 64, generator=generator, device=config.device)

        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising"), 1):
            if config.guidance_scale:
                model_input = torch.cat([x] * 2)
                model_cond = torch.cat([uncond, cond])
                noise_pred = net(model_input, t, model_cond)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = net(x, t, cond)

            x = scheduler.step(noise_pred, t, x).prev_sample

            if should_visualize(i, config, scheduler):
                images = vae.decode(1 / 0.18215 * x).sample

                output_filename = (
                    f"e{config.model_filename.split('_')[-1].split('.')[0]}/"
                    f"seed{config.seed}"
                    f"-n{config.num_inference_steps}"
                    f"-g{config.guidance_scale}"
                    f"--{i}.png"
                )
                visualize_results(
                    images=[images],
                    titles=[f"Predictions (g:{config.guidance_scale}, step {i}/{config.num_inference_steps})"],
                    output_path=config.output_dir / output_filename,
                    transform_func=output_transform,
                    show_titles=config.vis_with_caption,
                )


@app.command("autoencoder")
@dataclass_cli
@torch.no_grad()
def inference_autoencoder(config: InferenceConfig):
    """Run inference on the ablation model: Autoencoder."""
    pass


@app.command("pixel")
@dataclass_cli
@torch.no_grad()
def inference_pixel(config: InferenceConfig):
    """Run inference on the ablation model: PixelModel."""
    pass


@app.command("ldm1")
@dataclass_cli
@torch.no_grad()
def inference_ldm1(config: InferenceConfig):
    """Run inference on the ablation model: LDM-1."""
    pass


@app.command("ldm2")
@dataclass_cli
@torch.no_grad()
def inference_ldm2(config: InferenceConfig):
    """Run inference on the ablation model: LDM-2."""
    pass


@app.command("ldm3")
@dataclass_cli
@torch.no_grad()
def inference_ldm3(config: InferenceConfig):
    """Run inference on the ablation model: LDM-3."""
    pass


if __name__ == "__main__":
    app()
