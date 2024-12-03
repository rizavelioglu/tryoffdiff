from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path

from accelerate import Accelerator
from diffusers import PNDMScheduler
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from loguru import logger
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import typer

from tryoffdiff.config import TrainingConfig, dataclass_cli
from tryoffdiff.features import VitonVAESigLIPDataset
from tryoffdiff.modeling.model import create_model

app = typer.Typer(pretty_exceptions_enable=False)


def save_config(config):
    def serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif is_dataclass(obj):
            return asdict(obj)
        return obj

    config_dict = {k: serialize(v) for k, v in config.__dict__.items()}

    # Add class variables that are not instance variables
    for k in dir(config):
        if not k.startswith("__") and k not in config_dict:
            config_dict[k] = serialize(getattr(config, k))

    with open(config.output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=serialize)


def map_precision_to_dtype(precision):
    match precision:
        case "no":
            dtype = torch.float32
        case "fp16":
            dtype = torch.float16
        case _:
            raise ValueError("Invalid 'mixed_precision' type, expected one of ['no', 'fp16'].")
    return dtype


def should_save_model(accelerator, config, epoch):
    """Determine if model should be saved at current epoch.

    Returns:
        bool: True if model should be saved, False otherwise
    """
    if not accelerator.is_main_process:
        return False

    is_intermediate = (epoch + 1) % config.save_model_epochs == 0
    is_last_epoch = epoch == config.num_epochs - 1
    return is_intermediate or is_last_epoch


def maybe_save_model(model, accelerator, noise_scheduler, epoch, config):
    """Save model checkpoint and noise scheduler if saving conditions are met.

    Args:
        model: The model to save
        accelerator: Accelerator instance for distributed training
        noise_scheduler: The noise scheduler to save
        epoch: Current training epoch (0-based)
        config: Config object containing output_dir, save_model_epochs, and num_epochs

    Notes:
        Alternative saving method is:
        >>> accelerator.unwrap_model(model).unet.save_pretrained(f'{config.output_dir}/model_epoch_{epoch + 1}')
        which saves both `config.json` & `diffusion_pytorch_model.safetensors`.
        The model can then be loaded using diffusers' `.from_pretrained()` method.
    """
    if should_save_model(accelerator, config, epoch):
        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            f"{config.output_dir}/model_epoch_{epoch + 1}.pth",
        )
        noise_scheduler.save_pretrained(f"{config.output_dir}/scheduler")


@app.command("tryoffdiff")
@dataclass_cli
def train_tryoffdiff(config: TrainingConfig):
    """Train the proposed TryOffDiff on VITON-HD dataset for adapting StableDiffusion-v1.4 to Virtual Try-Off task.

    Notes:
        - Add `resume_from_checkpoint` arg with a full path to the checkpoint directory to resume training.
        - See `TrainingConfig` class for all available arguments.
    """
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=config.logger,
        project_dir=config.log_dir,
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(config.model_id)

    # Prepare dataloader
    train_set = VitonVAESigLIPDataset(root_dir=config.train_img_dir)
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, num_workers=4)

    # Model, scheduler, and optimizer setup
    model = create_model(config.model_class_name)
    noise_scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.num_epochs),
        num_cycles=1,
    )

    # Save config
    save_config(config)

    # Prepare for distributed training
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    # Initialize training state
    starting_epoch = 0
    global_step = 0
    loss_fn = nn.MSELoss()

    # Handle checkpoint loading
    if config.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        accelerator.load_state(config.resume_from_checkpoint)

        # Extract epoch number and resume from next epoch
        starting_epoch = int(Path(config.resume_from_checkpoint).stem.replace("epoch_", "")) + 1
        logger.info(f"Starting from epoch {starting_epoch}")
        global_step = starting_epoch * len(train_loader)

    model.train()

    for epoch in range(starting_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        total_loss = 0

        for latents, cond_emb in train_loader:
            # Sample noise to add to the latents
            noise = torch.randn_like(latents)
            timesteps = (
                torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],))
                .long()
                .to(accelerator.device)
            )
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the model prediction for the noise
            with accelerator.accumulate(model):
                pred = model(noisy_latents, timesteps, cond_emb)
                loss = loss_fn(pred, noise)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.detach().float()
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        progress_bar.close()

        # Log epoch-level metrics
        accelerator.log({"train_loss": total_loss.item() / len(train_loader), "epoch": epoch}, step=global_step)

        # Save checkpoint based on checkpoint_every_n_epochs
        if epoch % config.checkpoint_every_n_epochs == 0 and accelerator.is_main_process:
            output_dir = os.path.join(config.output_dir, f"epoch_{epoch}")
            accelerator.save_state(output_dir)
            logger.info(f"Saved checkpoint for epoch {epoch}")

        # Save model and noise scheduler if saving conditions are met
        maybe_save_model(model, accelerator, noise_scheduler, epoch, config)

    accelerator.end_training()


@app.command("autoencoder")
@dataclass_cli
def train_autoencoder(config: TrainingConfig):
    """Train the ablation model: Autoencoder"""
    pass


@app.command("pixel")
@dataclass_cli
def train_pixel(config: TrainingConfig):
    """Train the ablation model: PixelModel"""
    pass


@app.command("ldm1")
@dataclass_cli
def train_ldm1(config: TrainingConfig):
    """Train the ablation model: LDM-1"""
    pass


@app.command("ldm2")
@dataclass_cli
def train_ldm2(config: TrainingConfig):
    """Train the ablation model: LDM-2"""
    pass


@app.command("ldm3")
@dataclass_cli
def train_ldm3(config: TrainingConfig):
    """Train the ablation model: LDM-3"""
    pass


if __name__ == "__main__":
    app()
