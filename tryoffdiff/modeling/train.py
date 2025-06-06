from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path

from accelerate import Accelerator
from diffusers import EulerDiscreteScheduler, PNDMScheduler, get_linear_schedule_with_warmup
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from loguru import logger
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import typer

from tryoffdiff.config import TrainingConfig, dataclass_cli
from tryoffdiff.features import DressCodeDataset, VitonVAESigLIPDataset
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
        # Filter out image encoder parameters (keys containing 'image_encoder')
        state_dict = accelerator.unwrap_model(model).state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if "image_encoder" not in k.lower()}
        torch.save(filtered_state_dict, f"{config.output_dir}/model_epoch_{epoch + 1}.pth")
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

        # Read epoch number from the saved metadata
        epoch_file = os.path.join(config.resume_from_checkpoint, "epoch.txt")
        if os.path.exists(epoch_file):
            with open(epoch_file) as f:
                starting_epoch = int(f.read().strip()) + 1
            logger.info(f"Starting from epoch {starting_epoch}")
            global_step = starting_epoch * len(train_loader)
        else:
            raise ValueError(f"Epoch metadata not found in {config.resume_from_checkpoint}")

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
            ckpt_dir = os.path.join(config.output_dir, "checkpoint")
            accelerator.save_state(ckpt_dir)

            # Save epoch metadata to track the current checkpoint
            epoch_file = os.path.join(ckpt_dir, "epoch.txt")
            with open(epoch_file, "w") as f:
                f.write(str(epoch))

            logger.info(f"Saved checkpoint for epoch {epoch} at {ckpt_dir}")

        # Save model and noise scheduler if saving conditions are met
        maybe_save_model(model, accelerator, noise_scheduler, epoch, config)

    accelerator.end_training()


@app.command("tryoffdiffv2")
@dataclass_cli
def train_tryoffdiffv2(config: TrainingConfig):
    """Train TryOffDiffv2 models on Dress Code dataset for Virtual Try-Off task.

    Notes:
        - See `TrainingConfig` class for all available arguments.
    """

    def get_dresscode_dataset(cfg):
        if cfg.dataset_type not in ["dresscode", "dc-upperbody", "dc-lowerbody", "dc-dresses"]:
            raise ValueError("Provide one of [dresscode, dc-upperbody, dc-lowerbody, dc-dresses]")
        category = None if cfg.dataset_type == "dresscode" else cfg.dataset_type.split("-")[1]
        return DressCodeDataset(root_dir=cfg.train_img_dir, category=category)

    dataset_map = {
        "TryOffDiffv2Single": get_dresscode_dataset,
        "TryOffDiffv2Multi": get_dresscode_dataset,
    }

    def create_param_groups(model, base_lr=1e-4):
        """
        Create parameter groups for different model variants.
        Handles both pretrained and from-scratch UNets, and different adapter architectures.
        """
        # Initialize parameter groups
        param_groups = []
        param_counts = {}

        def add_to_param_groups(params_decay, params_no_decay, group_name, lr_multiplier=1.0):
            if len(params_decay) > 0:
                param_groups.append(
                    {
                        "params": params_decay,
                        "weight_decay": 0.01,
                        "lr": base_lr * lr_multiplier,
                        "name": f"{group_name}_decay",  # For debugging
                    }
                )
            if len(params_no_decay) > 0:
                param_groups.append(
                    {
                        "params": params_no_decay,
                        "weight_decay": 0.0,
                        "lr": base_lr * lr_multiplier,
                        "name": f"{group_name}_no_decay",  # For debugging
                    }
                )

            # Store counts for debugging
            param_counts[f"{group_name}_decay"] = sum(p.numel() for p in params_decay)
            param_counts[f"{group_name}_no_decay"] = sum(p.numel() for p in params_no_decay)

        # Collect parameters by component
        unet_params_decay = []
        unet_params_no_decay = []
        adapter_params_decay = []
        adapter_params_no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine parameter type
            is_no_decay = any(nd in name for nd in ["bias", "LayerNorm.weight", "norm"])

            # Sort parameters into appropriate groups
            if "unet" in name:
                if is_no_decay:
                    unet_params_no_decay.append(param)
                else:
                    unet_params_decay.append(param)
            elif any(adapter_name in name for adapter_name in ["proj", "proj1", "proj2"]):
                if is_no_decay:
                    adapter_params_no_decay.append(param)
                else:
                    adapter_params_decay.append(param)

        # Determine if UNet is pretrained or from scratch
        is_pretrained_unet = config.model_class_name in [
            "Ablation2",
            "Ablation4",
            "Ablation11",
            "Ablation12",
            "Ablation14",
        ]

        # Add parameter groups with appropriate learning rates
        # UNet parameters
        unet_lr_multiplier = 1.0 if is_pretrained_unet else 5.0  # Higher LR for from-scratch UNet
        add_to_param_groups(unet_params_decay, unet_params_no_decay, "unet", unet_lr_multiplier)

        # Adapter parameters (if present)
        if len(adapter_params_decay) > 0 or len(adapter_params_no_decay) > 0:
            add_to_param_groups(adapter_params_decay, adapter_params_no_decay, "adapter", lr_multiplier=10.0)

        # Print parameter counts for debugging
        print("\nParameter group sizes:")
        total_params = 0
        for name, count in param_counts.items():
            print(f"{name}: {count:,} parameters")
            total_params += count
        print(f"Total trainable parameters: {total_params:,}")

        return param_groups

    # Performance optimizations
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.cache_size_limit = 128
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
    torch.backends.cudnn.deterministic = False  # Disable deterministic mode for better performance

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=config.logger,
        project_dir=config.log_dir,
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)  # Ensure output directory exists
        accelerator.init_trackers(config.model_id)  # initialize trackers
        save_config(config)  # Save training configuration

    # Handle dataset loading
    try:
        dataset_fn = dataset_map.get(config.model_class_name)
        if not dataset_fn:
            raise ValueError(f"Invalid model class name: {config.model_class_name}")
        train_set = dataset_fn(config)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}") from e

    train_loader = DataLoader(
        train_set,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # Enable pinned memory for faster data transfer
        persistent_workers=True,  # Keep worker processes alive between epochs
        drop_last=True,  # Avoid incomplete batches for better performance
        generator=torch.Generator().manual_seed(42),  # Reproducibility
    )

    # Model initialization and optimization
    model = create_model(config.model_class_name)
    model = torch.compile(model)

    # Optimizer and learning rate scheduler
    param_groups = create_param_groups(model, base_lr=config.learning_rate)
    optimizer = torch.optim.AdamW(param_groups, fused=True)
    total_training_steps = len(train_loader) * config.num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=total_training_steps * 0.05,
        num_training_steps=total_training_steps,
    )

    # Noise scheduler config
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler", use_karras_sigmas=True
    )

    # Prepare for distributed training
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    # Initialize training state
    global_step = 0
    loss_fn = nn.MSELoss()
    model.train()
    torch.cuda.empty_cache()

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        total_loss = 0

        for batch in train_loader:
            if config.dataset_type == "dresscode":
                cond_img, latents, cls = batch
            elif config.dataset_type.startswith("dc-"):
                cond_img, latents = batch
            else:  # Assuming VITON-HD dataset
                latents, cond_emb = batch

            # Sample noise to add to the latents
            noise = torch.randn_like(latents, device=accelerator.device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=accelerator.device
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Forward & backward pass, update model parameters
            if config.dataset_type == "dresscode":
                pred = model(noisy_latents, timesteps, cond_img, cls)
            elif config.dataset_type.startswith("dc-"):
                pred = model(noisy_latents, timesteps, cond_img)
            else:
                pred = model(noisy_latents, timesteps, cond_emb)
            loss = loss_fn(pred, noise)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # logging
            total_loss += loss.detach().float()
            logs = {"lr": lr_scheduler.get_last_lr()[0], "batch_loss": loss.detach().item()}
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"step: {global_step}, lr: {logs['lr']:.1e}, loss: {logs['batch_loss']:.5f}")
            accelerator.log(logs, step=global_step)
            global_step += 1

        progress_bar.close()

        # Log epoch-level metrics
        accelerator.log({"epoch_loss": total_loss.item() / len(train_loader), "epoch": epoch}, step=global_step)

        # Save checkpoint based on checkpoint_every_n_epochs
        if epoch % config.checkpoint_every_n_epochs == 0 and accelerator.is_main_process:
            ckpt_dir = os.path.join(config.output_dir, "checkpoint")
            accelerator.save_state(ckpt_dir)

            # Save epoch metadata to track the current checkpoint
            epoch_file = os.path.join(ckpt_dir, "epoch.txt")
            with open(epoch_file, "w") as f:
                f.write(str(epoch))

            logger.info(f"Saved checkpoint for epoch {epoch} at {ckpt_dir}")

        # Save model and noise scheduler if saving conditions are met
        maybe_save_model(model, accelerator, noise_scheduler, epoch, config)

    accelerator.end_training()


if __name__ == "__main__":
    app()
