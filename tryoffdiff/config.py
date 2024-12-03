from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
import inspect
import json
import os
from pathlib import Path

import typer


def dataclass_cli(func):
    """
    Converts a function taking a dataclass as its first argument into a
    dataclass that can be called via `typer` as a CLI.

    Modified from: https://gist.github.com/tbenthompson/9db0452445451767b59f5cb0611ab483
    """

    # The dataclass type is the first argument of the function.
    sig = inspect.signature(func)
    param = list(sig.parameters.values())[0]
    cls = param.annotation
    assert is_dataclass(cls)

    def wrapped(**kwargs):
        # Convert the kwargs directly to the dataclass instance.
        arg = cls(**kwargs)

        # Actually call the entry point function.
        return func(arg)

    # Construct the CLI signature from the dataclass fields.
    # Remove the first argument (self) from the dataclass __init__ signature.
    signature = inspect.signature(cls.__init__)
    parameters = list(signature.parameters.values())
    if len(parameters) > 0 and parameters[0].name == "self":
        del parameters[0]

    wrapped.__signature__ = signature.replace(parameters=parameters)

    # The docstring is used for the explainer text in the CLI.
    wrapped.__doc__ = func.__doc__ + "\n" + ""

    return wrapped


def load_training_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "training_config.json")
    try:
        with open(config_path) as f:
            train_cfg = json.load(f)
    except FileNotFoundError as err:
        raise ValueError(f"Training config file not found at {config_path}") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Error decoding JSON from {config_path}") from err

    expected_keys = ["data_dir", "train_img_dir", "val_img_dir", "mixed_precision", "model_class_name"]
    for key in expected_keys:
        if key not in train_cfg:
            raise KeyError(f"Missing expected key '{key}' in training config")

    return train_cfg


@dataclass
class TrainingConfig:
    save_dir: str = typer.Option(..., help="Directory to save model checkpoints and logs.")
    data_dir: str = typer.Option(..., help="Directory containing the dataset.")
    model_class_name: str = typer.Option(..., help="Model class name to instantiate.")
    train_batch_size: int = typer.Option(16, help="Batch size for training.")
    num_epochs: int = typer.Option(500, help="Number of training epochs")
    start_model: str = typer.Option(None, help="Path to a pre-trained model to start from.")
    gradient_accumulation_steps: int = typer.Option(
        1, help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    learning_rate: float = typer.Option(1e-4, help="Learning rate for the optimizer.")
    lr_warmup_steps: int = typer.Option(
        1000, help="Number of steps for the warmup phase of the learning rate scheduler."
    )
    save_model_epochs: int = typer.Option(50, help="Save model checkpoint every n epochs.")
    mixed_precision: str = typer.Option("fp16", help="Mixed precision mode, e.g., 'fp16' or 'no'")
    logger: str = typer.Option("tensorboard", help="Logging backend to use.")
    device: str = typer.Option("cuda", help="Device to use for training. Options are 'cuda' or 'cpu'.")
    checkpoint_every_n_epochs: int = typer.Option(50, help="Number of epochs between checkpoint saving.")
    resume_from_checkpoint: str = typer.Option("", help="Path to checkpoint directory.")

    train_img_dir: Path = field(init=False)
    val_img_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    model_id: str = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self):
        """
        Post-initialization method to set up derived attributes.

        This method is automatically called after the object is initialized.
        It sets up the directory structure and generates a unique model identifier.
        """
        self.train_img_dir = Path(self.data_dir) / "train/"
        self.val_img_dir = Path(self.data_dir) / "test/"
        self.log_dir = Path(self.save_dir) / "logs/"
        self.model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.save_dir) / f"model_{self.model_id}/"
        typer.secho(message=f"ModelID:{self.model_id}", fg=typer.colors.YELLOW)


@dataclass
class InferenceConfig:
    model_dir: str = typer.Option(..., help="Directory containing the saved model")
    model_filename: str = typer.Option(..., help="Filename of the model, e.g. 'model_epoch_100.pth'")
    batch_size: int = typer.Option(..., help="Batch size for inference")
    num_inference_steps: int = typer.Option(..., help="Number of denoising steps.")
    device: str = typer.Option("cuda", help="Device to run the inference on, e.g., 'cuda' or 'cpu'")
    seed: int = typer.Option(42, help="Seed for random number generation")
    guidance_scale: float = typer.Option(None, help="")
    vis_intermediate_steps: bool = typer.Option(False, "-v", help="When provided, plots 10 intermediate results.")
    vis_with_caption: bool = typer.Option(False, "-c", help="When provided, adds caption to figures.")
    process_all_samples: bool = typer.Option(
        False, "--all", help="When provided, inference is run for all test samples."
    )

    output_dir: Path = field(init=False)
    model_path: Path = field(init=False)
    scheduler_dir: Path = field(init=False)

    # Fields populated from training config
    data_dir: Path = field(init=False)
    train_img_dir: Path = field(init=False)
    val_img_dir: Path = field(init=False)
    mixed_precision: str = field(init=False)
    model_class: str = field(init=False)

    def __post_init__(self):
        self.output_dir = Path(self.model_dir) / "preds/"
        self.model_path = Path(self.model_dir) / self.model_filename
        self.scheduler_dir = Path(self.model_dir) / "scheduler/"

        # Load common args from the training config
        train_cfg = load_training_config(self.model_dir)
        self.data_dir = Path(train_cfg["data_dir"])
        self.train_img_dir = Path(train_cfg["train_img_dir"])
        self.val_img_dir = Path(train_cfg["val_img_dir"])
        self.mixed_precision = train_cfg["mixed_precision"]
        self.model_class = train_cfg["model_class_name"]
