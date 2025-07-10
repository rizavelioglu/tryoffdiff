from glob import glob
import os
import subprocess
import sys

from cleanfid import fid
from DISTS_pytorch import DISTS
import numpy as np

# pyiqa requires older version of packages, causing dependency issues during install. Therefore, we install it here.
# Specifically, it requires accelerate=1.1.0 and transformers=4.37.2.
try:
    import pyiqa
except ImportError:
    print("pyiqa not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyiqa==0.1.14.1", "--no-deps"])
    import pyiqa
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import typer

app = typer.Typer(pretty_exceptions_enable=False)


class ImageDataset(Dataset):
    def __init__(self, gt_dir: str, pred_dir: str, resize_to: tuple[int, int] = (1024, 768)):
        typer.secho(f"Predictions dir: {pred_dir}.", fg=typer.colors.BRIGHT_BLUE)

        # Check if ground-truth and prediction directories match
        if not check_directory_contents(gt_dir, pred_dir):
            typer.secho("Proceeding with the available predictions...", fg=typer.colors.YELLOW)
            typer.confirm("Do you want to continue despite the mismatch?", abort=True)

        self.pred_files = sorted(glob(f"{pred_dir}/*.[jp][pn]g"))  # allow both jpg and png
        self.gt_files = [
            os.path.join(
                gt_dir,
                os.path.basename(f).replace(".png", ".jpg").replace("_0", "_1")
                if "dresscode" in gt_dir
                else os.path.basename(f).replace(".png", ".jpg"),
            )
            for f in self.pred_files
        ]

        # Create transforms based on the first predicted image dimensions
        self.gt_transform, self.pred_transform = self._create_transforms(resize_to)

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt = self.gt_transform(read_image(self.gt_files[idx]))
        pred = self.pred_transform(read_image(self.pred_files[idx]))
        return gt, pred

    def _create_transforms(self, resize_to):
        """Create and return gt and pred transforms based on resize dimensions."""
        # Get dimensions from the first predicted image
        h, w = read_image(self.pred_files[0]).shape[1:]

        # Define transform for ground-truth images
        self.gt_transform = transforms.Compose(
            [
                transforms.ToDtype(dtype=torch.float32, scale=True),
                transforms.Resize(resize_to),
            ]
        )

        # Define transform for predicted images based on their original dimensions
        if resize_to == (1024, 768):
            if (h, w) in [(256, 256), (256, 176), (512, 384)]:
                typer.secho(f"Resizing predictions ({h}x{w}) to 1024x768.", fg=typer.colors.YELLOW)
                self.pred_transform = transforms.Compose(
                    [
                        transforms.ToDtype(dtype=torch.float32, scale=True),
                        transforms.Resize(resize_to),
                    ]
                )
            elif (h, w) in [(512, 512), (128, 128)]:
                typer.secho(
                    f"Resizing predictions ({h}x{w}) to 1024x1024 and cropping to 1024x768.", fg=typer.colors.YELLOW
                )
                self.pred_transform = transforms.Compose(
                    [
                        transforms.ToDtype(dtype=torch.float32, scale=True),
                        transforms.Resize((1024, 1024)),
                        transforms.CenterCrop(resize_to),
                    ]
                )
            elif (h, w) == (1024, 1024):
                typer.secho(f"Cropping predictions ({h}x{w}) to 1024x768.", fg=typer.colors.YELLOW)
                self.pred_transform = transforms.Compose(
                    [
                        transforms.ToDtype(dtype=torch.float32, scale=True),
                        transforms.CenterCrop(resize_to),
                    ]
                )
            elif (h, w) == (1024, 768):
                self.pred_transform = transforms.ToDtype(dtype=torch.float32, scale=True)
            else:
                raise ValueError(f"Unexpected image size: ({h}, {w})")

        # Required for DISTS metric, which uses VGG16 that expects inputs in (224x224).
        elif resize_to == (341, 256):
            if (h, w) in [(256, 256), (256, 176), (512, 384)]:
                typer.secho(f"Resizing predictions ({h}x{w}) to 341x256.", fg=typer.colors.YELLOW)
                self.pred_transform = transforms.Compose(
                    [
                        transforms.Resize(resize_to),
                        transforms.ToDtype(dtype=torch.float32, scale=True),
                    ]
                )
            elif (h, w) == (512, 512):
                typer.secho(
                    f"Cropping predictions ({h}x{w}) to 512x384 and resizing to 341x256.", fg=typer.colors.YELLOW
                )
                self.pred_transform = transforms.Compose(
                    [
                        transforms.CenterCrop((512, 384)),
                        transforms.Resize(resize_to),
                        transforms.ToDtype(dtype=torch.float32, scale=True),
                    ]
                )
            elif (h, w) == (1024, 1024):
                typer.secho(f"Cropping predictions ({h}x{w}) to 341x256.", fg=typer.colors.YELLOW)
                self.pred_transform = transforms.Compose(
                    [
                        transforms.CenterCrop((1024, 768)),
                        transforms.Resize(resize_to),
                        transforms.ToDtype(dtype=torch.float32, scale=True),
                    ]
                )
            elif (h, w) == (1024, 768):
                typer.secho(f"Resizing predictions ({h}x{w}) to 341x256.", fg=typer.colors.YELLOW)
                self.pred_transform = transforms.Compose(
                    [
                        transforms.Resize(resize_to),
                        transforms.ToDtype(dtype=torch.float32, scale=True),
                    ]
                )
            else:
                raise ValueError(f"Unexpected image size: ({h}, {w})")

        else:
            raise ValueError(f"Unsupported resize dimensions: {resize_to}")

        return self.gt_transform, self.pred_transform


def check_directory_contents(gt_dir: str, pred_dir: str) -> bool:
    num_gt_files = len(os.listdir(gt_dir))
    num_pred_files = len(os.listdir(pred_dir))

    if num_gt_files != num_pred_files:
        typer.secho(
            "Warning: Mismatch in directory contents!\n"
            f" - Ground truth directory: {num_gt_files} files\n"
            f" - Prediction directory: {num_pred_files} files",
            fg=typer.colors.YELLOW,
        )
        return False
    return True


def print_results(metric_names: list[str], metric_values: list[float], source: str = None):
    typer.secho(source, fg=typer.colors.YELLOW)
    typer.echo("   Metric   |   Value  ")
    typer.echo("------------|----------")
    for name, value in zip(metric_names, metric_values, strict=True):
        typer.echo(f"{name:<11} | {value:.4f}")
    typer.echo("-----------------------")


class PYIQAEvaluator:
    """Image Quality Assessment evaluator using multiple metrics from PYIQA."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics, self.metric_names = self._initialize_metrics()
        self._reset_state()

    def _initialize_metrics(self) -> tuple[list[torch.nn.Module], list[str]]:
        metrics = [
            pyiqa.create_metric("ssim"),
            pyiqa.create_metric("ms_ssim"),
            pyiqa.create_metric("cw_ssim"),
            pyiqa.create_metric("lpips"),
        ]
        metric_names = [
            "\u2191 SSIM",
            "\u2191 MS-SSIM",
            "\u2191 CW-SSIM",
            "\u2193 LPIPS",
        ]
        return [metric.to(self.device) for metric in metrics], metric_names

    def _reset_state(self) -> None:
        """Reset accumulated values."""
        self.metric_values = torch.zeros(len(self.metrics), device=self.device)
        self.total = 0

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        """Update accumulated metrics with a new batch of images.

        Args:
            gt: Ground truth images tensor, shape (B, C, H, W)
            pred: Predicted images tensor, shape (B, C, H, W)
        """
        for i, metric in enumerate(self.metrics):
            self.metric_values[i] += metric(gt, pred).sum()
        self.total += gt.shape[0]

    def compute(self) -> list[float]:
        """Compute final averaged metrics."""
        return (self.metric_values / self.total).cpu().tolist()

    def reset(self) -> None:
        """Reset the evaluator to its initial state."""
        self._reset_state()


def compute_cleanfid(gt_dir, pred_dir):
    """
    Computes FID, CLIP-FID, and KID metrics between two directories (ground-truth and predicted images)
    using the `clean-fid` package: https://github.com/GaParmar/clean-fid.

    Args:
        gt_dir (str): Path to the directory containing ground-truth images.
        pred_dir (str): Path to the directory containing predicted images.
    """
    metric_names = ["\u2193 FID", "\u2193 CLIP-FID", "\u2193 KID"]
    metric_values = [
        fid.compute_fid(gt_dir, pred_dir, num_workers=8, verbose=False),
        fid.compute_fid(gt_dir, pred_dir, num_workers=8, verbose=False, model_name="clip_vit_b_32"),
        fid.compute_kid(gt_dir, pred_dir, num_workers=8, verbose=False),
    ]

    # Print results in a formatted way
    print_results(metric_names, metric_values, source="`clean-fid`")


@torch.no_grad()
def compute_dists(gt_dir: str, pred_dir: str, batch_size: int, num_workers: int):
    """Computes DISTS (Deep Image Structure and Texture Similarity) between ground-truth and predicted images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_names = ["\u2193 DISTS"]
    metric = DISTS().to(device)

    dataset = ImageDataset(gt_dir, pred_dir, resize_to=(341, 256))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    metric_values = []

    # Iterate over dataloader and compute metrics
    for gt_batch, pred_batch in tqdm(dataloader, desc="Evaluating Batches"):
        gt_batch, pred_batch = gt_batch.to(device), pred_batch.to(device)

        # Compute DISTS score for the current batch
        value = metric(gt_batch, pred_batch).mean().item()  # Compute batch mean
        metric_values.append(value)

    # Print results in a formatted way
    print_results(metric_names, [np.mean(metric_values)], source="`DISTS_pytorch`")


@app.command()
@torch.no_grad()
def main(
    gt_dir: str = typer.Option(..., help="Path to ground-truth directory."),
    pred_dir: str = typer.Option(..., help="Path to predictions directory."),
    batch_size: int = typer.Option(32, help="Batch size for processing."),
    num_workers: int = typer.Option(4, help="Number of worker processes for data loading."),
):
    # Initialize evaluator
    evaluator = PYIQAEvaluator()

    # Initialize dataloader
    dataset = ImageDataset(gt_dir, pred_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    # Accumulate metrics across all batches
    for gt_batch, pred_batch in tqdm(dataloader, desc="Evaluating Batches"):
        gt_batch = gt_batch.to(evaluator.device, non_blocking=True)
        pred_batch = pred_batch.to(evaluator.device, non_blocking=True)

        # Update metrics
        evaluator.update(pred_batch, gt_batch)

    # Compute final results
    results = evaluator.compute()
    print_results(evaluator.metric_names, results, source="`pyiqa`")

    # Compute metrics using `clean-fid`
    compute_cleanfid(gt_dir=gt_dir, pred_dir=pred_dir)

    # Compute DISTS metric using `DISTS_pytorch`
    compute_dists(gt_dir=gt_dir, pred_dir=pred_dir, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    app()
