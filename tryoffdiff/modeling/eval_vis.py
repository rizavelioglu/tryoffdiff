import tkinter as tk

from PIL import ImageTk
import pyiqa
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import typer

from tryoffdiff.modeling.eval import ImageDataset

app = typer.Typer()


class ImageEvalGUI:
    def __init__(self, gt_dir: str, pred_dir: str):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.device = "cuda"
        self.dataloader = self.prepare_dataloader()
        self.metrics, self.metric_names = self.prepare_metrics()

        self.root = tk.Tk()
        self.root.title("Evaluation")
        self.root.geometry("1800x1200")

        self.frame = tk.Frame(self.root)
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.frame.columnconfigure(1, weight=1)

        self.gt_label = tk.Label(self.frame, text="Ground Truth")
        self.gt_label.grid(column=0, row=1, pady=10)
        self.pred_label = tk.Label(self.frame, text="Prediction")
        self.pred_label.grid(column=2, row=1, pady=10)

        self.metrics_frame = tk.Frame(self.frame)
        self.metrics_frame.grid(column=1, row=1, pady=20)

        self.next_button = tk.Button(self.frame, text="Next (n)", command=self.next_image)
        self.next_button.grid(column=2, row=2, pady=10)
        self.quit_button = tk.Button(self.frame, text="Quit (q)", command=self.quit)
        self.quit_button.grid(column=0, row=2, pady=10)

        self.root.bind("<Key>", self.handle_key_press)

    def prepare_dataloader(self):
        dataset = ImageDataset(self.gt_dir, self.pred_dir)
        return iter(DataLoader(dataset, batch_size=1))

    def prepare_metrics(self):
        metrics = [
            pyiqa.create_metric("ssim"),
            pyiqa.create_metric("ms_ssim"),
            pyiqa.create_metric("cw_ssim"),
            pyiqa.create_metric("lpips"),
            pyiqa.create_metric("dists"),
        ]
        metric_names = [
            "\u2191 SSIM",
            "\u2191 MS-SSIM",
            "\u2191 CW-SSIM",
            "\u2193 LPIPS",
            "\u2193 DISTS",
        ]
        return [metric.to(self.device) for metric in metrics], metric_names

    @torch.no_grad()
    def compute_metrics(self, gt, pred):
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        # Create two columns for metric names and values
        name_column = tk.Frame(self.metrics_frame)
        name_column.pack(side=tk.LEFT, padx=(0, 10))
        value_column = tk.Frame(self.metrics_frame)
        value_column.pack(side=tk.LEFT)

        for name, metric in zip(self.metric_names, self.metrics, strict=False):
            result = metric(pred, gt).item()

            # Add metric name to the left column
            tk.Label(name_column, text=f"{name}:", font=("Courier", 12), anchor="w").pack(fill=tk.X, pady=2)

            # Add metric value to the right column
            tk.Label(value_column, text=f"{result:.4f}", font=("Courier", 12), anchor="w").pack(fill=tk.X, pady=2)

    def prepare_image(self, tensor):
        img = transforms.ToPILImage()(tensor.squeeze())
        return ImageTk.PhotoImage(img)

    def display_image(self):
        # Compute metrics
        gt, pred = next(self.dataloader)
        self.compute_metrics(gt.to(self.device), pred.to(self.device))

        gt_img = self.prepare_image(gt)
        pred_img = self.prepare_image(pred)

        self.gt_label.config(image=gt_img, compound=tk.TOP)
        self.gt_label.image = gt_img
        self.pred_label.config(image=pred_img, compound=tk.TOP)
        self.pred_label.image = pred_img

    def next_image(self):
        try:
            self.display_image()
        except StopIteration:
            self.quit()

    def quit(self):
        self.root.quit()

    def handle_key_press(self, event):
        if event.char.lower() == "n":
            self.next_image()
        elif event.char.lower() == "q":
            self.quit()

    def run(self):
        self.display_image()
        self.root.mainloop()


@app.command()
def main(
    gt_dir: str = typer.Option(..., help="Path to ground-truth directory."),
    pred_dir: str = typer.Option(..., help="Path to predictions directory."),
):
    """Evaluation visualized."""
    gui = ImageEvalGUI(gt_dir=gt_dir, pred_dir=pred_dir)
    gui.run()


if __name__ == "__main__":
    app()
