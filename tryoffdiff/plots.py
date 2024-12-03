import os

from matplotlib import pyplot as plt
from PIL import Image
import torch
import torchvision


def should_visualize(i, config, scheduler):
    is_intermediate = config.vis_intermediate_steps and i % max(1, config.num_inference_steps // 10) == 0
    is_final = i == len(scheduler.timesteps)
    return is_intermediate or is_final


def visualize_results(images, titles, output_path, transform_func=None, show_titles=False, figsize=(20, 8)):
    """Visualize a list of images with corresponding titles and save the result.

    Args:
        images: List of image tensors to visualize.
        titles: List of titles for each image.
        output_path: Path to save the output image.
        transform_func: Function to transform images before visualization.
        show_titles: Whether to show titles for each image.
        figsize: Figure size (width, height) in inches.
    """

    os.makedirs(output_path.parent, exist_ok=True)

    if transform_func:
        images = transform_func(images)

    if show_titles:
        _visualize_with_titles(images, titles, output_path, figsize)
    else:
        grids = [torchvision.utils.make_grid(img, nrow=img.shape[0], normalize=True, scale_each=True) for img in images]
        grids = torch.cat(grids, dim=1)
        torchvision.utils.save_image(grids, output_path)


def _visualize_with_titles(images, titles, output_path, figsize):
    if len(images) != len(titles):
        raise ValueError(f"Number of images must match number of titles. images: {len(images)}, titles: {len(titles)}")

    # Create a figure with subplots for each grid
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    # Make axs iterable if there's only one subplot
    axs = [axs] if len(images) == 1 else axs

    for i, (img, title) in enumerate(zip(images, titles, strict=True)):
        grid = torchvision.utils.make_grid(img, nrow=img.shape[0], normalize=True, scale_each=True)

        # Convert the grid tensor to a numpy array and transpose it for plotting
        grid_np = grid.cpu().numpy().transpose((1, 2, 0))

        # Plot the grid on the corresponding subplot
        axs[i].imshow(grid_np)
        axs[i].set_title(title)
        axs[i].axis("off")  # Turn off axis labels

    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200, pad_inches=0.1)
    plt.close(fig)


def convert_images_to_gif(image_files: list[str], output_path: str, duration: int = 500, loop: int = 0) -> None:
    """
    Creates a GIF from a list of image file paths.

    Args:
        image_files (List[str]): A list of file paths to the images.
        output_path (str): The file path where the GIF will be saved. Must include the .gif extension.
        duration (int, optional): Duration for each frame in milliseconds. Default is 500 ms.
        loop (int, optional): The number of times the GIF should loop. Default is 0 (infinite loop).

    Raises:
        FileNotFoundError: If any of the image files cannot be found.
    """
    # Load images
    images = []
    for file in image_files:
        try:
            img = Image.open(file)
            images.append(img)
        except FileNotFoundError as e:
            print(f"Error: {file} not found.")
            raise e

    # Ensure at least one image is loaded
    if not images:
        raise ValueError("No valid images were loaded. Please check the image file paths.")

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=loop,
        duration=duration,
    )
