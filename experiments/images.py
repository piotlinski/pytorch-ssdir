import argparse

from pathlib import Path
import PIL.Image as PILImage
from matplotlib import patches
import matplotlib.pyplot as plt

import numpy as np
from pytorch_ssd.data.datasets import datasets
import torch
from tqdm import tqdm
from pytorch_ssdir.modeling.model import SSDIR
from pytorch_ssd.modeling.visualize import denormalize


def save_image(image: torch.Tensor, path: str):
    pilimage = PILImage.fromarray((image.numpy() * 255).astype(np.uint8))
    pilimage.save(path)


def draw_boxes_and_save(image: torch.Tensor, boxes: torch.Tensor, path: str):
    npimage = (image.numpy() * 255).astype(np.uint8)
    boxes = boxes.numpy() * npimage.shape[0]
    size = 3 / 231 * 300
    fig, ax = plt.subplots(figsize=(size, size), dpi=100)
    ax.set_axis_off()
    ax.imshow(npimage)
    for box in boxes:
        x, y, w, h = box
        rect = patches.Rectangle((int(x - w / 2), int(y - h / 2)), w, h, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    fig.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0, dpi=100)
    plt.close(fig)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to the dataset")
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path to directory where results should be stored",
        default="results/images",
        type=Path,
    )
    parser.add_argument("--checkpoint", help="Path to SSDIR checkpoint")
    parser.add_argument("--dataset_name", help="Name of the dataset")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssdir = SSDIR.load_from_checkpoint(args.checkpoint).eval().to(device)
    ssdir.dataset = datasets[args.dataset_name]
    ssdir.data_dir = args.data_dir
    ssdir.batch_size = 1
    data_loader = ssdir.val_dataloader()
    for idx, (image, *_) in enumerate(tqdm(data_loader), start=1):
        if idx % 5 != 0:
            continue
        name = f"{idx:04d}"
        images_dir = args.output_dir / name
        images_dir.mkdir(exist_ok=True, parents=True)
        save_image(
            denormalize(image[0].permute(1, 2, 0), pixel_mean=ssdir.pixel_means, pixel_std=ssdir.pixel_stds),
            images_dir / f"{name}_input.png"
        )
        with torch.no_grad():
            latents = ssdir.encoder_forward(image.cuda())
            z_what, z_where, z_present, z_depth = ssdir.decoder.handle_latents(*latents)
            decoded_images, z_where_flat = ssdir.decoder.decode_objects(z_what, z_where)

            for decoded_idx, decoded in enumerate(decoded_images, start=1):
                save_image(decoded.cpu().permute(1, 2, 0), images_dir / f"{name}_obj{decoded_idx}.png")

            reconstructions, depths = ssdir.decoder.transform_objects(
                decoded_images, z_where_flat, z_present, z_depth
            )

            output = ssdir.decoder.merge_reconstructions(reconstructions, depths)
            output_to_save = output.cpu()[0].permute(1, 2, 0) / output.cpu().max()
            draw_boxes_and_save(output_to_save, z_where.cpu(), images_dir / f"{name}_with_boxes.png")
            save_image(output_to_save, images_dir / f"{name}_output.png")
