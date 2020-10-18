"""Visualization tools for training SSDIR."""
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from matplotlib import patches

from ssdir.modeling.models import SSDIR


def plot_image(
    image: torch.Tensor,
    boxes: Optional[torch.Tensor] = None,
    details: Optional[List[torch.Tensor]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot an image with details.
    :param image: image to plot
    :param boxes: xywh boxes tensor of shape (n_boxes x 4)
    :param details: list of tensors of shape (n_boxes x 1) containing additional info
    :param ax: optional axis to plot on
    :return: matplotlib axis with plotted image
    """
    if ax is None:
        ax = plt.gca()
    ax.axis("off")
    numpy_image = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(numpy_image)
    if boxes is not None:
        if details is not None:
            plot_iter = zip(boxes, *details)
        else:
            plot_iter = zip(boxes)
        for box, *detail in plot_iter:
            x, y, w, h = box.int()
            x0, y0 = x - w // 2, y - h // 2
            rect = patches.Rectangle(
                (x0, y0),
                w,
                h,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.text(
                x0,
                y0 + h,
                ", ".join(f"{d.item():1.3f}" for d in detail),
                verticalalignment="top",
                color="w",
                fontsize="x-small",
                fontweight="semibold",
                clip_on=True,
                bbox=dict(pad=0, facecolor="r", alpha=0.8),
            )
            ax.add_patch(rect)
    return ax


def visualize_latents(
    images: torch.Tensor, boxes: torch.Tensor, model: SSDIR, n_objects: int = 8
) -> plt.Figure:
    """Visualize model inference.

    :param images: input images to perform infer on
        shape (n_images x 3 x image_size x image_size)
    :param boxes: boxes predicted by SSD (n_images x n_boxes x 4)
    :param model: SSDIR model
    :param n_objects: max number of latent objects to plot
    :return: figure with visualization
    """
    with torch.no_grad():
        latents = model.encoder_forward(images)
        z_what, z_where, z_present, z_depth = model.decoder.pad_latents(latents)
        reconstructions = model.decoder_forward(latents)
        n_cols, n_rows = n_objects + 2, images.shape[0]
        fig = plt.Figure(figsize=(4 * n_cols, 4 * n_rows))
        for idx, (image, box, reconstruction, what, where, present, depth) in enumerate(
            zip(images, boxes, reconstructions, z_what, z_where, z_present, z_depth)
        ):
            subplot_idx = idx * n_cols + 1
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx)
            plot_image(image=image, boxes=box * image.shape[-1], ax=ax)
            ax.set_title("gt")

            present_mask = present == 1
            what = what[present_mask.expand_as(what)].view(-1, what.shape[-1])
            where = where[present_mask.expand_as(where)].view(-1, where.shape[-1])
            depth = depth[present_mask.expand_as(depth)].view(-1, depth.shape[-1])
            sorted_depth, sort_index = torch.sort(depth, dim=0, descending=True)
            sorted_what = what.gather(dim=0, index=sort_index.expand_as(what))
            sorted_where = where.gather(dim=0, index=sort_index.expand_as(where))

            decoded_images = model.decoder.what_dec(sorted_what[:n_objects])
            transformed_images = model.decoder.where_stn(
                decoded_images, sorted_where[:n_objects]
            )

            for obj_idx, (transformed_image, depth_info) in enumerate(
                zip(transformed_images, sorted_depth), start=1
            ):
                ax = fig.add_subplot(n_rows, n_cols, subplot_idx + obj_idx)
                plot_image(image=transformed_image, ax=ax)
                ax.set_title(f"depth={depth_info.item():1.3f}")

            subplot_idx += n_objects + 1
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx)
            plot_image(
                image=reconstruction,
                boxes=where * image.shape[-1],
                details=[depth],
                ax=ax,
            )
            ax.set_title("reconstruction")
        return fig
