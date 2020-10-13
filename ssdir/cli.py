"""Command Line Interface."""
import logging

import click
import horovod.torch as hvd
import numpy as np
import pyro.optim as optim
import torch
from pyro.infer import SVI, Trace_ELBO
from pyssd.config import get_config
from pyssd.data.datasets import datasets
from pyssd.data.transforms import TrainDataTransform
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ssdir import SSDIR
from ssdir.run.utils import HorovodOptimizer, corner_to_center_target_transform
from ssdir.run.visualize import visualize_latents

logger = logging.getLogger(__name__)


@click.group(help="SSDIR")
@click.pass_context
def main(ctx: click.Context):
    """Main group for subcommands."""
    ctx.ensure_object(dict)
    logging.basicConfig(level=logging.INFO)


@main.command(help="Train model")
@click.option("--n-epochs", default=100, help="number of epochs", type=int)
@click.option("--lr", default=1e-4, help="learning rate", type=float)
@click.option("--bs", default=4, help="batch size", type=int)
@click.option("--z-what-size", default=64, help="z_what size", type=int)
@click.option("--drop/--no-drop", default=True)
@click.option("--device", default="cuda", help="device for training", type=str)
@click.option(
    "--ssd-config-file",
    default=(
        "assets/pretrained"
        "/vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST/vgglite_mnist_sc.yml"
    ),
    help="ssd config to be used",
    type=str,
)
@click.option(
    "--tb-dir",
    default="assets/runs/latest",
    help="folder for storing TB output",
    type=str,
)
@click.option(
    "--logging-step", default=10, help="number of steps between logging", type=int
)
@click.option(
    "--vis-step", default=500, help="number of steps between visualization", type=int
)
@click.pass_obj
def train(
    obj,
    n_epochs: int,
    lr: float,
    bs: int,
    z_what_size: int,
    drop: bool,
    device: str,
    ssd_config_file: str,
    tb_dir: str,
    logging_step: int,
    vis_step: int,
):
    """Train the model."""
    hvd.init()
    if device == "cuda":
        torch.cuda.set_device(hvd.local_rank())
    epoch_loss = float("nan")
    logging_loss = float("nan")
    logging_losses = []

    tb_writer = SummaryWriter(log_dir=tb_dir)

    ssd_config = get_config(config_file=ssd_config_file)

    global_step = 0

    model = SSDIR(ssd_config=ssd_config, z_what_size=z_what_size, drop_empty=drop).to(
        device
    )
    optimizer = optim.Adam({"lr": lr})
    dataset = datasets[ssd_config.DATA.DATASET](
        f"{ssd_config.ASSETS_DIR}/{ssd_config.DATA.DATASET_DIR}",
        data_transform=TrainDataTransform(ssd_config),
        target_transform=corner_to_center_target_transform,
        subset="train",
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = DataLoader(
        dataset=dataset,
        num_workers=ssd_config.RUNNER.NUM_WORKERS,
        sampler=sampler,
        pin_memory=ssd_config.RUNNER.PIN_MEMORY,
        batch_size=bs,
    )

    vis_images = None
    vis_boxes = None

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    optimizer = HorovodOptimizer(optimizer)

    loss_fn = Trace_ELBO()
    svi = SVI(model=model.model, guide=model.guide, optim=optimizer, loss=loss_fn)

    epochs = range(n_epochs)
    if hvd.rank() == 0:
        epochs = tqdm(
            epochs,
            desc="TRAINING",
            unit="epoch",
            postfix=dict(step=global_step, loss=epoch_loss),
        )

    for epoch in epochs:
        sampler.set_epoch(epoch)
        epoch_losses = []
        epoch += 1

        steps = train_loader
        if hvd.rank() == 0:
            steps = tqdm(
                steps,
                desc=f"  epoch {epoch:4d}",
                unit="step",
                postfix=dict(loss=logging_loss),
            )
        for images, boxes, _ in steps:
            images = images.to(device)
            loss = svi.step(images)
            if vis_images is None and vis_boxes is None:
                vis_images = images.detach()
                vis_boxes = boxes.detach()

            loss = hvd.allreduce(torch.tensor(loss), "loss")
            loss = loss.item()

            epoch_losses.append(loss)
            logging_losses.append(loss)

            if global_step % logging_step == 0 and hvd.rank() == 0:
                epoch_loss = np.average(epoch_losses)
                logging_loss = np.average(logging_losses)
                logging_losses = []

                tb_writer.add_scalar(
                    tag="elbo/train",
                    scalar_value=logging_loss,
                    global_step=global_step,
                )

                epochs.set_postfix(step=global_step, loss=epoch_loss)  # type: ignore
                steps.set_postfix(loss=loss)  # type: ignore

            if global_step % vis_step == 0 and hvd.rank() == 0:
                model.eval()
                tb_writer.add_figure(
                    tag="inference",
                    figure=visualize_latents(
                        vis_images.to(device), boxes=vis_boxes, model=model
                    ),
                    global_step=global_step,
                )
                model.train()

            global_step += 1

    hvd.shutdown()
