"""Command Line Interface."""
import logging
import time
import warnings
from typing import List

import click
import horovod.torch as hvd
import numpy as np
import torch
from pyro.infer import Trace_ELBO
from pyssd.config import get_config
from pyssd.data.datasets import datasets
from pyssd.data.transforms import TrainDataTransform
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ssdir import SSDIR
from ssdir.run.utils import HorovodOptimizer, corner_to_center_target_transform
from ssdir.run.visualize import visualize_latents

logger = logging.getLogger(__name__)


warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)
warnings.filterwarnings(
    "ignore",
    message=(
        "where_enc.anchors was not registered in the param store "
        "because requires_grad=False"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "indices was not registered in the param store because requires_grad=False"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "empty_obj_const was not registered in the param store "
        "because requires_grad=False"
    ),
)


@click.group(help="SSDIR")
@click.pass_context
def main(ctx: click.Context):
    """Main group for subcommands."""
    ctx.ensure_object(dict)
    logging.basicConfig(level=logging.INFO)


@main.command(help="Train model")
@click.option("--n-epochs", default=100, help="number of epochs", type=int)
@click.option("--lr", default=1e-3, help="learning rate", type=float)
@click.option("--ssd-lr", default=1e-6, help="ssd learning rate", type=float)
@click.option("--bs", default=4, help="batch size", type=int)
@click.option("--z-what-size", default=64, help="z_what size", type=int)
@click.option(
    "--drop/--no-drop",
    default=True,
    help="drop unused images when reconstructing",
    type=bool,
)
@click.option(
    "--horovod/--no-horovod",
    default=False,
    help="use horovod distributed training",
    type=bool,
)
@click.option("--device", default="cuda", help="device for training", type=str)
@click.option(
    "--ssd-config-file",
    default="assets/pretrained/simple_multimnist/config.yml",
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
    ssd_lr: float,
    bs: int,
    z_what_size: int,
    drop: bool,
    horovod: bool,
    device: str,
    ssd_config_file: str,
    tb_dir: str,
    logging_step: int,
    vis_step: int,
):
    """Train the model."""
    if horovod:
        hvd.init()
        if device == "cuda":
            torch.cuda.set_device(hvd.local_rank())

    tb_writer = SummaryWriter(log_dir=tb_dir)

    ssd_config = get_config(config_file=ssd_config_file)
    experiment = ssd_config_file.split("/")[-2].split("_")
    ssd_config.defrost()
    ssd_config.EXPERIMENT_NAME = experiment[0]
    ssd_config.CONFIG_STRING = experiment[1]

    model = SSDIR(ssd_config=ssd_config, z_what_size=z_what_size, drop_empty=drop).to(
        device
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model.filtered_parameters(exclude="ssd")},
            {"params": model.filtered_parameters(include="ssd"), "lr": ssd_lr},
        ],
        lr=lr,
    )
    dataset = datasets[ssd_config.DATA.DATASET](
        f"{ssd_config.ASSETS_DIR}/{ssd_config.DATA.DATASET_DIR}",
        data_transform=TrainDataTransform(ssd_config),
        target_transform=corner_to_center_target_transform,
        subset="train",
    )
    if horovod:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        optimizer = HorovodOptimizer(optimizer)
    else:
        sampler = RandomSampler(dataset)

    train_loader = DataLoader(
        dataset=dataset,
        num_workers=ssd_config.RUNNER.NUM_WORKERS,
        sampler=sampler,
        pin_memory=ssd_config.RUNNER.PIN_MEMORY,
        batch_size=bs,
    )

    loss_fn = Trace_ELBO().differentiable_loss

    global_step = 0
    logging_losses: List[float] = []
    iter_times: List[float] = []

    epochs = range(n_epochs)
    if (horovod and hvd.rank() == 0) or not horovod:
        epochs = tqdm(
            epochs,
            desc="TRAINING",
            unit="epoch",
            postfix=dict(step=global_step, loss=float("nan"), its="nan it/s"),
        )

    for epoch in epochs:
        if horovod:
            sampler.set_epoch(epoch)
        epoch += 1

        for images, boxes, _ in train_loader:
            step_start = time.perf_counter()
            images = images.to(device)

            loss = loss_fn(model.model, model.guide, images)

            if horovod:
                loss = hvd.allreduce(loss, "loss")

            loss.backward()
            optimizer.step()

            logging_losses.append(loss.item())

            if global_step % logging_step == 0 and (
                (horovod and hvd.rank() == 0) or not horovod
            ):
                logging_loss = (
                    np.average(logging_losses) if logging_losses else float("nan")
                )
                logging_losses = []
                iter_time = np.average(iter_times) if iter_times else float("nan")
                iter_times = []
                if iter_time > 1:
                    iter_time = f"{iter_time:4.2f} s/it"
                else:
                    iter_time = f"{1 / iter_time:4.2f} it/s"

                tb_writer.add_scalar(
                    tag="elbo/train",
                    scalar_value=logging_loss,
                    global_step=global_step,
                )

                epochs.set_postfix(  # type: ignore
                    step=global_step, loss=logging_loss, its=iter_time
                )

            if global_step % vis_step == 0 and (
                (horovod and hvd.rank() == 0) or not horovod
            ):
                model.eval()
                tb_writer.add_figure(
                    tag="inference",
                    figure=visualize_latents(
                        images.detach().to(device), boxes=boxes.detach(), model=model
                    ),
                    global_step=global_step,
                )
                model.train()

            global_step += 1
            step_end = time.perf_counter()
            iter_times.append(step_end - step_start)

    if horovod:
        hvd.shutdown()
