"""Command Line Interface."""
import logging

import click
import numpy as np
import pyro.optim as optim
import torch
from pyro.infer import SVI, Trace_ELBO
from pyssd.config import get_config
from pyssd.data.loaders import TrainDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from ssdir import SSDIR
from ssdir.modeling.utils import per_param_lr

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
@click.option("--device", default="cuda", help="device for training", type=str)
@click.option(
    "--ssd-config-file",
    default="assets/pretrained/vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST/config.yml",
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
@click.pass_obj
def train(
    obj,
    n_epochs: int,
    lr: float,
    bs: int,
    z_what_size: int,
    device: str,
    ssd_config_file: str,
    tb_dir: str,
    logging_step: int,
):
    """Train the model."""
    epoch_loss = float("nan")
    logging_loss = float("nan")
    logging_losses = []

    tb_writer = SummaryWriter(log_dir=tb_dir)

    ssd_config = get_config(config_file=ssd_config_file)
    ssd_config.defrost()
    ssd_config.RUNNER.BATCH_SIZE = bs

    train_loader = TrainDataLoader(config=ssd_config)

    global_step = 0

    model = SSDIR(ssd_config=ssd_config, z_what_size=z_what_size).to(device)
    optimizer = optim.Adam(
        per_param_lr(
            lr_dict={"z_where": 1e-6 / 100, "z_present": lr / 10}, default_lr=lr
        )
    )
    loss_fn = Trace_ELBO()
    svi = SVI(model=model.model, guide=model.guide, optim=optimizer, loss=loss_fn)

    add_graph(model=model, tb_dir=tb_dir, device=device)

    with trange(
        n_epochs,
        desc="TRAINING",
        unit="epoch",
        postfix=dict(step=global_step, loss=epoch_loss),
    ) as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_losses = []
            epoch += 1

            with tqdm(
                train_loader,
                desc=f"  epoch {epoch:4d}",
                unit="step",
                postfix=dict(loss=logging_loss),
            ) as step_pbar:
                for images, _, _ in step_pbar:
                    images = images.to(device)
                    loss = svi.step(images)

                    epoch_losses.append(loss)
                    logging_losses.append(loss)

                    if global_step % logging_step == 0:
                        epoch_loss = np.average(epoch_losses)
                        logging_loss = np.average(logging_losses)
                        logging_losses = []

                        tb_writer.add_scalar(
                            tag="elbo/train",
                            scalar_value=logging_loss,
                            global_step=global_step,
                        )

                        epoch_pbar.set_postfix(step=global_step, loss=epoch_loss)
                        step_pbar.set_postfix(loss=loss)

                    global_step += 1


def add_graph(model: SSDIR, tb_dir: str, device: str = "cpu"):
    """Write encoder and decoder graphs to TB."""
    SummaryWriter(log_dir=f"{tb_dir}_encoder").add_graph(
        model.encoder, torch.zeros((1, 3, 300, 300), device=device)
    )
    SummaryWriter(log_dir=f"{tb_dir}_decoder").add_graph(
        model.decoder,
        (
            (
                torch.zeros((1, model.n_objects, model.z_what_size), device=device),
                torch.zeros((1, model.n_ssd_features, 4), device=device),
                torch.ones((1, model.n_ssd_features, 1), device=device),
                torch.ones((1, model.n_objects, 1), device=device),
            ),
        ),
    )
