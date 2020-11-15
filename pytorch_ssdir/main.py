"""Main function for SSDIR training."""
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_ssd.modeling.model import SSD

from pytorch_ssdir import SSDIR


def main(hparams):
    """Main function that creates and trains SSDIR model."""
    kwargs = vars(hparams)
    if hparams.ssd_checkpoint is not None:
        ssd = SSD.load_from_checkpoint(checkpoint_path=hparams.ssd_checkpoint, **kwargs)
    else:
        ssd = SSD(**kwargs)
    if hparams.ssdir_checkpoint is not None:
        model = SSDIR.load_from_checkpoint(checkpoint_path=hparams.ssdir_checkpoint)
    else:
        model = SSDIR(ssd_model=ssd, **kwargs)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}",
        save_top_k=hparams.n_checkpoints,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=hparams.early_stopping_patience
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor_callback]
    if hparams.early_stopping:
        callbacks.append(early_stopping_callback)

    logger = WandbLogger(
        name=(
            f"{hparams.dataset_name}-"
            f"SSDIR-{model.encoder.ssd_backbone.__class__.__name__}-"
            f"bs{hparams.batch_size}-lr{hparams.learning_rate}"
        ),
        save_dir=hparams.default_root_dir,
        project="ssdir",
    )
    logger.watch(model, log=hparams.watch, log_freq=hparams.watch_freq)

    trainer = Trainer.from_argparse_args(hparams, logger=logger, callbacks=callbacks)
    trainer.tune(model)
    trainer.fit(model)


def cli():
    """SSDIR CLI with argparse."""
    parser = ArgumentParser(conflict_handler="resolve")
    parser.add_argument(
        "-c",
        "--ssdir-checkpoint",
        type=str,
        default=None,
        help="Checkpoint to start training from",
    )
    parser.add_argument(
        "--ssd-checkpoint", type=str, default=None, help="SSD checkpoint file"
    )
    parser = SSDIR.add_model_specific_args(parser)
    parser.add_argument(
        "--n-checkpoints", type=int, default=3, help="Number of top checkpoints to save"
    )
    parser.add_argument(
        "--early-stopping",
        default=True,
        action="store_true",
        help="Enable early stopping",
    )
    parser.add_argument(
        "--no-early-stopping", dest="early_stopping", action="store_false"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of epochs with no improvements before stopping early",
    )
    parser.add_argument(
        "--watch",
        type=str,
        default=None,
        help="Log model topology as well as optionally gradients and weights. "
        "Available options: None, gradients, parameters, all",
    )
    parser.add_argument(
        "--watch-freq",
        type=int,
        default=100,
        help="How often to perform model watch.",
    )
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
