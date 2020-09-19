"""Modeling utils."""
from typing import Callable, Dict, Union

import numpy as np
import pyro
import torch
import torch.nn.functional as functional
from pyro.optim import PyroOptim
from pyssd.data.bboxes import corner_bbox_to_center_bbox


def per_param_lr(
    lr_dict: Dict[str, float], default_lr: float = 1e-3
) -> Callable[[str, str], Dict[str, float]]:
    """Get lr per param name for pyro optim."""

    def lr_callable(module_name: str, param_name: str) -> Dict[str, float]:
        if param_name in lr_dict:
            return {"lr": lr_dict[param_name]}
        else:
            return {"lr": default_lr}

    return lr_callable


def corner_to_center_target_transform(
    boxes: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]
):
    """Convert ground truth boxes from corner to center form."""
    n_objs = boxes.shape[0]
    pad_size = 200
    return (
        functional.pad(corner_bbox_to_center_bbox(boxes), [0, 0, 0, pad_size - n_objs]),
        functional.pad(labels, [0, pad_size - n_objs]),
    )


class HorovodOptimizer(PyroOptim):
    r"""
    Distributed wrapper for a :class:`~pyro.optim.optim.PyroOptim` optimizer.

    This class wraps a ``PyroOptim`` object similar to the way
    :func:`horovod.torch.DistributedOptimizer` wraps a
    :class:`torch.optim.Optimizer`.

    .. note::

        This requires :mod:`horovod.torch` to be installed, e.g. via
        ``pip install pyro[horovod]``. For details see
        https://horovod.readthedocs.io/en/stable/install.html

    :param: A Pyro optimizer instance.
    :type pyro_optim: ~pyro.optim.optim.PyroOptim
    :param \*\*horovod_kwargs: Extra parameters passed to
        :func:`horovod.torch.DistributedOptimizer`.
    """

    def __init__(self, pyro_optim, **horovod_kwargs):
        param_name = pyro.get_param_store().param_name

        def optim_constructor(params, **pt_kwargs):
            import horovod.torch as hvd

            pt_optim = pyro_optim.pt_optim_constructor(params, **pt_kwargs)
            named_parameters = [(param_name(p), p) for p in params]
            hvd_optim = hvd.DistributedOptimizer(
                pt_optim, named_parameters=named_parameters, **horovod_kwargs,
            )
            return hvd_optim

        super().__init__(
            optim_constructor, pyro_optim.pt_optim_args, pyro_optim.pt_clip_args
        )

    def __call__(self, params, *args, **kwargs):
        # Sort by name to ensure deterministic processing order.
        params = sorted(params, key=pyro.get_param_store().param_name)
        super().__call__(params, *args, **kwargs)
