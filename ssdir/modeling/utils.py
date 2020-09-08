"""Modeling utils."""
from typing import Callable, Dict


def per_param_lr(
    lr_dict: Dict[str, float], default_lr: float = 1e-3
) -> Callable[[str, str], Dict[str, float]]:
    """Get lr per param name for pyro optim."""

    def callable(module_name: str, param_name: str) -> Dict[str, float]:
        if param_name in lr_dict:
            return {"lr": lr_dict[param_name]}
        else:
            return {"lr": default_lr}

    return callable
