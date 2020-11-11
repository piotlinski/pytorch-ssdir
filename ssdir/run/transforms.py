"""Data transforms."""
from typing import Union

import numpy as np
import torch
import torch.nn.functional as functional
from pyssd.data.bboxes import corner_bbox_to_center_bbox


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
