from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pyro.distributions as dist
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensor
from pytorch_ssd.data.bboxes import center_bbox_to_corner_bbox
from torchvision.ops import nms

from pytorch_ssdir.modeling import SSDIR

BoundingBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    confidence: float
    bbox: BoundingBox
    label: int
    inference_size: Tuple[int, int]
    image_size: Tuple[int, int]
    what: np.ndarray
    present: np.ndarray
    depth: np.ndarray

    @property
    def xywh_norm(self) -> BoundingBox:
        x1, y1, x2, y2 = self.bbox
        x = (x1 + x2) / 2 / self.inference_size[1]
        y = (y1 + y2) / 2 / self.inference_size[0]
        w = (x2 - x1) / self.inference_size[1]
        h = (y2 - y1) / self.inference_size[0]
        return x, y, w, h

    @property
    def xywh(self) -> BoundingBox:
        x, y, w, h = self.xywh_norm
        height, width = self.image_size
        return x * width, y * height, w * width, h * height

    @property
    def x1y1x2y2_norm(self) -> BoundingBox:
        x1, y1, x2, y2 = self.bbox
        height, width = self.inference_size
        return x1 / width, y1 / height, x2 / width, y2 / height

    @property
    def x1y1x2y2(self) -> BoundingBox:
        x1, y1, x2, y2 = self.x1y1x2y2_norm
        height, width = self.image_size
        return x1 * width, y1 * height, x2 * width, y2 * height

    @property
    def centroid(self) -> np.ndarray:
        return np.array(self.xywh_norm[:2])

    @property
    def where(self) -> np.ndarray:
        return np.array(self.xywh_norm)

    @property
    def data(self) -> str:
        x1, y1, x2, y2 = self.x1y1x2y2
        return f"%d, %d, {x1}, {y1}, {x2}, {y2}, {self.confidence}, -1, -1, -1"


class Representer:
    def __init__(
        self,
        checkpoint: str,
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.45,
        max_per_image: int = 100,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ssdir = (
            SSDIR.load_from_checkpoint(checkpoint, normalize_z_present=False)
            .eval()
            .to(self.device)
        )
        self._prepare = Compose(
            [
                Resize(*self.ssdir.image_size),
                Normalize(
                    mean=self.ssdir.pixel_means,
                    std=self.ssdir.pixel_stds,
                    max_pixel_value=1.0,
                ),
                ToTensor(),
            ]
        )
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_per_image = max_per_image

    def _mask_confidence_threshold(
        self,
        z_what: Tuple[torch.Tensor, torch.Tensor],
        z_where: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_what_loc, z_what_scale = z_what
        z_depth_loc, z_depth_scale = z_depth

        z_what_loc = z_what_loc.view(-1, z_what_loc.shape[-1])
        z_what_scale = z_what_scale.view(-1, z_what_scale.shape[-1])
        z_where = z_where.view(-1, z_where.shape[-1])
        z_present = z_present.view(-1)
        z_depth_loc = z_depth_loc.view(-1)
        z_depth_scale = z_depth_scale.view(-1)

        mask = (
            torch.gt(z_present, self.confidence_threshold)
            .nonzero(as_tuple=False)
            .squeeze(1)
        )
        z_what_loc = z_what_loc[mask]
        z_what_scale = z_what_scale[mask]
        z_what = dist.Normal(z_what_loc, z_what_scale).sample()
        z_where = z_where[mask]
        z_present = z_present[mask]
        z_depth_loc = z_depth_loc[mask]
        z_depth_scale = z_depth_scale[mask]
        z_depth = dist.Normal(z_depth_loc, z_depth_scale).sample()
        return z_what, z_where, z_present, z_depth

    def _latents_nms(
        self,
        z_what: torch.Tensor,
        z_where: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_where = center_bbox_to_corner_bbox(z_where)
        z_where[:, 0::2] *= self.ssdir.image_size[1]
        z_where[:, 1::2] *= self.ssdir.image_size[0]
        keep_mask = nms(
            boxes=z_where, scores=z_present, iou_threshold=self.nms_threshold
        )
        keep_mask = keep_mask[: self.max_per_image]
        return (
            z_what[keep_mask],
            z_where[keep_mask],
            z_present[keep_mask],
            z_depth[keep_mask],
        )

    def __call__(self, frame: np.ndarray):
        height, width, _ = frame.shape
        prepared = self._prepare(image=frame)["image"].to(self.device)
        with torch.no_grad():
            latents = self.ssdir.encoder(prepared.unsqueeze(0))
        z_what, z_where, z_present, z_depth = self._latents_nms(
            *self._mask_confidence_threshold(*latents)
        )
        for bbox, confidence, what, depth in zip(
            z_where.cpu(), z_present.cpu(), z_what.cpu(), z_depth.cpu()
        ):
            yield Detection(
                confidence=confidence.item(),
                bbox=tuple(bbox.tolist()),
                label=1,
                inference_size=self.ssdir.image_size,
                image_size=(height, width),
                what=what.view(-1).numpy(),
                present=confidence.view(-1).numpy(),
                depth=depth.view(-1).numpy(),
            )
