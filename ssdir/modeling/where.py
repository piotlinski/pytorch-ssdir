"""$$z_{where}$$ encoder and decoder."""
from typing import Tuple

import torch
import torch.nn as nn
from ssd.data.bboxes import convert_locations_to_boxes
from ssd.modeling.box_predictors import SSDBoxPredictor


class WhereEncoder(nn.Module):
    """Module encoding input image to where latent params).

    .. converts regressional location results into boxes (center_x center_y, w, h)
       $$hat{center} * center_variance = \frac {center - center_prior} {hw_prior}$$
       $$exp(hat{hw} * size_variance) = \frac {hw} {hw_prior}$$
    """

    def __init__(
        self,
        ssd_box_predictor: SSDBoxPredictor,
        ssd_anchors: torch.Tensor,
        ssd_center_variance: float,
        ssd_size_variance: float,
    ):
        super().__init__()
        self.predictor = ssd_box_predictor
        self.anchors = ssd_anchors
        self.center_variance = ssd_center_variance
        self.size_variance = ssd_size_variance

    def forward(self, features: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        where = []
        batch_size = features[0].shape[0]
        for feature, reg_header in zip(features, self.predictor.reg_headers):
            where.append(
                reg_header(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, 4)
            )

        where_locations = torch.cat(where, dim=1)
        where_boxes = convert_locations_to_boxes(
            locations=where_locations,
            priors=self.anchors,
            center_variance=self.center_variance,
            size_variance=self.size_variance,
        )

        return where_boxes
