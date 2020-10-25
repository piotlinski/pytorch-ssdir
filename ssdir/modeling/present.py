"""$$z_{present}$$ encoder."""
from typing import Tuple

import torch
import torch.nn as nn
from pyssd.modeling.box_predictors import SSDBoxPredictor


class PresentEncoder(nn.Module):
    """Module encoding input image features to present latent param."""

    def __init__(self, ssd_box_predictor: SSDBoxPredictor):
        super().__init__()
        self.ssd_cls_headers = ssd_box_predictor.cls_headers
        self.n_classes = ssd_box_predictor.config.DATA.N_CLASSES
        self.background_present = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs probabilities tensor
        .. (batch_size x sum_features(grid*grid*n_boxes) x 1)
        """
        presents = []
        batch_size = features[0].shape[0]
        for feature, cls_header in zip(features, self.ssd_cls_headers):
            logits = torch.sigmoid(
                cls_header(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.n_classes)
            )
            max_values, max_indices = torch.max(logits, dim=-1, keepdim=True)
            present = torch.zeros_like(max_values)
            presents.append(present.where(max_indices == 0, max_values))

        presents.append(self.background_present.expand(batch_size, 1, 1))

        presents = torch.cat(presents, dim=1)
        return presents
