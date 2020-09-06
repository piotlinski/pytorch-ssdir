"""$$z_{present}$$ encoder."""
from typing import Tuple

import torch
import torch.nn as nn
from ssd.modeling.box_predictors import SSDBoxPredictor


class PresentEncoder(nn.Module):
    """Module encoding input image features to present latent param."""

    def __init__(self, ssd_box_predictor: SSDBoxPredictor):
        super().__init__()
        self.predictor = ssd_box_predictor

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """ Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs probabilities tensor (batch_size x sum_features(grid*grid) x 1)
        """
        presents = []
        batch_size = features[0].shape[0]
        for feature, cls_header in zip(features, self.predictor.cls_headers):
            logits = (
                cls_header(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.predictor.config.DATA.N_CLASSES)
            )
            max_values, max_indices = torch.max(logits, dim=-1, keepdim=True)
            present_indices = max_values > 0
            present = torch.zeros_like(max_values)
            present[present_indices] = max_values[present_indices]
            presents.append(present)

        presents = torch.cat(presents, dim=1)

        return presents
