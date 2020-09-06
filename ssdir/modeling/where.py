"""$$z_{where}$$ encoder and decoder."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from ssd.data.bboxes import convert_locations_to_boxes
from ssd.modeling.box_predictors import SSDBoxPredictor


class WhereEncoder(nn.Module):
    """Module encoding input image features to where latent params.

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


class WhereTransformer(nn.Module):
    """Transforms WhereDecoder output image using where box."""

    def __init__(self, image_size: int):
        super().__init__()
        self.image_size = image_size
        self.decoded_size = 64

    def convert_boxes_to_sxy(self, where_boxes: torch.Tensor) -> torch.Tensor:
        """ Convert where latents into sxy format.

        :param where_boxes: latent - detection box
        :return: sxy
        """
        s_where, _ = torch.max(where_boxes[..., 2:], dim=-1, keepdim=True)
        xy_where = where_boxes[..., :2] - s_where / 2
        return torch.cat(
            (s_where * self.image_size / self.decoded_size, xy_where * self.image_size),
            dim=-1,
        )

    @staticmethod
    def expand_where(sxy: torch.Tensor) -> torch.Tensor:
        """ Take sxy where latent and massage it into a transformation matrix.

        :param where: sxy boxes
        :return: transformation matrix for transposing and scaling
        """
        batch_size = sxy.shape[0]
        transformation_mtx = torch.cat(
            (torch.zeros((1, 1)).expand(batch_size, 1), sxy), dim=1
        )
        return torch.index_select(
            input=transformation_mtx, dim=1, index=[1, 0, 2, 0, 1, 3]
        ).view(batch_size, 2, 3)

    def forward(
        self, decoded_image: torch.Tensor, where_boxes: torch.Tensor
    ) -> torch.Tensor:
        batch_size = decoded_image.shape[0]
        channels = decoded_image.shape[1]
        sxy = self.convert_boxes_to_sxy(where_boxes=where_boxes)
        theta = self.expand_where(sxy)
        grid = functional.affine_grid(
            theta=theta, size=(batch_size, channels, self.image_size, self.image_size)
        )
        transformed_image = functional.grid_sample(input=decoded_image, grid=grid)
        return transformed_image
