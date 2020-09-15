"""$$z_{where}$$ encoder and decoder."""
import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from pyssd.data.bboxes import convert_locations_to_boxes
from pyssd.data.priors import process_prior
from pyssd.modeling.box_predictors import SSDBoxPredictor


class WhereEncoder(nn.Module):
    """Module encoding input image features to where latent params.

    .. converts regressional location results into boxes (center_x center_y, w, h)
       $$hat{center} * center_variance = \\frac {center - center_prior} {hw_prior}$$
       $$exp(hat{hw} * size_variance) = \\frac {hw} {hw_prior}$$
    """

    def __init__(self, ssd_box_predictor: SSDBoxPredictor):
        super().__init__()
        self.predictor = ssd_box_predictor
        self.anchors = nn.Parameter(
            process_prior(
                image_size=ssd_box_predictor.config.DATA.SHAPE,
                feature_maps=ssd_box_predictor.config.DATA.PRIOR.FEATURE_MAPS,
                min_sizes=ssd_box_predictor.config.DATA.PRIOR.MIN_SIZES,
                max_sizes=ssd_box_predictor.config.DATA.PRIOR.MAX_SIZES,
                strides=ssd_box_predictor.config.DATA.PRIOR.STRIDES,
                aspect_ratios=ssd_box_predictor.config.DATA.PRIOR.ASPECT_RATIOS,
                clip=ssd_box_predictor.config.DATA.PRIOR.CLIP,
            )
        )
        self.center_variance = ssd_box_predictor.config.MODEL.CENTER_VARIANCE
        self.size_variance = ssd_box_predictor.config.MODEL.SIZE_VARIANCE

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """ Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs bounding box parameters x_center, y_center, w, h tensor
        .. (batch_size x sum_features(grid*grid*n_boxes) x 4)
        """
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
        xy_where = where_boxes[..., :2]
        return torch.cat(
            (s_where * self.image_size / self.decoded_size, xy_where * self.image_size),
            dim=-1,
        )

    @staticmethod
    def expand_where(sxy: torch.Tensor) -> torch.Tensor:
        """ Take sxy where latent and massage it into a transformation matrix.

        :param sxy: sxy boxes
        :return: transformation matrix for transposing and scaling
        """
        n_boxes = sxy.shape[0]
        transformation_mtx = torch.cat(
            (torch.zeros((n_boxes, 1), device=sxy.device), sxy), dim=1
        )
        return transformation_mtx.index_select(
            dim=1, index=torch.tensor([1, 0, 2, 0, 1, 3], device=sxy.device),
        ).view(n_boxes, 2, 3)

    def forward(
        self, decoded_images: torch.Tensor, where_boxes: torch.Tensor
    ) -> torch.Tensor:
        """ Takes decoded images (sum_features(grid*grid) x 3 x 64 x 64)
        .. and bounding box parameters x_center, y_center, w, h tensor
        .. (sum_features(grid*grid*n_boxes) x 4)
        .. and outputs transformed images
        .. (sum_features(grid*grid*n_boxes) x 3 x image_size x image_size)
        """
        n_objects = decoded_images.shape[0]
        channels = decoded_images.shape[1]
        sxy = self.convert_boxes_to_sxy(where_boxes=where_boxes)
        theta = self.expand_where(sxy)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Default grid_sample and affine_grid behavior has changed ",
            )
            grid = functional.affine_grid(
                theta=theta,
                size=[n_objects, channels, self.image_size, self.image_size],
            )
            transformed_images = functional.grid_sample(
                input=decoded_images, grid=grid,
            )
        return transformed_images
