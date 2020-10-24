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
        self.ssd_loc_reg_headers = ssd_box_predictor.reg_headers
        ssd_config = ssd_box_predictor.config
        self.anchors = nn.Parameter(
            process_prior(
                image_size=ssd_config.DATA.SHAPE,
                feature_maps=ssd_config.DATA.PRIOR.FEATURE_MAPS,
                min_sizes=ssd_config.DATA.PRIOR.MIN_SIZES,
                max_sizes=ssd_config.DATA.PRIOR.MAX_SIZES,
                strides=ssd_config.DATA.PRIOR.STRIDES,
                aspect_ratios=ssd_config.DATA.PRIOR.ASPECT_RATIOS,
                clip=ssd_config.DATA.PRIOR.CLIP,
            ),
            requires_grad=False,
        )
        self.center_variance = ssd_config.MODEL.CENTER_VARIANCE
        self.size_variance = ssd_config.MODEL.SIZE_VARIANCE

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs bounding box parameters x_center, y_center, w, h tensor
        .. (batch_size x sum_features(grid*grid*n_boxes) x 4)
        """
        where = []
        batch_size = features[0].shape[0]
        for feature, reg_header in zip(features, self.ssd_loc_reg_headers):
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

    @staticmethod
    def scale_boxes(where_boxes: torch.Tensor) -> torch.Tensor:
        """Adjust scaled XYWH boxes to STN format.

        .. t_{XY} = (1 - 2 * {XY}) * s_{WH}
           s_{WH} = 1 / {WH}

        :param where_boxes: latent - detection box
        :return: scaled box
        """
        xy = where_boxes[..., :2]
        wh = where_boxes[..., 2:]
        scaled_wh = 1 / wh
        scaled_xy = (1 - 2 * xy) * scaled_wh
        return torch.cat((scaled_xy, scaled_wh), dim=-1)

    @staticmethod
    def convert_boxes_to_theta(where_boxes: torch.Tensor) -> torch.Tensor:
        """Convert where latents to transformation matrix.

        .. [ w_scale    0    x_translation ]
           [    0    h_scale y_translation ]

        :param where_boxes: latent - detection box
        :return: transformation matrix for transposing and scaling
        """
        n_boxes = where_boxes.shape[0]
        transformation_mtx = torch.cat(
            (torch.zeros((n_boxes, 1), device=where_boxes.device), where_boxes), dim=1
        )
        return transformation_mtx.index_select(
            dim=1,
            index=torch.tensor([3, 0, 1, 0, 4, 2], device=where_boxes.device),
        ).view(n_boxes, 2, 3)

    def forward(
        self, decoded_images: torch.Tensor, where_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Takes decoded images (sum_features(grid*grid) x 3 x 64 x 64)
        .. and bounding box parameters x_center, y_center, w, h tensor
        .. (sum_features(grid*grid*n_boxes) x 4)
        .. and outputs transformed images
        .. (sum_features(grid*grid*n_boxes) x 3 x image_size x image_size)
        """
        n_objects = decoded_images.shape[0]
        channels = decoded_images.shape[1]
        if where_boxes.numel():
            scaled_boxes = self.scale_boxes(where_boxes)
            theta = self.convert_boxes_to_theta(where_boxes=scaled_boxes)
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
                    input=decoded_images,
                    grid=grid,
                )
        else:
            transformed_images = decoded_images.view(
                -1, channels, self.image_size, self.image_size
            )
        return transformed_images
