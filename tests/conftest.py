"""Common tests tools."""
import pytest
import torch
from pyssd.config import CfgNode
from pyssd.modeling.model import SSD


@pytest.fixture
def ssd_config():
    """Default SSD config."""
    return CfgNode(
        {
            "ASSETS_DIR": "assets",
            "DATA": CfgNode(
                {
                    "DATASET": "MultiscaleMNIST",
                    "DATASET_DIR": "data",
                    "SHAPE": (300, 300),
                    "N_CLASSES": 2,
                    "CLASS_LABELS": ("digit",),
                    "PIXEL_MEAN": (0.0, 0.0, 0.0),
                    "PIXEL_STD": (1.0, 1.0, 1.0),
                    "AUGMENT_COLORS": False,
                    "PRIOR": CfgNode(
                        {
                            "BOXES_PER_LOC": (2, 2, 2, 2, 2),
                            "FEATURE_MAPS": (18, 9, 5, 3, 1),
                            "MIN_SIZES": (32, 80, 153, 207, 261),
                            "MAX_SIZES": (80, 153, 207, 261, 315),
                            "STRIDES": (16, 32, 64, 100, 300),
                            "ASPECT_RATIOS": ([], [], [], [], []),
                            "CLIP": True,
                        }
                    ),
                }
            ),
            "MODEL": CfgNode(
                {
                    "BATCH_NORM": False,
                    "USE_PRETRAINED": False,
                    "PRETRAINED_URL": "",
                    "PRETRAINED_DIR": "pretrained",
                    "CHECKPOINT_DIR": "checkpoints",
                    "CHECKPOINT_NAME": "",
                    "BACKBONE": "VGG11",
                    "BOX_PREDICTOR": "SSD",
                    "CENTER_VARIANCE": 0.1,
                    "SIZE_VARIANCE": 0.2,
                    "CONFIDENCE_THRESHOLD": 0.8,
                    "NMS_THRESHOLD": 0.45,
                    "MAX_PER_IMAGE": 100,
                    "IOU_THRESHOLD": 0.5,
                    "NEGATIVE_POSITIVE_RATIO": 3,
                }
            ),
            "RUNNER": CfgNode(
                {
                    "DEVICE": "cuda",
                    "EPOCHS": 15,
                    "BATCH_SIZE": 64,
                    "LR": 0.001,
                    "LR_REDUCE_PATIENCE": 5,
                    "LR_REDUCE_SKIP_EPOCHS": 10,
                    "LR_WARMUP_STEPS": 1000,
                    "NUM_WORKERS": 8,
                    "PIN_MEMORY": True,
                    "LOG_STEP": 10,
                    "USE_TENSORBOARD": True,
                    "TENSORBOARD_DIR": "runs",
                    "VIS_EPOCHS": 1,
                    "VIS_N_IMAGES": 4,
                    "VIS_CONFIDENCE_THRESHOLDS": (0.0, 0.2, 0.4, 0.6, 0.8),
                    "TRACK_MODEL_PARAMS": False,
                }
            ),
            "EXPERIMENT_NAME": "config",
            "CONFIG_STRING": "SSD-VGGLite_MultiscaleMNIST",
        }
    )


@pytest.fixture
def ssd_model(ssd_config):
    """Default SSD model."""
    return SSD(config=ssd_config)


@pytest.fixture
def ssd_features(ssd_config, ssd_model):
    """Sample ssd features tensors tuple."""
    features = [
        torch.rand(4, channels, feature_map, feature_map)
        for feature_map, channels in zip(
            ssd_config.DATA.PRIOR.FEATURE_MAPS, ssd_model.backbone.out_channels
        )
    ]
    return tuple(features)


@pytest.fixture
def n_ssd_features(ssd_config):
    """Total number of ssd features."""
    return sum(
        boxes * features ** 2
        for features, boxes in zip(
            ssd_config.DATA.PRIOR.FEATURE_MAPS, ssd_config.DATA.PRIOR.BOXES_PER_LOC
        )
    )
