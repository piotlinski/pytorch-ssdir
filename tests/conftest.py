"""Common tests tools."""
import pytest
import torch
from pyssd.config import get_config
from pyssd.modeling.model import SSD


@pytest.fixture
def ssd_config():
    """Default SSD config."""
    return get_config()


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
