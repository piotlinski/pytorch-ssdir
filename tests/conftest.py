"""Common tests tools."""
import pytest
import torch
from ssd.config import get_config
from ssd.modeling.model import SSD


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
