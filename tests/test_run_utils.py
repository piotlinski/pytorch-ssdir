"""Test modeling utils."""
import pytest

from ssdir.run.utils import per_param_lr


@pytest.mark.parametrize(
    "param, value", [("test", 0.1), ("testtest", 0.1), ("abc", 0.01), ("cba", 0.123)]
)
def test_per_param_lr(param, value):
    """Verify callable for pyro optimizer."""
    lr_dict = {"test": 0.1, "abc": 0.01}
    default_lr = 0.123
    lr_callable = per_param_lr(lr_dict=lr_dict, default_lr=default_lr)

    assert lr_callable("r", param)["lr"] == value
