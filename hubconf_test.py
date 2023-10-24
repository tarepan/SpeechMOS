"""Test torch.hub basic functionalities."""

from hubconf import utmos22_strong
from speechmos.utmos22.strong.model import UTMOS22Strong


def test_utmos22_strong_init():
    """Test `utmos22_strong` instantiation without weight load."""

    # Test - progress=True
    model = utmos22_strong(progress=True, pretrained=False)
    assert isinstance(model, UTMOS22Strong), "UTMOS22Strong not properly instantiated."

    # Test - progress=False
    model = utmos22_strong(progress=False, pretrained=False)
    assert isinstance(model, UTMOS22Strong), "UTMOS22Strong not properly instantiated."
