"""Test the utmos22-strong model."""

import torch

from .model import UTMOS22Strong


def test_model_init():
    """Test the `UTMOS22Strong` instantiation."""

    # Preparation
    UTMOS22Strong()

    # Test
    assert True, "UTMOS22Strong is not properly instantiated."


def test_model_forward():
    """Test the `UTMOS22Strong` forward run."""

    # Preparation
    model = UTMOS22Strong()
    sr = 16000
    ipt = torch.tensor([1. for _ in range(int(sr * 0.5))]).unsqueeze(0)

    # Prerequesite Test
    assert ipt.size() == (1, 8000), "Prerequesites are not satisfied."

    # Test
    model(ipt, sr)
    assert True, "UTMOS22Strong is not properly forwarded."


def test_model_output_shape():
    """Test the `UTMOS22Strong` forward output shape."""

    # Preparation
    model = UTMOS22Strong()
    sr = 16000
    ipt = torch.tensor([1. for _ in range(int(sr * 0.5))]).unsqueeze(0)

    # Prerequesite Test
    assert ipt.size() == (1, 8000), "Prerequesites are not satisfied."

    # Test
    opt = model(ipt, sr)
    assert opt.size() == (1,), "UTMOS22Strong is not properly forwarded."
