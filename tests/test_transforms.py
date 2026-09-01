from __future__ import annotations

import numpy as np
import torch

from mcfqpi.data.transforms import decode_phase, normalize_speckle, resize_phase, resize_speckle


def test_decode_uint8_to_pi() -> None:
    array = np.array([[0, 255]], dtype=np.uint8)
    phase = decode_phase(array, "auto")
    assert torch.allclose(phase[0, 0], torch.tensor(0.0))
    assert torch.allclose(phase[0, 1], torch.tensor(np.pi), atol=1e-6)


def test_phase_resize_preserves_range_and_shape() -> None:
    phase = torch.linspace(0, torch.pi, 20).view(4, 5)
    resized, mask = resize_phase(phase, 16, "fit_pad")
    assert resized.shape == (1, 16, 16)
    assert mask.shape == (1, 16, 16)
    assert float(resized.min()) >= 0
    assert float(resized.max()) <= np.pi + 1e-5


def test_speckle_normalization_is_finite() -> None:
    raw = torch.rand(1, 16, 16) * 1000
    output = normalize_speckle(raw, method="log_mean")
    assert output.shape == raw.shape
    assert torch.isfinite(output).all()
    assert 0 <= float(output.min()) <= float(output.max()) <= 1


def test_fit_pad_mask() -> None:
    raw = np.ones((10, 20), dtype=np.float32)
    image, mask = resize_speckle(raw, 32, "fit_pad")
    assert image.shape == (1, 32, 32)
    assert 0 < float(mask.mean()) < 1
