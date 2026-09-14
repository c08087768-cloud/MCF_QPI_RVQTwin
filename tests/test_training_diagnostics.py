from __future__ import annotations

import torch
import pytest

from mcfqpi.training.losses import InverseLoss, LossWeights


def test_weighted_terms_preserve_raw_terms() -> None:
    criterion = InverseLoss(LossWeights(phase_l1=1.0, gradient=0.15, token=0.2))
    raw = {
        "phase_l1": torch.tensor(2.0),
        "gradient": torch.tensor(4.0),
        "token": torch.tensor(3.0),
    }

    weighted = criterion.weighted_terms(raw)

    assert raw["gradient"].item() == 4.0
    assert weighted["weighted_phase_l1"].item() == 2.0
    assert weighted["weighted_gradient"].item() == pytest.approx(0.6)
    assert weighted["weighted_token"].item() == pytest.approx(0.6)


def test_refiner_diagnostics_are_optional() -> None:
    criterion = InverseLoss(LossWeights())
    diagnostics = criterion.model_diagnostics({
        "detail_scale": torch.tensor(0.05),
        "detail_phase": torch.tensor([[[[-2.0, 1.0]]]]),
        "phase_prior": torch.tensor([[[[0.2, -0.4]]]]),
    })

    assert diagnostics["detail_scale"].item() == pytest.approx(0.05)
    assert diagnostics["mean_abs_detail_phase"].item() == pytest.approx(1.5)
    assert diagnostics["mean_abs_phase_prior"].item() == pytest.approx(0.3)
    assert criterion.model_diagnostics({}) == {}
