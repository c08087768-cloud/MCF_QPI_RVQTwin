from __future__ import annotations

import pytest

from mcfqpi.models import DualDomainRVQTwin, PhaseRVQVAE
from scripts.evaluate_rvq_evidence import apply_residual_scale


def _model() -> DualDomainRVQTwin:
    prior = PhaseRVQVAE(
        latent_channels=16,
        base_channels=8,
        downsample_stages=3,
        codebook_size=16,
        num_quantizers=2,
        dropout=0.0,
    )
    return DualDomainRVQTwin(prior, base_channels=8, dropout=0.0, residual_scale=0.15)


def test_apply_residual_scale_overrides_model_without_reloading_weights() -> None:
    model = _model()
    original_prior = model.phase_prior

    apply_residual_scale(model, 0.05)

    assert model.residual_scale == pytest.approx(0.05)
    assert model.phase_prior is original_prior


def test_apply_residual_scale_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="非负"):
        apply_residual_scale(_model(), -0.01)
