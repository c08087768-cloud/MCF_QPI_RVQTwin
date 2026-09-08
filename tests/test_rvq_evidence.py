from __future__ import annotations

import pytest
import torch

from mcfqpi.models import DualDomainRVQTwin, PhaseRVQVAE
from scripts.evaluate_rvq_evidence import apply_residual_scale, collect_code_indices


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


def test_collect_code_indices_groups_bqhw_tensor_by_quantizer_level() -> None:
    indices = torch.tensor(
        [
            [[[0, 1]], [[4, 5]]],
            [[[2, 3]], [[6, 7]]],
        ]
    )

    result = collect_code_indices(indices, num_quantizers=2)

    assert len(result) == 2
    assert result[0].tolist() == [[[0, 1]], [[2, 3]]]
    assert result[1].tolist() == [[[4, 5]], [[6, 7]]]
