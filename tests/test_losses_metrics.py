from __future__ import annotations

import pandas as pd
import torch

from mcfqpi.evaluation.metrics import phase_metrics_batch, risk_coverage_curve, summarize_frame
from mcfqpi.training.losses import InverseLoss, LossWeights, PhasePriorLoss


def test_phase_prior_loss_is_finite() -> None:
    target = torch.rand(2, 1, 32, 32)
    outputs = {"phase": target * 0.9, "loss": torch.tensor(0.2, requires_grad=True)}
    loss, terms = PhasePriorLoss()(outputs, target)
    assert torch.isfinite(loss)
    assert set(terms) == {"l1", "gradient", "ssim", "vq"}


def test_inverse_loss_baseline() -> None:
    target = torch.rand(2, 1, 32, 32)
    outputs = {"phase": target * 0.95, "log_scale": torch.zeros_like(target)}
    batch = {"phase": target, "speckle": torch.rand_like(target), "valid_mask": torch.ones_like(target)}
    loss, terms = InverseLoss(LossWeights(phase_nll=0.1))(outputs, batch)
    assert torch.isfinite(loss)
    assert "phase_l1" in terms


def test_metrics_perfect_prediction() -> None:
    target = torch.rand(3, 1, 32, 32)
    rows = phase_metrics_batch(target, target)
    assert len(rows) == 3
    assert max(row["mae_rad"] for row in rows) < 1e-7
    assert min(row["pearson_r"] for row in rows) > 0.999


def test_summary_and_risk_coverage() -> None:
    frame = pd.DataFrame({"mae_rad": [0.1, 0.2, 0.3], "uncertainty_mean": [0.1, 0.2, 0.3]})
    summary = summarize_frame(frame, bootstrap_samples=10)
    curve = risk_coverage_curve(frame, coverages=[1.0, 0.5])
    assert "mae_rad" in summary
    assert len(curve) == 2
    assert curve[1]["risk"] <= curve[0]["risk"]
