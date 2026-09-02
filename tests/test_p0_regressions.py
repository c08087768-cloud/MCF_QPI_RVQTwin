from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from mcfqpi.evaluation.runner import combine_predictive_uncertainty, laplace_mixture_nll
from mcfqpi.models.rvq import ResidualVectorQuantizer
from mcfqpi.training.engine import _run_epoch
from mcfqpi.training.losses import InverseLoss, LossWeights


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
def test_residual_vq_ste_gradient_is_independent_of_stage_count(stages: int) -> None:
    rvq = ResidualVectorQuantizer(codebook_size=4, embedding_dim=2, num_quantizers=stages)
    x = torch.randn(1, 2, 3, 3, requires_grad=True)

    rvq(x)["quantized"].sum().backward()

    assert torch.allclose(x.grad, torch.ones_like(x))


def test_residual_vq_uses_mean_loss_across_stages() -> None:
    one = ResidualVectorQuantizer(codebook_size=4, embedding_dim=2, num_quantizers=1)
    three = ResidualVectorQuantizer(codebook_size=4, embedding_dim=2, num_quantizers=3)
    with torch.no_grad():
        for module in (one, three):
            for quantizer in module.quantizers:
                quantizer.embedding.weight.zero_()
    x = torch.ones(1, 2, 2, 2)

    one_loss = one(x)["loss"]
    three_loss = three(x)["loss"]

    assert torch.allclose(three_loss, one_loss)


class _IdentityForwardTwin(nn.Module):
    def forward(self, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"speckle": phase}


def test_cycle_loss_uses_clean_speckle_target() -> None:
    prediction = torch.zeros(1, 1, 4, 4)
    batch = {
        "phase": prediction,
        "speckle": torch.ones_like(prediction),
        "speckle_clean": torch.full_like(prediction, 0.25),
        "valid_mask": torch.ones_like(prediction),
    }
    criterion = InverseLoss(LossWeights(cycle_l1=1.0))

    _, terms = criterion({"phase": prediction}, batch, forward_twin=_IdentityForwardTwin())

    assert terms["cycle_l1"].item() == pytest.approx(0.25)


def test_cycle_loss_rejects_missing_clean_target() -> None:
    prediction = torch.zeros(1, 1, 4, 4)
    batch = {
        "phase": prediction,
        "speckle": torch.ones_like(prediction),
        "valid_mask": torch.ones_like(prediction),
    }
    criterion = InverseLoss(LossWeights(cycle_l1=1.0))

    with pytest.raises(KeyError, match="speckle_clean"):
        criterion({"phase": prediction}, batch, forward_twin=_IdentityForwardTwin())


def test_predictive_uncertainty_combines_laplace_variance() -> None:
    phases = torch.tensor([[[[[0.0]]]], [[[[2.0]]]]])
    scales = torch.tensor([[[[[3.0]]]], [[[[3.0]]]]])

    uncertainty = combine_predictive_uncertainty(phases, scales)

    # Population epistemic variance=1; Laplace aleatoric variance=2*3^2=18.
    assert uncertainty.item() == pytest.approx(19.0**0.5)


def test_predictive_uncertainty_is_absent_without_scale_or_mc_sampling() -> None:
    phases = torch.zeros(1, 1, 1, 2, 2)

    assert combine_predictive_uncertainty(phases, None) is None


def test_laplace_mixture_nll_has_correct_density_units() -> None:
    phases = torch.zeros(2, 1, 1, 1, 1)
    scales = torch.ones_like(phases)
    target = torch.zeros(1, 1, 1, 1)

    nll = laplace_mixture_nll(phases, scales, target)

    assert nll.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def _constant_gradient_step(
    model: nn.Module, batch: dict[str, torch.Tensor], training: bool
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    del training
    assert isinstance(model, _ScalarModel)
    loss = model.weight * batch["phase"].mean()
    return loss, {"phase_l1": loss.detach()}, {"phase": batch["phase"]}


@pytest.mark.parametrize(
    ("batch_count", "max_batches", "expected_weight", "expected_steps"),
    [(4, 0, 0.8, 2), (5, 0, 0.7, 3), (5, 3, 0.8, 2)],
)
def test_gradient_accumulation_steps_tail_groups(
    batch_count: int, max_batches: int, expected_weight: float, expected_steps: int
) -> None:
    model = _ScalarModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    loader = DataLoader([{"phase": torch.ones(1)} for _ in range(batch_count)], batch_size=None)

    _run_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        step_function=_constant_gradient_step,
        optimizer=optimizer,
        scaler=None,
        amp_enabled=True,
        amp_dtype="float16",
        gradient_clip=0.0,
        accumulation_steps=2,
        max_batches=max_batches,
        batch_scheduler=scheduler,
    )

    assert model.weight.item() == pytest.approx(expected_weight)
    assert scheduler.last_epoch == expected_steps


def test_training_rejects_empty_loader() -> None:
    model = _ScalarModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    loader = DataLoader([], batch_size=None)

    with pytest.raises(ValueError, match="empty|为空"):
        _run_epoch(
            model,
            loader,
            device=torch.device("cpu"),
            step_function=_constant_gradient_step,
            optimizer=optimizer,
            scaler=None,
            amp_enabled=False,
            amp_dtype="float16",
            gradient_clip=0.0,
            accumulation_steps=2,
            max_batches=0,
        )
