from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mcfqpi.training.losses import token_cross_entropy


def test_token_cross_entropy_matches_pytorch_value_and_gradients() -> None:
    torch.manual_seed(7)
    actual_logits = [torch.randn(2, 5, 3, 4, requires_grad=True) for _ in range(2)]
    reference_logits = [value.detach().clone().requires_grad_(True) for value in actual_logits]
    targets = torch.randint(0, 5, (2, 2, 3, 4))

    actual = token_cross_entropy(actual_logits, targets)
    reference = torch.stack(
        [F.cross_entropy(value, targets[:, stage]) for stage, value in enumerate(reference_logits)]
    ).mean()
    actual.backward()
    reference.backward()

    assert actual.item() == pytest.approx(reference.item(), rel=1e-6)
    for actual_stage, reference_stage in zip(actual_logits, reference_logits, strict=True):
        assert torch.allclose(actual_stage.grad, reference_stage.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 验证严格确定性路径")
def test_token_cross_entropy_supports_strict_cuda_determinism() -> None:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        logits = [torch.randn(2, 7, 4, 4, device="cuda", requires_grad=True) for _ in range(2)]
        targets = torch.randint(0, 7, (2, 2, 4, 4), device="cuda")

        loss = token_cross_entropy(logits, targets)
        loss.backward()

        assert torch.isfinite(loss)
        assert all(value.grad is not None for value in logits)
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
