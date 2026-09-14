import torch

from mcfqpi.training.engine import nonfinite_loss_message


def test_nonfinite_loss_message_reports_batch_and_nonfinite_terms() -> None:
    message = nonfinite_loss_message(
        torch.tensor(float("nan")),
        {
            "phase_l1": torch.tensor(0.1),
            "phase_nll": torch.tensor(float("nan")),
            "latent": torch.tensor(float("inf")),
        },
        batch_index=17,
    )

    assert "batch=17" in message
    assert "loss=nan" in message
    assert "phase_nll=nan" in message
    assert "latent=inf" in message
    assert "phase_l1" not in message
