from pathlib import Path

import torch

from mcfqpi.factory import build_inverse_architecture, build_inverse_from_checkpoint
from mcfqpi.utils import save_inference_checkpoint


def _proposed_config() -> dict:
    return {
        "type": "proposed",
        "base_channels": 8,
        "dropout": 0.0,
        "phase_prior_model": {
            "latent_channels": 16,
            "base_channels": 8,
            "downsample_stages": 3,
            "codebook_size": 16,
            "num_quantizers": 2,
            "dropout": 0.0,
        },
    }


def test_proposed_architecture_does_not_require_phase_prior_file() -> None:
    model = build_inverse_architecture(_proposed_config())
    assert model(torch.rand(1, 1, 32, 32))["phase"].shape == (1, 1, 32, 32)


def test_inference_checkpoint_is_self_contained(tmp_path: Path) -> None:
    config = _proposed_config()
    model = build_inverse_architecture(config)
    path = tmp_path / "inference.pt"
    save_inference_checkpoint(model, path, model_config=config, metadata={"schema_version": "1.0"})

    loaded, checkpoint = build_inverse_from_checkpoint(path)
    assert checkpoint["model_config"] == config
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)
