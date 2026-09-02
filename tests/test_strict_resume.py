from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from mcfqpi.training.engine import build_loader, fit_model
from mcfqpi.utils import seed_everything


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict:
        value = torch.full((1, 2, 2), index / 8)
        return {"speckle": value, "phase": value * 0.5}


def _train(output: Path, epochs: int, resume: Path | None = None) -> nn.Module:
    loader_config = {"batch_size": 2, "num_workers": 0, "drop_last": False}
    train = build_loader(_TinyDataset(), loader_config, training=True, generator_seed=42)
    val = build_loader(_TinyDataset(), loader_config, training=False, generator_seed=43)
    model = nn.Conv2d(1, 1, 1)

    def step(current, batch, training):
        prediction = current(batch["speckle"])
        loss = (prediction - batch["phase"]).square().mean()
        return loss, {"phase_l1": loss}, {"phase": prediction}

    fit_model(
        model, train, val, device=torch.device("cpu"), step_function=step,
        training_config={
            "epochs": epochs, "amp": False, "early_stopping_patience": 0,
            "optimizer": {"name": "sgd", "lr": 0.1},
            "scheduler": {"name": "none"},
        },
        full_config={"model": {"type": "resunet", "base_channels": 1}},
        output_dir=output, resume_checkpoint=resume,
    )
    return model


def test_strict_resume_matches_continuous_training(tmp_path: Path) -> None:
    seed_everything(7, deterministic=True)
    continuous = _train(tmp_path / "continuous", 4)
    continuous_state = {key: value.clone() for key, value in continuous.state_dict().items()}

    seed_everything(7, deterministic=True)
    _train(tmp_path / "resumed", 2)
    resumed = _train(tmp_path / "resumed", 4, tmp_path / "resumed" / "last.pt")

    for key, expected in continuous_state.items():
        assert torch.equal(expected, resumed.state_dict()[key])
