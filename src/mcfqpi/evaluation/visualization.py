from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_phase_comparison(
    speckle: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: str | Path,
    *,
    uncertainty: torch.Tensor | None = None,
    max_items: int = 6,
) -> Path:
    """保存散斑、真值、预测、误差和可选不确定度图。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_items, speckle.shape[0])
    columns = 5 if uncertainty is not None else 4
    figure, axes = plt.subplots(count, columns, figsize=(3.2 * columns, 3.0 * count), squeeze=False)
    for row in range(count):
        s = speckle[row, 0].detach().cpu().numpy()
        t = target[row, 0].detach().cpu().numpy() * np.pi
        p = prediction[row, 0].detach().cpu().numpy() * np.pi
        error = np.abs(p - t)
        entries = [
            (s, "Input speckle", None),
            (t, "Ground-truth phase / rad", (0, np.pi)),
            (p, "Predicted phase / rad", (0, np.pi)),
            (error, "Absolute error / rad", None),
        ]
        if uncertainty is not None:
            u = uncertainty[row, 0].detach().cpu().numpy() * np.pi
            entries.append((u, "Predicted scale / rad", None))
        for column, (image, title, limits) in enumerate(entries):
            kwargs = {}
            if limits is not None:
                kwargs = {"vmin": limits[0], "vmax": limits[1]}
            axis = axes[row, column]
            rendered = axis.imshow(image, **kwargs)
            axis.set_title(title)
            axis.axis("off")
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def save_loss_curves(history: list[dict[str, float]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return path
    epochs = [int(row["epoch"]) for row in history]
    figure = plt.figure(figsize=(8, 5))
    for key in ("train_loss", "val_loss"):
        values = [row.get(key, np.nan) for row in history]
        plt.plot(epochs, values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
