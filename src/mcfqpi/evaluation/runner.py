from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import save_json
from ..training.engine import move_batch_to_device
from .metrics import phase_metrics_batch, risk_coverage_curve, summarize_frame
from .visualization import save_phase_comparison


def _enable_mc_dropout(model: nn.Module) -> None:
    """模型整体保持 eval，仅将 Dropout 层切回 train 以进行 MC Dropout。"""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def evaluate_phase_model(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    output_dir: str | Path,
    mc_samples: int = 1,
    amp: bool = True,
    max_batches: int = 0,
    bootstrap_samples: int = 2000,
    save_predictions: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    if mc_samples > 1:
        _enable_mc_dropout(model)
    else:
        model.eval()

    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    elapsed_total = 0.0
    sample_total = 0
    first_visualized = False

    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch_to_device(raw_batch, device)
        phase_samples: list[torch.Tensor] = []
        scale_samples: list[torch.Tensor] = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max(1, mc_samples)):
            with _autocast(device, amp):
                outputs = model(batch["speckle"])
            phase_samples.append(outputs["phase"].float())
            if "log_scale" in outputs:
                scale_samples.append(torch.exp(outputs["log_scale"].float()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        batch_size = int(batch["phase"].shape[0])
        elapsed_total += elapsed
        sample_total += batch_size

        phase_stack = torch.stack(phase_samples, dim=0)
        prediction = phase_stack.mean(dim=0)
        epistemic = phase_stack.std(dim=0, unbiased=False) if mc_samples > 1 else torch.zeros_like(prediction)
        if scale_samples:
            aleatoric = torch.stack(scale_samples, dim=0).mean(dim=0)
        else:
            aleatoric = torch.zeros_like(prediction)
        uncertainty_normalized = torch.sqrt(epistemic.square() + aleatoric.square())
        metric_rows = phase_metrics_batch(
            prediction,
            batch["phase"],
            batch.get("valid_mask"),
            uncertainty=uncertainty_normalized * torch.pi,
        )
        ids = list(raw_batch["sample_id"])
        domains = list(raw_batch["domain"])
        splits = list(raw_batch["split"])
        classes = list(raw_batch.get("class_id", [""] * batch_size))
        per_sample_ms = elapsed * 1000.0 / max(batch_size * max(1, mc_samples), 1)
        for index, metrics in enumerate(metric_rows):
            metrics.update(
                {
                    "sample_id": str(ids[index]),
                    "domain": str(domains[index]),
                    "split": str(splits[index]),
                    "class_id": str(classes[index]),
                    "inference_ms_per_forward": per_sample_ms,
                }
            )
            rows.append(metrics)

        if not first_visualized:
            save_phase_comparison(
                batch["speckle"].cpu(),
                batch["phase"].cpu(),
                prediction.cpu(),
                output_dir / "qualitative_examples.png",
                uncertainty=(uncertainty_normalized * torch.pi).cpu(),
            )
            first_visualized = True
        if save_predictions:
            predictions.append(prediction.cpu().numpy().astype(np.float16))
            targets.append(batch["phase"].cpu().numpy().astype(np.float16))
            sample_ids.extend(str(value) for value in ids)

    if not rows:
        raise RuntimeError("评估没有产生任何样本")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and column != "inference_ms_per_forward"
    ]
    overall = summarize_frame(frame, metrics=numeric_columns, bootstrap_samples=bootstrap_samples)
    by_domain = {
        str(domain): summarize_frame(group, metrics=numeric_columns, bootstrap_samples=bootstrap_samples)
        for domain, group in frame.groupby("domain")
    }
    risk_coverage = []
    if "uncertainty_mean" in frame:
        risk_coverage = risk_coverage_curve(frame)
        pd.DataFrame(risk_coverage).to_csv(output_dir / "risk_coverage.csv", index=False)
    summary = {
        "samples": int(len(frame)),
        "overall": overall,
        "by_domain": by_domain,
        "mc_samples": int(mc_samples),
        "wall_seconds": elapsed_total,
        "mean_forward_ms_per_sample": elapsed_total * 1000.0 / max(sample_total * max(1, mc_samples), 1),
        "risk_coverage": risk_coverage,
    }
    save_json(summary, output_dir / "summary.json")
    if save_predictions:
        np.savez_compressed(
            output_dir / "predictions.npz",
            prediction=np.concatenate(predictions),
            target=np.concatenate(targets),
            sample_id=np.asarray(sample_ids),
        )
    return summary
