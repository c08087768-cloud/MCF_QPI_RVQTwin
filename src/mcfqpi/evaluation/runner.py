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
from .metrics import (
    expected_calibration_error,
    phase_metrics_batch,
    risk_coverage_curve,
    summarize_frame,
)
from .visualization import save_phase_comparison


def combine_predictive_uncertainty(
    phase_samples: torch.Tensor,
    scale_samples: torch.Tensor | None,
) -> torch.Tensor | None:
    """返回归一化相位标准差；Laplace(scale=b) 的方差为 2b²。"""
    if phase_samples.ndim < 2:
        raise ValueError("phase_samples 必须以 MC sample 为第一维")
    has_epistemic = phase_samples.shape[0] > 1
    if scale_samples is None and not has_epistemic:
        return None
    epistemic_var = (
        phase_samples.var(dim=0, unbiased=False)
        if has_epistemic
        else torch.zeros_like(phase_samples[0])
    )
    aleatoric_var = (
        2.0 * scale_samples.square().mean(dim=0)
        if scale_samples is not None
        else torch.zeros_like(epistemic_var)
    )
    return torch.sqrt((epistemic_var + aleatoric_var).clamp_min(0.0))


def laplace_mixture_nll(
    phase_samples: torch.Tensor,
    scale_samples: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算逐样本 MC-Laplace 混合负对数似然（归一化相位单位）。"""
    if phase_samples.shape != scale_samples.shape:
        raise ValueError("phase_samples 与 scale_samples 的形状必须一致")
    if phase_samples.ndim != target.ndim + 1 or phase_samples.shape[1:] != target.shape:
        raise ValueError("MC 样本形状必须为 [M, *target.shape]")
    scale = scale_samples.clamp_min(torch.finfo(scale_samples.dtype).eps)
    log_probability = -(
        (target.unsqueeze(0) - phase_samples).abs() / scale
        + torch.log(2.0 * scale)
    )
    mixture_log_probability = torch.logsumexp(log_probability, dim=0) - np.log(
        phase_samples.shape[0]
    )
    if mask is None:
        mask = torch.ones_like(target)
    valid = mask.to(dtype=torch.bool).expand_as(target)
    losses: list[torch.Tensor] = []
    for index in range(target.shape[0]):
        if not valid[index].any():
            raise ValueError("mask 中存在没有有效像素的样本")
        losses.append(-mixture_log_probability[index][valid[index]].mean())
    return torch.stack(losses)


def fit_laplace_temperature(
    validation_batches: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
    ],
) -> float:
    """仅用 validation 的 MC 预测拟合一个正标量 scale 温度。"""
    if not validation_batches:
        raise ValueError("温度校准至少需要一个 validation batch")
    device = validation_batches[0][0].device
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.5, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-6.0, 6.0).exp()
        losses = [
            laplace_mixture_nll(phases, scales * temperature, target, mask).mean()
            for phases, scales, target, mask in validation_batches
        ]
        loss = torch.stack(losses).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-6.0, 6.0).exp().cpu())


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
    uncertainty_temperature: float = 1.0,
) -> dict[str, Any]:
    if uncertainty_temperature <= 0:
        raise ValueError("uncertainty_temperature 必须为正数")
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
        scale_stack = (
            torch.stack(scale_samples, dim=0) * uncertainty_temperature
            if scale_samples
            else None
        )
        uncertainty_normalized = combine_predictive_uncertainty(
            phase_stack,
            scale_stack,
        )
        metric_rows = phase_metrics_batch(
            prediction,
            batch["phase"],
            batch.get("valid_mask"),
            uncertainty=(uncertainty_normalized * torch.pi)
            if uncertainty_normalized is not None
            else None,
        )
        ids = list(raw_batch["sample_id"])
        domains = list(raw_batch["domain"])
        splits = list(raw_batch["split"])
        classes = list(raw_batch.get("class_id", [""] * batch_size))
        per_sample_ms = elapsed * 1000.0 / max(batch_size * max(1, mc_samples), 1)
        mixture_nll = (
            laplace_mixture_nll(
                phase_stack, scale_stack, batch["phase"], batch.get("valid_mask")
            )
            if scale_stack is not None
            else None
        )
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
            if mixture_nll is not None:
                metrics["mixture_nll_normalized"] = float(mixture_nll[index].cpu())
            if uncertainty_normalized is not None:
                valid = (
                    batch["valid_mask"][index] > 0.5
                    if "valid_mask" in batch
                    else torch.ones_like(batch["phase"][index], dtype=torch.bool)
                )
                error = (prediction[index] - batch["phase"][index]).abs()
                std = uncertainty_normalized[index].clamp_min(1e-12)
                for label, multiplier in (("68", 1.0), ("90", 1.645), ("95", 1.96)):
                    metrics[f"coverage_{label}"] = float(
                        (error[valid] <= multiplier * std[valid]).float().mean().cpu()
                    )
            rows.append(metrics)

        if not first_visualized:
            save_phase_comparison(
                batch["speckle"].cpu(),
                batch["phase"].cpu(),
                prediction.cpu(),
                output_dir / "qualitative_examples.png",
                uncertainty_rad=(uncertainty_normalized * torch.pi).cpu()
                if uncertainty_normalized is not None
                else None,
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
    uncertainty_summary: dict[str, float] = {}
    if "uncertainty_mean" in frame:
        risk_coverage = risk_coverage_curve(frame)
        pd.DataFrame(risk_coverage).to_csv(output_dir / "risk_coverage.csv", index=False)
        uncertainty_summary = {
            "error_uncertainty_spearman": float(
                frame[["mae_rad", "uncertainty_mean"]].corr(method="spearman").iloc[0, 1]
            ),
            "ece_rad": expected_calibration_error(
                frame["mae_rad"].to_numpy(), frame["uncertainty_mean"].to_numpy()
            ),
        }
    summary = {
        "samples": int(len(frame)),
        "overall": overall,
        "by_domain": by_domain,
        "mc_samples": int(mc_samples),
        "wall_seconds": elapsed_total,
        "mean_forward_ms_per_sample": elapsed_total * 1000.0 / max(sample_total * max(1, mc_samples), 1),
        "risk_coverage": risk_coverage,
        "uncertainty_temperature": float(uncertainty_temperature),
        "uncertainty": uncertainty_summary,
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
