#!/usr/bin/env python3
"""在 validation 上检验离散码是否真正影响 proposed 模型。"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mcfqpi.config import load_config, save_json
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.metrics import phase_metrics_batch
from mcfqpi.factory import build_inverse_from_checkpoint
from mcfqpi.models import DualDomainRVQTwin
from mcfqpi.training.engine import build_loader, move_batch_to_device
from mcfqpi.utils import select_device


def apply_residual_scale(model: DualDomainRVQTwin, residual_scale: float) -> None:
    """覆盖连续旁路强度，不重新加载或修改 checkpoint 权重。"""
    if not math.isfinite(residual_scale) or residual_scale < 0.0:
        raise ValueError("residual_scale 必须是有限的非负数")
    model.residual_scale = float(residual_scale)


def collect_code_indices(
    indices: torch.Tensor, *, num_quantizers: int
) -> list[torch.Tensor]:
    """按 RVQ 级别拆分 [B,Q,H,W] 的 token 索引。"""
    if indices.ndim != 4 or indices.shape[1] != num_quantizers:
        raise ValueError(
            f"indices 应为 [B,{num_quantizers},H,W]，实际 {tuple(indices.shape)}"
        )
    return [indices[:, level] for level in range(num_quantizers)]


def main() -> None:
    parser = argparse.ArgumentParser(description="RVQ 占用、旁路比例和 token 干预诊断")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="仅在评估时覆盖连续残差尺度；不修改 checkpoint。",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = select_device(str(config.get("device", "auto")))
    model, _ = build_inverse_from_checkpoint(args.checkpoint, map_location="cpu")
    if not isinstance(model, DualDomainRVQTwin):
        raise TypeError("该诊断只适用于 DualDomainRVQTwin")
    if args.residual_scale is not None:
        apply_residual_scale(model, args.residual_scale)
    model.to(device).eval()
    dataset = build_dataset(config["data"], "val", training=False)
    loader = build_loader(dataset, config.get("loader", {}), training=False)
    errors: dict[str, list[float]] = defaultdict(list)
    ratios: list[float] = []
    code_indices: list[list[np.ndarray]] = [
        [] for _ in range(model.phase_prior.quantizer.num_quantizers)
    ]
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            for intervention in ("none", "shuffle", "mean"):
                output = model(batch["speckle"], token_intervention=intervention)
                rows = phase_metrics_batch(output["phase"], batch["phase"], batch.get("valid_mask"))
                errors[intervention].extend(row["mae_rad"] for row in rows)
                if intervention == "none":
                    ratios.extend(
                        (output["continuous_residual"].flatten(1).norm(dim=1)
                         / output["z_q"].flatten(1).norm(dim=1).clamp_min(1e-12)).cpu().tolist()
                    )
                    indices = output["indices"]
                    assert isinstance(indices, torch.Tensor)
                    for level, stage_indices in enumerate(
                        collect_code_indices(
                            indices,
                            num_quantizers=model.phase_prior.quantizer.num_quantizers,
                        )
                    ):
                        code_indices[level].append(stage_indices.detach().cpu().numpy())
    occupancy = []
    for level, chunks in enumerate(code_indices):
        values = np.concatenate(chunks).reshape(-1)
        counts = np.bincount(values, minlength=model.phase_prior.quantizer.codebook_size)
        probabilities = counts[counts > 0] / counts.sum()
        occupancy.append({
            "level": level + 1,
            "used_codes": int((counts > 0).sum()),
            "codebook_size": int(counts.size),
            "perplexity": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
        })
    baseline = float(np.mean(errors["none"]))
    report = {
        "split": "val",
        "residual_scale": model.residual_scale,
        "mae_rad": {key: float(np.mean(value)) for key, value in errors.items()},
        "relative_mae_change": {
            key: float(np.mean(value) / baseline - 1.0) for key, value in errors.items()
        },
        "continuous_to_quantized_norm_ratio": {
            "mean": float(np.mean(ratios)), "std": float(np.std(ratios))
        },
        "codebooks": occupancy,
    }
    save_json(report, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
