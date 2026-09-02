#!/usr/bin/env python3
"""在 validation 上检验离散码是否真正影响 proposed 模型。"""
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="RVQ 占用、旁路比例和 token 干预诊断")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    device = select_device(str(config.get("device", "auto")))
    model, _ = build_inverse_from_checkpoint(args.checkpoint, map_location="cpu")
    if not isinstance(model, DualDomainRVQTwin):
        raise TypeError("该诊断只适用于 DualDomainRVQTwin")
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
                    for level, indices in enumerate(output["indices"]):
                        code_indices[level].append(indices.detach().cpu().numpy())
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
