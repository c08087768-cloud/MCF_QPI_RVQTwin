#!/usr/bin/env python3
"""评估 Residual-VQ-VAE 相位先验的重建上限与码本使用情况。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mcfqpi.config import load_config, save_json
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.metrics import phase_metrics_batch, summarize_frame
from mcfqpi.evaluation.visualization import save_phase_comparison
from mcfqpi.factory import build_phase_prior, load_model_weights
from mcfqpi.training.engine import build_loader, move_batch_to_device
from mcfqpi.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估相位 RVQ-VAE 的重建性能与码本占用")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="outputs/phase_prior_evaluation")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def _token_statistics(counts: list[np.ndarray]) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    for stage, current in enumerate(counts):
        total = int(current.sum())
        probability = current / max(total, 1)
        nonzero = probability > 0
        entropy = float(-(probability[nonzero] * np.log(probability[nonzero])).sum())
        perplexity = float(np.exp(entropy))
        result.append(
            {
                "stage": int(stage),
                "codebook_size": int(current.size),
                "used_codes": int(nonzero.sum()),
                "dead_codes": int((~nonzero).sum()),
                "dead_code_fraction": float((~nonzero).mean()),
                "empirical_perplexity": perplexity,
                "normalized_perplexity": perplexity / max(int(current.size), 1),
                "token_count": total,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=True)
    device = select_device(str(config.get("device", "auto")))

    dataset = build_dataset(config["data"], args.split, training=False)
    loader = build_loader(dataset, config.get("loader", {}), training=False, generator_seed=seed)
    model = build_phase_prior(config.get("model", {})).to(device).eval()
    load_model_weights(model, args.checkpoint, map_location=device, strict=True)

    codebook_size = int(config.get("model", {}).get("codebook_size", 256))
    num_quantizers = int(config.get("model", {}).get("num_quantizers", 2))
    token_counts = [np.zeros(codebook_size, dtype=np.int64) for _ in range(num_quantizers)]
    rows: list[dict[str, Any]] = []
    first: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    with torch.inference_mode():
        for batch_index, raw in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            batch = move_batch_to_device(raw, device)
            outputs = model(batch["phase"])
            prediction = outputs["phase"].float()
            metrics = phase_metrics_batch(prediction, batch["phase"], batch.get("valid_mask"))
            indices = outputs["indices"]
            assert isinstance(indices, torch.Tensor)
            for stage in range(indices.shape[1]):
                values = indices[:, stage].detach().cpu().reshape(-1).numpy()
                token_counts[stage] += np.bincount(values, minlength=codebook_size)
            batch_size = prediction.shape[0]
            for index in range(batch_size):
                metrics[index].update(
                    {
                        "sample_id": str(raw["sample_id"][index]),
                        "domain": str(raw["domain"][index]),
                        "split": str(raw["split"][index]),
                    }
                )
                rows.append(metrics[index])
            if first is None:
                # 可视化函数第一列名为 Input speckle；这里传入真值相位，仅用于展示先验输入。
                first = (batch["phase"][:6].cpu(), batch["phase"][:6].cpu(), prediction[:6].cpu())

    if not rows:
        raise RuntimeError("相位先验评估没有产生样本")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    summary = {
        "split": args.split,
        "samples": int(len(frame)),
        "overall": summarize_frame(frame, metrics=numeric, bootstrap_samples=2000, seed=seed),
        "by_domain": {
            str(domain): summarize_frame(group, metrics=numeric, bootstrap_samples=2000, seed=seed)
            for domain, group in frame.groupby("domain")
        },
        "codebook": _token_statistics(token_counts),
        "interpretation": (
            "该结果是相位先验对真实标签的自编码上限，不是散斑到相位的逆问题结果。"
            "若先验自身误差过大或死码率过高，应先调码本和潜变量尺寸，再训练主模型。"
        ),
    }
    save_json(summary, output_dir / "summary.json")
    pd.DataFrame(summary["codebook"]).to_csv(output_dir / "codebook_statistics.csv", index=False)
    if first is not None:
        save_phase_comparison(*first, output_dir / "qualitative_examples.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
