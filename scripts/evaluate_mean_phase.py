#!/usr/bin/env python3
"""最弱的 sanity baseline：所有测试散斑都输出训练集逐像素平均相位。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from mcfqpi.config import load_config, save_json
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.metrics import phase_metrics_batch, summarize_frame
from mcfqpi.evaluation.visualization import save_phase_comparison
from mcfqpi.training.engine import build_loader, move_batch_to_device
from mcfqpi.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估训练集平均相位 sanity baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/mean_phase_baseline")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=True)
    device = select_device(str(config.get("device", "auto")))

    train_set = build_dataset(config["data"], "train", training=False)
    test_split = str(config.get("evaluation", {}).get("split", "test"))
    test_set = build_dataset(config["data"], test_split, training=False)
    loader_config = dict(config.get("loader", {}))
    train_loader = build_loader(train_set, loader_config, training=False, generator_seed=seed)
    test_loader = build_loader(test_set, loader_config, training=False, generator_seed=seed + 1)

    numerator: torch.Tensor | None = None
    denominator: torch.Tensor | None = None
    with torch.inference_mode():
        for raw in train_loader:
            batch = move_batch_to_device(raw, device)
            mask = batch.get("valid_mask", torch.ones_like(batch["phase"]))
            current_num = (batch["phase"] * mask).sum(dim=0, keepdim=True)
            current_den = mask.sum(dim=0, keepdim=True)
            numerator = current_num if numerator is None else numerator + current_num
            denominator = current_den if denominator is None else denominator + current_den
    if numerator is None or denominator is None:
        raise RuntimeError("训练集为空，无法计算平均相位")
    mean_phase = numerator / denominator.clamp_min(1.0)

    rows: list[dict[str, Any]] = []
    first: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    with torch.inference_mode():
        for raw in test_loader:
            batch = move_batch_to_device(raw, device)
            prediction = mean_phase.expand(batch["phase"].shape[0], -1, -1, -1)
            metrics = phase_metrics_batch(prediction, batch["phase"], batch.get("valid_mask"))
            for index, row in enumerate(metrics):
                row.update(
                    {
                        "sample_id": str(raw["sample_id"][index]),
                        "domain": str(raw["domain"][index]),
                        "split": str(raw["split"][index]),
                    }
                )
                rows.append(row)
            if first is None:
                first = (batch["speckle"][:6].cpu(), batch["phase"][:6].cpu(), prediction[:6].cpu())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    summary = {
        "split": test_split,
        "samples": int(len(frame)),
        "overall": summarize_frame(frame, metrics=numeric, bootstrap_samples=2000, seed=seed),
        "by_domain": {
            str(domain): summarize_frame(group, metrics=numeric, bootstrap_samples=2000, seed=seed)
            for domain, group in frame.groupby("domain")
        },
        "interpretation": "该模型完全忽略散斑输入，只用于发现数据泄漏或评价指标虚高。",
    }
    save_json(summary, output_dir / "summary.json")
    torch.save(mean_phase.cpu(), output_dir / "mean_phase.pt")
    if first is not None:
        save_phase_comparison(*first, output_dir / "qualitative_examples.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
