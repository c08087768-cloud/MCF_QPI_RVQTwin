#!/usr/bin/env python3
"""聚合多个评估目录的 summary.json，生成论文主表草稿。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总多个模型的评估结果")
    parser.add_argument("summaries", nargs="+", help="一个或多个 evaluation summary.json")
    parser.add_argument("--output", default="outputs/comparison_table.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = [
        "mae_normalized", "mae_rad", "rmse_rad", "fidelity_2d_correlation",
        "ssim", "psnr_db", "gradient_mae_rad_per_pixel", "uncertainty_mean",
    ]
    rows = []
    for raw_path in args.summaries:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"experiment": path.parent.name, "summary_path": str(path)}
        overall = payload.get("overall", {})
        for metric in metrics:
            stats = overall.get(metric)
            if stats:
                row[metric] = stats.get("mean")
                row[f"{metric}_ci_low"] = stats.get("ci_low")
                row[f"{metric}_ci_high"] = stats.get("ci_high")
        row["mean_forward_ms_per_sample"] = payload.get("mean_forward_ms_per_sample")
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"\n写入：{output}")


if __name__ == "__main__":
    main()
