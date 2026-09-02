#!/usr/bin/env python3
"""同 seed 配对、seed×sample 两层 bootstrap，并对多个模型做 Holm 校正。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcfqpi.config import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs=3, required=True)
    parser.add_argument("--model", action="append", nargs=4, metavar=("NAME", "SEED42", "SEED123", "SEED2026"), required=True)
    parser.add_argument("--metric", default="mae_rad")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    baseline = [pd.read_csv(path) for path in args.baseline]
    results = []
    for name, *paths in args.model:
        candidates = [pd.read_csv(path) for path in paths]
        paired = []
        for seed, (left, right) in enumerate(zip(baseline, candidates, strict=True)):
            merged = left[["sample_id", args.metric]].merge(
                right[["sample_id", args.metric]], on="sample_id", suffixes=("_baseline", "_model")
            )
            paired.append((merged[f"{args.metric}_model"] - merged[f"{args.metric}_baseline"]).to_numpy())
        draws = np.empty(args.samples)
        for draw in range(args.samples):
            chosen_seeds = rng.integers(0, len(paired), len(paired))
            seed_means = []
            for seed in chosen_seeds:
                values = paired[seed]
                seed_means.append(values[rng.integers(0, len(values), len(values))].mean())
            draws[draw] = np.mean(seed_means)
        p_value = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
        results.append({
            "model": name, "mean_paired_difference": float(np.mean([x.mean() for x in paired])),
            "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "p_value": float(p_value),
        })
    ordered = sorted(range(len(results)), key=lambda index: results[index]["p_value"])
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, results[index]["p_value"] * (len(results) - rank))
        running = max(running, adjusted)
        results[index]["holm_adjusted_p"] = running
    payload = {"metric": args.metric, "bootstrap_samples": args.samples, "comparisons": results}
    save_json(payload, Path(args.output))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
