#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mcfqpi.config import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="按预注册门槛决定 clean-cycle 能否进入主模型")
    parser.add_argument("--forward-summary", required=True)
    parser.add_argument("--cycle-off", nargs=3, required=True, metavar="CSV")
    parser.add_argument("--clean-cycle", nargs=3, required=True, metavar="CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    forward = json.loads(Path(args.forward_summary).read_text(encoding="utf-8"))
    forward_groups = [forward["overall"], *forward.get("by_domain", {}).values()]
    forward_pass = (
        all(group["pearson_r"]["mean"] >= 0.99 for group in forward_groups)
        and forward["overall"]["mae"]["mean"] <= 0.01
    )
    seed_rows = []
    domain_accumulator = {domain: {"off_mae": [], "cycle_mae": [], "off_ssim": [], "cycle_ssim": []}
                          for domain in ("digits", "fashion")}
    for off_path, cycle_path in zip(args.cycle_off, args.clean_cycle, strict=True):
        off, cycle = pd.read_csv(off_path), pd.read_csv(cycle_path)
        seed_rows.append({
            "off_mae": float(off.mae_rad.mean()),
            "cycle_mae": float(cycle.mae_rad.mean()),
            "relative_improvement": float(1 - cycle.mae_rad.mean() / off.mae_rad.mean()),
        })
        for domain in domain_accumulator:
            left, right = off[off.domain == domain], cycle[cycle.domain == domain]
            domain_accumulator[domain]["off_mae"].append(left.mae_rad.mean())
            domain_accumulator[domain]["cycle_mae"].append(right.mae_rad.mean())
            domain_accumulator[domain]["off_ssim"].append(left.ssim.mean())
            domain_accumulator[domain]["cycle_ssim"].append(right.ssim.mean())
    mean_improvement = sum(row["relative_improvement"] for row in seed_rows) / 3
    improved_seeds = sum(row["relative_improvement"] > 0 for row in seed_rows)
    domain_pass = all(
        sum(values["cycle_mae"]) / 3 <= sum(values["off_mae"]) / 3
        and sum(values["cycle_ssim"]) / 3 >= sum(values["off_ssim"]) / 3
        for values in domain_accumulator.values()
    )
    passed = forward_pass and mean_improvement >= 0.02 and improved_seeds >= 2 and domain_pass
    report = {
        "passed": passed,
        "decision": "clean-cycle" if passed else "cycle-off",
        "forward_pass": forward_pass,
        "mean_relative_mae_improvement": mean_improvement,
        "improved_seed_count": improved_seeds,
        "domain_non_degradation_pass": domain_pass,
        "seed_results": seed_rows,
    }
    save_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
