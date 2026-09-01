#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从训练 split 构造可复现的少样本 manifest")
    parser.add_argument("manifest")
    parser.add_argument("--shots", type=int, required=True, help="每个 domain 最多保留多少个训练样本")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest, dtype=str).fillna("")
    parts: list[pd.DataFrame] = []
    for domain, group in frame.groupby("domain", sort=True):
        train = group[group.split == "train"].sample(frac=1, random_state=args.seed).iloc[: args.shots]
        # val/test 全部保留，确保不同 shots 的比较使用完全相同评估集。
        parts.extend([train, group[group.split != "train"]])
    output_frame = pd.concat(parts, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False)
    print(output_frame.groupby(["domain", "split"]).size())
    print(f"写入：{output}")


if __name__ == "__main__":
    main()
