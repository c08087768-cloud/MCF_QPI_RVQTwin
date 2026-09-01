#!/usr/bin/env python3
"""随机可视化 MCF-QPI 散斑—相位配对，人工排查错配。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机显示 manifest 中的真实散斑—相位图像对")
    parser.add_argument("manifest")
    parser.add_argument("--split", default="train")
    parser.add_argument("--domain", default="all", choices=["all", "digits", "fashion", "unknown"])
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/data_inspection/random_pairs.png")
    return parser.parse_args()


def _load(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("F"), dtype=np.float32)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest, dtype=str).fillna("")
    if args.split != "all":
        frame = frame[frame.split == args.split]
    if args.domain != "all":
        frame = frame[frame.domain == args.domain]
    if frame.empty:
        raise RuntimeError("筛选后没有样本")
    sample = frame.sample(n=min(args.count, len(frame)), random_state=args.seed)

    rows = len(sample)
    figure, axes = plt.subplots(rows, 2, figsize=(8, max(2.2 * rows, 4)), squeeze=False)
    for row_index, row in enumerate(sample.itertuples(index=False)):
        speckle = _load(row.speckle_path)
        phase = _load(row.phase_path)
        axes[row_index, 0].imshow(speckle, cmap="gray")
        axes[row_index, 0].set_title(f"Speckle | {row.sample_id}", fontsize=8)
        axes[row_index, 1].imshow(phase, cmap="gray")
        axes[row_index, 1].set_title(f"Phase | {row.domain}/{row.split}", fontsize=8)
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"已保存配对抽查图：{output}")
    print("请人工确认同一行的散斑和相位索引一致；若配对依赖 natural_order_explicit，应多换几个 seed。")


if __name__ == "__main__":
    main()
