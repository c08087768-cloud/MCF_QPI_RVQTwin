#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mcfqpi.config import save_json
from mcfqpi.data.pairing import discover_pairs
from mcfqpi.data.split import assign_group_splits, split_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动配对 MCF-QPI 散斑和相位图，生成 manifest.csv")
    parser.add_argument("root", help="解压后的数据根目录")
    parser.add_argument("--output", default="data/processed/mcf_qpi_manifest.csv")
    parser.add_argument("--diagnostics", default="outputs/data_inspection/pairing_diagnostics.json")
    parser.add_argument(
        "--allow-order-pairing",
        action="store_true",
        help="仅当同组 speckle/phase 数量严格相等时允许按自然顺序配对；使用后必须人工抽查",
    )
    parser.add_argument("--compute-hashes", action="store_true", help="计算图像 SHA-256/感知哈希，较慢但可审计泄漏")
    parser.add_argument("--assign-unknown-splits", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.88)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, diagnostics = discover_pairs(
        args.root,
        allow_order_pairing=args.allow_order_pairing,
        compute_hashes=args.compute_hashes,
    )
    if not records:
        raise RuntimeError(
            "没有发现配对。先运行 inspect_raw_dataset.py 查看目录关键词；必要时修改 "
            "src/mcfqpi/data/pairing.py 的 SPECKLE_WORDS/PHASE_WORDS，或明确使用 --allow-order-pairing。"
        )
    if args.assign_unknown_splits:
        records = assign_group_splits(
            records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            preserve_known=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([record.to_dict() for record in records])
    frame.to_csv(output, index=False)
    diagnostics["split_counts"] = split_counts(records)
    diagnostics["manifest"] = str(output.resolve())
    save_json(diagnostics, args.diagnostics)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"\n已写入 {len(frame):,} 对：{output}")
    print("重要：若使用 natural_order_explicit，请随机可视化至少 100 对确认没有错配。")


if __name__ == "__main__":
    main()
