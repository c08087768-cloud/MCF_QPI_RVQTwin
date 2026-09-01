#!/usr/bin/env python3
"""创建严格的按标签哈希分组划分，作为官方 split 之外的防泄漏实验。"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import pandas as pd

from mcfqpi.data.pairing import PairRecord
from mcfqpi.data.split import assign_group_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="忽略原 split，按 phase hash 重新划分 train/val/test")
    parser.add_argument("manifest")
    parser.add_argument("--output", default="data/processed/mcf_qpi_manifest_strict.csv")
    parser.add_argument("--train-ratio", type=float, default=0.88)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest, dtype=str).fillna("")
    if "phase_sha256" not in frame or (frame["phase_sha256"] == "").any():
        raise RuntimeError(
            "严格重新划分要求每行都有 phase_sha256。请重新运行 build_manifest.py --compute-hashes。"
        )
    allowed = {field.name for field in fields(PairRecord)}
    records = []
    for row in frame.to_dict(orient="records"):
        payload = {key: str(value) for key, value in row.items() if key in allowed}
        payload["split"] = "unknown"
        payload["group_id"] = payload.get("phase_sha256", "") or payload.get("phase_phash", "")
        records.append(PairRecord(**payload))
    records = assign_group_splits(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        preserve_known=False,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame([record.to_dict() for record in records])
    result.to_csv(output, index=False)
    print(result.groupby(["domain", "split"]).size())
    print(f"写入严格分组划分：{output}")


if __name__ == "__main__":
    main()
