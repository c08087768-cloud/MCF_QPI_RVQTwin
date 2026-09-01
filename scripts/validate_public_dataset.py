#!/usr/bin/env python3
"""对照公开论文中的样本数量，检查 manifest，但不把论文内部矛盾强行当作错误。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mcfqpi.config import save_json

# 方法段给出的 split 数量。
PAPER_SPLIT_COUNTS = {
    ("digits", "train"): 26001,
    ("digits", "val"): 3300,
    ("digits", "test"): 78,
    ("fashion", "train"): 18001,
    ("fashion", "val"): 2700,
    ("fashion", "test"): 96,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 MCF-QPI manifest 的样本数量与元数据")
    parser.add_argument("manifest")
    parser.add_argument("--output", default="outputs/data_inspection/public_dataset_validation.json")
    parser.add_argument("--strict-paper-split", action="store_true", help="与方法段 split 不一致时返回非零状态")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest, dtype=str).fillna("")
    observed = frame.groupby(["domain", "split"]).size().to_dict()
    differences = []
    for key, expected in PAPER_SPLIT_COUNTS.items():
        actual = int(observed.get(key, 0))
        if actual != expected:
            differences.append({"domain": key[0], "split": key[1], "expected": expected, "actual": actual})

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "rows": int(len(frame)),
        "observed_counts": {f"{domain}/{split}": int(count) for (domain, split), count in observed.items()},
        "paper_method_split_counts": {f"{domain}/{split}": count for (domain, split), count in PAPER_SPLIT_COUNTS.items()},
        "differences": differences,
        "known_paper_text_inconsistencies": [
            "论文摘要/数据说明称总计 50,176 对。",
            "方法段写 29,379 digits + 20,796 fashion = 50,175。",
            "同一方法段给出的 split 求和为 29,379 digits + 20,797 fashion = 50,176。",
            "结果段把 96/78 的 digits/fashion 测试数量与方法段写反。",
        ],
        "guidance": (
            "以实际解压目录、文件名、配对可视化和哈希审计为准。不要为了匹配某一句文字而移动测试样本。"
        ),
    }
    save_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict_paper_split and differences:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
