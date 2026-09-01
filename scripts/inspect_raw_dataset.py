#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from mcfqpi.config import save_json
from mcfqpi.data.pairing import IMAGE_EXTENSIONS, classify_role, infer_domain, infer_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描下载解压后的 MCF-QPI 原始目录")
    parser.add_argument("root")
    parser.add_argument("--output", default="outputs/data_inspection/raw_structure.json")
    parser.add_argument("--dimension-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    files = [path for path in root.rglob("*") if path.is_file()]
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    extension_counts = Counter(path.suffix.lower() for path in files)
    role_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    unknown_examples: list[str] = []
    dimensions: Counter[str] = Counter()
    for index, path in enumerate(images):
        relative = path.relative_to(root)
        role = classify_role(relative) or "unknown"
        role_counts[role] += 1
        group_counts[f"{infer_domain(relative)}/{infer_split(relative)}/{role}"] += 1
        if role == "unknown" and len(unknown_examples) < 100:
            unknown_examples.append(str(relative))
        if index < args.dimension_samples:
            try:
                with Image.open(path) as image:
                    dimensions[f"{image.mode}:{image.width}x{image.height}"] += 1
            except Exception as exc:  # 结构检查不能因一张坏图中断
                dimensions[f"ERROR:{type(exc).__name__}"] += 1
    report = {
        "root": str(root),
        "total_files": len(files),
        "image_files": len(images),
        "extension_counts": dict(extension_counts),
        "role_counts": dict(role_counts),
        "group_counts": dict(group_counts),
        "sampled_dimensions": dict(dimensions),
        "unknown_examples": unknown_examples,
        "first_200_files": [str(path.relative_to(root)) for path in files[:200]],
    }
    save_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
