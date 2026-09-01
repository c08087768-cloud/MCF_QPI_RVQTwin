#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mcfqpi.data.archive import list_tar, safe_extract_tar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="安全解压 MCF-QPI tar 文件")
    parser.add_argument("archives", nargs="+", help="一个或多个 .tar/.tar.gz 文件")
    parser.add_argument("--output-dir", default="data/raw/extracted")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--list-limit", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    for raw in args.archives:
        archive = Path(raw)
        print(f"\n=== {archive} ===")
        for row in list_tar(archive, args.list_limit):
            print(f"{row['size']:>12}  {row['name']}")
        if not args.list_only:
            destination = output_root / archive.stem.replace(".tar", "")
            safe_extract_tar(archive, destination)
            print(f"已解压到：{destination}")


if __name__ == "__main__":
    main()
