#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mcfqpi.data.download import MCF_DATASETS, download_with_resume, selected_sources
from mcfqpi.utils import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 MCF-QPI 公开真实散斑—相位数据集")
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", *MCF_DATASETS])
    parser.add_argument("--output-dir", default="data/raw/downloads")
    parser.add_argument("--list", action="store_true", help="只列出链接和元数据，不下载")
    parser.add_argument("--hash", action="store_true", help="下载后计算 SHA-256（约需额外几分钟）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = selected_sources(args.datasets)
    for source in sources:
        print(
            f"[{source.key}] {source.filename}\n"
            f"  DOI: {source.doi}\n"
            f"  许可: {source.license_name}\n"
            f"  约: {source.approximate_size_mb:.2f} MB\n"
            f"  下载: {source.url}"
        )
    if args.list:
        return
    output_dir = Path(args.output_dir)
    for source in sources:
        destination = download_with_resume(source, output_dir / source.filename)
        print(f"下载完成：{destination}")
        if args.hash:
            print(f"SHA-256：{sha256_file(destination)}")


if __name__ == "__main__":
    main()
