#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mcfqpi.config import save_json
from mcfqpi.data.cache import cache_manifest_to_hdf5, inspect_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 MCF-QPI manifest 缓存为训练友好的 HDF5")
    parser.add_argument("manifest")
    parser.add_argument("--output", default="data/processed/mcf_qpi_128.h5")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--phase-encoding", default="auto", choices=["auto", "uint8", "uint16", "normalized", "radian"])
    parser.add_argument("--resize-mode", default="fit_pad", choices=["stretch", "fit_pad", "center_crop"])
    parser.add_argument("--normalization", default="log_mean", choices=["log_mean", "robust_log", "linear_mean"])
    parser.add_argument("--dynamic-range", type=float, default=20.0)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default="outputs/data_inspection/hdf5_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = cache_manifest_to_hdf5(
        args.manifest,
        args.output,
        size=args.size,
        phase_encoding=args.phase_encoding,
        resize_mode=args.resize_mode,
        speckle_normalization=args.normalization,
        dynamic_range=args.dynamic_range,
        compression=None if args.compression == "none" else args.compression,
        overwrite=args.overwrite,
    )
    report = inspect_hdf5(path)
    save_json(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
