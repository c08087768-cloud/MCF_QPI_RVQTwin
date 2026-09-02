#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from mcfqpi.config import load_config, save_json
from mcfqpi.protocol import create_freeze_record


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结正式 test 前的代码、配置、数据和权重")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/freezes")
    args = parser.parse_args()
    config = load_config(args.config)
    data_paths = [config["data"]["hdf5"]] if config.get("data", {}).get("hdf5") else []
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = create_freeze_record(
        args.config, args.checkpoint, data_paths=data_paths, git_sha=git_sha
    )
    output = Path(args.output_dir) / f"{record['freeze_id']}.json"
    save_json(record, output)
    print(f"freeze_id={record['freeze_id']}")
    print(output)


if __name__ == "__main__":
    main()
