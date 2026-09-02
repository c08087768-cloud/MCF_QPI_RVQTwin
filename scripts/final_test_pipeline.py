#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mcfqpi.config import load_config
from mcfqpi.protocol import verify_freeze_record


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结后的唯一正式 test 入口")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--freeze", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if str(config.get("evaluation", {}).get("split", "")) != "test":
        raise ValueError("final_test_pipeline 要求 evaluation.split=test")
    record = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    data_paths = [config["data"]["hdf5"]] if config.get("data", {}).get("hdf5") else []
    verify_freeze_record(
        record, args.config, args.checkpoint, data_paths=data_paths
    )
    subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--config", args.config,
         "--checkpoint", args.checkpoint],
        check=True,
    )


if __name__ == "__main__":
    main()
