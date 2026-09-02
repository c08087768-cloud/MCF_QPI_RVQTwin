#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys

from mcfqpi.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="开发评估：强制只访问 validation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if str(config.get("evaluation", {}).get("split", "val")) == "test":
        raise ValueError("dev_pipeline 禁止配置 test split")
    subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--config", args.config,
         "--checkpoint", args.checkpoint, "--set", "evaluation.split=val"],
        check=True,
    )


if __name__ == "__main__":
    main()
