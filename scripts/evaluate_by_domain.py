#!/usr/bin/env python3
"""对同一训练协议内的 digits/fashion 子集分别报告，不表示跨域迁移。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mcfqpi.config import load_config, save_json
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.runner import evaluate_phase_model
from mcfqpi.factory import build_inverse_from_checkpoint
from mcfqpi.training.engine import build_loader
from mcfqpi.utils import seed_everything, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="同一模型在 digits/fashion 子集上的分域报告")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/by_domain")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=True)
    device = select_device(str(config.get("device", "auto")))
    model, _ = build_inverse_from_checkpoint(args.checkpoint)
    summaries = {}
    for domain in ("digits", "fashion"):
        local = copy.deepcopy(config)
        local["data"]["domains"] = [domain]
        dataset = build_dataset(local["data"], str(local["evaluation"]["split"]), training=False)
        loader = build_loader(dataset, local.get("loader", {}), training=False, generator_seed=seed)
        summaries[domain] = evaluate_phase_model(
            model, loader, device=device, output_dir=Path(args.output_dir) / domain,
            mc_samples=int(local.get("evaluation", {}).get("mc_samples", 1)),
            amp=bool(local.get("evaluation", {}).get("amp", True)),
            bootstrap_samples=int(local.get("evaluation", {}).get("bootstrap_samples", 2000)),
        )
    save_json(summaries, Path(args.output_dir) / "summary.json")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
