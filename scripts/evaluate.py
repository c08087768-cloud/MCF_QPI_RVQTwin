#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mcfqpi.config import load_config
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.runner import evaluate_phase_model
from mcfqpi.factory import build_inverse_model, load_model_weights
from mcfqpi.training.engine import build_loader
from mcfqpi.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 MCF-QPI 保留测试集上评估相位恢复网络")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=True)
    device = select_device(str(config.get("device", "auto")))
    split = str(config.get("evaluation", {}).get("split", "test"))
    dataset = build_dataset(config["data"], split, training=False)
    loader = build_loader(dataset, config.get("loader", {}), training=False, generator_seed=seed)
    model = build_inverse_model(config.get("model", {}), map_location="cpu")
    load_model_weights(model, args.checkpoint, strict=True)
    eval_config = config.get("evaluation", {})
    summary = evaluate_phase_model(
        model,
        loader,
        device=device,
        output_dir=config.get("output_dir", "outputs/evaluation"),
        mc_samples=int(eval_config.get("mc_samples", 1)),
        amp=bool(eval_config.get("amp", True)),
        max_batches=int(eval_config.get("max_batches", 0)),
        bootstrap_samples=int(eval_config.get("bootstrap_samples", 2000)),
        save_predictions=bool(eval_config.get("save_predictions", False)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
