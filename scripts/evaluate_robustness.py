#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcfqpi.config import load_config, save_json
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.robustness import DeterministicCorruptionDataset
from mcfqpi.evaluation.runner import evaluate_phase_model
from mcfqpi.factory import build_inverse_model, load_model_weights
from mcfqpi.training.engine import build_loader
from mcfqpi.utils import seed_everything, select_device


DEFAULT_CONDITIONS = [
    ("clean", 0.0),
    ("gaussian_noise", 0.02),
    ("gaussian_noise", 0.05),
    ("gain", 0.20),
    ("background", 0.05),
    ("blur", 3.0),
    ("shift", 2.0),
    ("saturation", 0.25),
    ("dead_pixels", 0.01),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估相机噪声、模糊、平移和饱和鲁棒性")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/robustness")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=True)
    device = select_device(str(config.get("device", "auto")))
    base = build_dataset(config["data"], str(config.get("evaluation", {}).get("split", "test")), training=False)
    model = build_inverse_model(config.get("model", {}), map_location="cpu")
    load_model_weights(model, args.checkpoint, strict=True)
    summaries = {}
    for name, severity in DEFAULT_CONDITIONS:
        condition = DeterministicCorruptionDataset(base, corruption=name, severity=severity, seed=seed)
        loader = build_loader(condition, config.get("loader", {}), training=False, generator_seed=seed)
        label = f"{name}_{severity:g}"
        summary = evaluate_phase_model(
            model,
            loader,
            device=device,
            output_dir=Path(args.output_dir) / label,
            mc_samples=int(config.get("evaluation", {}).get("mc_samples", 1)),
            amp=bool(config.get("evaluation", {}).get("amp", True)),
            bootstrap_samples=int(config.get("evaluation", {}).get("bootstrap_samples", 500)),
        )
        summaries[label] = summary
    save_json(summaries, Path(args.output_dir) / "all_conditions.json")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
