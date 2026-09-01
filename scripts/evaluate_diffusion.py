#!/usr/bin/env python3
"""在保留测试集上用 DDIM 采样评估扩散基线。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcfqpi.config import load_config, save_json
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.runner import evaluate_phase_model
from mcfqpi.factory import build_diffusion, load_model_weights
from mcfqpi.models import DiffusionPhaseReconstructor
from mcfqpi.training.engine import build_loader
from mcfqpi.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 MCF-QPI 条件扩散基线")
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

    model_config = config.get("model", {})
    diffusion = build_diffusion(model_config)
    load_model_weights(diffusion, args.checkpoint, strict=True)
    model = DiffusionPhaseReconstructor(
        diffusion,
        sample_steps=int(model_config.get("sample_steps", 100)),
        eta=float(model_config.get("eta", 0.0)),
    )

    evaluation = config.get("evaluation", {})
    summary = evaluate_phase_model(
        model,
        loader,
        device=device,
        output_dir=config.get("output_dir", "outputs/evaluation_diffusion"),
        # 每次扩散采样本身已包含随机性。需要多样本不确定度时，可设 mc_samples>1；
        # 这会线性增加采样时间，正式延迟对比应同时报告采样步数。
        mc_samples=int(evaluation.get("mc_samples", 1)),
        amp=bool(evaluation.get("amp", True)),
        max_batches=int(evaluation.get("max_batches", 0)),
        bootstrap_samples=int(evaluation.get("bootstrap_samples", 2000)),
        save_predictions=bool(evaluation.get("save_predictions", False)),
    )
    summary["diffusion_sample_steps"] = int(model_config.get("sample_steps", 100))
    summary["diffusion_eta"] = float(model_config.get("eta", 0.0))
    save_json(summary, Path(config.get("output_dir", "outputs/evaluation_diffusion")) / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
