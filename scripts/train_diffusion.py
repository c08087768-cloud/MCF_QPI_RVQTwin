#!/usr/bin/env python3
"""训练 speckle-conditioned 扩散基线。

这是一套清晰、可运行的 DDPM 训练 + DDIM 推理实现，用于与 ResUNet 和
MCF-RVQTwin 做生成式基线比较。它不是 SpecDiffusion 官方代码的逐层复现。
"""
from __future__ import annotations

import argparse

from mcfqpi.config import load_config
from mcfqpi.data.dataset import build_dataset
from mcfqpi.factory import build_diffusion
from mcfqpi.training.engine import build_loader, fit_model
from mcfqpi.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MCF-QPI 条件扩散基线")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=bool(config.get("deterministic", False)))
    device = select_device(str(config.get("device", "auto")))

    train_dataset = build_dataset(config["data"], "train", training=True)
    val_dataset = build_dataset(config["data"], "val", training=False)
    train_loader = build_loader(train_dataset, config.get("loader", {}), training=True, generator_seed=seed)
    val_loader = build_loader(val_dataset, config.get("loader", {}), training=False, generator_seed=seed + 1)
    model = build_diffusion(config.get("model", {}))

    def step(model, batch, training):
        loss, terms = model.training_loss(batch["phase"], batch["speckle"])
        # generic engine 只将少量可视化字段移到 CPU；这里不需要输出完整噪声张量。
        return loss, terms, {}

    fit_model(
        model,
        train_loader,
        val_loader,
        device=device,
        step_function=step,
        training_config=config.get("training", {}),
        full_config=config,
        output_dir=config.get("output_dir", "outputs/diffusion"),
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
