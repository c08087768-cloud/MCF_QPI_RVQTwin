#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mcfqpi.config import load_config
from mcfqpi.data.dataset import build_dataset
from mcfqpi.factory import build_forward_twin
from mcfqpi.training.engine import build_loader, fit_model
from mcfqpi.training.losses import ForwardTwinLoss
from mcfqpi.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练真实配对数据的经验相位→散斑前向代理")
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
    model = build_forward_twin(config.get("model", {}))
    criterion = ForwardTwinLoss(**config.get("loss", {}))

    def step(model, batch, training):
        outputs = model(batch["phase"])
        loss, terms = criterion(outputs, batch["speckle"], batch.get("valid_mask"))
        return loss, terms, outputs

    fit_model(
        model,
        train_loader,
        val_loader,
        device=device,
        step_function=step,
        training_config=config.get("training", {}),
        full_config=config,
        output_dir=config.get("output_dir", "outputs/forward_twin"),
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
