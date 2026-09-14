#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mcfqpi.config import load_config
from mcfqpi.data.dataset import build_dataset
from mcfqpi.factory import build_forward_twin, build_inverse_model, load_model_weights
from mcfqpi.models.blocks import set_requires_grad
from mcfqpi.training.engine import build_loader, fit_model
from mcfqpi.training.losses import InverseLoss, LossWeights
from mcfqpi.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MCF-QPI 基线或 DualDomain-RVQ-Twin 逆网络")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed = int(config.get("seed", 42))
    strict_repro = str(config.get("reproducibility", {}).get("mode", "fast")) == "strict"
    seed_everything(seed, deterministic=strict_repro or bool(config.get("deterministic", False)))
    device = select_device(str(config.get("device", "auto")))
    train_dataset = build_dataset(config["data"], "train", training=True)
    val_dataset = build_dataset(config["data"], "val", training=False)
    loader_config = dict(config.get("loader", {}))
    if strict_repro:
        loader_config.update({"num_workers": 0, "persistent_workers": False})
    train_loader = build_loader(train_dataset, loader_config, training=True, generator_seed=seed)
    val_loader = build_loader(val_dataset, loader_config, training=False, generator_seed=seed + 1)
    model = build_inverse_model(config.get("model", {}), map_location="cpu")
    criterion = InverseLoss(LossWeights.from_dict(config.get("loss", {})))

    forward_twin = None
    forward_config = config.get("forward_twin", {})
    if forward_config.get("enabled", False):
        checkpoint = forward_config.get("checkpoint")
        if not checkpoint:
            raise ValueError("forward_twin.enabled=true 时必须提供 forward_twin.checkpoint")
        forward_twin = build_forward_twin(forward_config.get("model", {}))
        load_model_weights(forward_twin, checkpoint, strict=True)
        set_requires_grad(forward_twin, False)
        forward_twin.to(device).eval()

    def step(model, batch, training):
        outputs = model(batch["speckle"], target_phase=batch["phase"])
        loss, raw_terms = criterion(outputs, batch, forward_twin=forward_twin)
        terms = {
            **raw_terms,
            **criterion.weighted_terms(raw_terms),
            **criterion.model_diagnostics(outputs),
        }
        return loss, terms, outputs

    fit_model(
        model,
        train_loader,
        val_loader,
        device=device,
        step_function=step,
        training_config=config.get("training", {}),
        full_config=config,
        output_dir=config.get("output_dir", "outputs/inverse"),
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
