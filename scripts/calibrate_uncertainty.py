#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcfqpi.config import load_config, save_json
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.runner import _enable_mc_dropout, fit_laplace_temperature
from mcfqpi.factory import build_inverse_from_checkpoint
from mcfqpi.training.engine import build_loader, move_batch_to_device
from mcfqpi.utils import select_device, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="只在 validation 上拟合 MC-Laplace 温度")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    device = select_device(str(config.get("device", "auto")))
    model, _ = build_inverse_from_checkpoint(args.checkpoint)
    model.to(device)
    _enable_mc_dropout(model)
    dataset = build_dataset(config["data"], "val", training=False)
    loader = build_loader(dataset, config.get("loader", {}), training=False)
    batches = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            phases, scales = [], []
            for _ in range(args.mc_samples):
                output = model(batch["speckle"])
                if "log_scale" not in output:
                    raise ValueError("模型没有 log_scale，不能进行不确定度校准")
                phases.append(output["phase"].float())
                scales.append(output["log_scale"].float().exp())
            batches.append((
                torch.stack(phases), torch.stack(scales), batch["phase"],
                batch.get("valid_mask"),
            ))
    temperature = fit_laplace_temperature(batches)
    payload = {
        "schema_version": "mcfqpi-uncertainty-calibration-1.0",
        "split": "val",
        "temperature": temperature,
        "mc_samples": args.mc_samples,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
    }
    save_json(payload, Path(args.output))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
