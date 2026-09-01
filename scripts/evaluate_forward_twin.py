#!/usr/bin/env python3
"""独立评估经验 forward twin；只有验证质量足够好时才启用 cycle loss。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from mcfqpi.config import load_config, save_json
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.metrics import summarize_frame
from mcfqpi.factory import build_forward_twin, load_model_weights
from mcfqpi.training.engine import build_loader, move_batch_to_device
from mcfqpi.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估真实配对数据学习的 phase→speckle 经验代理")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="outputs/forward_twin_evaluation")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def _pearson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = prediction.flatten(1) - prediction.flatten(1).mean(dim=1, keepdim=True)
    t = target.flatten(1) - target.flatten(1).mean(dim=1, keepdim=True)
    return (p * t).sum(dim=1) / torch.sqrt(p.square().sum(dim=1) * t.square().sum(dim=1)).clamp_min(1e-8)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed_everything(int(config.get("seed", 42)), deterministic=True)
    device = select_device(str(config.get("device", "auto")))
    dataset = build_dataset(config["data"], args.split, training=False)
    loader = build_loader(dataset, config.get("loader", {}), training=False, generator_seed=42)
    model = build_forward_twin(config.get("model", {})).to(device).eval()
    load_model_weights(model, args.checkpoint, map_location=device, strict=True)

    rows = []
    first = None
    with torch.inference_mode():
        for raw in loader:
            batch = move_batch_to_device(raw, device)
            prediction = model(batch["phase"])["speckle"].float()
            target = batch["speckle"].float()
            mae = (prediction - target).abs().flatten(1).mean(dim=1)
            mse = (prediction - target).square().flatten(1).mean(dim=1)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            corr = _pearson(prediction, target)
            spectral = (
                torch.log1p(torch.fft.fft2(prediction, norm="ortho").abs())
                - torch.log1p(torch.fft.fft2(target, norm="ortho").abs())
            ).abs().flatten(1).mean(dim=1)
            for index in range(prediction.shape[0]):
                rows.append({
                    "sample_id": str(raw["sample_id"][index]),
                    "domain": str(raw["domain"][index]),
                    "mae": float(mae[index].cpu()),
                    "psnr_db": float(psnr[index].cpu()),
                    "pearson_r": float(corr[index].cpu()),
                    "spectral_l1": float(spectral[index].cpu()),
                })
            if first is None:
                first = (batch["phase"][:4].cpu(), target[:4].cpu(), prediction[:4].cpu())

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_sample_metrics.csv", index=False)
    summary = {
        "split": args.split,
        "overall": summarize_frame(frame, bootstrap_samples=1000),
        "recommendation": (
            "只有当验证集 speckle Pearson/PSNR 稳定、且 cycle-loss 消融确实提升逆问题测试指标时，"
            "才在主模型中启用经验闭环；否则将 cycle 权重设为 0。"
        ),
    }
    save_json(summary, output / "summary.json")
    if first is not None:
        phase, target, prediction = first
        fig, axes = plt.subplots(len(phase), 3, figsize=(9, 2.5 * len(phase)), squeeze=False)
        for i in range(len(phase)):
            axes[i, 0].imshow(phase[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[i, 1].imshow(target[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(prediction[i, 0], cmap="gray", vmin=0, vmax=1)
            for axis in axes[i]:
                axis.axis("off")
        axes[0, 0].set_title("Phase")
        axes[0, 1].set_title("Measured speckle")
        axes[0, 2].set_title("Twin speckle")
        fig.tight_layout()
        fig.savefig(output / "qualitative.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
