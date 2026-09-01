#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from mcfqpi.config import load_config
from mcfqpi.factory import build_inverse_model, load_model_weights


class PhaseOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, speckle: torch.Tensor) -> torch.Tensor:
        return self.model(speckle)["phase"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 phase-only ONNX 推理模型")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/model.onnx")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = build_inverse_model(config.get("model", {}), map_location="cpu").eval()
    load_model_weights(model, args.checkpoint, map_location="cpu", strict=True)
    wrapper = PhaseOnlyWrapper(model)
    size = int(config.get("data", {}).get("size", 128))
    dummy = torch.rand(1, 1, size, size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        input_names=["speckle"],
        output_names=["phase_normalized"],
        dynamic_axes={"speckle": {0: "batch"}, "phase_normalized": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"已导出：{output}")


if __name__ == "__main__":
    main()
