#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import torch

from mcfqpi.config import load_config, save_json
from mcfqpi.factory import build_inverse_model, load_model_weights
from mcfqpi.utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测量纯网络前向延迟和吞吐率")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", default="outputs/latency.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    device = select_device(str(config.get("device", "auto")))
    model = build_inverse_model(config.get("model", {}), map_location="cpu").to(device).eval()
    load_model_weights(model, args.checkpoint, map_location=device, strict=True)
    size = int(config.get("data", {}).get("size", 128))
    results = []
    for batch_size in args.batch_sizes:
        x = torch.rand(batch_size, 1, size, size, device=device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.iterations):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
        results.append(
            {
                "batch_size": batch_size,
                "milliseconds_per_batch": elapsed * 1000 / args.iterations,
                "milliseconds_per_sample": elapsed * 1000 / (args.iterations * batch_size),
                "samples_per_second": args.iterations * batch_size / elapsed,
            }
        )
    payload = {
        "device": str(device),
        "warmup_iterations": int(args.warmup),
        "timed_iterations": int(args.iterations),
        "timing_scope": "纯模型前向；不含数据读取、预处理、相机曝光和主机到GPU传输",
        "results": results,
    }
    save_json(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
