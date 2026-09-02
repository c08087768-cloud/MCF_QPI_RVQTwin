#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcfqpi.config import load_config
from mcfqpi.data.dataset import build_dataset
from mcfqpi.evaluation.runner import evaluate_phase_model
from mcfqpi.factory import build_inverse_from_checkpoint
from mcfqpi.training.engine import build_loader
from mcfqpi.utils import seed_everything, select_device, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 MCF-QPI 保留测试集上评估相位恢复网络")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--calibration", help="validation 生成的冻结温度 JSON")
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
    model, checkpoint = build_inverse_from_checkpoint(args.checkpoint, map_location="cpu")
    eval_config = config.get("evaluation", {})
    temperature = float(eval_config.get("uncertainty_temperature", 1.0))
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if calibration.get("split") != "val":
            raise ValueError("不确定度校准文件必须来自 validation")
        if calibration.get("checkpoint_sha256") != sha256_file(args.checkpoint):
            raise ValueError("校准文件与当前模型权重哈希不匹配")
        temperature = float(calibration["temperature"])
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
        uncertainty_temperature=temperature,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
