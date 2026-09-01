#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from mcfqpi.data.cache import cache_manifest_to_hdf5
from mcfqpi.data.pairing import discover_pairs


def _phase_pattern(size: int, rng: np.random.Generator, domain: str) -> np.ndarray:
    """生成仅用于软件连通性测试的 [0,1] 程序化相位标签。

    这些图不是 MCF-QPI 科学训练数据，不能用于论文结果。
    """
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    if domain == "digits":
        # 用若干粗线段模拟数字类简单结构。
        width = max(2, size // 12)
        points = [(int(rng.uniform(0.2, 0.8) * size), int(rng.uniform(0.15, 0.85) * size)) for _ in range(5)]
        draw.line(points, fill=int(rng.integers(150, 256)), width=width, joint="curve")
        if rng.random() < 0.6:
            box = [size // 4, size // 5, 3 * size // 4, 4 * size // 5]
            draw.ellipse(box, outline=int(rng.integers(120, 240)), width=width)
    else:
        # 用轮廓和填充块模拟 Fashion 类更复杂结构。
        x0, y0 = int(0.15 * size), int(0.15 * size)
        x1, y1 = int(0.85 * size), int(0.88 * size)
        polygon = [
            (size // 2, y0),
            (x1, int(0.38 * size)),
            (int(0.70 * size), y1),
            (int(0.30 * size), y1),
            (x0, int(0.38 * size)),
        ]
        draw.polygon(polygon, fill=int(rng.integers(100, 230)))
        for _ in range(3):
            cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
            radius = int(rng.integers(max(2, size // 20), max(3, size // 8)))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=int(rng.integers(0, 255)))
    phase = np.asarray(canvas, dtype=np.float32) / 255.0
    # 轻微平滑以形成连续相位；使用 FFT 高斯低通，避免额外依赖。
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    transfer = np.exp(-(fx**2 + fy**2) / (2 * 0.08**2))
    phase = np.fft.ifft2(np.fft.fft2(phase) * transfer).real
    phase -= phase.min()
    phase /= max(float(phase.max()), 1e-8)
    return phase.astype(np.float32)


def _speckle_from_phase(phase: np.ndarray, diffuser: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = phase.shape[0]
    yy, xx = np.mgrid[:size, :size]
    radius = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    aperture = (radius < 0.44 * size).astype(np.float32)
    field = aperture * np.exp(1j * (np.pi * phase + diffuser))
    # 固定 diffuser 代表稳定光纤系统；不同 phase 产生不同远场强度。
    spectrum = np.fft.fftshift(np.fft.fft2(field, norm="ortho"))
    intensity = np.abs(spectrum) ** 2
    intensity = intensity / max(float(intensity.mean()), 1e-8)
    # 软件 smoke 中加入少量相机噪声，测试鲁棒性代码路径。
    photons = 1000.0
    intensity = rng.poisson(np.clip(intensity, 0, 20) * photons) / photons
    intensity += rng.normal(0, 0.01, size=intensity.shape)
    intensity = np.log1p(np.clip(intensity, 0, 20)) / np.log1p(20.0)
    return np.clip(intensity, 0, 1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成仅供 smoke test 的小型程序化数据")
    parser.add_argument("--output-dir", default="data/smoke/raw")
    parser.add_argument("--manifest", default="data/smoke/manifest.csv")
    parser.add_argument("--hdf5", default="data/smoke/mcf_smoke_64.h5")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--val", type=int, default=12)
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    if root.exists() and any(root.rglob("*.png")) and not args.overwrite:
        print(f"Smoke 原始数据已存在，跳过生成：{root}")
    else:
        rng = np.random.default_rng(args.seed)
        diffuser = rng.uniform(-np.pi, np.pi, size=(args.size, args.size)).astype(np.float32)
        for domain in ("digits", "fashion"):
            for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
                speckle_dir = root / domain / split / "speckles"
                phase_dir = root / domain / split / "phases"
                speckle_dir.mkdir(parents=True, exist_ok=True)
                phase_dir.mkdir(parents=True, exist_ok=True)
                for index in range(count):
                    phase = _phase_pattern(args.size, rng, domain)
                    speckle = _speckle_from_phase(phase, diffuser, rng)
                    # 标签按公开数据的常见 8-bit 图像形式保存，对应 [0,π]。
                    Image.fromarray(np.round(phase * 255).astype(np.uint8)).save(phase_dir / f"phase_{index:05d}.png")
                    Image.fromarray(np.round(speckle * 65535).astype(np.uint16)).save(
                        speckle_dir / f"speckle_{index:05d}.png"
                    )
        print(f"已生成 smoke 原始数据：{root}")

    records, diagnostics = discover_pairs(root, allow_order_pairing=False, compute_hashes=True)
    if not records:
        raise RuntimeError(f"Smoke 配对失败：{diagnostics}")
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record.to_dict() for record in records]).to_csv(manifest, index=False)
    hdf5 = Path(args.hdf5)
    if hdf5.exists() and args.overwrite:
        hdf5.unlink()
    if not hdf5.exists():
        cache_manifest_to_hdf5(
            manifest,
            hdf5,
            size=args.size,
            phase_encoding="uint8",
            resize_mode="stretch",
            speckle_normalization="robust_log",
            overwrite=False,
        )
    print(f"Manifest：{manifest}；HDF5：{hdf5}；pairs={len(records)}")


if __name__ == "__main__":
    main()
