#!/usr/bin/env python3
"""合并三随机种子的 main_table_seedN.csv，输出 main_table_3seeds.csv。

对每个 (模型, 指标)：跨三种子取 mean ± std（样本标准差，ddof=1）。
- 主指标列（mae_rad / fidelity_2d_correlation / ssim / psnr_db 等）→ "mean±std" 字符串；
- bootstrap CI 列（*_ci_low / *_ci_high）→ 取三种子对应 CI 的均值（CI 是单种子内的 bootstrap 置信区间，跨种子取均值代表整体置信带）；
- summary_path 列保留 seed42 的（仅作指向）。

用法：
    python scripts/merge_three_seeds.py
默认读 SEEDS=42,123,2026，输出 outputs/research/main_table_3seeds.csv。
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

SEEDS = [42, 123, 2026]
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "outputs" / "research"
OUTPUT = INPUT_DIR / "main_table_3seeds.csv"

# 这些列是"主指标"（点估计），跨种子算 mean±std。
POINT_METRICS = [
    "mae_normalized",
    "mae_rad",
    "rmse_rad",
    "fidelity_2d_correlation",
    "ssim",
    "psnr_db",
    "gradient_mae_rad_per_pixel",
    "uncertainty_mean",
]
# 这些列是 bootstrap CI 端点，跨种子取均值。
CI_METRICS = [
    "mae_normalized_ci_low",
    "mae_normalized_ci_high",
    "mae_rad_ci_low",
    "mae_rad_ci_high",
    "rmse_rad_ci_low",
    "rmse_rad_ci_high",
    "fidelity_2d_correlation_ci_low",
    "fidelity_2d_correlation_ci_high",
    "ssim_ci_low",
    "ssim_ci_high",
    "psnr_db_ci_low",
    "psnr_db_ci_high",
    "gradient_mae_rad_per_pixel_ci_low",
    "gradient_mae_rad_per_pixel_ci_high",
    "uncertainty_mean_ci_low",
    "uncertainty_mean_ci_high",
]
# 这些列每种子只有一份（延迟等），不跨种子统计，直接取 seed42。
SINGLE_VALUED = ["mean_forward_ms_per_sample"]


def load_seed(seed: int) -> list[dict[str, str]]:
    path = INPUT_DIR / f"main_table_seed{seed}.csv"
    if not path.exists():
        print(f"[警告] 缺少 {path}，跳过 seed={seed}", file=sys.stderr)
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def model_key(row: dict[str, str]) -> str:
    """把 mean_phase_seed42 / eval_paper_style_seed42 等去掉 _seedN 后缀，得到模型名。"""
    name = row["experiment"]
    for seed in SEEDS:
        suffix = f"_seed{seed}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def fmt_mean_std(values: list[float]) -> str:
    nums = [v for v in values if v is not None]
    if not nums:
        return ""
    if len(nums) == 1:
        return f"{nums[0]:.6f}"
    mean = statistics.mean(nums)
    std = statistics.stdev(nums)  # ddof=1
    return f"{mean:.6f}±{std:.6f}"


def fmt_mean(values: list[float]) -> str:
    nums = [v for v in values if v is not None]
    if not nums:
        return ""
    return f"{statistics.mean(nums):.6f}"


def to_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> int:
    # 按模型名分组，收集三种子该模型的行。
    by_model: dict[str, list[dict[str, str]]] = {}
    header: list[str] | None = None
    for seed in SEEDS:
        rows = load_seed(seed)
        if not rows:
            continue
        if header is None:
            header = list(rows[0].keys())
        for row in rows:
            by_model.setdefault(model_key(row), []).append(row)

    if header is None or not by_model:
        print("[错误] 没有读到任何种子的 main_table，无法合并。", file=sys.stderr)
        return 1

    # 保持 mean_phase → eval_paper_style → eval_baseline → eval_proposed 的出现顺序。
    order = ["mean_phase", "eval_paper_style", "eval_baseline", "eval_proposed"]
    models = [m for m in order if m in by_model] + [m for m in by_model if m not in order]

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", *header[2:]])  # 去掉原 summary_path 单列，保留其余
        for model in models:
            rows = by_model[model]
            out: dict[str, str] = {}
            # summary_path 取第一个（seed42），仅作指向。
            out["summary_path"] = rows[0].get("summary_path", "")
            for col in header[2:]:
                values = [to_float(r.get(col, "")) for r in rows]
                if col in POINT_METRICS:
                    out[col] = fmt_mean_std(values)
                elif col in CI_METRICS:
                    out[col] = fmt_mean(values)
                elif col in SINGLE_VALUED:
                    out[col] = rows[0].get(col, "")
                else:
                    out[col] = rows[0].get(col, "")
            writer.writerow([model, *[out[c] for c in header[2:]]])

    print(f"写入：{OUTPUT}")
    print(f"合并 {len(models)} 个模型 × {len(SEEDS)} 种子")
    # 打印预览
    with OUTPUT.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 6:
                print(line.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
