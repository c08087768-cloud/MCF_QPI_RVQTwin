from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _flatten_masked(value: torch.Tensor, mask: torch.Tensor | None) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for index in range(value.shape[0]):
        current = value[index]
        if mask is None:
            result.append(current.reshape(-1))
        else:
            valid = mask[index].expand_as(current) > 0.5
            result.append(current[valid])
    return result


def _global_circular_offset(prediction_rad: torch.Tensor, target_rad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """估计使全局圆周残差最小的每样本常数相位偏置。"""
    delta = target_rad - prediction_rad
    real = (torch.cos(delta) * mask).flatten(1).sum(dim=1)
    imag = (torch.sin(delta) * mask).flatten(1).sum(dim=1)
    return torch.atan2(imag, real)


def _circular_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def _ssim_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """以 [0,1] 数据范围计算逐样本 SSIM。

    当预处理采用 fit-pad 时，填充区域不属于真实观测；因此这里使用 valid_mask
    对 SSIM map 做加权平均，避免大量零填充把结果虚高。
    """
    window = 11
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, padding)
    mu_y = F.avg_pool2d(target, window, 1, padding)
    var_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, padding) - mu_y.square()
    cov = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    if mask is None:
        return value.flatten(1).mean(dim=1)
    weight = mask.to(value.dtype)
    numerator = (value * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def phase_metrics_batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    """计算每个样本的相位指标。

    prediction/target 使用 [0,1]，内部乘 π 后报告弧度误差。
    同时报告 raw 与 global-offset-aligned 指标。散斑强度对全局相位常数不敏感，
    因此 aligned 指标可以区分“结构错误”和“相位零点差异”。
    """
    prediction = prediction.detach().float().clamp(0, 1)
    target = target.detach().float().clamp(0, 1)
    if mask is None:
        mask = torch.ones_like(target)
    else:
        mask = mask.detach().float()
    pred_rad = prediction * torch.pi
    target_rad = target * torch.pi
    raw_residual = pred_rad - target_rad
    circular_residual = _circular_difference(pred_rad, target_rad)
    offsets = _global_circular_offset(pred_rad, target_rad, mask)
    aligned_pred_rad = pred_rad + offsets[:, None, None, None]
    aligned_residual = _circular_difference(aligned_pred_rad, target_rad)
    ssim = _ssim_per_sample(prediction, target, mask)

    rows: list[dict[str, float]] = []
    for index in range(prediction.shape[0]):
        valid = mask[index].expand_as(prediction[index]) > 0.5
        p = prediction[index][valid]
        t = target[index][valid]
        raw = raw_residual[index][valid]
        circular = circular_residual[index][valid]
        aligned = aligned_residual[index][valid]
        p_center = p - p.mean()
        t_center = t - t.mean()
        correlation = (p_center * t_center).sum() / torch.sqrt(
            p_center.square().sum() * t_center.square().sum()
        ).clamp_min(1e-8)
        mse_norm = (p - t).square().mean()
        psnr = -10.0 * torch.log10(mse_norm.clamp_min(1e-12))
        pred_dx = pred_rad[index, :, :, 1:] - pred_rad[index, :, :, :-1]
        target_dx = target_rad[index, :, :, 1:] - target_rad[index, :, :, :-1]
        pred_dy = pred_rad[index, :, 1:, :] - pred_rad[index, :, :-1, :]
        target_dy = target_rad[index, :, 1:, :] - target_rad[index, :, :-1, :]
        mask_x = (mask[index, :, :, 1:] * mask[index, :, :, :-1]) > 0.5
        mask_y = (mask[index, :, 1:, :] * mask[index, :, :-1, :]) > 0.5
        gradient_mae = 0.5 * (
            (pred_dx - target_dx).abs()[mask_x].mean() + (pred_dy - target_dy).abs()[mask_y].mean()
        )
        row = {
            # 与原 MCF-QPI 论文的 MAE 训练量纲一致：标签先归一化到 [0,1]。
            "mae_normalized": float((p - t).abs().mean().cpu()),
            "rmse_normalized": float(torch.sqrt((p - t).square().mean()).cpu()),
            # 便于光学解释的弧度指标；公开标签被定义在 [0, π]。
            "mae_rad": float(raw.abs().mean().cpu()),
            "rmse_rad": float(torch.sqrt(raw.square().mean()).cpu()),
            "circular_mae_rad": float(circular.abs().mean().cpu()),
            "aligned_mae_rad": float(aligned.abs().mean().cpu()),
            "aligned_rmse_rad": float(torch.sqrt(aligned.square().mean()).cpu()),
            "global_offset_rad": float(offsets[index].cpu()),
            "residual_std_rad": float(raw.std(unbiased=False).cpu()),
            "gradient_mae_rad_per_pixel": float(gradient_mae.cpu()),
            "psnr_db": float(psnr.cpu()),
            "ssim": float(ssim[index].cpu()),
            # 原论文把二维相关系数称为 reconstruction fidelity。两列数值相同，
            # 同时保留可避免读者把它误认成复光场保真度。
            "pearson_r": float(correlation.cpu()),
            "fidelity_2d_correlation": float(correlation.cpu()),
            "n_valid_pixels": int(valid.sum().cpu()),
        }
        if uncertainty is not None:
            current = uncertainty[index].detach().float()
            if current.shape != mask[index].shape:
                current = F.interpolate(current.unsqueeze(0), size=mask.shape[-2:], mode="bilinear", align_corners=False)[0]
            row["uncertainty_mean"] = float(current[mask[index] > 0.5].mean().cpu())
        rows.append(row)
    return rows


def summarize_frame(
    frame: pd.DataFrame,
    *,
    metrics: Iterable[str] | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """输出均值、标准差、中位数和 bootstrap 置信区间。"""
    if metrics is None:
        metrics = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        if values.size == 1 or bootstrap_samples <= 0:
            low = high = float(values.mean())
        else:
            indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
            means = values[indices].mean(axis=1)
            low, high = np.quantile(means, [alpha, 1.0 - alpha])
        summary[metric] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "ci_low": float(low),
            "ci_high": float(high),
        }
    return summary


def risk_coverage_curve(
    frame: pd.DataFrame,
    *,
    error_column: str = "mae_rad",
    uncertainty_column: str = "uncertainty_mean",
    coverages: Iterable[float] = (1.0, 0.9, 0.8, 0.7, 0.5, 0.3),
) -> list[dict[str, float]]:
    """按不确定度从低到高保留样本，评估拒识是否有效。"""
    clean = frame[[error_column, uncertainty_column]].dropna().sort_values(uncertainty_column)
    result: list[dict[str, float]] = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage 必须位于 (0,1]")
        count = max(1, int(round(len(clean) * coverage)))
        selected = clean.iloc[:count]
        result.append(
            {
                "coverage": float(coverage),
                "n": int(count),
                "risk": float(selected[error_column].mean()),
                "uncertainty_threshold": float(selected[uncertainty_column].max()),
            }
        )
    return result


def expected_calibration_error(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    bins: int = 10,
) -> float:
    """简单回归校准误差：比较每个 uncertainty quantile 中的平均误差与平均不确定度。

    该数值只有在 uncertainty 与误差使用相同单位时才具有绝对意义。主模型输出的
    Laplace scale 是归一化相位单位，评估脚本会乘 π 后传入。
    """
    errors = np.asarray(errors, dtype=np.float64)
    uncertainties = np.asarray(uncertainties, dtype=np.float64)
    quantiles = np.quantile(uncertainties, np.linspace(0, 1, bins + 1))
    result = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (uncertainties >= quantiles[index]) & (
            uncertainties <= quantiles[index + 1] if upper_inclusive else uncertainties < quantiles[index + 1]
        )
        if not mask.any():
            continue
        result += mask.mean() * abs(errors[mask].mean() - uncertainties[mask].mean())
    return float(result)
