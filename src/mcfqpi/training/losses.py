from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """在有效像素上求均值，mask 可为 [B,1,H,W]。"""
    if mask is None:
        return value.mean()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=value.dtype)
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return masked_mean((prediction - target).abs(), mask)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return masked_mean((prediction - target).square(), mask)


def spatial_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def gradient_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred_dx, pred_dy = spatial_gradients(prediction)
    target_dx, target_dy = spatial_gradients(target)
    mask_x = mask[..., :, 1:] * mask[..., :, :-1] if mask is not None else None
    mask_y = mask[..., 1:, :] * mask[..., :-1, :] if mask is not None else None
    return 0.5 * (
        masked_l1(pred_dx, target_dx, mask_x) + masked_l1(pred_dy, target_dy, mask_y)
    )


def _ssim_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回逐像素 SSIM map 和 contrast-structure map。

    使用平均池化实现，避免额外依赖；训练时梯度稳定，最终报告仍由 metrics.py
    计算逐样本指标。
    """
    if window_size % 2 == 0:
        raise ValueError("SSIM window_size 必须为奇数")
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, 1, padding)
    mu_y = F.avg_pool2d(target, window_size, 1, padding)
    sigma_x = F.avg_pool2d(prediction.square(), window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    luminance = (2 * mu_x * mu_y + c1) / (mu_x.square() + mu_y.square() + c1)
    cs = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    return luminance * cs, cs


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    ssim_map, _ = _ssim_components(prediction, target)
    return 1.0 - masked_mean(ssim_map, mask)


def pearson_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """1 - 二维 Pearson 相关系数，逐样本后求平均。"""
    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.to(target.dtype)
    count = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    pred_mean = (prediction * mask).flatten(1).sum(dim=1) / count
    target_mean = (target * mask).flatten(1).sum(dim=1) / count
    pred_centered = (prediction - pred_mean[:, None, None, None]) * mask
    target_centered = (target - target_mean[:, None, None, None]) * mask
    numerator = (pred_centered * target_centered).flatten(1).sum(dim=1)
    denominator = torch.sqrt(
        pred_centered.square().flatten(1).sum(dim=1)
        * target_centered.square().flatten(1).sum(dim=1)
    ).clamp_min(1e-8)
    return (1.0 - numerator / denominator).mean()


def laplace_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_scale: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """异方差 Laplace NLL，输出的 scale 同样是归一化相位单位。"""
    log_scale = log_scale.clamp(-7.0, 1.0)
    value = (prediction - target).abs() * torch.exp(-log_scale) + log_scale
    return masked_mean(value, mask)


def spectral_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """对数幅度谱差异，用于经验前向 twin 的频谱约束。"""
    pred_fft = torch.fft.fft2(prediction, norm="ortho")
    target_fft = torch.fft.fft2(target, norm="ortho")
    return F.l1_loss(torch.log1p(pred_fft.abs()), torch.log1p(target_fft.abs()))


def token_cross_entropy(
    logits: list[torch.Tensor] | tuple[torch.Tensor, ...],
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """Residual VQ 每一级码本的 token 交叉熵。"""
    if target_indices.ndim != 4:
        raise ValueError(f"target_indices 应为 [B,Q,H,W]，得到 {tuple(target_indices.shape)}")
    if len(logits) != target_indices.shape[1]:
        raise ValueError("预测码本级数与 target token 级数不一致")
    losses = [F.cross_entropy(stage_logits, target_indices[:, stage]) for stage, stage_logits in enumerate(logits)]
    return torch.stack(losses).mean()


@dataclass
class LossWeights:
    phase_l1: float = 1.0
    phase_nll: float = 0.0
    gradient: float = 0.15
    ssim: float = 0.20
    pearson: float = 0.05
    vq: float = 0.10
    token: float = 0.10
    latent: float = 0.10
    residual: float = 0.01
    cycle_l1: float = 0.0
    cycle_spectral: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "LossWeights":
        values = values or {}
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(values) - known)
        if unknown:
            raise KeyError(f"未知损失权重：{unknown}")
        return cls(**{key: float(value) for key, value in values.items()})


class InverseLoss(nn.Module):
    """基线和主模型共用的逆问题损失。"""

    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        *,
        forward_twin: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = outputs["phase"]
        target = batch["phase"]
        mask = batch.get("valid_mask")
        terms: dict[str, torch.Tensor] = {}
        terms["phase_l1"] = masked_l1(prediction, target, mask)
        terms["gradient"] = gradient_l1(prediction, target, mask)
        terms["ssim"] = ssim_loss(prediction, target, mask)
        terms["pearson"] = pearson_loss(prediction, target, mask)

        if "log_scale" in outputs:
            terms["phase_nll"] = laplace_nll(prediction, target, outputs["log_scale"], mask)
        else:
            terms["phase_nll"] = prediction.new_zeros(())

        vq_loss = outputs.get("vq_loss")
        terms["vq"] = vq_loss if isinstance(vq_loss, torch.Tensor) else prediction.new_zeros(())

        if "token_logits" in outputs and "target_indices" in outputs:
            terms["token"] = token_cross_entropy(outputs["token_logits"], outputs["target_indices"])
        else:
            terms["token"] = prediction.new_zeros(())

        if "z_e" in outputs and "target_z_e" in outputs:
            terms["latent"] = F.smooth_l1_loss(outputs["z_e"], outputs["target_z_e"])
        else:
            terms["latent"] = prediction.new_zeros(())

        residual = outputs.get("continuous_residual")
        terms["residual"] = residual.abs().mean() if isinstance(residual, torch.Tensor) else prediction.new_zeros(())

        if forward_twin is not None and (self.weights.cycle_l1 > 0 or self.weights.cycle_spectral > 0):
            reconstructed = forward_twin(prediction)["speckle"]
            terms["cycle_l1"] = masked_l1(reconstructed, batch["speckle"], mask)
            terms["cycle_spectral"] = spectral_l1(reconstructed, batch["speckle"])
            outputs["reconstructed_speckle"] = reconstructed
        else:
            terms["cycle_l1"] = prediction.new_zeros(())
            terms["cycle_spectral"] = prediction.new_zeros(())

        total = sum(getattr(self.weights, name) * value for name, value in terms.items())
        return total, terms


class PhasePriorLoss(nn.Module):
    def __init__(self, *, l1: float = 1.0, gradient: float = 0.15, ssim: float = 0.20, vq: float = 1.0) -> None:
        super().__init__()
        self.weights = {"l1": float(l1), "gradient": float(gradient), "ssim": float(ssim), "vq": float(vq)}

    def forward(
        self,
        outputs: dict[str, Any],
        phase: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = outputs["phase"]
        terms = {
            "l1": masked_l1(prediction, phase, mask),
            "gradient": gradient_l1(prediction, phase, mask),
            "ssim": ssim_loss(prediction, phase, mask),
            "vq": outputs["loss"],
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        return total, terms


class ForwardTwinLoss(nn.Module):
    def __init__(self, *, l1: float = 1.0, ssim: float = 0.20, spectral: float = 0.20) -> None:
        super().__init__()
        self.weights = {"l1": float(l1), "ssim": float(ssim), "spectral": float(spectral)}

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = outputs["speckle"]
        terms = {
            "l1": masked_l1(prediction, target, mask),
            "ssim": ssim_loss(prediction, target, mask),
            "spectral": spectral_l1(prediction, target),
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        return total, terms
