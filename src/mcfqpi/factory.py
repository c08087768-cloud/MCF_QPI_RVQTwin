from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .models import (
    ConditionalDenoiser,
    DiffusionPhaseReconstructor,
    DualDomainPriorRefiner,
    DualDomainResUNet,
    DualDomainRVQTwin,
    EmpiricalForwardTwin,
    GaussianDiffusion,
    PhaseRVQVAE,
    ResUNetPhase,
)
from .utils import load_checkpoint


def _model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    # 兼容直接保存 state_dict 的情况。
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]
    raise KeyError("checkpoint 中找不到 model/state_dict")


def build_phase_prior(config: dict[str, Any]) -> PhaseRVQVAE:
    return PhaseRVQVAE(
        latent_channels=int(config.get("latent_channels", 64)),
        base_channels=int(config.get("base_channels", 32)),
        downsample_stages=int(config.get("downsample_stages", 3)),
        codebook_size=int(config.get("codebook_size", 256)),
        num_quantizers=int(config.get("num_quantizers", 2)),
        commitment_beta=float(config.get("commitment_beta", 0.25)),
        dropout=float(config.get("dropout", 0.05)),
    )


def load_phase_prior(
    checkpoint_path: str | Path,
    *,
    model_config: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> PhaseRVQVAE:
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    if model_config is None:
        saved_config = checkpoint.get("config", {})
        model_config = saved_config.get("model", saved_config.get("phase_prior", {}))
    model = build_phase_prior(model_config or {})
    missing, unexpected = model.load_state_dict(_model_state(checkpoint), strict=strict)
    if not strict and (missing or unexpected):
        print(f"[警告] Phase prior 非严格加载：missing={missing}, unexpected={unexpected}")
    return model


def build_forward_twin(config: dict[str, Any]) -> EmpiricalForwardTwin:
    return EmpiricalForwardTwin(
        base_channels=int(config.get("base_channels", 32)),
        dropout=float(config.get("dropout", 0.05)),
    )


def build_diffusion(config: dict[str, Any]) -> GaussianDiffusion:
    """构建可训练的 speckle-conditioned DDPM/DDIM 基线。

    该实现用于公平的“生成式基线”比较，不声称逐层复现 SpecDiffusion 论文。
    """
    denoiser = ConditionalDenoiser(
        base_channels=int(config.get("base_channels", 32)),
        time_dim=int(config.get("time_dim", 128)),
        dropout=float(config.get("dropout", 0.05)),
    )
    return GaussianDiffusion(
        denoiser,
        timesteps=int(config.get("timesteps", 1000)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
        loss_type=str(config.get("loss_type", "mse")),
    )


def build_diffusion_reconstructor(config: dict[str, Any]) -> DiffusionPhaseReconstructor:
    diffusion = build_diffusion(config)
    return DiffusionPhaseReconstructor(
        diffusion,
        sample_steps=int(config.get("sample_steps", 100)),
        eta=float(config.get("eta", 0.0)),
    )


def build_inverse_model(config: dict[str, Any], *, map_location: str | torch.device = "cpu") -> nn.Module:
    model = build_inverse_architecture(config)
    if isinstance(model, (DualDomainRVQTwin, DualDomainPriorRefiner)):
        checkpoint = config.get("phase_prior_checkpoint")
        if not checkpoint:
            raise ValueError("新训练主模型需要 model.phase_prior_checkpoint")
        prior = load_phase_prior(
            checkpoint,
            model_config=config.get("phase_prior_model"),
            map_location=map_location,
            strict=True,
        )
        model.phase_prior.load_state_dict(prior.state_dict(), strict=True)
    return model


def build_inverse_architecture(config: dict[str, Any]) -> nn.Module:
    """只按配置构建架构，不读取任何初始化权重。"""
    model_type = str(config.get("type", "resunet")).lower()
    if model_type in {"resunet", "baseline", "unet"}:
        return ResUNetPhase(
            in_channels=1,
            base_channels=int(config.get("base_channels", 32)),
            dropout=float(config.get("dropout", 0.10)),
            predict_uncertainty=bool(config.get("predict_uncertainty", True)),
        )
    if model_type in {"dual_domain_resunet", "dual_resunet"}:
        return DualDomainResUNet(
            base_channels=int(config.get("base_channels", 24)),
            dropout=float(config.get("dropout", 0.10)),
            use_spatial=bool(config.get("use_spatial", True)),
            use_frequency=bool(config.get("use_frequency", True)),
        )
    if model_type in {"dual_domain_rvq_twin", "rvqtwin", "proposed"}:
        prior = build_phase_prior(config.get("phase_prior_model", {}))
        return DualDomainRVQTwin(
            prior,
            base_channels=int(config.get("base_channels", 32)),
            dropout=float(config.get("dropout", 0.10)),
            residual_scale=float(config.get("residual_scale", 0.15)),
            freeze_prior_encoder=bool(config.get("freeze_prior_encoder", True)),
            freeze_codebook=bool(config.get("freeze_codebook", True)),
            freeze_decoder=bool(config.get("freeze_decoder", False)),
            use_spatial=bool(config.get("use_spatial", True)),
            use_frequency=bool(config.get("use_frequency", True)),
            use_quantization=bool(config.get("use_quantization", True)),
        )
    if model_type in {"dual_domain_prior_refiner", "prior_refiner"}:
        prior = build_phase_prior(config.get("phase_prior_model", {}))
        return DualDomainPriorRefiner(
            prior,
            base_channels=int(config.get("base_channels", 32)),
            dropout=float(config.get("dropout", 0.10)),
            residual_scale=float(config.get("residual_scale", 0.15)),
            detail_scale_max=float(config.get("detail_scale_max", 0.25)),
            detail_scale_init=float(config.get("detail_scale_init", 0.05)),
            freeze_prior_encoder=bool(config.get("freeze_prior_encoder", True)),
            freeze_codebook=bool(config.get("freeze_codebook", True)),
            freeze_decoder=bool(config.get("freeze_decoder", False)),
            use_spatial=bool(config.get("use_spatial", True)),
            use_frequency=bool(config.get("use_frequency", True)),
            use_quantization=bool(config.get("use_quantization", True)),
        )
    raise ValueError(f"未知 model.type：{model_type}")


def build_inverse_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """从自包含纯推理 checkpoint 构建架构并严格加载权重。"""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_config" not in checkpoint:
        raise ValueError("不是自包含 MCF-QPI 推理 checkpoint")
    model = build_inverse_architecture(checkpoint["model_config"])
    model.load_state_dict(_model_state(checkpoint), strict=True)
    return model, checkpoint


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    missing, unexpected = model.load_state_dict(_model_state(checkpoint), strict=strict)
    if not strict and (missing or unexpected):
        print(f"[警告] 非严格加载：missing={missing}, unexpected={unexpected}")
    return checkpoint
