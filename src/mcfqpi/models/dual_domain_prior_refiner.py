from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .blocks import ConvNormAct, DownBlock, ResidualBlock, UpBlock, set_requires_grad
from .dual_domain_rvq_twin import FourierMagnitude
from .phase_rvqvae import PhaseRVQVAE


class PyramidEncoder(nn.Module):
    """生成供细节解码器使用的四尺度特征金字塔。"""

    def __init__(self, *, base_channels: int, dropout: float) -> None:
        super().__init__()
        self.stem = ConvNormAct(1, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, dropout=dropout),
            ResidualBlock(base_channels * 8, dropout=dropout, dilation=2),
        )

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(image)
        x, scale1 = self.down1(x)
        x, scale2 = self.down2(x)
        x, scale3 = self.down3(x)
        return [scale1, scale2, scale3, self.bottleneck(x)]


class GatedScaleFusion(nn.Module):
    """同尺度空间/频域特征的逐位置门控融合。"""

    def __init__(self, channels: int, *, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            ResidualBlock(channels, dropout=dropout),
        )

    def forward(self, spatial: torch.Tensor, frequency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat([spatial, frequency], dim=1)
        gate = torch.sigmoid(self.gate(joint))
        fused = gate * spatial + (1.0 - gate) * frequency + self.fusion(joint)
        return fused, gate


class DualDomainPriorRefiner(nn.Module):
    """双域编码、相位 RVQ 先验与多尺度细节修正的相位反演网络。"""

    def __init__(
        self,
        phase_prior: PhaseRVQVAE,
        *,
        base_channels: int = 32,
        dropout: float = 0.10,
        residual_scale: float = 0.15,
        detail_scale_max: float = 0.25,
        detail_scale_init: float = 0.05,
        freeze_prior_encoder: bool = True,
        freeze_codebook: bool = True,
        freeze_decoder: bool = False,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_quantization: bool = True,
    ) -> None:
        super().__init__()
        if not use_spatial and not use_frequency:
            raise ValueError("空间域和频域分支至少启用一个")
        if not math.isfinite(detail_scale_max) or detail_scale_max <= 0.0:
            raise ValueError("detail_scale_max 必须是有限正数")
        if not math.isfinite(detail_scale_init) or not 0.0 < detail_scale_init < detail_scale_max:
            raise ValueError("detail_scale_init 必须位于 (0, detail_scale_max) 内")

        self.phase_prior = phase_prior
        self.residual_scale = float(residual_scale)
        self.detail_scale_max = float(detail_scale_max)
        self.use_spatial = bool(use_spatial)
        self.use_frequency = bool(use_frequency)
        self.use_quantization = bool(use_quantization)
        self.fourier = FourierMagnitude()
        self.spatial_encoder = PyramidEncoder(base_channels=base_channels, dropout=dropout)
        self.frequency_encoder = PyramidEncoder(base_channels=base_channels, dropout=dropout)
        channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8]
        self.fusions = nn.ModuleList([GatedScaleFusion(channel, dropout=dropout) for channel in channels])
        latent_channels = phase_prior.latent_channels
        self.latent_projection = nn.Conv2d(channels[-1], latent_channels, kernel_size=1)
        self.continuous_residual = nn.Sequential(
            ResidualBlock(latent_channels, dropout=dropout),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
            nn.Tanh(),
        )
        self.up3 = UpBlock(channels[3], channels[2], channels[1], dropout=dropout)
        self.up2 = UpBlock(channels[1], channels[1], channels[0], dropout=dropout)
        self.up1 = UpBlock(channels[0], channels[0], base_channels, dropout=dropout)
        self.detail_refine = ResidualBlock(base_channels, dropout=dropout)
        self.detail_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        initial_fraction = detail_scale_init / detail_scale_max
        self.detail_scale_logit = nn.Parameter(torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction))))

        if freeze_prior_encoder:
            set_requires_grad(self.phase_prior.encoder, False)
        if freeze_codebook:
            set_requires_grad(self.phase_prior.quantizer, False)
        if freeze_decoder:
            set_requires_grad(self.phase_prior.decoder, False)

    def train(self, mode: bool = True) -> "DualDomainPriorRefiner":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.phase_prior.encoder.parameters()):
            self.phase_prior.encoder.eval()
        if not any(parameter.requires_grad for parameter in self.phase_prior.quantizer.parameters()):
            self.phase_prior.quantizer.eval()
        if not any(parameter.requires_grad for parameter in self.phase_prior.decoder.parameters()):
            self.phase_prior.decoder.eval()
        return self

    def _encode(self, speckle: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        spatial_scales = self.spatial_encoder(speckle) if self.use_spatial else []
        frequency_scales = self.frequency_encoder(self.fourier(speckle)) if self.use_frequency else []
        fused_scales: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for level, fusion in enumerate(self.fusions):
            if self.use_spatial and self.use_frequency:
                fused, gate = fusion(spatial_scales[level], frequency_scales[level])
            elif self.use_spatial:
                fused = spatial_scales[level]
                frequency_scales.append(torch.zeros_like(fused))
                gate = torch.ones_like(fused)
            else:
                fused = frequency_scales[level]
                spatial_scales.append(torch.zeros_like(fused))
                gate = torch.zeros_like(fused)
            fused_scales.append(fused)
            gates.append(gate)
        return spatial_scales, frequency_scales, fused_scales, gates

    def _decode_detail(self, fused_scales: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        detail = self.up3(fused_scales[3], fused_scales[2])
        detail = self.up2(detail, fused_scales[1])
        detail = self.up1(detail, fused_scales[0])
        detail = self.detail_refine(detail)
        return self.detail_head(detail), self.uncertainty_head(detail).clamp(-7.0, 1.0)

    def forward(
        self,
        speckle: torch.Tensor,
        *,
        target_phase: torch.Tensor | None = None,
        token_intervention: str | None = None,
    ) -> dict[str, Any]:
        spatial_scales, frequency_scales, fused_scales, gates = self._encode(speckle)
        z_e = self.latent_projection(fused_scales[-1])
        continuous = self.continuous_residual(z_e) * self.residual_scale
        output: dict[str, Any]
        if self.use_quantization:
            quantized = self.phase_prior.quantizer(z_e)
            z_q = quantized["quantized"]
            if token_intervention == "shuffle":
                z_q = torch.flip(z_q, dims=(-2, -1))
            elif token_intervention == "mean":
                z_q = z_q.mean(dim=(-2, -1), keepdim=True).expand_as(z_q)
            elif token_intervention not in {None, "none"}:
                raise ValueError(f"未知 token_intervention：{token_intervention}")
            output = {
                "indices": quantized["indices"],
                "token_logits": quantized["logits"],
                "vq_loss": quantized["loss"],
                "perplexity": quantized["perplexity"],
            }
        else:
            z_q = z_e
            output = {"vq_loss": z_e.new_zeros(()), "perplexity": z_e.new_zeros((0,))}

        phase_prior = self.phase_prior.decode(z_q + continuous)
        detail_phase, log_scale = self._decode_detail(fused_scales)
        detail_scale = self.detail_scale_max * torch.sigmoid(self.detail_scale_logit)
        phase = phase_prior + detail_scale * detail_phase
        output.update({
            "phase": phase,
            "phase_prior": phase_prior,
            "detail_phase": detail_phase,
            "detail_scale": detail_scale,
            "log_scale": log_scale,
            "z_e": z_e,
            "z_q": z_q,
            "continuous_residual": continuous,
            "spatial_latent": spatial_scales[-1],
            "frequency_latent": frequency_scales[-1],
            "fusion_gate": gates[-1],
            "fused_scales": fused_scales,
        })
        if target_phase is not None:
            with torch.no_grad():
                target = self.phase_prior.encode(target_phase, quantize=True)
            output["target_z_e"] = target["z_e"]
            output["target_z_q"] = target["quantized"]
            if self.use_quantization:
                output["target_indices"] = target["indices"]
        return output
