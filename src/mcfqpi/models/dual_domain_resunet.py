from __future__ import annotations

import torch
from torch import nn

from .blocks import ResidualBlock, UpBlock
from .dual_domain_prior_refiner import GatedScaleFusion, PyramidEncoder
from .dual_domain_rvq_twin import FourierMagnitude


class DualDomainResUNet(nn.Module):
    """不使用 phase prior 的双域、多尺度 ResUNet 基线。"""

    def __init__(
        self,
        *,
        base_channels: int = 24,
        dropout: float = 0.10,
        use_spatial: bool = True,
        use_frequency: bool = True,
    ) -> None:
        super().__init__()
        if not use_spatial and not use_frequency:
            raise ValueError("空间域和频域分支至少启用一个")
        self.use_spatial = bool(use_spatial)
        self.use_frequency = bool(use_frequency)
        self.fourier = FourierMagnitude()
        self.spatial_encoder = PyramidEncoder(base_channels=base_channels, dropout=dropout)
        self.frequency_encoder = PyramidEncoder(base_channels=base_channels, dropout=dropout)
        channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8]
        self.fusions = nn.ModuleList([GatedScaleFusion(channel, dropout=dropout) for channel in channels])
        self.up3 = UpBlock(channels[3], channels[2], channels[1], dropout=dropout)
        self.up2 = UpBlock(channels[1], channels[1], channels[0], dropout=dropout)
        self.up1 = UpBlock(channels[0], channels[0], base_channels, dropout=dropout)
        self.refine = ResidualBlock(base_channels, dropout=dropout)
        self.phase_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, speckle: torch.Tensor, **_: object) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        spatial = self.spatial_encoder(speckle) if self.use_spatial else None
        frequency = self.frequency_encoder(self.fourier(speckle)) if self.use_frequency else None
        fused_scales: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        spatial_scales: list[torch.Tensor] = []
        frequency_scales: list[torch.Tensor] = []
        for level, fusion in enumerate(self.fusions):
            if spatial is not None and frequency is not None:
                spatial_feature = spatial[level]
                frequency_feature = frequency[level]
                fused, gate = fusion(spatial_feature, frequency_feature)
            elif spatial is not None:
                spatial_feature = spatial[level]
                frequency_feature = torch.zeros_like(spatial_feature)
                fused = spatial_feature
                gate = torch.ones_like(spatial_feature)
            else:
                assert frequency is not None
                frequency_feature = frequency[level]
                spatial_feature = torch.zeros_like(frequency_feature)
                fused = frequency_feature
                gate = torch.zeros_like(frequency_feature)
            spatial_scales.append(spatial_feature)
            frequency_scales.append(frequency_feature)
            fused_scales.append(fused)
            gates.append(gate)

        x = self.up3(fused_scales[3], fused_scales[2])
        x = self.up2(x, fused_scales[1])
        x = self.up1(x, fused_scales[0])
        features = self.refine(x)
        return {
            "phase": torch.sigmoid(self.phase_head(features)),
            "log_scale": self.uncertainty_head(features).clamp(-7.0, 1.0),
            "features": features,
            "spatial_latent": spatial_scales[-1],
            "frequency_latent": frequency_scales[-1],
            "fusion_gate": gates[-1],
            "fused_scales": fused_scales,
        }
