from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    """选择能够整除通道数的 GroupNorm 分组数。

    MCF-QPI 在 128×128 分辨率上通常可使用较大的 batch，但为了让 smoke test、
    小样本微调和单卡低显存训练同样稳定，本项目默认使用 GroupNorm 而不是 BatchNorm。
    """
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """卷积 + GroupNorm + SiLU。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    """两层残差卷积块，可用于空间域和频域分支。"""

    def __init__(self, channels: int, *, dropout: float = 0.0, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, dilation=dilation, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DownBlock(nn.Module):
    """先提取局部特征，再用步长卷积降采样。"""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels, out_channels)
        self.residual = ResidualBlock(out_channels, dropout=dropout)
        self.down = ConvNormAct(out_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.residual(self.project(x))
        return self.down(skip), skip


class UpBlock(nn.Module):
    """双线性上采样后与编码器 skip 拼接，避免反卷积棋盘格伪影。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels + skip_channels, out_channels)
        self.residual = ResidualBlock(out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.residual(self.project(torch.cat([x, skip], dim=1)))


class LatentDecoder(nn.Module):
    """将低分辨率潜变量逐级恢复到相位图尺寸。

    该解码器不使用来自输入散斑的 skip connection。这样，VQ 码本必须真正承载
    相位结构，而不能依靠逆网络的高分辨率旁路偷偷传递信息。
    """

    def __init__(
        self,
        latent_channels: int,
        base_channels: int = 64,
        upsample_stages: int = 3,
        out_channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = base_channels * 2
        self.stem = nn.Sequential(
            ConvNormAct(latent_channels, channels),
            ResidualBlock(channels, dropout=dropout),
        )
        stages: list[nn.Module] = []
        for stage in range(upsample_stages):
            next_channels = max(base_channels // (2 ** max(stage - 1, 0)), 32)
            stages.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ConvNormAct(channels, next_channels),
                    ResidualBlock(next_channels, dropout=dropout),
                ]
            )
            channels = next_channels
        self.body = nn.Sequential(*stages)
        self.head = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.feature_channels = channels

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.body(self.stem(z))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(z))


class ConvEncoder(nn.Module):
    """固定降采样倍数的卷积编码器。"""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        *,
        base_channels: int = 32,
        downsample_stages: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = base_channels
        layers: list[nn.Module] = [ConvNormAct(in_channels, channels)]
        for _ in range(downsample_stages):
            next_channels = min(channels * 2, base_channels * 4)
            layers.extend(
                [
                    ResidualBlock(channels, dropout=dropout),
                    ConvNormAct(channels, next_channels, stride=2),
                ]
            )
            channels = next_channels
        layers.extend(
            [
                ResidualBlock(channels, dropout=dropout),
                nn.Conv2d(channels, latent_channels, kernel_size=1),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_requires_grad(modules: nn.Module | Iterable[nn.Module], requires_grad: bool) -> None:
    if isinstance(modules, nn.Module):
        modules = [modules]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)
