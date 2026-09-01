from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct, DownBlock, ResidualBlock, UpBlock


class EmpiricalForwardTwin(nn.Module):
    """由真实配对数据学习的“相位 → 归一化散斑”经验前向代理。

    注意：它不是多芯光纤的解析传播模型，也不等于实测 transmission matrix。
    它只近似当前公开数据采集系统中的统计映射。只有当独立验证集上的前向重建
    足够好时，才应把它冻结并用于逆网络的 cycle consistency。
    """

    def __init__(self, base_channels: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.stem = ConvNormAct(1, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, dropout=dropout, dilation=1),
            ResidualBlock(base_channels * 8, dropout=dropout, dilation=2),
        )
        self.up3 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, dropout=dropout)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, dropout=dropout)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels, dropout=dropout)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.stem(phase)
        x1, s1 = self.down1(x0)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x = self.bottleneck(x3)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        # 数据预处理后的散斑位于 [0,1]，用 sigmoid 保持一致。
        return {"speckle": torch.sigmoid(self.head(x))}
