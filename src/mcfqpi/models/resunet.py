from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct, DownBlock, ResidualBlock, UpBlock


class ResUNetPhase(nn.Module):
    """复现级监督基线：单通道散斑直接回归 [0,1] 相位。

    原 MCF-QPI 论文使用 U-Net 与 ResNet 的混合结构。这里实现一个清晰、可复现的
    ResUNet 基线，并额外输出像素级 Laplace 尺度参数，便于与主模型采用同一套
    不确定度评估；关闭 uncertainty 时不会计算对应损失。
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.10,
        predict_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, base_channels)
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
        self.refine = ResidualBlock(base_channels, dropout=dropout)
        self.phase_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(base_channels, 1, kernel_size=1) if predict_uncertainty else None

    def forward(self, speckle: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        x0 = self.stem(speckle)
        x1, skip1 = self.down1(x0)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x = self.bottleneck(x3)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        x = self.refine(x)
        phase = torch.sigmoid(self.phase_head(x))
        output: dict[str, torch.Tensor] = {"phase": phase, "features": x}
        if self.uncertainty_head is not None:
            # 归一化相位单位下的 log(b)，限制极端值防止 NLL 数值发散。
            output["log_scale"] = self.uncertainty_head(x).clamp(-7.0, 1.0)
        return output
