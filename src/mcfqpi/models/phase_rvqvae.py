from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvEncoder, LatentDecoder
from .rvq import ResidualVectorQuantizer


class PhaseRVQVAE(nn.Module):
    """只在真实相位标签上训练的残差 VQ 自编码器。

    MCF-QPI 标签由灰度图映射到 [0,π]。代码中统一把它除以 π 表示为 [0,1]，
    因此解码器末端使用 sigmoid。该先验的目的不是生成“好看的图”，而是形成一个
    离散、受约束的相位状态空间，减少逆网络对训练类别纹理的无约束记忆。
    """

    def __init__(
        self,
        *,
        latent_channels: int = 64,
        base_channels: int = 32,
        downsample_stages: int = 3,
        codebook_size: int = 256,
        num_quantizers: int = 2,
        commitment_beta: float = 0.25,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.downsample_stages = int(downsample_stages)
        self.encoder = ConvEncoder(
            1,
            latent_channels,
            base_channels=base_channels,
            downsample_stages=downsample_stages,
            dropout=dropout,
        )
        self.quantizer = ResidualVectorQuantizer(
            codebook_size,
            latent_channels,
            num_quantizers=num_quantizers,
            commitment_beta=commitment_beta,
        )
        self.decoder = LatentDecoder(
            latent_channels,
            base_channels=base_channels * 2,
            upsample_stages=downsample_stages,
            out_channels=1,
            dropout=dropout,
        )

    def encode(self, phase: torch.Tensor, *, quantize: bool = True) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        z_e = self.encoder(phase)
        output: dict[str, torch.Tensor | list[torch.Tensor]] = {"z_e": z_e}
        if quantize:
            output.update(self.quantizer(z_e))
        return output

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(latent))

    def forward(self, phase: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        encoded = self.encode(phase, quantize=True)
        z_q = encoded["quantized"]
        assert isinstance(z_q, torch.Tensor)
        reconstruction = self.decode(z_q)
        return {"phase": reconstruction, "reconstruction": reconstruction, **encoded}
