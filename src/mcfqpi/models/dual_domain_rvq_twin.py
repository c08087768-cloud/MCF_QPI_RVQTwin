from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvEncoder, LatentDecoder, ResidualBlock, set_requires_grad
from .phase_rvqvae import PhaseRVQVAE


class FourierMagnitude(nn.Module):
    """把散斑转换为归一化的对数 Fourier 幅度图。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        magnitude = torch.log1p(spectrum.abs())
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (magnitude - mean) / std


class DualDomainSpeckleEncoder(nn.Module):
    """空间域 + Fourier 域双分支散斑编码器。"""

    def __init__(
        self,
        latent_channels: int,
        *,
        base_channels: int = 32,
        downsample_stages: int = 3,
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
        self.spatial_encoder = ConvEncoder(
            1,
            latent_channels,
            base_channels=base_channels,
            downsample_stages=downsample_stages,
            dropout=dropout,
        )
        self.frequency_encoder = ConvEncoder(
            1,
            latent_channels,
            base_channels=base_channels,
            downsample_stages=downsample_stages,
            dropout=dropout,
        )
        self.gate = nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=1),
            ResidualBlock(latent_channels, dropout=dropout),
        )

    def forward(self, speckle: torch.Tensor) -> dict[str, torch.Tensor]:
        # 消融实验需要能够真正关闭某一分支，而不是仅把其损失权重设为零。
        # 为保持 checkpoint 结构稳定，模块仍被构造，但关闭的分支不会执行前向。
        spatial: torch.Tensor | None = None
        frequency: torch.Tensor | None = None
        if self.use_spatial:
            spatial = self.spatial_encoder(speckle)
        if self.use_frequency:
            frequency = self.frequency_encoder(self.fourier(speckle))

        if spatial is not None and frequency is not None:
            joint = torch.cat([spatial, frequency], dim=1)
            gate = torch.sigmoid(self.gate(joint))
            gated = gate * spatial + (1.0 - gate) * frequency
            fused = gated + self.fusion(joint)
        elif spatial is not None:
            fused = spatial
            frequency = torch.zeros_like(spatial)
            gate = torch.ones_like(spatial)
        elif frequency is not None:
            fused = frequency
            spatial = torch.zeros_like(frequency)
            gate = torch.zeros_like(frequency)
        else:  # __init__ 已拦截，仅用于静态类型收窄和防御式编程。
            raise RuntimeError("空间域和频域分支均未启用")

        return {
            "latent": fused,
            "spatial_latent": spatial,
            "frequency_latent": frequency,
            "fusion_gate": gate,
        }


class DualDomainRVQTwin(nn.Module):
    """本文建议的 MCF-only 主模型。

    1. 双域编码器从真实远场散斑提取潜变量；
    2. 使用相位 RVQ-VAE 的离散码本投影到“合法相位”状态空间；
    3. 小幅连续残差补偿硬量化造成的细节损失；
    4. 输出相位和像素级不确定度；
    5. 训练时可用相位先验编码器提供 token/latent 对齐监督。
    """

    def __init__(
        self,
        phase_prior: PhaseRVQVAE,
        *,
        base_channels: int = 32,
        dropout: float = 0.10,
        residual_scale: float = 0.15,
        freeze_prior_encoder: bool = True,
        freeze_codebook: bool = True,
        freeze_decoder: bool = False,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_quantization: bool = True,
    ) -> None:
        super().__init__()
        self.phase_prior = phase_prior
        self.residual_scale = float(residual_scale)
        self.use_quantization = bool(use_quantization)
        self.speckle_encoder = DualDomainSpeckleEncoder(
            phase_prior.latent_channels,
            base_channels=base_channels,
            downsample_stages=phase_prior.downsample_stages,
            dropout=dropout,
            use_spatial=use_spatial,
            use_frequency=use_frequency,
        )
        latent_channels = phase_prior.latent_channels
        self.continuous_residual = nn.Sequential(
            ResidualBlock(latent_channels, dropout=dropout),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
            nn.Tanh(),
        )
        # 单独的不确定度解码器，避免改动预训练相位解码器的结构。
        self.uncertainty_decoder = LatentDecoder(
            latent_channels,
            base_channels=base_channels * 2,
            upsample_stages=phase_prior.downsample_stages,
            out_channels=1,
            dropout=dropout,
        )
        if freeze_prior_encoder:
            set_requires_grad(self.phase_prior.encoder, False)
        if freeze_codebook:
            set_requires_grad(self.phase_prior.quantizer, False)
        if freeze_decoder:
            set_requires_grad(self.phase_prior.decoder, False)

    def train(self, mode: bool = True) -> "DualDomainRVQTwin":
        super().train(mode)
        # 冻结模块即使在 model.train() 后也保持 eval，避免其中 dropout 改变 target token。
        if not any(p.requires_grad for p in self.phase_prior.encoder.parameters()):
            self.phase_prior.encoder.eval()
        if not any(p.requires_grad for p in self.phase_prior.quantizer.parameters()):
            self.phase_prior.quantizer.eval()
        if not any(p.requires_grad for p in self.phase_prior.decoder.parameters()):
            self.phase_prior.decoder.eval()
        return self

    def forward(
        self,
        speckle: torch.Tensor,
        *,
        target_phase: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        encoded = self.speckle_encoder(speckle)
        z_e = encoded["latent"]
        continuous = self.continuous_residual(z_e) * self.residual_scale
        output: dict[str, torch.Tensor | list[torch.Tensor]]
        if self.use_quantization:
            quantized = self.phase_prior.quantizer(z_e)
            z_q = quantized["quantized"]
            assert isinstance(z_q, torch.Tensor)
            latent = z_q + continuous
            output = {
                "indices": quantized["indices"],
                "token_logits": quantized["logits"],
                "vq_loss": quantized["loss"],
                "perplexity": quantized["perplexity"],
            }
        else:
            # 连续潜空间消融：仍复用相位先验解码器，但不经过离散码本。
            z_q = z_e
            latent = z_e + continuous
            output = {
                "vq_loss": z_e.new_zeros(()),
                "perplexity": z_e.new_zeros((0,)),
            }
        phase = self.phase_prior.decode(latent)
        log_scale = self.uncertainty_decoder(latent).clamp(-7.0, 1.0)

        output.update({
            "phase": phase,
            "log_scale": log_scale,
            "z_e": z_e,
            "z_q": z_q,
            "continuous_residual": continuous,
            **encoded,
        })
        if target_phase is not None:
            # target token 是监督信号，不需要反向传播到相位先验编码器。
            with torch.no_grad():
                target = self.phase_prior.encode(target_phase, quantize=True)
            output["target_z_e"] = target["z_e"]
            output["target_z_q"] = target["quantized"]
            if self.use_quantization:
                output["target_indices"] = target["indices"]
        return output
