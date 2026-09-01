from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConvNormAct, _group_count


class SinusoidalTimeEmbedding(nn.Module):
    """DDPM 标准正弦时间步嵌入。"""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(-scale * torch.arange(half, device=time.device, dtype=torch.float32))
        values = time.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([values.sin(), values.cos()], dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, dropout=dropout)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x)
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.conv2(hidden)
        return hidden + self.skip(x)


class ConditionalDenoiser(nn.Module):
    """用于 MCF-QPI 的轻量 speckle-conditioned noise predictor。

    该实现是可复现的条件扩散基线，不声称逐层复现 SpecDiffusion 官方网络。
    条件散斑在输入端与带噪相位拼接，并在各尺度重新下采样后注入。
    """

    def __init__(self, base_channels: int = 32, time_dim: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.stem = ConvNormAct(2, base_channels)
        self.block1 = TimeResidualBlock(base_channels + 1, base_channels, time_dim, dropout)
        self.down1 = ConvNormAct(base_channels, base_channels * 2, stride=2)
        self.block2 = TimeResidualBlock(base_channels * 2 + 1, base_channels * 2, time_dim, dropout)
        self.down2 = ConvNormAct(base_channels * 2, base_channels * 4, stride=2)
        self.block3 = TimeResidualBlock(base_channels * 4 + 1, base_channels * 4, time_dim, dropout)
        self.down3 = ConvNormAct(base_channels * 4, base_channels * 8, stride=2)
        self.middle = nn.ModuleList(
            [
                TimeResidualBlock(base_channels * 8 + 1, base_channels * 8, time_dim, dropout),
                TimeResidualBlock(base_channels * 8, base_channels * 8, time_dim, dropout),
            ]
        )
        self.up3 = TimeResidualBlock(base_channels * 8 + base_channels * 4 + 1, base_channels * 4, time_dim, dropout)
        self.up2 = TimeResidualBlock(base_channels * 4 + base_channels * 2 + 1, base_channels * 2, time_dim, dropout)
        self.up1 = TimeResidualBlock(base_channels * 2 + base_channels + 1, base_channels, time_dim, dropout)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    @staticmethod
    def _condition_at(speckle: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return F.interpolate(speckle, size=feature.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, noisy_phase: torch.Tensor, speckle: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embed(time)
        x0 = self.stem(torch.cat([noisy_phase, speckle], dim=1))
        s1 = self.block1(torch.cat([x0, self._condition_at(speckle, x0)], dim=1), time_embedding)
        x1 = self.down1(s1)
        s2 = self.block2(torch.cat([x1, self._condition_at(speckle, x1)], dim=1), time_embedding)
        x2 = self.down2(s2)
        s3 = self.block3(torch.cat([x2, self._condition_at(speckle, x2)], dim=1), time_embedding)
        x3 = self.down3(s3)
        x = self.middle[0](torch.cat([x3, self._condition_at(speckle, x3)], dim=1), time_embedding)
        x = self.middle[1](x, time_embedding)
        x = F.interpolate(x, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, s3, self._condition_at(speckle, s3)], dim=1), time_embedding)
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, s2, self._condition_at(speckle, s2)], dim=1), time_embedding)
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, s1, self._condition_at(speckle, s1)], dim=1), time_embedding)
        return self.head(x)


class GaussianDiffusion(nn.Module):
    """线性 beta DDPM 训练 + DDIM 推理。"""

    def __init__(
        self,
        denoiser: ConditionalDenoiser,
        *,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"mse", "l1"}:
            raise ValueError("loss_type 必须是 mse 或 l1")
        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    @staticmethod
    def _extract(values: torch.Tensor, time: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        result = values.gather(0, time)
        return result.view(time.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, phase_minus_one_one: torch.Tensor, time: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_bar, time, phase_minus_one_one.shape) * phase_minus_one_one
            + self._extract(self.sqrt_one_minus_alpha_bar, time, phase_minus_one_one.shape) * noise
        )

    def training_loss(self, phase: torch.Tensor, speckle: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = phase * 2.0 - 1.0
        time = torch.randint(0, self.timesteps, (phase.shape[0],), device=phase.device)
        noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, time, noise)
        prediction = self.denoiser(noisy, speckle, time)
        if self.loss_type == "mse":
            loss = F.mse_loss(prediction, noise)
            return loss, {"noise_mse": loss}
        loss = F.l1_loss(prediction, noise)
        return loss, {"noise_l1": loss}

    @torch.no_grad()
    def sample(
        self,
        speckle: torch.Tensor,
        *,
        steps: int = 50,
        eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDIM 采样，返回 [0,1] 相位。"""
        batch, _, height, width = speckle.shape
        x = initial_noise if initial_noise is not None else torch.randn(batch, 1, height, width, device=speckle.device)
        steps = max(1, min(int(steps), self.timesteps))
        # unique_consecutive 避免 steps 接近 timesteps 时 round 产生重复时间步。
        schedule = torch.unique_consecutive(
            torch.linspace(self.timesteps - 1, 0, steps, device=speckle.device).round().long()
        )
        for index, current in enumerate(schedule):
            time = torch.full((batch,), int(current.item()), device=speckle.device, dtype=torch.long)
            alpha_current = self.alpha_bar[current]
            noise_prediction = self.denoiser(x, speckle, time)
            x0 = (x - torch.sqrt(1.0 - alpha_current) * noise_prediction) / torch.sqrt(alpha_current)
            x0 = x0.clamp(-1.0, 1.0)
            if index == len(schedule) - 1:
                x = x0
                break
            next_time = schedule[index + 1]
            alpha_next = self.alpha_bar[next_time]
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_next) / (1.0 - alpha_current))
                * (1.0 - alpha_current / alpha_next)
            ).clamp_min(0)
            direction = torch.sqrt((1.0 - alpha_next - sigma.square()).clamp_min(0)) * noise_prediction
            random_term = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_next) * x0 + direction + random_term
        return ((x + 1.0) / 2.0).clamp(0, 1)


class DiffusionPhaseReconstructor(nn.Module):
    """把 GaussianDiffusion 包装成与其他逆网络一致的评估接口。

    forward 只接收散斑并返回 {"phase": ...}。由于扩散采样天然随机，正式评估应
    固定随机种子，并在需要时重复采样报告均值/方差和推理延迟。
    """

    def __init__(self, diffusion: GaussianDiffusion, *, sample_steps: int = 100, eta: float = 0.0) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.sample_steps = int(sample_steps)
        self.eta = float(eta)

    def forward(self, speckle: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        phase = self.diffusion.sample(speckle, steps=self.sample_steps, eta=self.eta)
        return {"phase": phase}
