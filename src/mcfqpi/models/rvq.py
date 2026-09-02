from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    raw_quantized: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor
    logits: torch.Tensor
    perplexity: torch.Tensor


class VectorQuantizer(nn.Module):
    """可微矢量量化器。

    输入形状为 [B,C,H,W]。距离的负值可作为 token logits，用于让散斑编码器
    对齐相位先验得到的离散 token。码本采用普通梯度更新，结构直观且易于复现。
    """

    def __init__(self, codebook_size: int, embedding_dim: int, commitment_beta: float = 0.25) -> None:
        super().__init__()
        if codebook_size < 2:
            raise ValueError("codebook_size 必须至少为 2")
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.commitment_beta = float(commitment_beta)
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def distance_logits(self, x: torch.Tensor) -> torch.Tensor:
        """返回 [B,K,H,W] 的负平方距离。"""
        if x.ndim != 4 or x.shape[1] != self.embedding_dim:
            raise ValueError(f"量化器期望 [B,{self.embedding_dim},H,W]，得到 {tuple(x.shape)}")
        flat = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        code = self.embedding.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + code.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ code.t()
        )
        batch, _, height, width = x.shape
        return (-distances).view(batch, height, width, self.codebook_size).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> QuantizerOutput:
        logits = self.distance_logits(x)
        indices = logits.argmax(dim=1)
        raw_quantized = F.embedding(indices, self.embedding.weight).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(raw_quantized, x.detach())
        commitment_loss = F.mse_loss(x, raw_quantized.detach())
        loss = codebook_loss + self.commitment_beta * commitment_loss

        # Straight-through estimator：前向使用离散码，反向把梯度传给编码器。
        quantized = x + (raw_quantized - x).detach()
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
        average_prob = one_hot.mean(dim=(0, 1, 2))
        perplexity = torch.exp(-(average_prob * torch.log(average_prob + 1e-10)).sum())
        return QuantizerOutput(
            quantized=quantized,
            raw_quantized=raw_quantized,
            loss=loss,
            indices=indices,
            logits=logits,
            perplexity=perplexity,
        )


class ResidualVectorQuantizer(nn.Module):
    """Residual VQ：多个码本依次量化前一层留下的残差。"""

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        num_quantizers: int = 2,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()
        if num_quantizers < 1:
            raise ValueError("num_quantizers 必须至少为 1")
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.num_quantizers = int(num_quantizers)
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, embedding_dim, commitment_beta)
                for _ in range(self.num_quantizers)
            ]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        residual = x
        raw_sum = torch.zeros_like(x)
        losses: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        perplexities: list[torch.Tensor] = []
        for quantizer in self.quantizers:
            output = quantizer(residual)
            raw_sum = raw_sum + output.raw_quantized
            # 下一层只看未被前面码本解释的残差；detach 避免残差路径重复更新旧码本。
            residual = residual - output.raw_quantized.detach()
            losses.append(output.loss)
            indices.append(output.indices)
            logits.append(output.logits)
            perplexities.append(output.perplexity)
        return {
            "quantized": x + (raw_sum - x).detach(),
            "raw_quantized": raw_sum,
            "loss": torch.stack(losses).mean(),
            "indices": torch.stack(indices, dim=1),  # [B,Q,H,W]
            "logits": logits,
            "perplexity": torch.stack(perplexities),
            "residual": residual,
        }

    @torch.no_grad()
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """由 [B,Q,H,W] token 索引恢复量化潜变量。"""
        if indices.ndim != 4 or indices.shape[1] != self.num_quantizers:
            raise ValueError(
                f"indices 应为 [B,{self.num_quantizers},H,W]，实际 {tuple(indices.shape)}"
            )
        result: torch.Tensor | None = None
        for stage, quantizer in enumerate(self.quantizers):
            value = F.embedding(indices[:, stage], quantizer.embedding.weight).permute(0, 3, 1, 2)
            result = value if result is None else result + value
        assert result is not None
        return result.contiguous()
