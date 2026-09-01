from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DeterministicCorruptionDataset(Dataset[dict[str, Any]]):
    """在测试时施加可复现的相机/对准扰动，不改变相位标签。"""

    def __init__(
        self,
        base: Dataset[dict[str, Any]],
        *,
        corruption: str,
        severity: float,
        seed: int = 42,
    ) -> None:
        self.base = base
        self.corruption = corruption
        self.severity = float(severity)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        image = sample["speckle"].clone()
        generator = torch.Generator().manual_seed(self.seed + index * 1009)
        name = self.corruption
        value = self.severity
        if name == "clean":
            pass
        elif name == "gaussian_noise":
            noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
            image = image + value * noise
        elif name == "gain":
            sign = -1.0 if torch.rand((), generator=generator) < 0.5 else 1.0
            image = image * (1.0 + sign * value)
        elif name == "background":
            image = image + value
        elif name == "blur":
            kernel = max(1, int(round(value)))
            if kernel % 2 == 0:
                kernel += 1
            image = F.avg_pool2d(image.unsqueeze(0), kernel, stride=1, padding=kernel // 2)[0]
        elif name == "shift":
            pixels = int(round(value))
            dy = int(torch.randint(-pixels, pixels + 1, (), generator=generator).item()) if pixels else 0
            dx = int(torch.randint(-pixels, pixels + 1, (), generator=generator).item()) if pixels else 0
            image = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
        elif name == "saturation":
            threshold = max(0.05, 1.0 - value)
            image = image.clamp_max(threshold) / threshold
        elif name == "dead_pixels":
            mask = torch.rand(image.shape, generator=generator) < value
            image = image.masked_fill(mask, 0)
        else:
            raise ValueError(f"未知 corruption：{name}")
        sample["speckle"] = image.clamp(0, 1)
        sample["corruption"] = name
        sample["severity"] = value
        return sample
