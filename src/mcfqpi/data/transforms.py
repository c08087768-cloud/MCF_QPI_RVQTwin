from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ResizeMode = Literal["stretch", "fit_pad", "center_crop"]
PhaseEncoding = Literal["auto", "uint8", "uint16", "normalized", "radian"]


def read_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("F"), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"无法读取二维灰度图：{path}，shape={array.shape}")
    return array


def _resize_tensor(image: torch.Tensor, size: int, mode: ResizeMode, interpolation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """返回缩放后的 [C,H,W] 图像和 [1,H,W] 有效掩膜。"""
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(f"期望 [C,H,W]，实际 {tuple(image.shape)}")
    channels, height, width = image.shape
    align_corners = False if interpolation in {"bilinear", "bicubic"} else None

    def interpolate(tensor: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
        kwargs = {"mode": interpolation, "size": target}
        if align_corners is not None:
            kwargs["align_corners"] = align_corners
        return F.interpolate(tensor.unsqueeze(0), **kwargs)[0]

    if mode == "stretch":
        return interpolate(image, (size, size)), torch.ones(1, size, size, dtype=image.dtype)

    if mode == "fit_pad":
        scale = min(size / height, size / width)
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))
        resized = interpolate(image, (new_height, new_width))
        output = torch.zeros(channels, size, size, dtype=image.dtype)
        mask = torch.zeros(1, size, size, dtype=image.dtype)
        top = (size - new_height) // 2
        left = (size - new_width) // 2
        output[:, top : top + new_height, left : left + new_width] = resized
        mask[:, top : top + new_height, left : left + new_width] = 1
        return output, mask

    if mode == "center_crop":
        scale = max(size / height, size / width)
        new_height = max(size, round(height * scale))
        new_width = max(size, round(width * scale))
        resized = interpolate(image, (new_height, new_width))
        top = (new_height - size) // 2
        left = (new_width - size) // 2
        return resized[:, top : top + size, left : left + size], torch.ones(1, size, size, dtype=image.dtype)
    raise ValueError(f"未知 resize mode：{mode}")


def decode_phase(array: np.ndarray, encoding: PhaseEncoding = "auto") -> torch.Tensor:
    """把公开数据的相位标签转换成弧度，默认目标范围为 [0, π]。

    论文说明标签由 MNIST/Fashion-MNIST 灰度图映射到 [0, π]。auto 只根据 dtype
    和数值范围推断编码；转换报告应保存推断结果，正式实验不要默默改变编码。
    """
    array = np.asarray(array)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("相位图没有有限数值")
    minimum, maximum = float(finite.min()), float(finite.max())
    if encoding == "auto":
        if np.issubdtype(array.dtype, np.uint8) or (minimum >= 0 and maximum > 1.5 and maximum <= 255.5):
            encoding = "uint8"
        elif np.issubdtype(array.dtype, np.uint16) or (minimum >= 0 and maximum > 255.5 and maximum <= 65535.5):
            encoding = "uint16"
        elif minimum >= -1e-6 and maximum <= 1.0 + 1e-6:
            encoding = "normalized"
        elif minimum >= -1e-6 and maximum <= np.pi + 1e-3:
            encoding = "radian"
        else:
            raise ValueError(
                f"无法自动判断相位编码：dtype={array.dtype}, min={minimum:.4g}, max={maximum:.4g}。"
                "请显式设置 --phase-encoding。"
            )
    tensor = torch.from_numpy(np.ascontiguousarray(array.astype(np.float32)))
    if encoding == "uint8":
        return tensor.clamp(0, 255) / 255.0 * np.pi
    if encoding == "uint16":
        return tensor.clamp(0, 65535) / 65535.0 * np.pi
    if encoding == "normalized":
        return tensor.clamp(0, 1) * np.pi
    if encoding == "radian":
        return tensor.clamp(0, float(np.pi))
    raise ValueError(f"未知相位编码：{encoding}")


def resize_phase(phase_rad: torch.Tensor, size: int, mode: ResizeMode) -> tuple[torch.Tensor, torch.Tensor]:
    """缩放 [0,π] 相位标签。

    当前公开标签不跨越 2π 包裹边界，故默认可直接双线性插值。仍以复单位向量方式
    实现，便于以后扩展到 [0,2π] 标签，避免跨界平均错误。
    """
    phase_complex = torch.stack([torch.sin(phase_rad), torch.cos(phase_rad)], dim=0)
    resized, mask = _resize_tensor(phase_complex, size, mode, "bilinear")
    norm = resized.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    resized = resized / norm
    recovered = torch.atan2(resized[0], resized[1])
    recovered = torch.remainder(recovered, 2 * np.pi).clamp(0, np.pi)
    return recovered.unsqueeze(0), mask


def normalize_speckle(
    intensity: torch.Tensor,
    *,
    method: Literal["log_mean", "robust_log", "linear_mean"] = "log_mean",
    dynamic_range: float = 20.0,
) -> torch.Tensor:
    """将相机强度变成网络输入 [1,H,W]。

    log_mean 先除以均值，再进行 log1p 压缩；它保留散斑相对结构并弱化曝光变化。
    robust_log 使用每幅图 0.1%/99.9% 分位数裁剪，适合存在坏点时使用。
    """
    if intensity.ndim == 2:
        intensity = intensity.unsqueeze(0)
    intensity = intensity.float().clamp_min(0)
    if method == "robust_log":
        flat = intensity.flatten()
        low = torch.quantile(flat, 0.001)
        high = torch.quantile(flat, 0.999).clamp_min(low + 1e-8)
        normalized = ((intensity - low) / (high - low)).clamp(0, 1)
        return torch.log1p(dynamic_range * normalized) / np.log1p(dynamic_range)
    normalized = intensity / intensity.mean().clamp_min(1e-8)
    if method == "linear_mean":
        return normalized.clamp(0, dynamic_range) / dynamic_range
    if method == "log_mean":
        return torch.log1p(normalized.clamp(0, dynamic_range)) / np.log1p(dynamic_range)
    raise ValueError(f"未知 speckle normalization：{method}")


def resize_speckle(array: np.ndarray, size: int, mode: ResizeMode) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.from_numpy(np.ascontiguousarray(array.astype(np.float32)))
    return _resize_tensor(tensor, size, mode, "bilinear")


@dataclass
class SpeckleAugment:
    """只对散斑测量域施加扰动，不对标签做虚假的几何增强。"""

    gain: float = 0.15
    background: float = 0.03
    gaussian_noise: float = 0.02
    poisson_peak: float = 0.0
    blur_probability: float = 0.2
    max_shift_pixels: int = 2
    saturation_probability: float = 0.1
    dead_pixel_probability: float = 0.0

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clone()
        if self.gain > 0:
            factor = 1.0 + (2 * torch.rand(()) - 1.0) * self.gain
            image = image * factor
        if self.background > 0:
            image = image + torch.rand(()) * self.background
        if self.max_shift_pixels > 0:
            dy = int(torch.randint(-self.max_shift_pixels, self.max_shift_pixels + 1, ()).item())
            dx = int(torch.randint(-self.max_shift_pixels, self.max_shift_pixels + 1, ()).item())
            # 相机/光纤端面轻微错位应表现为“平移后新区域为空”，而不是 torch.roll
            # 那种把右边像素绕回左边的周期边界。下面显式复制有效重叠区域。
            shifted = torch.zeros_like(image)
            height, width = image.shape[-2:]
            src_y0, src_y1 = max(0, -dy), min(height, height - dy)
            src_x0, src_x1 = max(0, -dx), min(width, width - dx)
            dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
            dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
            if src_y1 > src_y0 and src_x1 > src_x0:
                shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = image[..., src_y0:src_y1, src_x0:src_x1]
            image = shifted
        if self.blur_probability > 0 and torch.rand(()) < self.blur_probability:
            image = F.avg_pool2d(image.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
        if self.poisson_peak > 0:
            scaled = image.clamp_min(0) / image.max().clamp_min(1e-8) * self.poisson_peak
            image = torch.poisson(scaled) / self.poisson_peak
        if self.gaussian_noise > 0:
            image = image + torch.randn_like(image) * self.gaussian_noise
        if self.saturation_probability > 0 and torch.rand(()) < self.saturation_probability:
            threshold = 0.7 + 0.25 * torch.rand(())
            image = image.clamp_max(threshold) / threshold
        if self.dead_pixel_probability > 0:
            mask = torch.rand_like(image) < self.dead_pixel_probability
            image = image.masked_fill(mask, 0)
        return image.clamp(0, 1)
