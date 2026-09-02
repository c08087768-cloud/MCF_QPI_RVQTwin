from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        # 图像尺寸固定时 benchmark 通常更快；论文复现实验可开启 deterministic。
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def select_device(name: str = "auto") -> torch.device:
    """选择训练/评估设备。

    选卡优先级（从高到低）：
    1. 环境变量 MCFQPI_DEVICE（若已设置）—— 方便不动 yaml 选卡，例如：
         MCFQPI_DEVICE=cuda:1 bash scripts/run_research_pipeline.sh
    2. 显式传入的 name（来自 yaml 的 device: 字段）
    3. "auto"：有 GPU 用 GPU，否则 CPU

    支持的取值："cpu" / "cuda" / "cuda:0" / "cuda:1" / "cuda:N"。
    注意这里的 N 是经 CUDA_VISIBLE_DEVICES 重映射后的本地索引，
    例如设了 CUDA_VISIBLE_DEVICES=1 后，"cuda:0" 就对应物理 1 号卡。
    """
    env = os.environ.get("MCFQPI_DEVICE", "").strip()
    if env:
        name = env

    if name in ("", "auto"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "配置要求 CUDA，但当前 PyTorch 未检测到可用 GPU。"
                "请安装与显卡驱动匹配的 CUDA 版 PyTorch，或改用 device: auto 在 CPU 上运行。"
            )
        gpu_count = torch.cuda.device_count()
        if device.index is not None and device.index >= gpu_count:
            raise RuntimeError(
                f"指定的 GPU 索引 {device.index} 超出范围：本机只检测到 {gpu_count} 张可用 GPU。"
                "提示：若有多块卡但看不到全部，检查 CUDA_VISIBLE_DEVICES 是否限制了可见范围。"
            )
    return device


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    """先写临时文件再原子替换，减少中断导致的损坏 checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def save_inference_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    model_config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """保存无需 sidecar、可用 weights_only 安全读取的纯推理 checkpoint。"""
    atomic_torch_save(
        {
            "schema_version": "mcfqpi-inference-1.0",
            "model": model.state_dict(),
            "model_config": model_config,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        return {"model": checkpoint}
    return checkpoint


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_reproducibility_metadata(full_config: dict[str, Any]) -> dict[str, Any]:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    data = full_config.get("data", {})
    fingerprints = {}
    for key in ("manifest", "hdf5"):
        raw_path = data.get(key)
        if raw_path and Path(raw_path).is_file():
            fingerprints[key] = {"path": str(Path(raw_path).resolve()), "sha256": sha256_file(raw_path)}
    return {
        "git_sha": git_sha,
        "data_fingerprints": fingerprints,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count(),
        },
    }


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


@contextlib.contextmanager
def timer() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def worker_seed_init(worker_id: int) -> None:
    """为 DataLoader worker 设置不同、可复现的随机数状态。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
