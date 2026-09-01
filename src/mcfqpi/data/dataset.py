from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import (
    PhaseEncoding,
    ResizeMode,
    SpeckleAugment,
    decode_phase,
    normalize_speckle,
    read_grayscale,
    resize_phase,
    resize_speckle,
)


class MCFManifestDataset(Dataset[dict[str, Any]]):
    """从 CSV manifest 读取真实 MCF-QPI 图像对。"""

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str | None = None,
        domains: Iterable[str] | None = None,
        size: int = 128,
        phase_encoding: PhaseEncoding = "auto",
        resize_mode: ResizeMode = "fit_pad",
        speckle_normalization: str = "log_mean",
        dynamic_range: float = 20.0,
        augment: SpeckleAugment | None = None,
        limit: int = 0,
    ) -> None:
        frame = pd.read_csv(manifest, dtype=str).fillna("")
        if split and split != "all":
            frame = frame[frame["split"] == split]
        if domains:
            domain_set = set(domains)
            frame = frame[frame["domain"].isin(domain_set)]
        frame = frame.reset_index(drop=True)
        if limit > 0:
            frame = frame.iloc[:limit].copy()
        if frame.empty:
            raise ValueError(f"数据集为空：manifest={manifest}, split={split}, domains={domains}")
        self.frame = frame
        self.size = int(size)
        self.phase_encoding = phase_encoding
        self.resize_mode = resize_mode
        self.speckle_normalization = speckle_normalization
        self.dynamic_range = float(dynamic_range)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        speckle_array = read_grayscale(row.speckle_path)
        phase_array = read_grayscale(row.phase_path)
        speckle, speckle_mask = resize_speckle(speckle_array, self.size, self.resize_mode)
        speckle = normalize_speckle(
            speckle,
            method=self.speckle_normalization,  # type: ignore[arg-type]
            dynamic_range=self.dynamic_range,
        )
        if self.augment is not None:
            speckle = self.augment(speckle)
        phase_rad = decode_phase(phase_array, self.phase_encoding)
        phase_rad, phase_mask = resize_phase(phase_rad, self.size, self.resize_mode)
        phase = phase_rad / np.pi
        valid_mask = speckle_mask * phase_mask
        return {
            "speckle": speckle.contiguous(),
            "phase": phase.contiguous(),
            "phase_rad": phase_rad.contiguous(),
            "valid_mask": valid_mask.contiguous(),
            "sample_id": str(row.sample_id),
            "domain": str(row.domain),
            "split": str(row.split),
            "class_id": str(row.get("class_id", "")),
            "index": int(index),
        }


class MCFH5Dataset(Dataset[dict[str, Any]]):
    """读取由 scripts/cache_to_hdf5.py 生成的缓存。每个 worker 延迟打开 HDF5。"""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str | None = None,
        domains: Iterable[str] | None = None,
        augment: SpeckleAugment | None = None,
        limit: int = 0,
    ) -> None:
        self.path = str(Path(path))
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as file:
            def decode_array(name: str) -> np.ndarray:
                values = np.asarray(file[name])
                return np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])

            splits = decode_array("split")
            domain_values = decode_array("domain")
            indices = np.arange(len(splits))
            if split and split != "all":
                indices = indices[splits == split]
            if domains:
                indices = indices[np.isin(domain_values[indices], list(domains))]
            if limit > 0:
                indices = indices[:limit]
            self.indices = indices.astype(np.int64)
        if self.indices.size == 0:
            raise ValueError(f"HDF5 数据集为空：{path}, split={split}, domains={domains}")
        self.augment = augment

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        file = self._ensure_open()
        row = int(self.indices[index])
        speckle = torch.from_numpy(np.asarray(file["speckle"][row], dtype=np.float32))
        if self.augment is not None:
            speckle = self.augment(speckle)
        phase = torch.from_numpy(np.asarray(file["phase"][row], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(file["valid_mask"][row], dtype=np.float32))
        decode = lambda key: file[key][row].decode("utf-8") if isinstance(file[key][row], bytes) else str(file[key][row])
        return {
            "speckle": speckle,
            "phase": phase,
            "phase_rad": phase * np.pi,
            "valid_mask": mask,
            "sample_id": decode("sample_id"),
            "domain": decode("domain"),
            "split": decode("split"),
            "class_id": decode("class_id"),
            "index": row,
        }

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()


def build_augmentation(config: dict[str, Any] | None) -> SpeckleAugment | None:
    if not config or not config.get("enabled", False):
        return None
    kwargs = {key: value for key, value in config.items() if key != "enabled"}
    return SpeckleAugment(**kwargs)


def build_dataset(config: dict[str, Any], split: str, *, training: bool) -> Dataset[dict[str, Any]]:
    augment = build_augmentation(config.get("augmentation")) if training else None
    common = {
        "split": split,
        "domains": config.get("domains"),
        "augment": augment,
        "limit": int(config.get("limit", 0)),
    }
    if config.get("hdf5"):
        return MCFH5Dataset(config["hdf5"], **common)
    return MCFManifestDataset(
        config["manifest"],
        size=int(config.get("size", 128)),
        phase_encoding=config.get("phase_encoding", "auto"),
        resize_mode=config.get("resize_mode", "fit_pad"),
        speckle_normalization=config.get("speckle_normalization", "log_mean"),
        dynamic_range=float(config.get("dynamic_range", 20.0)),
        **common,
    )
