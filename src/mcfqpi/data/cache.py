from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import MCFManifestDataset


def cache_manifest_to_hdf5(
    manifest: str | Path,
    output: str | Path,
    *,
    size: int = 128,
    phase_encoding: str = "auto",
    resize_mode: str = "fit_pad",
    speckle_normalization: str = "log_mean",
    dynamic_range: float = 20.0,
    compression: str | None = "lzf",
    overwrite: bool = False,
) -> Path:
    """把图像目录转换为随机访问更快的 HDF5。

    缓存中保存的是已经完成 resize/normalization 的 float32 张量；训练阶段的随机
    传感器增强仍在 Dataset 中在线执行。这样可以避免每个 epoch 重复解码数万张图。
    """
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出已存在：{output}；如需覆盖请使用 --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = MCFManifestDataset(
        manifest,
        split=None,
        domains=None,
        size=size,
        phase_encoding=phase_encoding,  # type: ignore[arg-type]
        resize_mode=resize_mode,  # type: ignore[arg-type]
        speckle_normalization=speckle_normalization,
        dynamic_range=dynamic_range,
        augment=None,
    )
    string_dtype = h5py.string_dtype("utf-8")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w", libver="latest") as file:
        count = len(dataset)
        image_shape = (count, 1, size, size)
        chunk = (min(64, count), 1, size, size)
        speckle_ds = file.create_dataset(
            "speckle", shape=image_shape, dtype="float32", chunks=chunk, compression=compression
        )
        phase_ds = file.create_dataset(
            "phase", shape=image_shape, dtype="float32", chunks=chunk, compression=compression
        )
        mask_ds = file.create_dataset(
            "valid_mask", shape=image_shape, dtype="float32", chunks=chunk, compression=compression
        )
        metadata = {
            key: file.create_dataset(key, shape=(count,), dtype=string_dtype)
            for key in ("sample_id", "domain", "split", "class_id")
        }
        for index in tqdm(range(count), desc="缓存 MCF-QPI", dynamic_ncols=True):
            sample = dataset[index]
            speckle_ds[index] = sample["speckle"].numpy()
            phase_ds[index] = sample["phase"].numpy()
            mask_ds[index] = sample["valid_mask"].numpy()
            for key, target in metadata.items():
                target[index] = str(sample[key])
        file.attrs["manifest"] = str(Path(manifest).resolve())
        file.attrs["size"] = int(size)
        file.attrs["phase_range_rad"] = "[0, pi]"
        file.attrs["phase_encoding_requested"] = phase_encoding
        file.attrs["resize_mode"] = resize_mode
        file.attrs["speckle_normalization"] = speckle_normalization
        file.attrs["dynamic_range"] = float(dynamic_range)
        file.attrs["schema_version"] = "1.0"
        file.flush()
    temporary.replace(output)
    return output


def inspect_hdf5(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    report: dict[str, Any] = {"path": str(path.resolve()), "datasets": {}, "attributes": {}}
    with h5py.File(path, "r") as file:
        for name, dataset in file.items():
            report["datasets"][name] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
        report["attributes"] = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in file.attrs.items()
        }
        if "split" in file:
            values = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in file["split"]]
            report["split_counts"] = pd.Series(values).value_counts().to_dict()
        if "domain" in file:
            values = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in file["domain"]]
            report["domain_counts"] = pd.Series(values).value_counts().to_dict()
    return report
