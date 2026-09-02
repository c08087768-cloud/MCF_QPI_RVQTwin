from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mcfqpi.data.cache import cache_manifest_to_hdf5
from mcfqpi.data.dataset import MCFH5Dataset
from mcfqpi.utils import sha256_file


def test_hdf5_roundtrip(tmp_path: Path) -> None:
    rows = []
    for index, split in enumerate(["train", "val", "test"]):
        speckle = tmp_path / f"speckle_{index}.png"
        phase = tmp_path / f"phase_{index}.png"
        Image.fromarray(np.full((16, 16), 100 + index, dtype=np.uint8)).save(speckle)
        Image.fromarray(np.full((16, 16), 50 + index, dtype=np.uint8)).save(phase)
        rows.append(
            {
                "sample_id": f"x{index}",
                "domain": "digits",
                "split": split,
                "speckle_path": str(speckle),
                "phase_path": str(phase),
                "class_id": "",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    output = tmp_path / "cache.h5"
    cache_manifest_to_hdf5(manifest, output, size=16, phase_encoding="uint8")
    dataset = MCFH5Dataset(output, split="test")
    sample = dataset[0]
    assert sample["sample_id"] == "x2"
    assert sample["speckle"].shape == (1, 16, 16)

    with h5py.File(output, "r") as file:
        assert file.attrs["schema_version"] == "2.0"
        assert file.attrs["manifest_sha256"] == sha256_file(manifest)
        assert file.attrs["phase_encoding"] == "uint8"


def test_hdf5_metadata_mismatch_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    speckle = tmp_path / "speckle.png"
    phase = tmp_path / "phase.png"
    Image.fromarray(np.full((8, 8), 100, dtype=np.uint8)).save(speckle)
    Image.fromarray(np.full((8, 8), 50, dtype=np.uint8)).save(phase)
    pd.DataFrame([{
        "sample_id": "x", "domain": "digits", "split": "train",
        "speckle_path": str(speckle), "phase_path": str(phase), "class_id": "",
    }]).to_csv(manifest, index=False)
    output = tmp_path / "cache.h5"
    cache_manifest_to_hdf5(manifest, output, size=8, phase_encoding="uint8")

    with pytest.raises(ValueError, match="HDF5 元数据不一致"):
        MCFH5Dataset(output, expected_metadata={"size": 16})


def test_hdf5_v2_rejects_auto_phase_encoding(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(columns=["sample_id", "domain", "split", "speckle_path", "phase_path"]).to_csv(
        manifest, index=False
    )
    with pytest.raises(ValueError, match="显式 phase_encoding"):
        cache_manifest_to_hdf5(manifest, tmp_path / "cache.h5", phase_encoding="auto")
