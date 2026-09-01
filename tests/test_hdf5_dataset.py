from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mcfqpi.data.cache import cache_manifest_to_hdf5
from mcfqpi.data.dataset import MCFH5Dataset


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
