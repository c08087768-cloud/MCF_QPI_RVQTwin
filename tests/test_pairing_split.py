from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mcfqpi.data.pairing import discover_pairs
from mcfqpi.data.split import assign_group_splits


def _save(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), value, dtype=np.uint8)).save(path)


def test_discover_pairs_by_index(tmp_path: Path) -> None:
    for index in range(3):
        _save(tmp_path / "digits" / "train" / "speckles" / f"speckle_{index}.png", index)
        _save(tmp_path / "digits" / "train" / "phases" / f"phase_{index}.png", index)
    records, diagnostics = discover_pairs(tmp_path, compute_hashes=True)
    assert len(records) == 3
    assert diagnostics["paired_total"] == 3
    assert all(record.domain == "digits" for record in records)
    assert all(record.split == "train" for record in records)


def test_group_split_is_deterministic_and_leak_free(tmp_path: Path) -> None:
    for index in range(6):
        _save(tmp_path / "fashion" / "speckles" / f"speckle_{index}.png", index)
        _save(tmp_path / "fashion" / "phases" / f"phase_{index}.png", index)
    records, _ = discover_pairs(tmp_path, compute_hashes=True)
    first = assign_group_splits(records, preserve_known=False, seed=123)
    second = assign_group_splits(records, preserve_known=False, seed=123)
    assert [record.split for record in first] == [record.split for record in second]
    by_group: dict[str, set[str]] = {}
    for record in first:
        by_group.setdefault(record.group_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_group.values())
