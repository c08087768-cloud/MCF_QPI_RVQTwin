from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .pairing import PairRecord


def stable_fraction(text: str, seed: int = 42) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_group_splits(
    records: Iterable[PairRecord],
    *,
    train_ratio: float = 0.88,
    val_ratio: float = 0.10,
    seed: int = 42,
    preserve_known: bool = True,
) -> list[PairRecord]:
    """按 domain 和 group_id 做确定性划分，避免重复相位跨集合泄漏。

    剩余比例自动作为 test。若 group_id 缺失，则退化到 sample_id。
    """
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio 和 val_ratio 设置不合法")
    result: list[PairRecord] = []
    group_to_split: dict[tuple[str, str], str] = {}
    for record in records:
        if preserve_known and record.split in {"train", "val", "test"}:
            result.append(record)
            continue
        group = record.group_id or record.phase_sha256 or record.phase_phash or record.sample_id
        key = (record.domain, group)
        split = group_to_split.get(key)
        if split is None:
            value = stable_fraction(f"{record.domain}:{group}", seed)
            if value < train_ratio:
                split = "train"
            elif value < train_ratio + val_ratio:
                split = "val"
            else:
                split = "test"
            group_to_split[key] = split
        result.append(replace(record, split=split))
    return result


def split_counts(records: Iterable[PairRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        counts[record.domain][record.split] += 1
    return {domain: dict(splits) for domain, splits in counts.items()}
