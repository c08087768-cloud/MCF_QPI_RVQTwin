from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

import pandas as pd

from .pairing import PairRecord


def validate_manual_duplicate_review(
    review: pd.DataFrame,
    *,
    minimum_per_stratum: int = 100,
) -> None:
    """严格协议门禁：候选重复与阈值附近非重复都必须完成人工核查。"""
    required = {"decision", "review_stratum"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"人工核查门禁缺少列：{sorted(missing)}")
    counts = review.groupby("review_stratum").size().to_dict()
    for stratum in ("candidate", "threshold_negative"):
        if int(counts.get(stratum, 0)) < minimum_per_stratum:
            raise ValueError(
                f"人工核查门禁未通过：{stratum} 至少需要 {minimum_per_stratum} 对"
            )
    if not set(review["decision"]).issubset({"duplicate", "distinct"}):
        raise ValueError("人工核查 decision 只能是 duplicate 或 distinct")


def build_strict_group_ids(
    frame: pd.DataFrame,
    review: pd.DataFrame,
) -> dict[str, str]:
    """按精确哈希和经人工确认的保守近重复边构造连通分量。"""
    required = {"sample_id", "phase_sha256"}
    if missing := required - set(frame.columns):
        raise ValueError(f"严格分组缺少列：{sorted(missing)}")
    ids = [str(value) for value in frame["sample_id"]]
    parent = {sample_id: sample_id for sample_id in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for _, group in frame[frame["phase_sha256"] != ""].groupby("phase_sha256"):
        members = [str(value) for value in group["sample_id"]]
        for member in members[1:]:
            union(members[0], member)

    required_review = {
        "sample_a", "sample_b", "phash_hamming", "ssim", "normalized_mae", "decision"
    }
    if missing := required_review - set(review.columns):
        raise ValueError(f"近重复核查缺少列：{sorted(missing)}")
    for row in review.itertuples(index=False):
        if str(row.decision) != "duplicate":
            continue
        qualifies = (
            int(row.phash_hamming) <= 4
            and float(row.ssim) >= 0.995
            and float(row.normalized_mae) <= 0.005
        )
        if not qualifies:
            raise ValueError("标记为 duplicate 的样本对不满足预注册近重复阈值")
        left, right = str(row.sample_a), str(row.sample_b)
        if left not in parent or right not in parent:
            raise ValueError(f"人工核查引用未知 sample_id：{left}, {right}")
        union(left, right)
    return {sample_id: find(sample_id) for sample_id in ids}


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
