from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .utils import sha256_file, sha256_text


def _fingerprints(
    config_path: str | Path,
    checkpoint_path: str | Path,
    data_paths: Iterable[str | Path],
    git_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": "mcfqpi-freeze-1.0",
        "git_sha": str(git_sha),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "data_sha256": {
            str(Path(path).resolve()): sha256_file(path) for path in sorted(map(Path, data_paths))
        },
    }


def create_freeze_record(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    data_paths: Iterable[str | Path],
    git_sha: str,
) -> dict[str, Any]:
    payload = _fingerprints(config_path, checkpoint_path, data_paths, git_sha)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**payload, "freeze_id": sha256_text(canonical)}


def verify_freeze_record(
    record: dict[str, Any],
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    data_paths: Iterable[str | Path],
) -> None:
    actual = _fingerprints(
        config_path, checkpoint_path, data_paths, str(record.get("git_sha", ""))
    )
    canonical = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual_id = sha256_text(canonical)
    if actual_id != record.get("freeze_id"):
        raise ValueError("freeze 验证失败：配置、权重或数据指纹已改变")
