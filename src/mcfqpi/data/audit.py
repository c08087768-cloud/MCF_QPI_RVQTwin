from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def audit_manifest(path: str | Path, *, inspect_dimensions: bool = True) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"sample_id", "domain", "split", "speckle_path", "phase_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest 缺少列：{missing}")

    report: dict[str, Any] = {
        "rows": len(frame),
        "counts_by_domain_split": {
            f"{domain}/{split}": int(count)
            for (domain, split), count in frame.groupby(["domain", "split"]).size().items()
        },
        "missing_files": [],
        "duplicate_sample_ids": frame[frame.sample_id.duplicated(keep=False)].sample_id.tolist(),
        "cross_split_exact_phase_duplicates": [],
        "cross_split_perceptual_phase_duplicates": [],
        "dimension_counts": defaultdict(int),
        "pair_dimension_mismatches": [],
    }
    for column, output_key in [
        ("phase_sha256", "cross_split_exact_phase_duplicates"),
        ("phase_phash", "cross_split_perceptual_phase_duplicates"),
    ]:
        if column not in frame.columns:
            continue
        nonempty = frame[frame[column] != ""]
        for value, group in nonempty.groupby(column):
            splits = sorted(set(group.split))
            if len(splits) > 1:
                report[output_key].append({
                    "hash": value,
                    "splits": splits,
                    "sample_ids": group.sample_id.tolist()[:20],
                })

    if inspect_dimensions:
        for row in frame.itertuples(index=False):
            speckle = Path(row.speckle_path)
            phase = Path(row.phase_path)
            if not speckle.exists() or not phase.exists():
                report["missing_files"].append({"sample_id": row.sample_id, "speckle": str(speckle), "phase": str(phase)})
                continue
            with Image.open(speckle) as image:
                s_size = image.size
            with Image.open(phase) as image:
                p_size = image.size
            report["dimension_counts"][f"speckle:{s_size[0]}x{s_size[1]}"] += 1
            report["dimension_counts"][f"phase:{p_size[0]}x{p_size[1]}"] += 1
            if s_size != p_size:
                report["pair_dimension_mismatches"].append({"sample_id": row.sample_id, "speckle": s_size, "phase": p_size})
    report["dimension_counts"] = dict(report["dimension_counts"])
    report["passed_core_checks"] = not (
        report["missing_files"]
        or report["duplicate_sample_ids"]
        or report["cross_split_exact_phase_duplicates"]
    )
    return report
