from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPECKLE_WORDS = {
    "speckle", "speckles", "input", "inputs", "distal", "farfield", "far_field",
    "intensity", "measurement", "measurements", "cam", "camera", "datax", "xdata",
    "exp", "experiment", "experimental",
}
PHASE_WORDS = {
    "phase", "phases", "label", "labels", "target", "targets", "proximal", "groundtruth",
    "ground_truth", "gt", "datay", "ydata", "original",
}
GENERIC_WORDS = {
    "image", "images", "img", "train", "training", "val", "valid", "validation", "test",
    "dataset", "data", "mcf", "qpi", "fashion", "mnist", "digit", "digits",
}


def _de_camel(text: str) -> str:
    """把 camelCase 目录名转成 snake_case，便于按词匹配（trainInput → train_input）。"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text).lower()


@dataclass
class PairRecord:
    sample_id: str
    domain: str
    split: str
    speckle_path: str
    phase_path: str
    pair_method: str
    class_id: str = ""
    group_id: str = ""
    speckle_sha256: str = ""
    phase_sha256: str = ""
    phase_phash: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def natural_key(path: Path | str) -> list[object]:
    text = str(path).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def infer_domain(path: Path) -> str:
    text = "/".join(part.lower() for part in path.parts)
    if "fashion" in text or "clothes" in text:
        return "fashion"
    if "digit" in text or "mnist" in text or "number" in text:
        return "digits"
    return "unknown"


def infer_split(path: Path) -> str:
    tokens = {token for part in path.parts for token in re.split(r"[^a-z0-9]+", _de_camel(part)) if token}
    if "test" in tokens or "testing" in tokens:
        return "test"
    if tokens & {"val", "valid", "validation"}:
        return "val"
    if "train" in tokens or "training" in tokens:
        return "train"
    # MCF-QPI 的主目录（exp_128、label_128、labelInput）没有 train 标记，但确实是训练集。
    return "train"


def infer_class_id(path: Path) -> str:
    """尽力从目录/文件名推断类别。失败时返回空字符串，不把猜测当真值。"""
    candidates = list(path.parts[-3:]) + [path.stem]
    for candidate in candidates:
        match = re.search(r"(?:class|label|digit|category)[-_ ]?(\d+)", candidate.lower())
        if match:
            return match.group(1)
    parent = path.parent.name
    if parent.isdigit() and 0 <= int(parent) <= 99:
        return parent
    return ""


def role_score(path: Path) -> tuple[int, int]:
    text = _de_camel("/".join(path.parts))
    normalized = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = set(normalized.split())
    compact = normalized.replace(" ", "")
    speckle = sum(word in tokens or word.replace("_", "") in compact for word in SPECKLE_WORDS)
    phase = sum(word in tokens or word.replace("_", "") in compact for word in PHASE_WORDS)
    return speckle, phase


def classify_role(path: Path) -> str | None:
    speckle, phase = role_score(path)
    if speckle > phase:
        return "speckle"
    # tie 时判为 phase：实际数据中 labelInput 同时含 label/input，但它是相位标签目录。
    if phase > 0:
        return "phase"
    return None


def normalized_pair_key(path: Path) -> str:
    words = re.split(r"[^a-z0-9]+", path.stem.lower())
    ignored = SPECKLE_WORDS | PHASE_WORDS | GENERIC_WORDS
    remaining = [word for word in words if word and word not in ignored]
    digits = re.findall(r"\d+", path.stem)
    if digits:
        # 数字索引通常是最可靠的配对标识；保留所有数字以避免不同类别索引冲突。
        return "_".join(digits)
    return "_".join(remaining) or path.stem.lower()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def image_perceptual_hash(path: Path, size: int = 16) -> str:
    """生成轻量感知哈希，用于把近似重复标签放到同一个 split。

    这不是密码学哈希，也不应替代精确 SHA-256；它只用于泄漏审计和分组。
    """
    with Image.open(path) as image:
        gray = image.convert("F").resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(gray, dtype=np.float32)
    array = array - array.mean()
    bits = (array >= 0).astype(np.uint8).reshape(-1)
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def _pair_one_group(
    speckles: list[Path],
    phases: list[Path],
    *,
    allow_order_pairing: bool,
) -> list[tuple[Path, Path, str]]:
    phase_map: dict[str, list[Path]] = defaultdict(list)
    for phase in phases:
        phase_map[normalized_pair_key(phase)].append(phase)
    for values in phase_map.values():
        values.sort(key=natural_key)

    result: list[tuple[Path, Path, str]] = []
    used: set[Path] = set()
    unmatched: list[Path] = []
    for speckle in sorted(speckles, key=natural_key):
        key = normalized_pair_key(speckle)
        candidates = [candidate for candidate in phase_map.get(key, []) if candidate not in used]
        if len(candidates) == 1:
            phase = candidates[0]
            used.add(phase)
            result.append((speckle, phase, "normalized_key"))
        else:
            unmatched.append(speckle)

    remaining_phases = [phase for phase in sorted(phases, key=natural_key) if phase not in used]
    if unmatched and allow_order_pairing and len(unmatched) == len(remaining_phases):
        result.extend((s, p, "natural_order_explicit") for s, p in zip(sorted(unmatched, key=natural_key), remaining_phases))
    return result


def discover_pairs(
    root: str | Path,
    *,
    allow_order_pairing: bool = False,
    compute_hashes: bool = False,
) -> tuple[list[PairRecord], dict[str, object]]:
    root = Path(root)
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    grouped: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(lambda: {"speckle": [], "phase": [], "unknown": []})
    for path in files:
        relative = path.relative_to(root)
        role = classify_role(relative) or "unknown"
        grouped[(infer_domain(relative), infer_split(relative))][role].append(path)

    records: list[PairRecord] = []
    diagnostics: dict[str, object] = {"root": str(root), "image_files": len(files), "groups": {}, "unclassified": []}
    counter = 0
    for (domain, split), roles in sorted(grouped.items()):
        pairs = _pair_one_group(roles["speckle"], roles["phase"], allow_order_pairing=allow_order_pairing)
        diagnostics["groups"][f"{domain}/{split}"] = {
            "speckles": len(roles["speckle"]),
            "phases": len(roles["phase"]),
            "unknown": len(roles["unknown"]),
            "paired": len(pairs),
        }
        diagnostics["unclassified"].extend(str(path.relative_to(root)) for path in roles["unknown"][:100])
        for speckle, phase, method in pairs:
            key = normalized_pair_key(speckle)
            sample_id = f"{domain}_{split}_{key}_{counter:06d}"
            record = PairRecord(
                sample_id=sample_id,
                domain=domain,
                split=split,
                speckle_path=str(speckle.resolve()),
                phase_path=str(phase.resolve()),
                pair_method=method,
                class_id=infer_class_id(phase),
            )
            if compute_hashes:
                record.speckle_sha256 = file_sha256(speckle)
                record.phase_sha256 = file_sha256(phase)
                record.phase_phash = image_perceptual_hash(phase)
                record.group_id = record.phase_sha256 or record.phase_phash
            records.append(record)
            counter += 1
    diagnostics["paired_total"] = len(records)
    diagnostics["allow_order_pairing"] = allow_order_pairing
    return records, diagnostics
