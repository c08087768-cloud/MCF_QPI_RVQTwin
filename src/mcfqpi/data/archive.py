from __future__ import annotations

import tarfile
from pathlib import Path


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar(archive: str | Path, destination: str | Path) -> Path:
    """安全解压 tar，拒绝绝对路径、目录穿越和指向目录外的链接。"""
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar.getmembers():
            member_path = destination / member.name
            if not _is_within_directory(destination, member_path):
                raise RuntimeError(f"拒绝危险 tar 条目：{member.name}")
            if member.issym() or member.islnk():
                link_target = member_path.parent / member.linkname
                if not _is_within_directory(destination, link_target):
                    raise RuntimeError(f"拒绝指向目录外的链接：{member.name} -> {member.linkname}")
        # filter='data' 在较新 Python 中进一步限制危险元数据；旧版本回退到前面的手工检查。
        try:
            tar.extractall(destination, filter="data")
        except TypeError:
            tar.extractall(destination)
    return destination


def list_tar(archive: str | Path, limit: int = 100) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar.getmembers()[:limit]:
            rows.append({"name": member.name, "size": member.size, "type": member.type.decode(errors="ignore")})
    return rows
