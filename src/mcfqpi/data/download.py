from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class PublicFile:
    key: str
    filename: str
    url: str
    doi: str
    approximate_size_mb: float
    license_name: str = "CC BY 4.0"


# 使用论文页面所列的两个 Optica/Figshare 条目。
# 这些 ndownloader 地址会自动重定向到短时有效的对象存储链接，故下载器必须跟随重定向。
MCF_DATASETS: dict[str, PublicFile] = {
    "fashion": PublicFile(
        key="fashion",
        filename="fashion_MCF_QPI_dataset.tar",
        url="https://opticapublishing.figshare.com/ndownloader/files/43882668",
        doi="10.6084/m9.figshare.24932583",
        approximate_size_mb=310.16,
    ),
    "digits": PublicFile(
        key="digits",
        filename="MCF_speckle_digits.tar",
        url="https://opticapublishing.figshare.com/ndownloader/files/43882719",
        doi="10.6084/m9.figshare.24932604",
        approximate_size_mb=395.45,
    ),
}


def download_with_resume(
    source: PublicFile,
    destination: str | Path,
    *,
    chunk_size: int = 4 * 1024 * 1024,
    retries: int = 5,
    timeout: tuple[int, int] = (20, 120),
) -> Path:
    """带断点续传地下载一个公开文件。

    Figshare 的下载地址会重定向。若服务器支持 HTTP Range，则从已有临时文件末尾继续；
    若服务器忽略 Range 并返回 200，则安全地从头覆盖临时文件。
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        existing = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
        try:
            with requests.get(
                source.url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                resumed = existing > 0 and response.status_code == 206
                if existing > 0 and not resumed:
                    # 服务器未接受 Range。避免把完整文件追加到残缺文件后面。
                    existing = 0
                mode = "ab" if resumed else "wb"
                content_length = int(response.headers.get("Content-Length", "0") or 0)
                total = existing + content_length if content_length else None
                with temporary.open(mode) as file, tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    desc=source.filename,
                    dynamic_ncols=True,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        file.write(chunk)
                        progress.update(len(chunk))
            temporary.replace(destination)
            return destination
        except (requests.RequestException, OSError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"下载失败：{source.url}\n最后错误：{exc}") from exc
            wait = min(2**attempt, 30)
            print(f"下载中断，第 {attempt}/{retries} 次尝试失败；{wait}s 后重试：{exc}")
            time.sleep(wait)
    raise AssertionError("不可达")


def selected_sources(keys: Iterable[str]) -> list[PublicFile]:
    result: list[PublicFile] = []
    for key in keys:
        if key == "all":
            return list(MCF_DATASETS.values())
        if key not in MCF_DATASETS:
            raise KeyError(f"未知数据集 {key!r}；可选：{sorted(MCF_DATASETS)}")
        result.append(MCF_DATASETS[key])
    return result
