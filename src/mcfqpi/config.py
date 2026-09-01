from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并保证顶层为字典。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"配置文件顶层必须是字典：{path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _parse_scalar(value: str) -> Any:
    """使用 YAML 语法解析命令行标量，例如 true、3、[digits]。"""
    parsed = yaml.safe_load(value)
    return parsed


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """应用 key.subkey=value 形式的命令行覆盖。"""
    if not overrides:
        return config
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项必须是 key=value：{item}")
        key, raw_value = item.split("=", 1)
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"无效覆盖项：{item}")
        cursor: dict[str, Any] = config
        for part in parts[:-1]:
            child = cursor.get(part)
            if child is None:
                child = {}
                cursor[part] = child
            if not isinstance(child, dict):
                raise TypeError(f"无法覆盖 {key}：{part} 不是字典")
            cursor = child
        cursor[parts[-1]] = _parse_scalar(raw_value)
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    return apply_overrides(load_yaml(path), overrides)
