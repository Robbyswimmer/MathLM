"""Lightweight YAML/JSON loader with include support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


from .config import parse_config


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", ""}:
        return None
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        if value.startswith("0") and len(value) > 1 and value[1].isdigit():
            raise ValueError
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if (value.startswith("\"") and value.endswith("\"")) or (
                value.startswith("'") and value.endswith("'")
            ):
                return value[1:-1]
            return value


def _minimal_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_part, _, value_part = line.lstrip().partition(":")
        key = key_part.strip()
        value = value_part.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value:
            new_dict: Dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _parse_scalar(value)
    return root


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if yaml is not None:  # type: ignore
        data = yaml.safe_load(text) or {}
    else:
        data = _minimal_yaml(text)
    include = data.pop("include", None)
    if include:
        include_path = path.parent / include
        base = load_config(include_path)
        data = _merge(base, data)
    return data


def load_experiment_config(path: Path) -> Any:
    config_dict = load_config(path)
    return parse_config(config_dict)
