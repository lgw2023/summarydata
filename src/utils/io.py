from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Any

try:  # Optional dependency: PyYAML
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback when PyYAML is unavailable
    yaml = None


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if yaml:
            return yaml.safe_load(f)
        return json.load(f)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
