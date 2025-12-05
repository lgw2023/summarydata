from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Any, List, Dict

try:
    # 配置文件和部分中间文件统一使用 YAML
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - 在缺少依赖的环境给出明确提示
    yaml = None


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """
    加载 YAML 配置文件。

    说明：
    - 项目中配置文件（如 configs/default.yaml）默认都是 YAML 格式；
    - 当 PyYAML 未安装时，不再尝试用 json 解析，而是直接给出清晰错误提示。
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML 未安装，无法解析 YAML 配置文件。"
            "请先安装依赖：pip install pyyaml，或通过 requirements.txt 统一安装。"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        # 统一返回 dict，避免 None 导致后续访问出错
        return data or {}


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """
    简单的 JSONL 读取工具，返回 dict 列表。
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

