from __future__ import annotations

"""
提示词加载工具。

当前需求：**不再从 `prompts/ground_prompt.txt` 与 `prompts/structure_prompt.txt` 等外部文件加载**，
统一使用 `score_prompt.py` 中定义的内置模板。
"""

from typing import Tuple

from score_prompt import GROUND_PROMPT_TPL, STRUCT_PROMPT_TPL


def load_prompts_from_config(cfg: dict) -> Tuple[str, str]:  # noqa: ARG001 - cfg 为兼容旧配置保留
    """
    根据 judge 配置加载提示词模板。

    现在会忽略以下配置字段：
    - `ground_prompt_path` / `struct_prompt_path`
    - `prompts_dir`

    无论是否配置这些字段，均直接返回 `score_prompt.py` 中的 `GROUND_PROMPT_TPL` 与 `STRUCT_PROMPT_TPL`。
    """
    _ = cfg  # 显式丢弃，避免未使用变量告警
    return GROUND_PROMPT_TPL, STRUCT_PROMPT_TPL



