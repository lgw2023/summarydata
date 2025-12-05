from __future__ import annotations

import json
from typing import Any, Dict, Tuple


def _extract_json_substring(text: str) -> str:
    """
    尝试从 LLM 输出中截取第一个完整的 JSON 子串。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text")
    return text[start : end + 1]


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    尝试对 judge 的原始文本输出进行 JSON 解析。
    - 先直接 json.loads
    - 失败时，尝试截取首尾大括号内的子串再解析
    - 若仍失败，则抛出异常，由上层决定 fallback 策略
    """
    try:
        return json.loads(text)
    except Exception:
        json_str = _extract_json_substring(text)
        return json.loads(json_str)


def compute_overall_score(parsed: Dict[str, Any], default_max_score: float = 5.0) -> Tuple[float, float]:
    """
    根据解析后的 JSON 计算整体得分。

    约定：
    - JSON 结构中包含 "checks": [{"score": ...}, ...]
    - 整体得分为所有 score 的平均值；
    - 最大分默认为 default_max_score。
    """
    checks = parsed.get("checks") or []
    scores = []
    for item in checks:
        try:
            s = float(item.get("score"))
        except Exception:
            continue
        scores.append(s)

    if not scores:
        return 0.0, default_max_score

    avg = sum(scores) / len(scores)
    return float(avg), default_max_score


