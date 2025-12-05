from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Iterable, Dict, Any, List


def compute_model_stats(judge_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    按 model_type / model_name 维度统计基础指标：
    - 样本数量
    - 平均分、标准差
    """
    buckets: Dict[str, List[float]] = defaultdict(list)

    for row in judge_rows:
        key = f'{row.get("model_type","")}/{row.get("model_name","")}'
        score = float(row.get("aggregate_score", 0.0))
        buckets[key].append(score)

    stats: Dict[str, Dict[str, Any]] = {}
    for key, scores in buckets.items():
        if not scores:
            continue
        stats[key] = {
            "count": len(scores),
            "mean": mean(scores),
            "std": pstdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }
    return stats


