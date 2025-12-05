from __future__ import annotations

from typing import Iterable, List, Dict, Any
from collections import defaultdict

from src.judges.base import JudgeScore


def aggregate_scores(judge_scores: Iterable[JudgeScore]) -> List[Dict[str, Any]]:
    """
    将 JudgeScore 列表聚合为便于排序与下游分析的结构。

    输出结构大致对齐 TASK.md 2.4 的约定：
    {
      "sample_id": "...",
      "candidate_id": "...",
      "model_type": "...",
      "model_name": "...",
      "scores": {
        "ground": {
          "score": 4.5,
          "max_score": 5,
          "raw_judge_output": "...",
          "raw_judge_prompt": "..."
        },
        "structure": {
          "score": 3.8,
          "max_score": 5,
          "raw_judge_output": "...",
          "raw_judge_prompt": "..."
        }
      },
      "aggregate_score": 4.15,
      "judge": "heuristic",
      "judge_meta": {...},
      "notes": "..."
    }
    """
    aggregated: List[Dict[str, Any]] = []
    for score in judge_scores:
        aggregate_score = (score.ground_score + score.structure_score) / 2
        aggregated.append(
            {
                "sample_id": score.sample_id,
                "candidate_id": score.candidate_id,
                # 兼容早期代码：保留 candidate 字段为别名
                "candidate": score.candidate_id,
                "model_type": score.model_type,
                "model_name": score.model_name,
                "scores": {
                    "ground": {
                        "score": score.ground_score,
                        "max_score": score.ground_max_score,
                        "raw_judge_output": score.ground_raw,
                        "raw_judge_prompt": score.ground_prompt,
                    },
                    "structure": {
                        "score": score.structure_score,
                        "max_score": score.structure_max_score,
                        "raw_judge_output": score.structure_raw,
                        "raw_judge_prompt": score.structure_prompt,
                    },
                },
                "ground_score": score.ground_score,
                "structure_score": score.structure_score,
                "aggregate_score": aggregate_score,
                "judge": score.judge,
                "judge_meta": {
                    "judge_name": score.judge,
                },
                "notes": score.notes,
            }
        )
    return aggregated


def group_scores_by_sample(
    aggregated_rows: Iterable[Dict[str, Any]],
    results_key: str = "results",
) -> List[Dict[str, Any]]:
    """
    将按「候选为一行」的打分结果，聚合为「每个 sample 一行」的层级结构。

    输出结构示例：
    {
      "sample_id": "...",
      "results": [
        {
          "candidate_id": "...",
          "candidate": "...",
          "model_type": "...",
          "model_name": "...",
          "scores": {...},
          "ground_score": ...,
          "structure_score": ...,
          "aggregate_score": ...,
          "judge": "...",
          "judge_meta": {...},
          "notes": "..."
        },
        ...
      ]
    }
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        sample_id = str(row.get("sample_id"))
        # 内层无需重复 sample_id，减小冗余
        inner = {k: v for k, v in row.items() if k != "sample_id"}
        grouped[sample_id].append(inner)

    return [
        {
            "sample_id": sample_id,
            results_key: rows,
        }
        for sample_id, rows in grouped.items()
    ]


def flatten_grouped_scores(
    grouped_rows: Iterable[Dict[str, Any]],
    results_key: str = "results",
) -> List[Dict[str, Any]]:
    """
    将按 sample 分组的层级结构「还原」为按候选一行的扁平结构。

    - 若传入的本身已经是旧版的扁平结构（没有 results_key），则原样返回；
    - 这样可以兼容历史生成的 judge_results.jsonl。
    """
    flat: List[Dict[str, Any]] = []
    for row in grouped_rows:
        # 旧格式：一行一个候选
        if results_key not in row:
            # 确保 sample_id 为字符串
            if "sample_id" in row:
                row = dict(row)
                row["sample_id"] = str(row["sample_id"])
            flat.append(row)
            continue

        sample_id = str(row.get("sample_id"))
        for inner in row.get(results_key, []) or []:
            flat_row = dict(inner)
            flat_row["sample_id"] = sample_id
            flat.append(flat_row)
    return flat


