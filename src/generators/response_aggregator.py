from __future__ import annotations

from typing import Iterable, Dict, List

from src.data_loader.context_builder import build_context_text
from src.data_loader.excel_loader import Sample
from src.generators.base import Candidate


def build_generated_response_rows(
    samples: Iterable[Sample],
    candidates: Iterable[Candidate],
) -> List[Dict]:
    """
    将 Candidate 列表按 sample_id 聚合，生成写入
    `data/processed/generated_responses.jsonl` 所需的行结构。

    结构对齐 TASK.md 2.3：
    {
      "sample_id": "...",
      "context": "...",
      "question": "...",
      "candidates": [ {candidate...}, ... ]
    }
    """
    sample_lookup: Dict[str, Sample] = {s.sample_id: s for s in samples}

    grouped: Dict[str, List[Candidate]] = {}
    for c in candidates:
        grouped.setdefault(c.sample_id, []).append(c)

    rows: List[Dict] = []
    for sample_id, cand_list in grouped.items():
        sample = sample_lookup.get(sample_id)
        if not sample:
            # 理论上不应发生，忽略异常样本以避免中断整个 pipeline
            continue
        row = {
            "sample_id": sample_id,
            "context": build_context_text(sample),
            "question": sample.query,
            "candidates": [c.to_dict() for c in cand_list],
        }
        rows.append(row)
    return rows


