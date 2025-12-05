from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict


@dataclass
class RankedCandidate:
    sample_id: str
    candidate_id: str
    aggregate_score: float
    rank: int


@dataclass
class Pair:
    sample_id: str
    positive: str
    negative: str


def rank_candidates(scores: Iterable[dict]) -> List[RankedCandidate]:
    """
    根据 aggregate_score 为每个 sample 内的候选进行排序。
    输入可以是 aggregate_scores 的单行 dict。
    """
    grouped: Dict[str, List[dict]] = {}
    for score in scores:
        grouped.setdefault(str(score["sample_id"]), []).append(score)

    ranked_rows: List[RankedCandidate] = []
    for sample_id, group in grouped.items():
        sorted_group = sorted(group, key=lambda row: row["aggregate_score"], reverse=True)
        for idx, row in enumerate(sorted_group):
            candidate_id = row.get("candidate_id") or row.get("candidate")
            ranked_rows.append(
                RankedCandidate(
                    sample_id=str(sample_id),
                    candidate_id=str(candidate_id),
                    aggregate_score=float(row["aggregate_score"]),
                    rank=idx + 1,
                )
            )
    return ranked_rows


def build_pairs(
    ranked: List[RankedCandidate],
    top_k: int = 1,
    bottom_k: int = 1,
    min_score_diff: float = 0.0,
) -> List[Pair]:
    """
    根据排序结果构建正负样本对。

    - top_k/bottom_k：控制每个样本取前 k 与后 k 参与配对。
    - min_score_diff：若同一样本内最高分与最低分差值小于该阈值，则认为排序不可信，直接丢弃该样本。
    """
    pairs: List[Pair] = []
    grouped: Dict[str, List[RankedCandidate]] = {}
    for rc in ranked:
        grouped.setdefault(rc.sample_id, []).append(rc)

    for sample_id, group in grouped.items():
        if not group:
            continue

        max_score = max(rc.aggregate_score for rc in group)
        min_score = min(rc.aggregate_score for rc in group)
        if max_score - min_score < min_score_diff:
            # 排序差值过小，认为难以区分优劣，跳过该样本
            continue

        sorted_group = sorted(group, key=lambda r: r.rank)
        top = sorted_group[:top_k]
        bottom = sorted(group, key=lambda r: r.rank, reverse=True)[:bottom_k]
        for pos in top:
            for neg in bottom:
                pairs.append(
                    Pair(
                        sample_id=sample_id,
                        positive=pos.candidate_id,
                        negative=neg.candidate_id,
                    )
                )
    return pairs

