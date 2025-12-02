from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict


@dataclass
class RankedCandidate:
    sample_id: str
    candidate: str
    aggregate_score: float
    rank: int


@dataclass
class Pair:
    sample_id: str
    positive: str
    negative: str


def rank_candidates(scores: Iterable[dict]) -> List[RankedCandidate]:
    grouped: Dict[str, List[dict]] = {}
    for score in scores:
        grouped.setdefault(str(score["sample_id"]), []).append(score)

    ranked_rows: List[RankedCandidate] = []
    for sample_id, group in grouped.items():
        sorted_group = sorted(group, key=lambda row: row["aggregate_score"], reverse=True)
        for idx, row in enumerate(sorted_group):
            ranked_rows.append(
                RankedCandidate(
                    sample_id=str(sample_id),
                    candidate=row["candidate"],
                    aggregate_score=float(row["aggregate_score"]),
                    rank=idx + 1,
                )
            )
    return ranked_rows


def build_pairs(ranked: List[RankedCandidate], top_k: int = 1, bottom_k: int = 1) -> List[Pair]:
    pairs: List[Pair] = []
    grouped: Dict[str, List[RankedCandidate]] = {}
    for rc in ranked:
        grouped.setdefault(rc.sample_id, []).append(rc)

    for sample_id, group in grouped.items():
        sorted_group = sorted(group, key=lambda r: r.rank)
        top = sorted_group[:top_k]
        bottom = sorted(group, key=lambda r: r.rank, reverse=True)[:bottom_k]
        for pos in top:
            for neg in bottom:
                pairs.append(Pair(sample_id=sample_id, positive=pos.candidate, negative=neg.candidate))
    return pairs
