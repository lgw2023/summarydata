from __future__ import annotations

from typing import Iterable, List

from src.judges.base import JudgeScore


def aggregate_scores(judge_scores: Iterable[JudgeScore]) -> List[dict]:
    aggregated = []
    for score in judge_scores:
        aggregate_score = (score.ground_score + score.structure_score) / 2
        aggregated.append(
            {
                "sample_id": score.sample_id,
                "candidate": score.candidate,
                "judge": score.judge,
                "ground_score": score.ground_score,
                "structure_score": score.structure_score,
                "aggregate_score": aggregate_score,
                "notes": score.notes,
            }
        )
    return aggregated
