from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.data_loader.excel_loader import Sample
from src.generators.base import Candidate


@dataclass
class JudgeScore:
    sample_id: str
    candidate: str
    judge: str
    ground_score: float
    structure_score: float
    notes: str | None = None


class BaseJudge:
    name: str = "judge"

    def judge(self, sample: Sample, candidate: Candidate) -> JudgeScore:  # pragma: no cover - interface
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    name = "heuristic"

    def judge(self, sample: Sample, candidate: Candidate) -> JudgeScore:
        ground = 4.0
        structure = 4.0

        if sample.service and sample.service not in candidate.content:
            ground -= 1.0
            note = "未引用课程库内容"
        else:
            note = None

        if len(candidate.content) < 10:
            structure -= 1.0

        return JudgeScore(
            sample_id=sample.sample_id,
            candidate=candidate.generator,
            judge=self.name,
            ground_score=max(0.0, ground),
            structure_score=max(0.0, structure),
            notes=note,
        )


def build_judges(configs: List[dict]) -> List[BaseJudge]:
    registry = {
        "heuristic": HeuristicJudge,
    }
    judges: List[BaseJudge] = []
    for cfg in configs:
        name = cfg.get("name")
        judge_cls = registry.get(name)
        if not judge_cls:
            raise ValueError(f"Unknown judge: {name}")
        judges.append(judge_cls())
    return judges
