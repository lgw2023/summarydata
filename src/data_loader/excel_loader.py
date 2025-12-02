from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Sample:
    sample_id: str
    query: str
    category_level1: str | None = None
    category_level2: str | None = None
    category_level3: str | None = None
    domain: str | None = None
    data: str | None = None
    suggest: str | None = None
    rag: str | None = None
    service: str | None = None
    last_answer_phone: str | None = None
    reference_answers: dict[str, str] | None = None


REQUIRED_COLUMNS = {
    "典型query": "query",
}

OPTIONAL_COLUMNS = {
    "一级分类": "category_level1",
    "二级分类": "category_level2",
    "三级分类": "category_level3",
    "domain": "domain",
    "data": "data",
    "suggest": "suggest",
    "rag": "rag",
    "service": "service",
    "last_answer_phone": "last_answer_phone",
    "a_answer": "a_answer",
    "b_answer": "b_answer",
    "winner": "winner",
}


class SampleLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> List[Sample]:
        with self.path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing required column(s): {missing}")

            samples: List[Sample] = []
            for idx, row in enumerate(reader):
                mapping = {internal: row.get(external) for external, internal in OPTIONAL_COLUMNS.items()}
                reference = {
                    key: row.get(key)
                    for key in ("a_answer", "b_answer")
                    if row.get(key)
                }
                samples.append(
                    Sample(
                        sample_id=str(row.get("sample_id", idx + 1)),
                        query=row["典型query"],
                        category_level1=mapping["category_level1"],
                        category_level2=mapping["category_level2"],
                        category_level3=mapping["category_level3"],
                        domain=mapping["domain"],
                        data=mapping["data"],
                        suggest=mapping["suggest"],
                        rag=mapping["rag"],
                        service=mapping["service"],
                        last_answer_phone=mapping["last_answer_phone"],
                        reference_answers=reference or None,
                    )
                )
        return samples


def export_samples(samples: Iterable[Sample]) -> List[dict]:
    return [sample.__dict__ for sample in samples]
