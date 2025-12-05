from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Sample:
    sample_id: str
    query: str
    # 上一轮对话（可选）
    last_query: str | None = None
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
    "query": "query",
}

OPTIONAL_COLUMNS = {
    "一级分类": "category_level1",
    "二级分类": "category_level2",
    "三级分类": "category_level3",
    "domain": "domain",
    "last_query": "last_query",
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

    def load(self, max_rows: int | None = None) -> List[Sample]:
        with self.path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing required column(s): {missing}")

            samples: List[Sample] = []
            for idx, row in enumerate(reader):
                if max_rows is not None and idx >= max_rows and max_rows != 0:
                    break
                mapping = {internal: row.get(external) for external, internal in OPTIONAL_COLUMNS.items()}

                # 参考答案列兼容两套命名：
                # - 旧数据：a_answer / b_answer
                # - 新数据：answer_phone / think_phone
                reference_keys: list[str] = []
                if any(col in reader.fieldnames for col in ("a_answer", "b_answer")):
                    reference_keys = [col for col in ("a_answer", "b_answer") if row.get(col)]
                elif any(col in reader.fieldnames for col in ("answer_phone", "think_phone")):
                    reference_keys = [col for col in ("answer_phone", "think_phone") if row.get(col)]

                reference = {key: row.get(key) for key in reference_keys if row.get(key)}

                samples.append(
                    Sample(
                        sample_id=str(row.get("sample_id", idx + 1)),
                        query=row["query"],
                        last_query=mapping.get("last_query"),
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
