from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.data_loader.excel_loader import Sample


@dataclass
class Candidate:
    sample_id: str
    generator: str
    content: str


class BaseGenerator:
    name: str = "base"

    def generate(self, sample: Sample) -> Candidate:  # pragma: no cover - interface
        raise NotImplementedError


class EchoGenerator(BaseGenerator):
    name = "echo"

    def generate(self, sample: Sample) -> Candidate:
        parts = [f"问：{sample.query}"]
        if sample.last_answer_phone:
            parts.append(f"上一轮助手回复：{sample.last_answer_phone}")
        if sample.suggest:
            parts.append(f"专家建议：{sample.suggest}")
        content = "\n".join(parts)
        return Candidate(sample_id=sample.sample_id, generator=self.name, content=content)


def build_generators(configs: List[dict]) -> List[BaseGenerator]:
    registry = {
        "echo": EchoGenerator,
    }
    generators: List[BaseGenerator] = []
    for cfg in configs:
        name = cfg.get("name")
        generator_cls = registry.get(name)
        if not generator_cls:
            raise ValueError(f"Unknown generator: {name}")
        generators.append(generator_cls())
    return generators
