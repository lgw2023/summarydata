from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import load_yaml


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


@dataclass
class OutputFiles:
    samples: Path
    judge_results: Path
    ranked_pairs: Path


@dataclass
class RankingConfig:
    top_k: int = 1
    bottom_k: int = 1


@dataclass
class PipelineConfig:
    raw_data: Path
    processed_dir: Path
    intermediate_dir: Path
    prompts_dir: Path
    generators: list[dict[str, Any]]
    judges: list[dict[str, Any]]
    ranking: RankingConfig
    output_files: OutputFiles

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        base_dir = Path(path).resolve().parent
        config = load_yaml(path)

        paths_cfg = config.get("paths", {})
        pipeline_cfg = config.get("pipeline", {})

        raw_data = _resolve_path(base_dir, paths_cfg["raw_data"])
        processed_dir = _resolve_path(base_dir, paths_cfg["processed_dir"])
        intermediate_dir = _resolve_path(base_dir, paths_cfg["intermediate_dir"])
        prompts_dir = _resolve_path(base_dir, paths_cfg["prompts_dir"])

        ranking_cfg = pipeline_cfg.get("ranking", {})
        output_cfg = pipeline_cfg.get("output_files", {})

        return cls(
            raw_data=raw_data,
            processed_dir=processed_dir,
            intermediate_dir=intermediate_dir,
            prompts_dir=prompts_dir,
            generators=pipeline_cfg.get("generators", []),
            judges=pipeline_cfg.get("judges", []),
            ranking=RankingConfig(
                top_k=ranking_cfg.get("top_k", 1),
                bottom_k=ranking_cfg.get("bottom_k", 1),
            ),
            output_files=OutputFiles(
                samples=_resolve_path(base_dir, output_cfg.get("samples", "data/processed/samples.jsonl")),
                judge_results=_resolve_path(base_dir, output_cfg.get("judge_results", "data/processed/judge_results.jsonl")),
                ranked_pairs=_resolve_path(base_dir, output_cfg.get("ranked_pairs", "data/processed/ranked_pairs.jsonl")),
            ),
        )
