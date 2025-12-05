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
    generated_responses: Path
    judge_results: Path
    ranked_pairs: Path


@dataclass
class RankingConfig:
    top_k: int = 1
    bottom_k: int = 1
    min_score_diff: float = 0.0


@dataclass
class PipelineConfig:
    raw_data: Path
    processed_dir: Path
    intermediate_dir: Path
    generators: list[dict[str, Any]]
    judges: list[dict[str, Any]]
    ranking: RankingConfig
    output_files: OutputFiles

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        base_dir = Path(path).resolve().parent
        config = load_yaml(path)

        if not isinstance(config, dict):
            raise ValueError(f"Invalid config file: {path}, expected a mapping at top level.")

        paths_cfg = config.get("paths")
        pipeline_cfg = config.get("pipeline")

        if not isinstance(paths_cfg, dict):
            raise ValueError("Config missing required 'paths' section.")
        if not isinstance(pipeline_cfg, dict):
            raise ValueError("Config missing required 'pipeline' section.")

        raw_data = _resolve_path(base_dir, paths_cfg["raw_data"])
        processed_dir = _resolve_path(base_dir, paths_cfg["processed_dir"])
        intermediate_dir = _resolve_path(base_dir, paths_cfg["intermediate_dir"])

        # 基础路径存在性校验
        if not raw_data.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_data}")

        ranking_cfg = pipeline_cfg.get("ranking", {}) or {}
        output_cfg = pipeline_cfg.get("output_files", {}) or {}

        # 排序配置校验
        top_k = int(ranking_cfg.get("top_k", 1))
        bottom_k = int(ranking_cfg.get("bottom_k", 1))
        min_score_diff = float(ranking_cfg.get("min_score_diff", 0.0))
        if top_k <= 0 or bottom_k <= 0:
            raise ValueError("ranking.top_k and ranking.bottom_k must be positive integers.")
        if min_score_diff < 0.0:
            raise ValueError("ranking.min_score_diff must be >= 0.")

        generators_cfg = pipeline_cfg.get("generators")
        judges_cfg = pipeline_cfg.get("judges")
        if not isinstance(generators_cfg, list) or not generators_cfg:
            raise ValueError("pipeline.generators must be a non-empty list.")
        if not isinstance(judges_cfg, list) or not judges_cfg:
            raise ValueError("pipeline.judges must be a non-empty list.")

        return cls(
            raw_data=raw_data,
            processed_dir=processed_dir,
            intermediate_dir=intermediate_dir,
            generators=generators_cfg,
            judges=judges_cfg,
            ranking=RankingConfig(
                top_k=top_k,
                bottom_k=bottom_k,
                min_score_diff=min_score_diff,
            ),
            output_files=OutputFiles(
                samples=_resolve_path(base_dir, output_cfg.get("samples", "data/processed/samples.jsonl")),
                generated_responses=_resolve_path(
                    base_dir,
                    output_cfg.get("generated_responses", "data/processed/generated_responses.jsonl"),
                ),
                judge_results=_resolve_path(base_dir, output_cfg.get("judge_results", "data/processed/judge_results.jsonl")),
                ranked_pairs=_resolve_path(base_dir, output_cfg.get("ranked_pairs", "data/processed/ranked_pairs.jsonl")),
            ),
        )
