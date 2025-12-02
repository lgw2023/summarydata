from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.data_loader.excel_loader import SampleLoader, export_samples
from src.generators.base import build_generators, Candidate
from src.judges.base import build_judges
from src.scoring.aggregator import aggregate_scores
from src.ranking.ranker import rank_candidates, build_pairs
from src.utils.io import write_jsonl, ensure_dir


def run_generation(samples, generator_configs) -> list[Candidate]:
    generators = build_generators(generator_configs)
    generated: list[Candidate] = []
    for sample in samples:
        for generator in generators:
            generated.append(generator.generate(sample))
    return generated


def run_judging(samples, candidates, judge_configs):
    judges = build_judges(judge_configs)
    results = []
    sample_lookup = {sample.sample_id: sample for sample in samples}
    for candidate in candidates:
        sample = sample_lookup[candidate.sample_id]
        for judge in judges:
            results.append(judge.judge(sample, candidate))
    return results


def run_pipeline(config_path: str | Path) -> None:
    config = PipelineConfig.from_yaml(config_path)
    ensure_dir(config.processed_dir)
    ensure_dir(config.intermediate_dir)

    samples = SampleLoader(config.raw_data).load()
    write_jsonl(config.output_files.samples, export_samples(samples))

    candidates = run_generation(samples, config.generators)
    judge_scores = run_judging(samples, candidates, config.judges)
    aggregated_scores = aggregate_scores(judge_scores)
    write_jsonl(config.output_files.judge_results, aggregated_scores)

    ranked = rank_candidates(aggregated_scores)
    pairs = build_pairs(ranked, top_k=config.ranking.top_k, bottom_k=config.ranking.bottom_k)
    write_jsonl(
        config.output_files.ranked_pairs,
        [
            {"sample_id": p.sample_id, "positive": p.positive, "negative": p.negative}
            for p in pairs
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end data generation and judging pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config)
