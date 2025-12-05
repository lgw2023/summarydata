from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.utils.io import read_jsonl
from src.ranking.ranker import rank_candidates
from src.scoring.aggregator import flatten_grouped_scores
from src.utils.logging_utils import init_logger
from src.utils.env import load_env


def inspect_sample(config_path: str | Path, sample_id: str) -> None:
    logger = logging.getLogger(__name__)
    config = PipelineConfig.from_yaml(config_path)
    raw_rows = read_jsonl(config.output_files.judge_results)
    rows = flatten_grouped_scores(raw_rows)
    ranked = rank_candidates(rows)
    ranked_for_sample = [r for r in ranked if r.sample_id == str(sample_id)]

    if not ranked_for_sample:
        print(f"No ranking found for sample_id={sample_id}")
        return

    print(f"Ranking for sample_id={sample_id}:")
    for rc in sorted(ranked_for_sample, key=lambda r: r.rank):
        print(f"  rank={rc.rank}, candidate_id={rc.candidate_id}, aggregate_score={rc.aggregate_score:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ranking result for a specific sample_id")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    parser.add_argument("--sample-id", required=True, help="Sample ID to inspect")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    init_logger()
    args = parse_args()
    inspect_sample(args.config, args.sample_id)


