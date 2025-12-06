from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.ranking.ranker import rank_candidates, build_pairs
from src.scoring.aggregator import flatten_grouped_scores
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env


def run_build_pairs(config_path: str | Path, raw_data_path: str | Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Loading config from %s (raw_data=%s)", config_path, raw_data_path)
    config = PipelineConfig.from_yaml(config_path, raw_data_path=raw_data_path)
    raw_rows = read_jsonl(config.output_files.judge_results)
    judge_rows = flatten_grouped_scores(raw_rows)
    ranked = rank_candidates(judge_rows)
    pairs = build_pairs(
        ranked,
        top_k=config.ranking.top_k,
        bottom_k=config.ranking.bottom_k,
        min_score_diff=config.ranking.min_score_diff,
    )
    write_jsonl(
        config.output_files.ranked_pairs,
        [
            {"sample_id": p.sample_id, "positive": p.positive, "negative": p.negative}
            for p in pairs
        ],
    )
    logger.info("Built %d pairs", len(pairs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pos/neg pairs from judge_results.jsonl")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file (仅包含排序与打分配置等，不再固定原始数据路径)",
    )
    parser.add_argument(
        "--raw-data",
        required=True,
        help="本次实验的输入数据文件路径（例如 CSV/Excel），用于决定从哪个 data/<输入文件名>/ 目录读取 judge_results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    init_logger()
    args = parse_args()
    run_build_pairs(args.config, raw_data_path=args.raw_data)


