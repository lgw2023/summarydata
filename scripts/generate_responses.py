from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.data_loader.excel_loader import SampleLoader
from src.data_loader.context_builder import build_context_samples, export_context_samples_jsonl
from src.generators.base import build_generators, Candidate, generate_candidates
from src.utils.io import write_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env


def run_generate(config_path: str | Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Loading config from %s", config_path)
    config = PipelineConfig.from_yaml(config_path)

    samples = SampleLoader(config.raw_data).load()
    logger.info("Loaded %d samples", len(samples))
    context_samples = build_context_samples(samples)
    export_context_samples_jsonl(config.intermediate_dir, context_samples)
    context_lookup: Dict[str, str] = {cs.sample_id: cs.context for cs in context_samples}

    generators = build_generators(config.generators)
    # 同一模型内：按样本批量执行；不同模型之间：并发执行
    candidates: List[Candidate] = list(generate_candidates(samples, generators))

    grouped_candidates: Dict[str, List[Candidate]] = defaultdict(list)
    for c in candidates:
        grouped_candidates[c.sample_id].append(c)

    generated_rows = []
    for sample in samples:
        generated_rows.append(
            {
                "sample_id": sample.sample_id,
                "context": context_lookup.get(sample.sample_id, ""),
                "question": sample.query,
                "candidates": [cand.to_dict() for cand in grouped_candidates.get(sample.sample_id, [])],
            }
        )

    write_jsonl(config.output_files.generated_responses, generated_rows)
    logger.info("Generated responses for %d samples", len(generated_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses only")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    init_logger()
    args = parse_args()
    run_generate(args.config)


