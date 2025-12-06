from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.data_loader.excel_loader import SampleLoader
from src.generators.base import build_generators, Candidate
from src.judges.base import build_judges, BaseJudge
from src.scoring.aggregator import aggregate_scores, group_scores_by_sample
from src.utils.io import write_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env


# 默认的 judge 并发 worker 数（按样本/候选切分任务）
DEFAULT_JUDGE_WORKERS = 8


def _judge_single(
    judge: BaseJudge,
    candidate: Candidate,
    sample_lookup,
):
    """
    对单个候选执行一次 judge 调用。
    """
    sample = sample_lookup[candidate.sample_id]
    return judge.judge(sample, candidate)


def run_judge(config_path: str | Path, raw_data_path: str | Path) -> None:
    """
    仅执行「生成 + 评估」阶段。

    注意：这里会根据当前配置**重新生成候选**，
    由于 candidate_id 是基于 (sample_id, model_name, index) 的稳定 hash，
    因此与 generate_responses.py 的输出在配置相同时是一致的。
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading config from %s (raw_data=%s)", config_path, raw_data_path)
    config = PipelineConfig.from_yaml(config_path, raw_data_path=raw_data_path)

    samples = SampleLoader(config.raw_data).load()
    logger.info("Loaded %d samples", len(samples))
    generators = build_generators(config.generators)
    candidates: List[Candidate] = []
    for sample in samples:
        for gen in generators:
            candidates.append(gen.generate(sample))

    judges = build_judges(config.judges)
    sample_lookup = {s.sample_id: s for s in samples}

    # 按样本/候选粒度批量并发打分：通常只有一个 LLM judge，此时对不同样本的打分任务并行执行。
    from concurrent.futures import ThreadPoolExecutor, as_completed

    judge_scores: List = []
    if judges and candidates:
        # 为每个 (judge, candidate) 组合创建一个任务，一般情况下 judge 只有一个，
        # 就等价于“按样本数并行”。
        tasks = [(judge, cand) for judge in judges for cand in candidates]
        desired_workers = DEFAULT_JUDGE_WORKERS
        worker_count = max(1, min(desired_workers, len(tasks)))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {
                executor.submit(_judge_single, judge, cand, sample_lookup): (judge, cand)
                for judge, cand in tasks
            }
            for future in as_completed(future_to_task):
                judge, cand = future_to_task[future]
                try:
                    result = future.result()
                    judge_scores.append(result)
                except Exception as exc:  # pragma: no cover - 防御性日志
                    logger.exception(
                        "Judge %s failed during batch judging on candidate %s: %r",
                        getattr(judge, "name", "unknown"),
                        getattr(cand, "candidate_id", "unknown"),
                        exc,
                    )

    aggregated = aggregate_scores(judge_scores)
    grouped = group_scores_by_sample(aggregated)
    write_jsonl(config.output_files.judge_results, grouped)
    logger.info("Wrote %d judge rows (grouped by sample_id)", len(grouped))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run judge only (generation + scoring)")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file (仅包含模型与打分配置等，不再固定原始数据路径)",
    )
    parser.add_argument(
        "--raw-data",
        required=True,
        help="本次实验的输入数据文件路径（例如 CSV/Excel），用于决定读取样本以及 data/<输入文件名>/ 下的输出目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    init_logger()
    args = parse_args()
    run_judge(args.config, raw_data_path=args.raw_data)


