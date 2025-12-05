from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Dict, List, Sequence

from collections import defaultdict
import logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.data_loader.excel_loader import Sample, SampleLoader, export_samples
from src.data_loader.context_builder import (
    build_context_samples,
    export_context_samples_jsonl,
)
from src.generators.base import build_generators, Candidate, generate_candidates
from src.judges.base import build_judges, BaseJudge
from src.scoring.aggregator import (
    aggregate_scores,
    group_scores_by_sample,
    flatten_grouped_scores,
)
from src.ranking.ranker import rank_candidates, build_pairs
from src.utils.io import write_jsonl, ensure_dir, read_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env


# 默认的 judge 并发 worker 数（按样本/候选切分任务）
DEFAULT_JUDGE_WORKERS = 16


# ===== Gemini 2.5 特殊后处理：裁剪掉「思考过程」 =====
_GEMINI_MODEL_NAME_FROM_ENV = os.getenv("LLM_MODEL_GEMINI25_NAME")
# 与默认配置文件中保持兼容
_GEMINI_MODEL_DEFAULT_NAME = "google/gemini-2.5-flash"
_GEMINI_MODEL_NAMES = {
    name for name in (_GEMINI_MODEL_NAME_FROM_ENV, _GEMINI_MODEL_DEFAULT_NAME) if name
}


def _postprocess_gemini_candidates(candidates: Iterable[Candidate]) -> List[Candidate]:
    """
    对 Gemini 2.5 模型的输出进行统一清洗：
    - 若响应中包含 "</think>"，仅保留该标记之后的内容（不含 "</think>" 本身）。
    """
    THINK_END = "</think>"
    processed: List[Candidate] = []
    for cand in candidates:
        if cand.model_name in _GEMINI_MODEL_NAMES and THINK_END in cand.response:
            idx = cand.response.rfind(THINK_END)
            # 仅保留 </think> 之后的文本，并去掉前后空白
            new_resp = cand.response[idx + len(THINK_END) :].lstrip()
            cand.response = new_resp
        processed.append(cand)
    return processed


def run_generation(samples, generator_configs) -> list[Candidate]:
    """
    在同一模型内按样本批量生成，不同模型之间并发执行。
    同时对特定模型（如 Gemini 2.5）做统一后处理。
    """
    generators = build_generators(generator_configs)
    raw_candidates = list(generate_candidates(samples, generators))
    return _postprocess_gemini_candidates(raw_candidates)


def _judge_single(
    judge: BaseJudge,
    candidate: Candidate,
    sample_lookup: Dict[str, Sample],
):
    """
    对单个候选执行一次 judge 调用。
    """
    sample = sample_lookup[candidate.sample_id]
    return judge.judge(sample, candidate)


def run_judging(
    samples: Sequence[Sample],
    candidates: Sequence[Candidate],
    judge_configs,
    max_workers: int | None = None,
) -> List:
    """
    按样本/候选粒度批量并发打分：
    - 通常只有一个 LLM judge，此时对不同样本的打分任务并行执行；
    - 若存在多个 judge，则会对 (judge, candidate) 的组合任务并行执行。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    judges = build_judges(judge_configs)
    if not judges or not candidates:
        return []

    sample_lookup: Dict[str, Sample] = {sample.sample_id: sample for sample in samples}

    logger = logging.getLogger(__name__)
    # 为每个 (judge, candidate) 组合创建一个任务，一般情况下 judge 只有一个，
    # 就等价于“按样本数并行”。
    tasks: List[tuple[BaseJudge, Candidate]] = [
        (judge, candidate) for judge in judges for candidate in candidates
    ]
    if not tasks:
        return []

    # 按样本数量设置并发 worker 数，默认 8，可通过 max_workers 参数覆盖
    desired_workers = max_workers or DEFAULT_JUDGE_WORKERS
    worker_count = max(1, min(desired_workers, len(tasks)))

    all_results: List = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {
            executor.submit(_judge_single, judge, candidate, sample_lookup): (judge, candidate)
            for judge, candidate in tasks
        }
        for future in as_completed(future_to_task):
            judge, candidate = future_to_task[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:  # pragma: no cover - 防御性日志
                logger.exception(
                    "Judge %s failed during batch judging on candidate %s: %r",
                    getattr(judge, "name", "unknown"),
                    getattr(candidate, "candidate_id", "unknown"),
                    exc,
                )

    return all_results


def _load_samples_from_jsonl(path: str | Path) -> list[Sample]:
    """
    从 samples.jsonl 重新构造 Sample 对象，供后续 stage 复用。
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Samples file not found: {path_obj}. 请先运行 `--stage generate` 或 `--stage all`。"
        )
    rows = read_jsonl(path_obj)
    if not rows:
        raise ValueError(f"Samples file {path_obj} is empty.")
    return [Sample(**row) for row in rows]


def _load_candidates_from_jsonl(path: str | Path) -> list[Candidate]:
    """
    从 generated_responses.jsonl 重新构造 Candidate 对象，供后续 stage 复用。
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Generated responses file not found: {path_obj}. 请先运行 `--stage generate` 或 `--stage all`。"
        )
    rows = read_jsonl(path_obj)
    if not rows:
        raise ValueError(f"Generated responses file {path_obj} is empty.")

    candidates: list[Candidate] = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        for cand in row.get("candidates", []):
            candidates.append(
                Candidate(
                    sample_id=sample_id,
                    candidate_id=str(cand.get("candidate_id")),
                    model_type=str(cand.get("model_type")),
                    model_name=str(cand.get("model_name")),
                    response=str(cand.get("response") or ""),
                    gen_config=cand.get("gen_config") or {},
                )
            )
    if not candidates:
        raise ValueError(f"No candidates found in {path_obj}.")
    return candidates


def run_pipeline(
    config_path: str | Path,
    stage: str = "all",
    max_rows: int | None = 3,
) -> None:
    """
    一键运行 pipeline，支持通过 --stage 只执行当前阶段。

    stage 取值：
    - generate：从原始数据构建样本 + 上下文，并生成候选回复，结果写入 JSONL；
    - judge：基于已生成的 samples.jsonl / generated_responses.jsonl 执行打分聚合；
    - rank：基于已生成的 judge_results.jsonl 执行排序与正负样本构建；
    - all（默认）：依次执行 generate → judge → rank，每个阶段完成后落盘，便于中断与续跑。
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading pipeline config from %s (stage=%s)", config_path, stage)
    config = PipelineConfig.from_yaml(config_path)
    ensure_dir(config.processed_dir)
    ensure_dir(config.intermediate_dir)

    # ==== Stage: generate ====
    if stage in ("all", "generate"):
        samples = SampleLoader(config.raw_data).load(max_rows=max_rows)
        logger.info("Loaded %d samples from %s", len(samples), config.raw_data)
        # 1) 导出原始样本快照，便于后续对齐 & 调试
        write_jsonl(config.output_files.samples, export_samples(samples))

        # 2) 构建并导出上下文样本（TASK.md 2.2）
        context_samples = build_context_samples(samples)
        export_context_samples_jsonl(config.intermediate_dir, context_samples)
        context_lookup: Dict[str, str] = {cs.sample_id: cs.context for cs in context_samples}

        # 3) 调用各类生成器生成候选回复，并导出统一结构的 generated_responses.jsonl
        candidates = run_generation(samples, config.generators)
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

        # 如果只跑 generate，则直接返回；否则继续后续阶段。
        if stage == "generate":
            return

    # ==== Stage: judge ====
    if stage in ("all", "judge"):
        if stage == "judge":
            # 独立 judge 模式：从前一阶段产物中恢复 samples 与 candidates。
            samples = _load_samples_from_jsonl(config.output_files.samples)
            candidates = _load_candidates_from_jsonl(config.output_files.generated_responses)

        judge_scores = run_judging(samples, candidates, config.judges)
        # 1) 先得到按候选一行的扁平结构，供后续 rank 阶段直接使用
        aggregated_scores = aggregate_scores(judge_scores)
        # 2) 写盘时再按 sample_id 分组，便于人工查看与下游分析
        grouped_scores = group_scores_by_sample(aggregated_scores)
        write_jsonl(config.output_files.judge_results, grouped_scores)
        logger.info(
            "Wrote %d judge rows (grouped by sample_id) to %s",
            len(grouped_scores),
            config.output_files.judge_results,
        )

        if stage == "judge":
            return

    # ==== Stage: rank ====
    if stage in ("all", "rank"):
        if stage == "rank":
            judge_results_path = config.output_files.judge_results
            raw_rows = read_jsonl(judge_results_path)
            if not raw_rows:
                raise FileNotFoundError(
                    f"Judge results not found or empty: {judge_results_path}. "
                    "请先运行 `--stage judge` 或 `--stage all`。"
                )
            # 兼容新的「按 sample_id 分组」结构，以及旧版本的扁平结构
            aggregated_scores = flatten_grouped_scores(raw_rows)

        ranked = rank_candidates(aggregated_scores)
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
        logger.info(
            "Pipeline finished: %d judge rows, %d pairs",
            len(aggregated_scores),
            len(pairs),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end data generation and judging pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "judge", "rank"],
        default="all",
        help="Which stage of the pipeline to run (default: all).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum number of rows to load from the raw data (default: 3).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 加载 .env 中的模型与代理配置（若存在）
    load_env()
    init_logger()
    args = parse_args()
    run_pipeline(args.config, stage=args.stage, max_rows=args.max_rows)
