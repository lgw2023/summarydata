from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Dict, List, Sequence, Any

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
from src.scoring.aggregator import aggregate_scores, group_scores_by_sample
from src.utils.io import write_jsonl, ensure_dir, read_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env


# 默认的 judge 并发 worker 数（按样本/候选切分任务）
DEFAULT_JUDGE_WORKERS = 16

# generate 阶段增量写盘时，每批并行生成的样本数（通过环境变量可覆盖）
try:
    DEFAULT_GENERATE_BATCH_SIZE = max(
        1, int(os.getenv("GENERATE_BATCH_SIZE", "16"))
    )
except ValueError:
    DEFAULT_GENERATE_BATCH_SIZE = 16


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

    logger = logging.getLogger(__name__)
    logger.info("Loaded %d samples from existing file %s", len(rows), path_obj)
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

    # 若存在同一 sample_id 的多条记录（例如在增量生成、多次运行后），
    # 优先使用**最后一条**，视为该样本的最新完整结果。
    rows_by_sample_id: Dict[str, Dict] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if not sample_id:
            continue
        rows_by_sample_id[sample_id] = row

    logger = logging.getLogger(__name__)
    logger.info(
        "Loaded generated responses for %d samples from existing file %s",
        len(rows_by_sample_id),
        path_obj,
    )

    candidates: list[Candidate] = []
    for sample_id, row in rows_by_sample_id.items():
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


def _load_completed_generated_rows(config: PipelineConfig) -> Dict[str, Dict]:
    """
    从已有的 generated_responses.jsonl 中筛选出「已完成所有当前配置生成器」的样本行。

    设计原则：
    - 仅当某个 sample_id 对所有当前 PipelineConfig.generators 中的 (model_type, model_name)
      至少各有一条候选时，才认为该样本“已完成”，后续跑 generate 阶段时可以跳过重新生成；
    - 若生成器配置发生变化（新增 / 删除模型，或修改 model_name），则旧结果不再视为完整，
      会对对应样本重新生成，从而保证配置变更后结果不会“错误复用”；
    - 对于此前中途失败、候选不全的样本，不会出现在返回结果里，后续会整条样本重新生成。
    """
    logger = logging.getLogger(__name__)
    output_path = config.output_files.generated_responses
    rows = read_jsonl(output_path)
    if not rows:
        return {}

    # 针对可能存在的「同一 sample_id 多条记录」场景，保留**最后一条**作为最新结果。
    latest_rows_by_id: Dict[str, Dict] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if not sample_id:
            continue
        latest_rows_by_id[sample_id] = row

    # 根据当前配置构建一次 generator，仅用于拿到 (model_type, model_name) 组合。
    generators = build_generators(config.generators)
    if not generators:
        return {}
    expected_pairs = {(g.model_type, g.model_name) for g in generators}
    if not expected_pairs:
        return {}

    completed: Dict[str, Dict] = {}
    for sample_id, row in latest_rows_by_id.items():
        candidates = row.get("candidates") or []
        seen_pairs = {
            (str(c.get("model_type")), str(c.get("model_name"))) for c in candidates
        }
        # 仅当所有当前配置的 (model_type, model_name) 都已出现时，视为该样本“完整”
        if expected_pairs.issubset(seen_pairs):
            completed[sample_id] = row
    if completed:
        sample_ids_preview = sorted(completed.keys())
        logger.info(
            "Loaded %d completed samples from existing generated_responses file %s. "
            "Sample IDs:\n%s",
            len(completed),
            output_path,
            ", ".join(sample_ids_preview),
        )
    return completed


def _manual_score_from_total_20(total_score_20: Any) -> int:
    """
    将 0–20 分的 total_score_20 映射为 0–5 的人工打分，阈值与 visualize_data_app.py 保持一致：
    - 20 分 -> 5 分
    - 19–16 分 -> 4 分
    - 15–14 分 -> 3 分
    - 13–12 分 -> 2 分
    - 1–11 分 -> 1 分
    - 0 分或异常值 -> 0 分
    """
    try:
        v = float(total_score_20)
    except (TypeError, ValueError):
        return 0

    if v <= 0:
        return 0
    if v >= 20:
        return 5
    if 16 <= v < 20:
        return 4
    if 14 <= v < 16:
        return 3
    if 12 <= v < 14:
        return 2
    if 0 < v < 12:
        return 1
    return 0


def _build_manual_rank_output_path_from_source(source_path: Path) -> Path:
    """
    根据 judge_results_kto.jsonl 的路径生成对应的 *_rank_manually.jsonl 路径。
    规则与 visualize_data_app.py 中 _build_manual_rank_output_path 保持一致。
    """
    suffix = source_path.suffix or ".jsonl"
    return source_path.parent / f"{source_path.stem}_rank_manually{suffix}"


def _convert_record_to_manual_rank_payload(
    record: dict[str, Any],
    source_rel: str,
) -> dict[str, Any] | None:
    """
    将单条 judge_results_kto 记录转换为 *_rank_manually.jsonl 的行结构。
    """
    if not isinstance(record, dict):
        return None

    sample_id = record.get("sample_id")
    if sample_id is None:
        return None

    results = record.get("results")
    if not isinstance(results, list):
        return None

    manual_ranks: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        cid = item.get("candidate_id")
        if cid is None:
            continue

        manual_ranks.append(
            {
                "candidate_id": cid,
                "model_name": item.get("model_name"),
                "model_type": item.get("model_type"),
                "aggregate_score": item.get("aggregate_score"),
                "ground_score": item.get("ground_score"),
                "structure_score": item.get("structure_score"),
                "manual_score": _manual_score_from_total_20(item.get("total_score_20")),
            }
        )

    if not manual_ranks:
        return None

    context_fields = {k: v for k, v in record.items() if k != "results"}

    return {
        "sample_id": str(sample_id),
        "source_file": source_rel,
        "context": context_fields,
        "manual_ranks": manual_ranks,
    }


def run_pipeline(
    config_path: str | Path,
    raw_data_path: str | Path,
    stage: str = "all",
    max_rows: int | None = 3,
) -> None:
    """
    一键运行 pipeline，支持通过 --stage 只执行当前阶段。

    stage 取值：
    - generate：从原始数据构建样本 + 上下文，并生成候选回复，结果写入 JSONL；
    - judge：基于已生成的 samples.jsonl / generated_responses.jsonl 执行打分聚合；
    - rank：对上一阶段生成的 judge_results_kto.jsonl 按总分阈值自动生成 *_rank_manually.jsonl；
    - all（默认）：依次执行 generate → judge → rank，每个阶段完成后落盘，便于中断与续跑。
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Loading pipeline config from %s (stage=%s, raw_data=%s)",
        config_path,
        stage,
        raw_data_path,
    )
    config = PipelineConfig.from_yaml(config_path, raw_data_path=raw_data_path)
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

        # 3) 断点续跑 + 实时写盘：
        #    - 先从已有 generated_responses.jsonl 中识别出“已完成”的样本；
        #    - 对未完成样本按批次调用原有的并行生成逻辑（run_generation）；
        #    - 每个样本生成完一整批候选后立刻 append 写入 JSONL；
        #    - 若本次运行中途被中断，下次重跑会自动跳过已完成样本，仅补齐缺失部分。
        completed_rows_by_id = _load_completed_generated_rows(config)
        completed_sample_ids = set(completed_rows_by_id.keys())
        if completed_sample_ids:
            logger.info(
                "Detected %d completed samples from previous run in %s, "
                "will skip regeneration for them.",
                len(completed_sample_ids),
                config.output_files.generated_responses,
            )

        # 仅对“未完成”的样本重新生成；完整样本后续 judge/rank 直接复用旧结果。
        samples_to_generate: List[Sample] = [
            s for s in samples if s.sample_id not in completed_sample_ids
        ]

        if not samples_to_generate:
            logger.info(
                "All %d samples already have completed generated responses for current config. "
                "Nothing to regenerate.",
                len(samples),
            )
        else:
            output_path = config.output_files.generated_responses
            ensure_dir(output_path.parent)

            total_samples = len(samples)
            reused_count = len(completed_sample_ids)
            newly_generated_count = 0

            batch_size = DEFAULT_GENERATE_BATCH_SIZE
            logger.info(
                "Starting incremental generation for %d samples (batch_size=%d, reused_from_previous_runs=%d)",
                len(samples_to_generate),
                batch_size,
                reused_count,
            )

            # 预先根据当前配置推导出“应生成的模型集合”，便于逐样本完成度判断
            expected_model_pairs: set[tuple[str, str]] = set()
            try:
                generators_for_check = build_generators(config.generators)
                expected_model_pairs = {
                    (gen.model_type, gen.model_name) for gen in generators_for_check
                }
            except Exception as exc:  # pragma: no cover - 防御性日志
                logger.warning("未能解析生成器配置以做完成度校验：%r", exc)

            # 采用 append 方式实时写盘：每处理完一批样本，就将这些样本的完整候选写入文件。
            with Path(output_path).open("a", encoding="utf-8") as f:
                for start in range(0, len(samples_to_generate), batch_size):
                    batch = samples_to_generate[start : start + batch_size]
                    try:
                        batch_candidates = run_generation(batch, config.generators)
                    except Exception as exc:  # pragma: no cover - 防御性日志
                        logger.exception(
                            "run_generation failed on batch starting at index %d: %r",
                            start,
                            exc,
                        )
                        continue

                    grouped_candidates: Dict[str, List[Candidate]] = defaultdict(list)
                    for cand in batch_candidates:
                        grouped_candidates[cand.sample_id].append(cand)

                    for sample in batch:
                        sid = sample.sample_id
                        per_sample_candidates = grouped_candidates.get(sid, [])
                        if not per_sample_candidates:
                            logger.warning(
                                "No candidates generated for sample %s in current batch, skipping write for this sample.",
                                sid,
                            )
                            continue

                        row = {
                            "sample_id": sid,
                            "context": context_lookup.get(sid, ""),
                            "question": sample.query,
                            "candidates": [c.to_dict() for c in per_sample_candidates],
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        newly_generated_count += 1

                        # 当前样本的所有模型回复已就绪，打印一次标记便于前端观察进度
                        if expected_model_pairs:
                            generated_pairs = {
                                (c.model_type, c.model_name) for c in per_sample_candidates
                            }
                            if expected_model_pairs.issubset(generated_pairs):
                                logger.info(
                                    "Sample %s 已完成所有模型生成（%d/%d），模型列表：%s",
                                    sid,
                                    len(per_sample_candidates),
                                    len(expected_model_pairs),
                                    ", ".join(sorted({c.model_name for c in per_sample_candidates})),
                                )

            logger.info(
                "Generation stage finished: total_samples=%d, newly_generated=%d, reused_from_previous_runs=%d",
                total_samples,
                newly_generated_count,
                reused_count,
            )

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
        judge_kto_path = config.processed_dir / "judge_results_kto.jsonl"
        if not judge_kto_path.exists():
            raise FileNotFoundError(
                f"未找到 judge_results_kto.jsonl：{judge_kto_path}。"
                "请先完成上一阶段的 KTO judge。"
            )

        raw_rows = read_jsonl(judge_kto_path)
        if not raw_rows:
            raise ValueError(f"文件为空或解析失败：{judge_kto_path}")

        data_dir = PROJECT_ROOT / "data"
        try:
            source_rel = str(judge_kto_path.relative_to(data_dir))
        except ValueError:
            source_rel = str(judge_kto_path)

        payloads: list[dict[str, Any]] = []
        for rec in raw_rows:
            payload = _convert_record_to_manual_rank_payload(rec, source_rel)
            if payload:
                payloads.append(payload)

        if not payloads:
            raise ValueError("未能从 judge_results_kto.jsonl 中提取任何有效记录，无法生成打分文件。")

        output_path = _build_manual_rank_output_path_from_source(judge_kto_path)
        ensure_dir(output_path.parent)

        with output_path.open("w", encoding="utf-8") as f:
            for item in payloads:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            "自动打分完成：输入=%s，输出=%s，样本数=%d",
            judge_kto_path,
            output_path,
            len(payloads),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end data generation and judging pipeline")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file (模型与打分配置等，不再内置原始数据路径)",
    )
    parser.add_argument(
        "--raw-data",
        required=True,
        help="本次实验的输入数据文件路径（例如 CSV/Excel），用于决定读取样本以及 data/<输入文件名>/ 下的输出目录",
    )
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
    run_pipeline(
        args.config,
        raw_data_path=args.raw_data,
        stage=args.stage,
        max_rows=args.max_rows,
    )
