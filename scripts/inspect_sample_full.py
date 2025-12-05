from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.data_loader.excel_loader import SampleLoader
from src.data_loader.context_builder import build_context_text
from src.utils.io import read_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env
from src.ranking.ranker import rank_candidates
from src.scoring.aggregator import flatten_grouped_scores


def _find_first(rows: List[Dict[str, Any]], sample_id: str) -> Dict[str, Any] | None:
    for row in rows:
        if str(row.get("sample_id")) == str(sample_id):
            return row
    return None


def inspect_sample_full(config_path: str | Path, sample_id: str) -> None:
    """
    串联查看单个 sample 在各阶段的完整链路：
    - 原始样本（data.csv / Excel）
    - 构建的上下文（现场重建）
    - 生成的候选回复（generated_responses.jsonl）
    - LLM-as-judge 打分（judge_results.jsonl）
    - 排序结果与正负样本位置
    """
    logger = logging.getLogger(__name__)
    config = PipelineConfig.from_yaml(config_path)

    # 1) 原始样本
    samples = SampleLoader(config.raw_data).load()
    sample_lookup: Dict[str, Any] = {s.sample_id: s for s in samples}
    sample = sample_lookup.get(str(sample_id))
    if sample is None:
        print(f"[raw] sample_id={sample_id} not found in raw data.")
        return

    print("=" * 80)
    print(f"RAW SAMPLE (sample_id={sample.sample_id})")
    print("- query:", sample.query)
    print("- category_level1/2/3:", sample.category_level1, "/", sample.category_level2, "/", sample.category_level3)
    print("- domain:", sample.domain)
    print("- has_reference_answers:", bool(sample.reference_answers))

    # 2) 上下文（现场重建，避免依赖中间文件是否存在）
    print("\n" + "=" * 80)
    print("CONTEXT (rebuilt)")
    context_text = build_context_text(sample)
    print(context_text)

    # 3) 生成候选（来自 generated_responses.jsonl）
    generated_rows = read_jsonl(config.output_files.generated_responses)
    gen_row = _find_first(generated_rows, sample.sample_id)
    print("\n" + "=" * 80)
    print("GENERATED CANDIDATES")
    if not gen_row:
        print("No generated_responses found for this sample.")
    else:
        candidates = gen_row.get("candidates", []) or []
        print(f"Total candidates: {len(candidates)}")
        for idx, cand in enumerate(candidates, 1):
            cid = cand.get("candidate_id")
            mtype = cand.get("model_type")
            mname = cand.get("model_name")
            resp = cand.get("response", "") or ""
            print("-" * 40)
            print(f"[{idx}] candidate_id={cid}")
            print(f"    model_type={mtype}, model_name={mname}")
            print(f"    response_preview={resp[:200].replace('\\n', ' ')}{'...' if len(resp) > 200 else ''}")

    # 4) Judge 打分结果
    raw_judge_rows = read_jsonl(config.output_files.judge_results)
    judge_rows = flatten_grouped_scores(raw_judge_rows)
    judge_for_sample = [r for r in judge_rows if str(r.get("sample_id")) == str(sample.sample_id)]
    print("\n" + "=" * 80)
    print("JUDGE RESULTS")
    if not judge_for_sample:
        print("No judge_results found for this sample.")
    else:
        for r in judge_for_sample:
            cid = r.get("candidate_id") or r.get("candidate")
            mtype = r.get("model_type")
            mname = r.get("model_name")
            agg = r.get("aggregate_score")
            g = r.get("ground_score")
            s = r.get("structure_score")
            print("-" * 40)
            print(f"candidate_id={cid}")
            print(f"    model_type={mtype}, model_name={mname}")
            print(f"    aggregate_score={agg}, ground={g}, structure={s}")

    # 5) 排序信息
    print("\n" + "=" * 80)
    print("RANKING (within this sample)")
    if not judge_for_sample:
        print("No ranking since no judge_results.")
    else:
        ranked_all = rank_candidates(judge_for_sample)
        ranked_for_sample = [r for r in ranked_all if str(r.sample_id) == str(sample.sample_id)]
        if not ranked_for_sample:
            print("rank_candidates returned empty list for this sample.")
        else:
            for rc in sorted(ranked_for_sample, key=lambda r: r.rank):
                print(
                    f"rank={rc.rank}, candidate_id={rc.candidate_id}, "
                    f"aggregate_score={rc.aggregate_score}"
                )

    print("\n" + "=" * 80)
    logger.info("Finished inspection for sample_id=%s", sample_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect full pipeline information for a single sample_id"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    parser.add_argument("--sample-id", required=True, help="Sample ID to inspect")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    init_logger()
    args = parse_args()
    inspect_sample_full(args.config, args.sample_id)


