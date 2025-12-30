from __future__ import annotations

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """
    简单的 JSONL 读取工具，返回 dict 列表。
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def update_outcome_counts(
    a_score: float,
    b_score: float,
    *,
    win_key: str,
    draw_key: str,
    lose_key: str,
    counter: Dict[str, int],
) -> None:
    if a_score > b_score:
        counter[win_key] += 1
    elif a_score == b_score:
        counter[draw_key] += 1
    else:
        counter[lose_key] += 1


def print_outcome_summary(counter: Dict[str, int], *, title: str, total: int, prefix: str) -> None:
    win = counter[f"{prefix}_win"]
    draw = counter[f"{prefix}_draw"]
    lose = counter[f"{prefix}_lose"]

    print(f"\n{title}:")
    print(f"  样本数: {total}")
    if total:
        print(f"  获胜: {win}/{total} ({win/total*100:.2f}%)")
        print(f"  平局: {draw}/{total} ({draw/total*100:.2f}%)")
        print(f"  失败: {lose}/{total} ({lose/total*100:.2f}%)")
    else:
        print("  获胜: N/A")
        print("  平局: N/A")
        print("  失败: N/A")


def main() -> None:
    path = 'data/data_diff_sample/processed/judge_results_kto_merger.jsonl'
    rows = read_jsonl(path)
    
    # 统计计数器
    counter: Dict[str, int] = {
        "sft_vs_ref_win": 0,
        "sft_vs_ref_draw": 0,
        "sft_vs_ref_lose": 0,
        "sft_vs_qwen_win": 0,
        "sft_vs_qwen_draw": 0,
        "sft_vs_qwen_lose": 0,
        "kto_vs_ref_win": 0,
        "kto_vs_ref_draw": 0,
        "kto_vs_ref_lose": 0,
        "kto_vs_qwen_win": 0,
        "kto_vs_qwen_draw": 0,
        "kto_vs_qwen_lose": 0,
        "sft_kto_vs_ref_win": 0,
        "sft_kto_vs_ref_draw": 0,
        "sft_kto_vs_ref_lose": 0,
        "sft_kto_vs_qwen_win": 0,
        "sft_kto_vs_qwen_draw": 0,
        "sft_kto_vs_qwen_lose": 0,
        "dpo_vs_ref_win": 0,
        "dpo_vs_ref_draw": 0,
        "dpo_vs_ref_lose": 0,
        "dpo_vs_qwen_win": 0,
        "dpo_vs_qwen_draw": 0,
        "dpo_vs_qwen_lose": 0,
    }
    total_samples = 0
    total_samples_sft_kto = 0
    total_samples_dpo = 0
    
    for i in rows:
        results = i.get("results") or []
        
        # 收集当前样本中各个模型的得分
        scores = {}
        for result in results:
            model_name = result.get("model_name")
            if model_name in [
                "qwen32_lora_kto_maxepochs30",
                "qwen32_lora_sft_maxepochs30",
                "qwen32_lora_sft_kto_maxepochs30",
                "qwen32_lora_dpo_maxepochs30",
                "qwen3-32b",
                "ref_answer_phone",
            ]:
                score = result.get("total_score_20")
                if score is not None:
                    scores[model_name] = score
        
        # 如果所有需要的模型得分都存在，进行比较
        if all(key in scores for key in ["qwen32_lora_sft_maxepochs30", "qwen32_lora_kto_maxepochs30", "ref_answer_phone", "qwen3-32b"]):
            total_samples += 1
            
            # 比较 qwen32_lora_sft
            update_outcome_counts(
                scores["qwen32_lora_sft_maxepochs30"],
                scores["ref_answer_phone"],
                win_key="sft_vs_ref_win",
                draw_key="sft_vs_ref_draw",
                lose_key="sft_vs_ref_lose",
                counter=counter,
            )
            update_outcome_counts(
                scores["qwen32_lora_sft_maxepochs30"],
                scores["qwen3-32b"],
                win_key="sft_vs_qwen_win",
                draw_key="sft_vs_qwen_draw",
                lose_key="sft_vs_qwen_lose",
                counter=counter,
            )
            
            # 比较 qwen32_lora_kto
            update_outcome_counts(
                scores["qwen32_lora_kto_maxepochs30"],
                scores["ref_answer_phone"],
                win_key="kto_vs_ref_win",
                draw_key="kto_vs_ref_draw",
                lose_key="kto_vs_ref_lose",
                counter=counter,
            )
            update_outcome_counts(
                scores["qwen32_lora_kto_maxepochs30"],
                scores["qwen3-32b"],
                win_key="kto_vs_qwen_win",
                draw_key="kto_vs_qwen_draw",
                lose_key="kto_vs_qwen_lose",
                counter=counter,
            )

        # 如果 sft_kto 所需得分存在，单独统计（避免影响上面的 total_samples 口径）
        if all(key in scores for key in ["qwen32_lora_sft_kto_maxepochs30", "ref_answer_phone", "qwen3-32b"]):
            total_samples_sft_kto += 1
            update_outcome_counts(
                scores["qwen32_lora_sft_kto_maxepochs30"],
                scores["ref_answer_phone"],
                win_key="sft_kto_vs_ref_win",
                draw_key="sft_kto_vs_ref_draw",
                lose_key="sft_kto_vs_ref_lose",
                counter=counter,
            )
            update_outcome_counts(
                scores["qwen32_lora_sft_kto_maxepochs30"],
                scores["qwen3-32b"],
                win_key="sft_kto_vs_qwen_win",
                draw_key="sft_kto_vs_qwen_draw",
                lose_key="sft_kto_vs_qwen_lose",
                counter=counter,
            )

        # dpo 口径单独统计
        if all(key in scores for key in ["qwen32_lora_dpo_maxepochs30", "ref_answer_phone", "qwen3-32b"]):
            total_samples_dpo += 1
            update_outcome_counts(
                scores["qwen32_lora_dpo_maxepochs30"],
                scores["ref_answer_phone"],
                win_key="dpo_vs_ref_win",
                draw_key="dpo_vs_ref_draw",
                lose_key="dpo_vs_ref_lose",
                counter=counter,
            )
            update_outcome_counts(
                scores["qwen32_lora_dpo_maxepochs30"],
                scores["qwen3-32b"],
                win_key="dpo_vs_qwen_win",
                draw_key="dpo_vs_qwen_draw",
                lose_key="dpo_vs_qwen_lose",
                counter=counter,
            )
    
    # 输出统计结果
    print(f"\n{'='*60}")
    print_outcome_summary(
        counter,
        title="qwen32_lora_sft vs ref_answer_phone",
        total=total_samples,
        prefix="sft_vs_ref",
    )
    print_outcome_summary(
        counter,
        title="qwen32_lora_sft vs qwen3-32b",
        total=total_samples,
        prefix="sft_vs_qwen",
    )
    print_outcome_summary(
        counter,
        title="qwen32_lora_kto vs ref_answer_phone",
        total=total_samples,
        prefix="kto_vs_ref",
    )
    print_outcome_summary(
        counter,
        title="qwen32_lora_kto vs qwen3-32b",
        total=total_samples,
        prefix="kto_vs_qwen",
    )

    print(f"\n{'-'*60}")
    print_outcome_summary(
        counter,
        title="qwen32_lora_sft_kto vs ref_answer_phone",
        total=total_samples_sft_kto,
        prefix="sft_kto_vs_ref",
    )
    print_outcome_summary(
        counter,
        title="qwen32_lora_sft_kto vs qwen3-32b",
        total=total_samples_sft_kto,
        prefix="sft_kto_vs_qwen",
    )

    print(f"\n{'-'*60}")
    print_outcome_summary(
        counter,
        title="qwen32_lora_dpo vs ref_answer_phone",
        total=total_samples_dpo,
        prefix="dpo_vs_ref",
    )
    print_outcome_summary(
        counter,
        title="qwen32_lora_dpo vs qwen3-32b",
        total=total_samples_dpo,
        prefix="dpo_vs_qwen",
    )
    print(f"{'='*60}")

if __name__ == "__main__":
    main()