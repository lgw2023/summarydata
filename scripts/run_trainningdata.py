from __future__ import annotations

"""
将生成阶段与人工打分结果拼接，导出 messages 训练数据：
- 输入 1：generated_responses.jsonl，包含每个样本的上下文与多模型候选回复；
- 输入 2：judge_results_kto_rank_manually.jsonl，包含候选回复的人工评分；
- 输出：messages.jsonl，每条记录为 messages（三段对话）+ manual_score。

运行示例：
python scripts/run_trainningdata.py \\
    --generated data/summary_train_v3/processed/generated_responses.jsonl \\
    --manual   data/summary_train_v3/processed/judge_results_kto_rank_manually.jsonl \\
    --output   data/summary_train_v3/processed/messages.jsonl
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.response_prompt_v4 import SYSTEMT_PROMPT_PHONE_GENERAL  # noqa: E402
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402


DEFAULT_DATASET_NAME = "summary_train_v3"
DEFAULT_GENERATED = (
    PROJECT_ROOT / f"data/{DEFAULT_DATASET_NAME}/processed/generated_responses.jsonl"
)
DEFAULT_MANUAL = (
    PROJECT_ROOT
    / f"data/{DEFAULT_DATASET_NAME}/processed/judge_results_kto_rank_manually.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / f"data/{DEFAULT_DATASET_NAME}/processed/messages.jsonl"
)
DEFAULT_OUTPUT_SFT = (
    PROJECT_ROOT / f"data/{DEFAULT_DATASET_NAME}/processed/messages_sft.jsonl"
)
DEFAULT_OUTPUT_KTO = (
    PROJECT_ROOT / f"data/{DEFAULT_DATASET_NAME}/processed/messages_kto.jsonl"
)

DEFAULT_TOKENIZER_NAME = "Qwen/Qwen3-32B"


def _count_tokens(messages: List[Dict[str, Any]], tokenizer: Any) -> int:
    """
    使用 Qwen/Qwen3-32B tokenizer 估算一条多轮对话 messages 的 token 长度。
    优先使用 chat template；失败时退化为简单拼接文本。
    """
    if tokenizer is None:
        return 0

    # try:
    #     # 优先走 chat 模板（Qwen 系列模型推荐用法）
    #     input_ids = tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=True,
    #         add_special_tokens=True,
    #     )
    #     return len(input_ids)
    # except Exception:
    # 兼容没有 chat_template 的情况：将多轮对话拼成一段文本
    text_parts: List[str] = []
    roles: List[str] = []
    for m in messages:
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        text_parts.append(f"{role}: {content}")
        roles.append(role)
    text = "\n".join(text_parts)
    encoded = tokenizer(text, add_special_tokens=True)
    input_ids = encoded.get("input_ids") or []
    return len(input_ids), ",".join(roles)


def _latest_rows_by_sample(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    若存在同一 sample_id 的多条记录，保留最后一条（与 run_pipeline 行为保持一致）。
    """
    rows = read_jsonl(path)
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        latest[sid] = row
    return latest


def _manual_score_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    构建 {sample_id: {candidate_id: manual_score}} 映射。
    """
    rows = read_jsonl(path)
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        scores: Dict[str, Any] = mapping.setdefault(sid, {})
        for item in row.get("manual_ranks") or []:
            cid = str(item.get("candidate_id") or "").strip()
            if not cid:
                continue
            scores[cid] = item.get("manual_score")
    return mapping


def build_messages_rows(
    generated_path: Path, manual_path: Path, tokenizer: Any
) -> Iterable[Dict[str, Any]]:
    """
    生成 messages 数据行：messages + manual_score。
    """
    latest_generated = _latest_rows_by_sample(generated_path)
    manual_scores = _manual_score_map(manual_path)

    for sample_id, row in latest_generated.items():
        context = row.get("context") or ""
        candidates: List[Dict[str, Any]] = row.get("candidates") or []
        score_map = manual_scores.get(sample_id, {})

        for cand in candidates:
            cid = str(cand.get("candidate_id") or "").strip()
            if not cid:
                continue
            if cid not in score_map:
                # 没有人工分的候选跳过
                continue
            score = score_map[cid]
            label: Optional[bool]
            if score == 5:
                label = True
            elif score in (0, 1):
                label = False
            else:
                label = None
            messages = [
                {"role": "system", "content": SYSTEMT_PROMPT_PHONE_GENERAL},
                {"role": "user", "content": context},
                {"role": "assistant", "content": str(cand.get("response") or "")},
            ]
            token_len, roles = _count_tokens(messages, tokenizer)
            if token_len >= 0:
                print(token_len, roles)
            yield {
                "messages": messages,
                "manual_score": score,
                "model_name": str(cand.get("model_name") or ""),
                "label": label,
                "token_len": token_len,
            }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将生成结果与人工打分合并，导出 messages jsonl 数据集。"
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        help=(
            "原始 CSV 路径，例如 ./文件名.csv；"
            "根据文件名自动推导 data/文件名/processed/ 下的输入输出路径。"
        ),
    )
    parser.add_argument(
        "--generated",
        default=DEFAULT_GENERATED,
        type=Path,
        help="生成阶段输出的 generated_responses.jsonl 路径",
    )
    parser.add_argument(
        "--manual",
        default=DEFAULT_MANUAL,
        type=Path,
        help="人工打分文件 judge_results_kto_rank_manually.jsonl 路径",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="输出 messages jsonl 路径（完整数据，含 manual_score / model_name / label）。",
    )
    parser.add_argument(
        "--output-sft",
        default=DEFAULT_OUTPUT_SFT,
        type=Path,
        help=(
            "输出 SFT 训练用 messages jsonl 路径；"
            "仅保留 manual_score=5 且在 selected_models 中的数据，且移除 manual_score / model_name / label 字段。"
        ),
    )
    parser.add_argument(
        "--output-kto",
        default=DEFAULT_OUTPUT_KTO,
        type=Path,
        help=(
            "输出 KTO 训练用 messages jsonl 路径；"
            "仅保留 label 为 true/false 且在 selected_models 中的数据，且移除 manual_score / model_name 字段。"
        ),
    )
    parser.add_argument(
        "--selected-models",
        type=str,
        default="anthropic/claude-4-sonnet,qwen3-max,qwen3-235b-a22b-instruct-2507,Qwen/Qwen3-Next-80B-A3B-Instruct,qwen3-30b-a3b,qwen3-32b,deepseek-chat,deepseek-reasoner,ref_answer_phone,ref_think_phone",
        help=(
            "用于过滤的模型名列表，逗号分隔，例如："
            "qwen2.5-7b-instruct,deepseek-v2.5；留空则不过滤 model_name。"
        ),
    )
    args = parser.parse_args()

    # 如果提供了 --raw-data，根据文件名自动推导 data/文件名/processed 下的路径，
    # 仅在各路径仍为默认值（summary_train_v3）时进行覆盖，保留手动传入的灵活性。
    if getattr(args, "raw_data", None):
        dataset_name = Path(args.raw_data).stem
        base_dir = PROJECT_ROOT / "data" / dataset_name / "processed"

        if args.generated == DEFAULT_GENERATED:
            args.generated = base_dir / "generated_responses.jsonl"
        if args.manual == DEFAULT_MANUAL:
            args.manual = base_dir / "judge_results_kto_rank_manually.jsonl"
        if args.output == DEFAULT_OUTPUT:
            args.output = base_dir / "messages.jsonl"
        if args.output_sft == DEFAULT_OUTPUT_SFT:
            args.output_sft = base_dir / "messages_sft.jsonl"
        if args.output_kto == DEFAULT_OUTPUT_KTO:
            args.output_kto = base_dir / "messages_kto.jsonl"

    generated_path = (
        args.generated if args.generated.is_absolute() else (PROJECT_ROOT / args.generated)
    ).resolve()
    manual_path = args.manual if args.manual.is_absolute() else (PROJECT_ROOT / args.manual)
    manual_path = manual_path.resolve()
    output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output)
    output_path = output_path.resolve()

    output_sft_path = (
        args.output_sft if args.output_sft.is_absolute() else (PROJECT_ROOT / args.output_sft)
    ).resolve()
    output_kto_path = (
        args.output_kto if args.output_kto.is_absolute() else (PROJECT_ROOT / args.output_kto)
    ).resolve()

    if args.selected_models:
        selected_models = [
            m.strip() for m in str(args.selected_models).split(",") if m.strip()
        ]
    else:
        selected_models = None

    # 初始化 Qwen/Qwen3-32B tokenizer，用于估算每条样本的 token 长度
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_TOKENIZER_NAME,
        trust_remote_code=True,
    )

    rows = list(build_messages_rows(generated_path, manual_path, tokenizer))
    write_jsonl(output_path, rows)

    # 构建 SFT / KTO 子集
    sft_rows = []
    kto_rows = []
    for row in rows:
        model_name = str(row.get("model_name") or "")
        if selected_models and model_name not in selected_models:
            continue

        score = row.get("manual_score")
        label = row.get("label")

        if score == 5:
            # SFT：只保留 messages 字段
            sft_rows.append(
                {
                    "messages": row.get("messages"),
                }
            )

        if isinstance(label, bool):
            # KTO：保留 messages + label 字段
            kto_rows.append(
                {
                    "messages": row.get("messages"),
                    "label": label,
                }
            )

    write_jsonl(output_sft_path, sft_rows)
    write_jsonl(output_kto_path, kto_rows)

    try:
        display_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output_path

    try:
        display_sft_path = output_sft_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_sft_path = output_sft_path

    try:
        display_kto_path = output_kto_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_kto_path = output_kto_path

    print(f"完成，写入 {len(rows)} 条记录到 {display_path}")
    print(f"完成，写入 {len(sft_rows)} 条记录到 {display_sft_path}")
    print(f"完成，写入 {len(kto_rows)} 条记录到 {display_kto_path}")


if __name__ == "__main__":
    main()