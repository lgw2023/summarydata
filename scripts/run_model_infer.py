import json
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List

import argparse
from concurrent.futures import ThreadPoolExecutor
import requests

from prompts.response_prompt_v4 import SYSTEMT_PROMPT_PHONE_GENERAL  # noqa: E402
# ================= 默认参数（可通过命令行覆盖） =================
# vLLM 服务的基础地址：
#   vllm serve ... --served-model-name qwen32_sft --port 8000
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_MODEL_NAME = "qwen32_sft"

# 推理输入 jsonl 路径（每行一个样本，格式见 example_of_test_context）
# 通常是已有的 generated_responses.jsonl
DEFAULT_TEST_CONTEXT = "data/data_diff_sample/processed/generated_responses.jsonl"

# 生成时的一些参数
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_BATCH_SIZE = 4  # 这里只是读写上的 batch，与 vLLM server 无强约束


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用 vLLM OpenAI 接口，对 generated_responses.jsonl 追加一个开源模型候选。"
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default=DEFAULT_VLLM_BASE_URL,
        help=f"vLLM OpenAI 兼容接口基础地址（默认：{DEFAULT_VLLM_BASE_URL}）",
    )
    parser.add_argument(
        "--vllm-model-name",
        type=str,
        default=DEFAULT_VLLM_MODEL_NAME,
        help=f"vLLM 启动时的 served-model-name（默认：{DEFAULT_VLLM_MODEL_NAME}）",
    )
    parser.add_argument(
        "--test-context",
        type=str,
        default=DEFAULT_TEST_CONTEXT,
        help="输入 jsonl 文件路径（默认：data/data_diff_sample/processed/generated_responses.jsonl）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"单次生成的最大 token 数（默认：{DEFAULT_MAX_TOKENS}）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"采样温度（默认：{DEFAULT_TEMPERATURE}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"脚本读写层面的 batch 大小（默认：{DEFAULT_BATCH_SIZE}）",
    )
    return parser.parse_args()


example_of_test_context = {
    "sample_id": "1",
    "context": ".......",
    "question": "配速最快的那次跑了多少距离",
    "candidates": [
        {
            "candidate_id": "1::ref_think_phone::b8fbdf8f",
            "model_type": "reference",
            "model_name": "ref_think_phone",
            "response": "......",
            "gen_config": {"model_name": "ref_think_phone", "answer_key": "think_phone"},
        },
        {
            "candidate_id": "1::ref_answer_phone::a9ebed41",
            "model_type": "reference",
            "model_name": "ref_answer_phone",
            "response": "......",
            "gen_config": {"model_name": "ref_answer_phone", "answer_key": "answer_phone"},
        },
        # 下面省略若干候选，仅作为格式示例
    ],
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _call_vllm_chat(
    messages: List[Dict[str, str]],
    base_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """通过 vLLM OpenAI 兼容接口完成一次对话请求，返回模型回复文本。"""
    url = base_url.rstrip("/") + "/chat/completions"
    # 调试：打印 messages 中每个字典的 content 前 10 个字符和后 10 个字符
    for i, msg in enumerate(messages):
        content = str(msg.get("content", ""))
        head = content[:10]
        tail = content[-10:] if len(content) > 10 else content
        print(
            f"[DEBUG messages[{i}]] role={msg.get('role', '')} content={head}......{tail}".replace("\n", "")
        )
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"].replace("<think>\n\n</think>\n\n", "")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"vLLM 返回格式异常: {data}") from e


def _process_row(
    row: Dict[str, Any],
    model_type: str,
    model_name: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """处理单条样本：调用 vLLM 并把新候选追加到 candidates 中。"""
    context = str(row.get("context") or "")
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": SYSTEMT_PROMPT_PHONE_GENERAL})
    messages.append({"role": "user", "content": context})

    response_text = _call_vllm_chat(
        messages=messages,
        base_url=base_url,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    sample_id = str(row.get("sample_id") or "")
    new_candidate = {
        "candidate_id": f"{sample_id}::{model_name}::{uuid4().hex[:8]}",
        "model_type": model_type,
        "model_name": model_name,
        "response": response_text,
        "gen_config": {
            "model_name": model_name,
            "base_url": base_url,
        },
    }

    # candidates = list(row.get("candidates") or [])
    candidates = list([])
    candidates.append(new_candidate)
    row["candidates"] = candidates
    return row


def main() -> None:
    """从输入 jsonl 中读取样本，调用 vLLM 接口生成回复并写回。"""
    args = _parse_args()

    input_path = Path(args.test_context)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    rows = _read_jsonl(input_path)

    # 输出文件规则：
    # - 若输入名为 generated_responses.jsonl，则输出名为 generated_responses.jsonl；
    # - 否则报错（如需其它规则，可自行扩展）。
    model_type = "open_source"
    model_name = args.vllm_model_name
    if input_path.name == "generated_responses.jsonl":
        output_path = input_path.with_name(f"generated_responses_{model_name}.jsonl")
        print(f"Output path: {output_path}")
    else:
        raise ValueError(f"输入文件名 {input_path.name} 不合法，只接受 generated_responses.jsonl")

    updated_rows: List[Dict[str, Any]] = []

    # 使用线程池并发调用 vLLM，每次最多并发 2 个请求
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]

            # 这里对 batch 内的样本做并发推理；线程池会限制全局最大并发为 2
            processed_batch = list(
                executor.map(
                    lambda r: _process_row(
                        r,
                        model_type=model_type,
                        model_name=model_name,
                        base_url=args.vllm_base_url,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    ),
                    batch,
                )
            )
            updated_rows.extend(processed_batch)

    _write_jsonl(output_path, updated_rows)
    print(f"Done. Wrote {len(updated_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

# cd /data1/liguowei/summarydata && python scripts/run_model_infer.py --vllm-model-name qwen32_lora_sft_kto --vllm-base-url http://localhost:8000/v1
