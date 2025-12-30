# import json

# def trim_leaf_strings(obj):
#     """
#     递归处理任意嵌套结构：
#     - 如果是 dict：递归处理 value
#     - 如果是 list：递归处理每个元素
#     - 如果是 str：替换为“前 10 个字 + 后 10 个字”（长度 <= 20 就原样返回）
#     - 其他类型：原样返回
#     """
#     if isinstance(obj, dict):
#         return {k: trim_leaf_strings(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [trim_leaf_strings(item) for item in obj]
#     elif isinstance(obj, str):
#         return obj[:10] + obj[-10:] if len(obj) > 20 else obj
#     else:
#         return obj


# def main(jsonl_path: str):
#     # 读取第一个样本（第一行）
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         first_line = f.readline().strip()

#     if not first_line:
#         print("文件为空或第一行是空行")
#         return

#     # 解析 JSON
#     data = json.loads(first_line)

#     # 处理字符串叶子节点
#     processed = trim_leaf_strings(data)

#     # 打印结果
#     print(json.dumps(processed, ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     # 换成你的 jsonl 文件路径
#     file_path = "data/data_diff_sample/processed/judge_results_kto.jsonl"
#     main(file_path)


from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer


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


DEFAULT_TOKENIZER_NAME = "/data1/Qwen3_32B"


def _count_tokens(messages: List[Dict[str, Any]], tokenizer: Any) -> int:
    """
    使用 Qwen/Qwen3-32B tokenizer 估算一条多轮对话 messages 的 token 长度。
    优先使用 chat template；失败时退化为简单拼接文本。
    """
    if tokenizer is None:
        return 0

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


def main() -> None:
    # 初始化 Qwen/Qwen3-32B tokenizer，用于估算每条样本的 token 长度
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_TOKENIZER_NAME,
        trust_remote_code=True,
    )
    path = '/data1/jiacheng/KTO_demo/data/judge_results_kto_8answers_discriminantia_sampling_v5_QAL_msswift_context.jsonl'
    rows = read_jsonl(path)
    for i in rows:
        messages = i.get("messages") or []
        token_len, roles = _count_tokens(messages, tokenizer)
        print(token_len, roles)

if __name__ == "__main__":
    main()