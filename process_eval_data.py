#!/usr/bin/env python3
"""
逐行处理 summary_eval_diff_data.xlsx 中的 last_answer_phone、query、data 三列，
使用 Claude Messages API 顺序推理（每行独立），结果保存至 summary_eval_diff_data_claude.xlsx
"""

import anthropic
import pandas as pd
import time

SYSTEM_PROMPT = (
    "你是一个专业的健康管理助手。"
    "请根据用户提供的个人健康数据，简洁、准确地回答用户的问题。"
    "回答语言与用户问题保持一致（中文）。"
    "不要编造数据中没有的内容。"
)


def build_messages(query: str, data: str, last_answer: str) -> list:
    """
    构造每行对应的 messages 列表。
    - 若存在历史回答，作为 assistant 消息插入（前面补一个占位 user 消息）。
    - 当前 query 和 data 合并为最终 user 消息。
    """
    messages = []

    if last_answer.strip():
        messages.append({
            "role": "user",
            "content": "[上一轮对话背景]"
        })
        messages.append({
            "role": "assistant",
            "content": last_answer.strip()
        })

    user_content = f"用户询问：{query.strip()}"
    if data.strip():
        user_content += f"\n\n相关个人健康数据：\n{data.strip()}"

    messages.append({
        "role": "user",
        "content": user_content
    })

    return messages


def call_claude(client: anthropic.Anthropic, messages: list, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages
            )
            # 兼容不同版本 SDK 的响应结构
            content = response.content
            if isinstance(content, str):
                return content
            texts = []
            for block in content:
                if hasattr(block, "type") and block.type == "text":
                    texts.append(block.text)
                elif hasattr(block, "text"):
                    texts.append(block.text)
            return "\n".join(texts)
        except anthropic.RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"  Rate limit，等待 {wait}s 后重试...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                wait = 10 * (attempt + 1)
                print(f"  服务器错误 {e.status_code}，等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                return f"[API ERROR {e.status_code}]: {e.message}"
        except Exception as e:
            print(f"  未知错误: {type(e).__name__}: {e}")
            return f"[ERROR]: {e}"
    return "[ERROR: 重试次数耗尽]"


def main():
    client = anthropic.Anthropic()

    # 1. 读取数据
    print("读取数据...")
    df = pd.read_excel("summary_eval_diff_data.xlsx")
    total = len(df)
    print(f"共 {total} 行数据，开始逐行推理...\n")

    answers = []

    for i, row in df.iterrows():
        query = str(row["query"]) if pd.notna(row["query"]) else ""
        data = str(row["data"]) if pd.notna(row["data"]) else ""
        last_answer = str(row["last_answer_phone"]) if pd.notna(row["last_answer_phone"]) else ""

        messages = build_messages(query, data, last_answer)
        answer = call_claude(client, messages)
        answers.append(answer)

        # 进度提示（每10行或最后一行）
        if (i + 1) % 10 == 0 or i + 1 == total:
            print(f"  进度: {i + 1}/{total}")

    # 2. 写入结果列并保存
    df["claude_answer"] = answers

    output_cols = ["last_answer_phone", "query", "data", "claude_answer"]
    output_df = df[output_cols].copy()

    output_path = "summary_eval_diff_data_claude.xlsx"
    output_df.to_excel(output_path, index=False)
    print(f"\n✓ 结果已保存至: {output_path}")
    print(f"  输出行数: {len(output_df)}, 列: {output_cols}")


if __name__ == "__main__":
    main()
