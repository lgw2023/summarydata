from __future__ import annotations

"""
对 sport_health_log2data_agent_result.xlsx 增加上一轮问答的“个人具体数据”标记列。

需求：
- 输入表包含多列，其中 last_query / last_answer 为上一轮问答。
- 用 LLM 判断该上一轮问答是否涉及“具体的个人运动/健康/生理/行为/作息等具体数据”。
- 新增一列 last_answer_personal=true/false。
- LLM 配置从 .env 获取（load_dotenv），本仓库用 src.utils.env.load_env() 封装。

用法示例：
  python scripts/sport_health_log2data_agent_result.py \
    --input sport_health_log2data_agent_result.xlsx
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from openai import DefaultHttpxClient, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import load_env  # noqa: E402


def _is_missing(v: Any) -> bool:
    try:
        return v is None or bool(pd.isna(v))
    except Exception:
        if isinstance(v, float) and v != v:
            return True
        return v is None


def _to_text(v: Any) -> str:
    if _is_missing(v):
        return ""
    return str(v).strip()


def _sha_key(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n---\n")
    return h.hexdigest()[:16]


def _extract_first_json_object(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if not t:
        return {}
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(t[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...\n（内容过长已截断）\n...\n{tail}"


def _resolve_llm_config() -> tuple[str, str, str]:
    """
    返回 (base_url, api_key, model_name)。

    优先从 .env 提供的 PERSONAL 专用变量读取：
      - LLM_MODEL_DSCHAT_URL / LLM_MODEL_DSCHAT_API_KEY / LLM_MODEL_DSCHAT_NAME
    回退到仓库常用的 judge / ground 变量：
      - LLM_MODEL_JUDGE_URL / LLM_MODEL_JUDGE_API_KEY / LLM_MODEL_JUDGE_NAME
      - LLM_MODEL_GROUND_URL / LLM_MODEL_GROUND_API_KEY / LLM_MODEL_GROUND_NAME
    """

    base_url = (os.getenv("LLM_MODEL_DSCHAT_URL") or "").strip()
    api_key = (os.getenv("LLM_MODEL_DSCHAT_API_KEY") or "").strip()
    model_name = (os.getenv("LLM_MODEL_DSCHAT_NAME") or "").strip()

    if not (base_url and api_key and model_name):
        base_url = base_url or (os.getenv("LLM_MODEL_JUDGE_URL") or "").strip()
        api_key = api_key or (os.getenv("LLM_MODEL_JUDGE_API_KEY") or "").strip()
        model_name = model_name or (os.getenv("LLM_MODEL_JUDGE_NAME") or "").strip()

    if not (base_url and api_key and model_name):
        base_url = base_url or (os.getenv("LLM_MODEL_GROUND_URL") or "").strip()
        api_key = api_key or (os.getenv("LLM_MODEL_GROUND_API_KEY") or "").strip()
        model_name = model_name or (os.getenv("LLM_MODEL_GROUND_NAME") or "").strip()

    if not base_url or not api_key or not model_name:
        raise ValueError(
            "缺少 LLM 配置。请至少设置以下环境变量之一组（推荐写入 .env）：\n"
            "- LLM_MODEL_DSCHAT_URL / LLM_MODEL_DSCHAT_API_KEY / LLM_MODEL_DSCHAT_NAME\n"
            "- LLM_MODEL_JUDGE_URL / LLM_MODEL_JUDGE_API_KEY / LLM_MODEL_JUDGE_NAME\n"
            "- LLM_MODEL_GROUND_URL / LLM_MODEL_GROUND_API_KEY / LLM_MODEL_GROUND_NAME\n"
        )

    return base_url, api_key, model_name


def _build_client(base_url: str, api_key: str) -> OpenAI:
    proxy = (os.getenv("LLM_HTTP_PROXY") or "").strip()
    if not proxy:
        proxy = (os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "").strip()
    if proxy:
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultHttpxClient(proxy=proxy),
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def infer_last_answer_personal_one(
    *,
    client: OpenAI,
    model_name: str,
    last_query: str,
    last_answer: str,
    max_chars: int = 24000,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    返回严格结构：
      {
        "last_answer_personal": true|false,
        "evidence": ["..."]   # 可选，短证据
      }
    """
    q = _clip_text((last_query or "").strip(), max_chars=max_chars)
    a = _clip_text((last_answer or "").strip(), max_chars=max_chars)

    system_prompt = (
        "你是一个严格的“个人具体数据识别器”。你的任务是判断：给定上一轮问答(last_query + last_answer)，"
        "是否涉及“具体的个人运动/健康/生理/行为/作息等的具体数据”。\n\n"
        "请只输出严格 JSON（不要任何多余文本），结构：\n"
        '{ "last_answer_personal": true|false, "evidence": ["..."] }\n\n'
        "判定规则（务必严格）：\n"
        "- true：出现了可被视为“个人记录/测量/日志”的具体信息，例如：步数、配速、距离、时长、心率、血压、血糖、体重、体脂、睡眠时长、入睡/起床时间、训练计划执行情况、具体日期/时间段的运动或作息、某次跑步/骑行/健身的详细数据等。可以是 last_query 或 last_answer 中出现。\n"
        "- false：只有泛泛的健康建议/运动科普/通用方案；或者只是说“我可以帮你查询/需要授权/没有数据”；或者只讨论功能不涉及任何个人记录；或者完全是闲聊。\n"
        "- “提到健康领域名词”本身不算具体数据，必须有个人化且具体的记录/数值/时间点/事件细节。\n"
        "- evidence 请给 0~2 条短证据（引用原文片段），如果为 false 可为空数组。\n"
    )

    user_prompt = (
        f"【last_query】\n{q}\n\n"
        f"【last_answer】\n{a}\n\n"
        "请按要求只输出 JSON。"
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = (resp.choices[0].message.content or "").strip()
            except Exception:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                content = (resp.choices[0].message.content or "").strip()

            obj = _extract_first_json_object(content)
            raw = obj.get("last_answer_personal", False)
            if isinstance(raw, str):
                raw = raw.strip().lower() in {"true", "1", "yes", "y"}
            val = bool(raw)
            ev = obj.get("evidence", [])
            if ev is None:
                ev = []
            if isinstance(ev, str):
                ev = [ev]
            if not isinstance(ev, list):
                ev = [str(ev)]
            ev = [str(x) for x in ev][:2]
            return {"last_answer_personal": val, "evidence": ev}
        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.8 * (2**attempt)))

    raise RuntimeError(f"LLM 调用失败（重试 {max_retries} 次后仍失败）: {last_err}") from last_err


def _read_jsonl_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            k = str(obj.get("key") or "").strip()
            v = obj.get("value")
            if k and isinstance(v, dict):
                cache[k] = v
    return cache


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="为 last_query/last_answer 增加 last_answer_personal 列。")
    p.add_argument("--input", type=str, default="sport_health_log2data_agent_result.xlsx", help="输入 xlsx 路径")
    p.add_argument("--output", type=str, default="", help="输出 xlsx 路径（默认自动命名）")
    p.add_argument("--sheet", type=str, default="0", help="sheet 名或索引（默认 0）")
    p.add_argument("--query-col", type=str, default="last_query", help="上一轮问题列名")
    p.add_argument("--answer-col", type=str, default="last_answer", help="上一轮回答列名")
    p.add_argument("--out-col", type=str, default="last_answer_personal", help="输出列名")
    p.add_argument("--workers", type=int, default=16, help="并发请求数（默认 4）")
    p.add_argument("--save-every", type=int, default=16, help="每处理 N 条保存一次（默认 30）")
    p.add_argument(
        "--cache",
        type=str,
        default="scripts/.cache_last_answer_personal.jsonl",
        help="jsonl 缓存路径（默认 scripts/.cache_last_answer_personal.jsonl）",
    )
    p.add_argument("--max-chars", type=int, default=24000, help="每个字段最大字符截断（默认 24000）")
    return p.parse_args()


def main() -> None:
    load_env()
    args = _parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    sheet: Any
    if str(args.sheet).isdigit():
        sheet = int(str(args.sheet))
    else:
        sheet = str(args.sheet)

    df = pd.read_excel(input_path, sheet_name=sheet, dtype=object)
    if args.query_col not in df.columns:
        raise ValueError(f"输入表缺少列: {args.query_col!r}")
    if args.answer_col not in df.columns:
        raise ValueError(f"输入表缺少列: {args.answer_col!r}")
    if args.out_col not in df.columns:
        df[args.out_col] = None

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}.with_last_answer_personal{input_path.suffix}")

    cache_path = Path(args.cache)
    cache = _read_jsonl_cache(cache_path)

    base_url, api_key, model_name = _resolve_llm_config()
    client = _build_client(base_url=base_url, api_key=api_key)

    # 收集需要推理的行
    todo: list[tuple[int, str, str, str]] = []
    for i in range(len(df)):
        existing = df.at[i, args.out_col] if args.out_col in df.columns else None
        if not _is_missing(existing):
            continue

        q = _to_text(df.at[i, args.query_col])
        a = _to_text(df.at[i, args.answer_col])
        if not q and not a:
            df.at[i, args.out_col] = False
            continue

        key = _sha_key([q, a])
        if key in cache and isinstance(cache[key], dict) and "last_answer_personal" in cache[key]:
            raw = cache[key].get("last_answer_personal", False)
            if isinstance(raw, str):
                raw = raw.strip().lower() in {"true", "1", "yes", "y"}
            df.at[i, args.out_col] = bool(raw)
            continue

        todo.append((i, key, q, a))

    if not todo:
        df.to_excel(output_path, index=False)
        print(f"\033[92m已完成：没有需要推理的行。输出文件: {output_path}\033[0m")
        return

    done = 0
    saved = 0
    save_every = max(1, int(args.save_every))

    def _work(row_idx: int, key: str, q: str, a: str) -> tuple[int, str, dict[str, Any]]:
        res = infer_last_answer_personal_one(
            client=client,
            model_name=model_name,
            last_query=q,
            last_answer=a,
            max_chars=int(args.max_chars),
        )
        return row_idx, key, res

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_work, row_idx, key, q, a) for (row_idx, key, q, a) in todo]
        for fut in as_completed(futs):
            row_idx, key, res = fut.result()
            val = bool(res.get("last_answer_personal", False))
            df.at[row_idx, args.out_col] = val
            cache[key] = res
            _append_jsonl(
                cache_path,
                {"key": key, "value": res, "row_idx": int(row_idx), "ts": int(time.time())},
            )
            done += 1

            if done - saved >= save_every:
                df.to_excel(output_path, index=False)
                saved = done
                print(f"已保存进度：{done}/{len(todo)} -> {output_path}")

    df.to_excel(output_path, index=False)
    print(f"\033[92m已完成：共推理 {done} 行。输出文件: {output_path}\033[0m")


if __name__ == "__main__":
    main()