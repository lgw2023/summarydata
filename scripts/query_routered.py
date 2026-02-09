from __future__ import annotations

"""
根据 query + data 反推“合理的提问时间”（query_time）。

使用方式（默认读写根目录的 query_routered.xlsx）：
  python scripts/query_routered.py

常用参数：
  --input query_routered.xlsx
  --output query_routered.with_query_time.xlsx
  --query-col query
  --data-col data
  --last-answer-col last_answer_phone
  --out-col query_time
  --id-col row_id
  --workers 1
  --save-every 20
  --cache scripts/.cache_query_time.jsonl
  --default-year 2025

LLM 连接信息优先从以下环境变量读取（推荐放到 .env）：
  - LLM_MODEL_QUERYTIME_URL / LLM_MODEL_QUERYTIME_API_KEY / LLM_MODEL_QUERYTIME_NAME
若未配置，则回退到：
  - LLM_MODEL_JUDGE_URL / LLM_MODEL_JUDGE_API_KEY / LLM_MODEL_JUDGE_NAME
再回退到（如果你想复用 KTO judge 配置）：
  - LLM_MODEL_GROUND_URL / LLM_MODEL_GROUND_API_KEY / LLM_MODEL_GROUND_NAME

可选代理：
  - LLM_HTTP_PROXY（优先）
  - HTTPS_PROXY / HTTP_PROXY（回退）
"""

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI, DefaultHttpxClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import load_env  # noqa: E402


_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_MD_RE = re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?!\d)")
_HMS_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)")


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
        h.update(p.encode("utf-8", errors="ignore"))
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


def _resolve_llm_config() -> tuple[str, str, str]:
    """
    返回 (base_url, api_key, model_name)。
    """
    # 优先：专用于本脚本的配置
    base_url = (os.getenv("LLM_MODEL_QUERYTIME_URL") or "").strip()
    api_key = (os.getenv("LLM_MODEL_QUERYTIME_API_KEY") or "").strip()
    model_name = (os.getenv("LLM_MODEL_QUERYTIME_NAME") or "").strip()

    # 回退：此前 classify_personal_data_llm_check.py 的配置
    if not (base_url and api_key and model_name):
        base_url = base_url or (os.getenv("LLM_MODEL_JUDGE_URL") or "").strip()
        api_key = api_key or (os.getenv("LLM_MODEL_JUDGE_API_KEY") or "").strip()
        model_name = model_name or (os.getenv("LLM_MODEL_JUDGE_NAME") or "").strip()

    # 再回退：KTO judge 常用的 GROUND 配置
    if not (base_url and api_key and model_name):
        base_url = base_url or (os.getenv("LLM_MODEL_GROUND_URL") or "").strip()
        api_key = api_key or (os.getenv("LLM_MODEL_GROUND_API_KEY") or "").strip()
        model_name = model_name or (os.getenv("LLM_MODEL_GROUND_NAME") or "").strip()

    if not base_url or not api_key or not model_name:
        raise ValueError(
            "缺少 LLM 配置。请至少设置以下环境变量之一组：\n"
            "- LLM_MODEL_QUERYTIME_URL / LLM_MODEL_QUERYTIME_API_KEY / LLM_MODEL_QUERYTIME_NAME\n"
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


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...\n（内容过长已截断）\n...\n{tail}"


def _contains_year(text: str) -> bool:
    return bool(_YEAR_RE.search(text or ""))


def _format_ts(ts: Any) -> str | None:
    """
    将可解析的 datetime/Timestamp 统一格式化为 'YYYY-MM-DD HH:MM:SS'。
    """
    if ts is None:
        return None
    try:
        # pandas Timestamp
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
    except Exception:
        pass
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return None


def _try_parse_dt(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        dt = pd.to_datetime(s, errors="coerce")
    except Exception:
        return None
    if dt is None or bool(pd.isna(dt)):
        return None
    try:
        return str(dt.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        try:
            return str(dt.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            return None


def _extract_latest_month_day(text: str) -> tuple[int, int] | None:
    """
    从文本中抽取所有 M/D 或 MM/DD，返回“最大”的那一个（按 month*100+day）。
    """
    best: tuple[int, int] | None = None
    best_score = -1
    for m_s, d_s in _MD_RE.findall(text or ""):
        try:
            m = int(m_s)
            d = int(d_s)
        except Exception:
            continue
        if not (1 <= m <= 12 and 1 <= d <= 31):
            continue
        score = m * 100 + d
        if score > best_score:
            best_score = score
            best = (m, d)
    return best


def _normalize_query_time_string(qt: str, default_year: int) -> str | None:
    """
    尝试将各种可能的输出归一为 'YYYY-MM-DD HH:MM:SS'。
    重点处理“缺少年份”的场景：如 '2/16 23:59:59' 或 '02-16'。
    """
    s = (qt or "").strip()
    if not s:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", s):
        return s
    # 若包含年份，优先交给解析器
    if _contains_year(s):
        return _try_parse_dt(s) or s

    # 无年份：自己抽 month/day + time
    md = _extract_latest_month_day(s)
    if not md:
        return None
    m, d = md
    hm = _HMS_RE.search(s)
    if hm:
        hh = int(hm.group(1))
        mm = int(hm.group(2))
        ss = int(hm.group(3) or "0")
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            hh, mm, ss = 23, 59, 59
        t = f"{hh:02d}:{mm:02d}:{ss:02d}"
    else:
        t = "23:59:59"
    return f"{int(default_year):04d}-{m:02d}-{d:02d} {t}"


def _fill_query_time_fallback(
    *,
    query_time: str | None,
    time_end_hint: str,
    data: str,
    default_year: int,
    should_default_year: bool,
) -> str | None:
    """
    当 LLM 输出 query_time 为 null / 或缺少年份时，做一次保守兜底：
    - 若有 time_end_hint：直接用它（通常已经含 YYYY）
    - 否则在 data 中找最新的 M/D，用 default_year 补齐
    """
    if isinstance(query_time, str) and query_time.strip():
        norm = _normalize_query_time_string(query_time, default_year=default_year)
        if norm:
            return norm
        # 兜底：原样返回（不阻断写盘）
        return query_time.strip()

    # query_time 为 None：只有在“应默认年份”时才尝试用 data/time_end 补
    if not should_default_year:
        return None

    te = _try_parse_dt(time_end_hint)
    if te:
        return te

    md = _extract_latest_month_day(data)
    if md:
        m, d = md
        return f"{int(default_year):04d}-{m:02d}-{d:02d} 23:59:59"
    return None


def infer_query_time_one(
    *,
    client: OpenAI,
    model_name: str,
    query: str,
    data: str,
    last_answer_phone: str = "",
    timezone: str,
    time_start_hint: str = "",
    time_end_hint: str = "",
    include_window_hints: bool = True,
    max_data_chars: int = 0,
    default_year: int = 2025,
) -> dict[str, Any]:
    """
    输出结构（严格 JSON）：
      {
        "query_time": "YYYY-MM-DD HH:MM:SS" | null,
        "timezone": "Asia/Shanghai" | "UTC+8" | ...,
        "confidence": 0.0-1.0,
        "evidence": ["..."]  # 证据片段（短）
      }
    """
    q = (query or "").strip()
    # 默认使用“全量 data/last_answer_phone”，不做截断。
    # 说明：`max_data_chars` 参数仅为兼容保留，不再用于截断逻辑。
    d = (data or "").strip()
    last_a = (last_answer_phone or "").strip()

    # 年份默认规则：仅当 query 与 data 都没有出现年份时启用
    should_default_year = (not _contains_year(q)) and (not _contains_year(d))

    system_prompt = (
        "你是一个严谨的时间标注助手。你的任务：根据用户问题(query)与已抓取到的个人健康数据(data)的内容，"
        "反向推理出“用户最可能提问的时间(query_time)”。\n\n"
        "要求：\n"
        "- 只输出严格 JSON（不要任何多余文本）。\n"
        "- query_time 统一输出为 'YYYY-MM-DD HH:MM:SS'（24小时制）。\n"
        "- 如果只能确定日期，时间部分用 '23:59:59' 作为该日结束时刻。\n"
        "- 如果年份缺失但能从 data 的上下文推断（例如同一段数据出现了明确年份/或时间窗口暗示），请补全年份。\n"
        f"- 若 query 与 data 都不包含年份信息，则年份默认使用 {int(default_year)} 年（不要因为缺少年份而返回 null）。\n"
        "- 若提供了 time_start/time_end（代表系统抓取数据的时间窗口），query_time 应当与该窗口相一致：\n"
        "  - 通常 query_time 应接近 time_end（很多系统会把查询时间取为“当日结束”以便取齐昨日/昨晚数据）。\n"
        "  - 避免输出明显早于 time_start 或晚于 time_end 很多的时间。\n"
        "- 如果无法合理推断，query_time 输出 null，confidence=0，并给出 evidence 说明缺失点。\n"
        "- timezone 需返回一个明确时区（默认按中国北京时间 Asia/Shanghai / UTC+8）。\n"
    )

    window_hint = ""
    if include_window_hints and (time_start_hint.strip() or time_end_hint.strip()):
        window_hint = (
            "\n【已知信息：本次抓取返回的数据时间窗口（可能来自系统侧计算）】\n"
            f"- time_start: {time_start_hint.strip() or '（空）'}\n"
            f"- time_end: {time_end_hint.strip() or '（空）'}\n"
        )

    last_answer_hint = ""
    if last_a:
        last_answer_hint = f"\n【上一轮助手回复 last_answer_phone】\n{last_a}\n"

    user_prompt = (
        f"【query】\n{q}\n\n"
        f"{last_answer_hint}\n"
        f"【data】\n{d}\n"
        f"{window_hint}\n"
        f"【年份默认规则】\n- query 与 data 都不含年份 -> 默认年份={int(default_year)}\n\n"
        "请输出 JSON，字段如下：\n"
        '{\n  "query_time": "YYYY-MM-DD HH:MM:SS" 或 null,\n  "timezone": "Asia/Shanghai",\n  "confidence": 0.0,\n  "evidence": ["..."]\n}\n'
        "其中 evidence 请引用 data / query / last_answer_phone 中的关键时间线索（尽量短）。"
    ).strip()

    # 优先尝试 response_format=json_object；若下游服务不支持则回退
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

    # 轻量规范化，保证下游写盘稳定
    out: dict[str, Any] = {}
    qt = obj.get("query_time", None)
    if isinstance(qt, str):
        qt = qt.strip()
        qt = qt or None
    # 兜底：缺少年份时默认 2025；或 LLM 输出 null 时尝试用 time_end/data 补齐
    out["query_time"] = _fill_query_time_fallback(
        query_time=qt,
        time_end_hint=time_end_hint if include_window_hints else "",
        data=d,
        default_year=int(default_year),
        should_default_year=should_default_year,
    )

    tz = obj.get("timezone", None)
    if not isinstance(tz, str) or not tz.strip():
        tz = timezone
    out["timezone"] = str(tz).strip()

    conf = obj.get("confidence", 0.0)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    out["confidence"] = max(0.0, min(1.0, conf_f))

    ev = obj.get("evidence", [])
    if ev is None:
        ev = []
    if isinstance(ev, str):
        ev = [ev]
    if not isinstance(ev, list):
        ev = [str(ev)]
    out["evidence"] = [str(x)[:200] for x in ev if str(x).strip()][:10]

    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据 query+data 反推 query_time（调用 LLM）")
    p.add_argument("--input", type=str, default="query_routered.xlsx")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--sheet", type=int, default=0)
    p.add_argument("--id-col", type=str, default="row_id")
    p.add_argument("--query-col", type=str, default="query")
    p.add_argument("--data-col", type=str, default="data")
    p.add_argument(
        "--last-answer-col",
        type=str,
        default="last_answer_phone",
        help="若该列存在且非空，会与 query/data 一起作为上下文传给 LLM",
    )
    p.add_argument("--out-col", type=str, default="query_time")
    p.add_argument("--meta-col", type=str, default="query_time_llm_meta")
    p.add_argument("--force", action="store_true", help="即使 out-col 非空也强制重算")
    p.add_argument("--max-rows", type=int, default=0, help="0 表示全量")
    p.add_argument("--workers", type=int, default=16, help="并发数（建议从 1 开始）")
    p.add_argument("--save-every", type=int, default=16, help="每处理 N 行就保存一次输出")
    p.add_argument("--cache", type=str, default="scripts/.cache_query_time.jsonl")
    p.add_argument("--timezone", type=str, default="Asia/Shanghai")
    p.add_argument(
        "--max-data-chars",
        type=int,
        default=0,
        help="已废弃：默认传全量 data，不再按字符数截断（为兼容旧命令保留该参数）",
    )
    p.add_argument("--no-window-hints", action="store_true", help="不把 time_start/time_end 作为提示传给模型")
    p.add_argument("--default-year", type=int, default=2025, help="当 query 与 data 都不含年份时默认使用的年份")
    return p.parse_args()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
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
            if not k or not isinstance(v, dict):
                continue
            cache[k] = v
    return cache


def _append_cache(path: Path, key: str, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def _safe_excel_write(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)


def _normalize_row_id(v: Any, fallback: int) -> str:
    if _is_missing(v):
        return str(fallback)
    if isinstance(v, float) and v == v and v.is_integer():
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    s = str(v).strip()
    return s or str(fallback)


def main() -> None:
    load_env()
    args = _parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {in_path}")

    out_path = Path(args.output) if args.output.strip() else in_path.with_name(f"{in_path.stem}.with_query_time{in_path.suffix}")
    cache_path = Path(args.cache)

    df = pd.read_excel(in_path, sheet_name=args.sheet, dtype=object, nrows=(None if args.max_rows == 0 else int(args.max_rows)))
    if args.query_col not in df.columns or args.data_col not in df.columns:
        raise ValueError(
            f"输入文件缺少必要列：需要 {args.query_col!r}, {args.data_col!r}；"
            f"实际列为：{list(df.columns)}"
        )

    # 确保输出列存在
    if args.out_col not in df.columns:
        df[args.out_col] = None
    if args.meta_col and args.meta_col not in df.columns:
        df[args.meta_col] = None

    base_url, api_key, model_name = _resolve_llm_config()
    client = _build_client(base_url=base_url, api_key=api_key)

    include_window_hints = not bool(args.no_window_hints)
    has_time_start = "time_start" in df.columns
    has_time_end = "time_end" in df.columns
    has_last_answer = bool(args.last_answer_col) and (args.last_answer_col in df.columns)

    cache = _load_cache(cache_path)

    # 任务列表：先挑出需要处理的行（支持断点续跑）
    todo_indices: list[int] = []
    for i in range(len(df)):
        cur = df.at[i, args.out_col]
        if not args.force and _to_text(cur):
            continue
        q = _to_text(df.at[i, args.query_col])
        if not q:
            continue
        todo_indices.append(i)

    if not todo_indices:
        _safe_excel_write(df, out_path)
        print(f"无需处理：已写出 {out_path}")
        return

    # 并发执行：每个任务内部仍是一次 chat 请求
    workers = max(1, int(args.workers))

    def _task(i: int) -> tuple[int, dict[str, Any], str]:
        query = _to_text(df.at[i, args.query_col])
        data = _to_text(df.at[i, args.data_col])
        last_answer_phone = _to_text(df.at[i, args.last_answer_col]) if has_last_answer else ""
        ts = _to_text(df.at[i, "time_start"]) if (include_window_hints and has_time_start) else ""
        te = _to_text(df.at[i, "time_end"]) if (include_window_hints and has_time_end) else ""

        cache_key = _sha_key([query, data, last_answer_phone, ts, te, args.timezone, str(int(args.default_year))])
        if cache_key in cache:
            return i, cache[cache_key], cache_key

        val = infer_query_time_one(
            client=client,
            model_name=model_name,
            query=query,
            data=data,
            last_answer_phone=last_answer_phone,
            timezone=args.timezone,
            time_start_hint=ts,
            time_end_hint=te,
            include_window_hints=include_window_hints,
            max_data_chars=int(args.max_data_chars),
            default_year=int(args.default_year),
        )
        cache[cache_key] = val
        _append_cache(cache_path, cache_key, val)
        return i, val, cache_key

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_task, i): i for i in todo_indices}
        for fut in as_completed(futures):
            i, val, _k = fut.result()
            df.at[i, args.out_col] = val.get("query_time")
            if args.meta_col:
                df.at[i, args.meta_col] = json.dumps(val, ensure_ascii=False)
            completed += 1
            if args.save_every > 0 and completed % int(args.save_every) == 0:
                _safe_excel_write(df, out_path)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 已处理 {completed}/{len(todo_indices)}，已保存到 {out_path}")

    _safe_excel_write(df, out_path)
    print(f"完成：共处理 {len(todo_indices)} 行，输出 {out_path}，缓存 {cache_path}")


if __name__ == "__main__":
    main()

