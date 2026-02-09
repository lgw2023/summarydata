from __future__ import annotations

import sys
import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING
import pandas as pd
from openai import OpenAI, DefaultHttpxClient
import tiktoken
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_clean import *  # noqa: F403
from src.data_clean import __all__  # noqa: F401
from src.utils.env import load_env

if TYPE_CHECKING:  # pragma: no cover
    from src.data_clean.models import PersonalDataPattern

from classify_personal_data import get_patterns_all
from classify_personal_data import aggregate_patterns_to_dataframes
from classify_personal_data import aggregate_dataframes_to_table
from classify_personal_data import aggregate_patterns_to_formatted_text
from classify_personal_data import aggregate_patterns_by_timejson
from classify_personal_data import aggregate_patterns_to_dataline_text

try:
    load_env()
except Exception:
    pass

def get_encoding(model: str):
    """
    获取与模型匹配的 tokenizer（tiktoken encoding）
    优先用 encoding_for_model；不支持时回退到 o200k_base
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # 常见回退（新一代模型通常用 o200k_base）
        return tiktoken.get_encoding("o200k_base")

def count_tokens_text(text: str, model: str | None = None) -> int:
    """统计纯文本 token 数（最常用）"""
    m = (model or "").strip()
    if not m:
        # 允许在未显式传入 model 时，使用全局 model_name（若存在）
        try:
            m = str(globals().get("model_name") or "").strip()
        except Exception:
            m = ""
    if not m:
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    enc = get_encoding(m)
    return len(enc.encode(text))

def _clip(text1: Any, text2: Any, max_tokens_total: int) -> tuple[str, str]:
    s1 = "" if text1 is None else str(text1)
    s2 = "" if text2 is None else str(text2)
    if count_tokens_text(s1) + count_tokens_text(s2) <= max_tokens_total:
        return s1, s2
    return "", ""

def _df_to_text(df: pd.DataFrame) -> str:
    try:
        return df.to_string(index=False)
    except Exception:
        return str(df)

def _dfs_to_text(dfs: list[pd.DataFrame]) -> str:
    parts: list[str] = []
    for j, df in enumerate(dfs):
        parts.append(
            f"[DataFrame {j}] \n{_df_to_text(df)}"
        )
    return "\n\n".join(parts).strip()

def _call_llm_compare(client: OpenAI, model_name: str, max_tokens_total: int, person_text: str, rebuild_text: str, system_prompt: str) -> dict[str, Any]:
    person_text, rebuild_text = _clip(person_text, rebuild_text, max_tokens_total)
    if not person_text or not rebuild_text:
        return {"same": True, "missing_info": ["超过上下文限制"]}

    user_prompt = (
        "【原始个人数据文本 person_datas】\n"
        f"{person_text}\n\n"
        "【重构结果文本】\n"
        f"{rebuild_text}\n\n"
        "请判断重构结果文本是否遗漏了原文中出现的指标记录。只输出 JSON。"
    )

    # 优先尝试 response_format=json_object；若下游服务不支持，则回退到纯文本解析
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

    # 解析 JSON（若夹杂额外文本，则尝试抽取第一个 { ... }）
    obj: dict[str, Any]
    try:
        obj = json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(content[start : end + 1])
        else:
            obj = {"same": True, "missing_info": ["（LLM 回复无法解析为 JSON） 无单位"]}

    # 轻量规范化字段，保证下游稳定
    check_val = obj.get("same", False)
    if isinstance(check_val, str):
        check_val = check_val.strip().lower() in {"true", "1", "yes", "y"}
    obj["same"] = bool(check_val)

    mi = obj.get("missing_info", [])
    if mi is None:
        mi = []
    if isinstance(mi, str):
        mi = [mi]
    if not isinstance(mi, list):
        mi = [str(mi)]
    obj["missing_info"] = [str(x) for x in mi][:10]
    return obj


def _time_jsons_to_text(time_jsons: list[dict[str, Any]]) -> str:
    """
    将 aggregate_time 的结构化输出压缩成“可读文本”，用于 LLM 对比是否漏指标。
    重点保留：时间桶(label) + events 的 name/value/unit/status + fallback（若有）。
    """
    if not time_jsons:
        return "（空）"

    def _safe(x: Any) -> str:
        try:
            return str(x if x is not None else "").strip()
        except Exception:
            return ""

    parts: list[str] = []
    for i, bucket in enumerate(time_jsons):
        t = bucket.get("time") or {}
        label = _safe(t.get("label")) or _safe(t.get("date")) or _safe(t.get("start")) or _safe(t.get("type"))
        parts.append(f"{label}".strip())

        events = bucket.get("events") or []
        if isinstance(events, list) and events:
            for ev in events:
                if not isinstance(ev, dict):
                    parts.append(f"- （event 非 dict）{_safe(ev)}")
                    continue
                name = _safe(ev.get("name"))
                value = ev.get("value")
                if isinstance(value, list):
                    value_s = " | ".join(_safe(v) for v in value if _safe(v))
                else:
                    value_s = _safe(value)
                unit = _safe(ev.get("unit")) or "无单位"
                status = _safe(ev.get("status"))
                logic = _safe(ev.get("logic"))
                diff_value = _safe(ev.get("diff_value"))
                extra = ""
                if logic or diff_value:
                    extra = f"（{logic}{diff_value}）".strip()
                seg = " ".join(x for x in [name, value_s, unit, status] if x).strip()
                if extra:
                    seg = f"{seg} {extra}".strip()
                parts.append(f"- {seg}".rstrip())

        fallback = bucket.get("fallback")
        if fallback:
            if isinstance(fallback, list):
                for fb in fallback[:20]:
                    parts.append(f"- {_safe(fb)}")
            else:
                parts.append(f"- {_safe(fallback)}")

        # summary = bucket.get("summary")
        # if isinstance(summary, str) and summary.strip():
        #     parts.append(f"[summary] {summary.strip()}")

        parts.append("")

    return "\n".join(parts).strip()

def check_aggregate_dataframes_by_llm(
    person_datas: list[str],
    client: OpenAI,
    model_name: str,
    max_tokens_total: int,
    skip_llm_check: bool = False,
) -> tuple[
    list[list[Any]],
    list[list[pd.DataFrame]],
    list[pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    对聚合结果做 LLM 一致性校验（检查 person_datas 与聚合表是否存在信息遗漏）。

    返回：
    - patterns_all: list[list[PersonalDataPattern]]
    - dataframes: list[list[pd.DataFrame]]
    - wide_tables: list[pd.DataFrame]
    - check_dataframes: list[dict]（每条样本一份，来自 LLM 的 JSON 解析结果）
    - check_wide_tables: list[dict]（每条样本一份，来自 LLM 的 JSON 解析结果）
    """

    #########################################################################################################
    # 解析：一串单行或者多行的个人数据文本（包括多指标数据） -> 若干 PersonalDataPattern 列表
    # 入口在 src/data_clean/parse.py：explode_newlines_and_route_to_dataclasses
    if not person_datas:
        raise ValueError("person_datas 不能为空")

    strict_uncovered_to_unparsed = True
    include_loose_lines = True

    # 关键优化点：每条样本只做一次 parse，后续所有聚合都复用 patterns
    patterns_all = [get_patterns_all(t, strict_uncovered_to_unparsed=strict_uncovered_to_unparsed) for t in person_datas]

    #########################################################################################################
    # 1) aggregate_dataframe：输出 DataFrame 列表 + 元信息（df.attrs）
    # 解析：若干 PersonalDataPattern 列表 -> DataFrame 列表
    dataframes = [
        aggregate_patterns_to_dataframes(pats, include_loose_lines=include_loose_lines)
        for pats in patterns_all
    ]
    
    # 1) aggregate_dataframe -> 单表（宽 schema）
    # 说明：dataframes 的结构是 list[list[DataFrame]]，其中 dataframes 的长度 == 输入样本条数；
    # 对每条样本，把其分表 dataframes[i] 合并为一张宽表（列并集），因此输出宽表数量也应与 dataframes 长度相同。
    wide_tables = [
        aggregate_dataframes_to_table(dfs, include_loose_lines=include_loose_lines)
        for dfs in dataframes
    ]

    #########################################################################################################
    # 2) 使用 LLM 校验：person_datas vs dataframes / person_datas vs wide_tables
    # 说明：要求分开做两轮校验，输出 check_dataframes 与 check_wide_tables（list[dict]）
    system_prompt = (
        "你是一个严谨的“数据一致性审查员”。你的任务是比较：\n"
        "- 原始个人数据文本\n"
        "- 重构结果文本\n"
        "判断重构结果文本是否遗漏了原文中出现的指标记录。\n\n"
        "输出必须是严格 JSON（不要包含任何多余文字）。JSON 结构要求：\n"
        "{\n"
        '  "same": true|false,\n'
        '  "missing_info": ["指标名称 数据 单位", "..."]\n'
        "}\n\n"
        "判定规则：\n"
        "- 若未发现遗漏：same=true，missing_info=[]\n"
        "- 若发现遗漏：same=false，missing_info 只需给出少量样例（例如 1~5 条），不要求穷尽。\n"
        "- missing_info 每条按“指标名称 数据 单位”拼接；若原文缺单位则用“无单位”。\n"
        "- 指标名称可按原始个人数据照写，不要因为格式差异误判。\n"
        "- 重构结果文本里若有同一指标但单位不同，不算信息遗漏。\n"
    )


    check_dataframes: list[dict[str, Any]] = []
    check_wide_tables: list[dict[str, Any]] = []
    rebuild_dataframes_texts: list[str] = []
    rebuild_wide_tables_texts: list[str] = []

    for i, person_text in enumerate(person_datas):
        dfs = dataframes[i] if i < len(dataframes) else []
        # 移除列名 'data_type'（如果存在）
        for j, df in enumerate(dfs):
            if hasattr(df, "columns") and "data_type" in df.columns:
                dfs[j] = df.drop(columns=["data_type"])
        wt = wide_tables[i] if i < len(wide_tables) else pd.DataFrame()
        cols_to_remove = {"entity_type", "data_type", "title", "table_idx", "row_idx"}
        if not wt.empty:
            remaining = [col for col in wt.columns if col not in cols_to_remove]
            wt = wt[remaining]

        dfs_text = _dfs_to_text(dfs) if dfs else "（空）"
        wt_text = _df_to_text(wt) if wt is not None else "（空）"
        wt_text = wt_text.replace("不适用", " ")
        rebuild_dataframes_texts.append(dfs_text)
        rebuild_wide_tables_texts.append(wt_text)

        # if i not in [108, 47, 105, 25]:
        #     continue
        # if i in [108, 47, 105, 25]:
        #     print((f"【原始个人数据文本 person_datas】【{i}】\n{person_text}\n"))
        #     print(("【聚合结果 dfs_text】\n" f"{dfs_text}\n"))
        #     print(("【聚合结果 wt_text】\n" f"{wt_text}\n"))

        if not skip_llm_check:
            res_dfs = _call_llm_compare(
                client=client,
                model_name=model_name,
                max_tokens_total=max_tokens_total,
                person_text=person_text,
                rebuild_text=dfs_text,
                system_prompt=system_prompt,
                )
            check_dataframes.append(res_dfs)

            res_wide_tables = _call_llm_compare(
                client=client,
                model_name=model_name,
                max_tokens_total=max_tokens_total,
                person_text=person_text,
                rebuild_text=wt_text,
                system_prompt=system_prompt,
            )
            check_wide_tables.append(res_wide_tables)
            if not res_dfs["same"] or not res_wide_tables["same"]:
                print((
                f"【原始个人数据文本 person_datas】【{i}】\n"
                f"{person_text}\n"
                ))
                print(f"\033[92m分表 DataFrames（list[pd.DataFrame]） 存在信息遗漏: {res_dfs['missing_info']}\033[0m")
                print((
                "【聚合结果表格分表 DataFrames（list[pd.DataFrame]）】\n"
                f"{dfs_text}\n"
                ))
                print(f"\033[92m宽表 WideTable（pd.DataFrame） 存在信息遗漏: {res_wide_tables['missing_info']}\033[0m")
                print((
                "【聚合结果表格 WideTable（pd.DataFrame）】\n"
                f"{wt_text}\n"
                ))
                raise ValueError(f"【{i}】分表 DataFrames（list[pd.DataFrame]） 或 宽表 WideTable（pd.DataFrame） 存在信息遗漏")
            else:
                print(f"\033[92m【{i}】分表 DataFrames（list[pd.DataFrame]） 没有信息遗漏\033[0m")
                print(f"\033[92m【{i}】宽表 WideTable（pd.DataFrame） 没有信息遗漏\033[0m")

    if skip_llm_check:
        return (
            patterns_all,
            dataframes,
            wide_tables,
            check_dataframes,
            check_wide_tables,
            rebuild_dataframes_texts,
            rebuild_wide_tables_texts,
        )
    return patterns_all, dataframes, wide_tables, check_dataframes, check_wide_tables

def check_aggregate_markdown_tables_by_llm(
    person_datas: list[str],
    client: OpenAI,
    model_name: str,
    max_tokens_total: int,
    skip_llm_check: bool = False,
) -> tuple[
    list[list[Any]],
    list[str],
    list[dict[str, Any]],
]:
    """
    对 aggregate_format 输出的 Markdown 表格做 LLM 一致性校验（检查 person_datas 与 markdown_tables 是否存在信息遗漏）。

    返回：
    - patterns_all: list[list[PersonalDataPattern]]
    - markdown_tables: list[str]
    - check_markdown_tables: list[dict]（每条样本一份，来自 LLM 的 JSON 解析结果）
    """
    if not person_datas:
        raise ValueError("person_datas 不能为空")

    strict_uncovered_to_unparsed = True
    include_loose_lines = True

    patterns_all = [get_patterns_all(t, strict_uncovered_to_unparsed=strict_uncovered_to_unparsed) for t in person_datas]
    markdown_tables = [
        aggregate_patterns_to_formatted_text(pats, include_loose_lines=include_loose_lines)
        for pats in patterns_all
    ]

    system_prompt = (
        "你是一个严谨的“数据一致性审查员”。你的任务是比较：\n"
        "- 原始个人数据文本\n"
        "- 重构结果文本\n"
        "判断重构结果文本是否遗漏了原文中出现的指标记录。\n\n"
        "输出必须是严格 JSON（不要包含任何多余文字）。JSON 结构要求：\n"
        "{\n"
        '  "same": true|false,\n'
        '  "missing_info": ["指标名称 数据 单位", "..."]\n'
        "}\n\n"
        "判定规则：\n"
        "- 若未发现遗漏：same=true，missing_info=[]\n"
        "- 若发现遗漏：same=false，missing_info 只需给出少量样例（例如 1~5 条），不要求穷尽。\n"
        "- missing_info 每条按“指标名称 数据 单位”拼接；若原文缺单位则用“无单位”。\n"
        "- 指标名称可按原始个人数据照写，不要因为格式差异误判。\n"
        "- 重构结果文本里若有同一指标但单位不同，不算信息遗漏。\n"
    )

    check_markdown_tables: list[dict[str, Any]] = []
    for i, person_text in enumerate(person_datas):
        md_text = markdown_tables[i] if i < len(markdown_tables) else ""
        md_text = md_text.replace("零散或无法聚合", "其他个人数据")
        if not md_text:
            md_text = ""

        # if i not in [108, 47, 105, 25]:
        #     continue
        # if i in [108, 47, 105, 25]:
        #     print((f"【原始个人数据文本 person_datas】【{i}】\n{person_text}\n"))
        #     print(("【聚合结果 md_text】\n" f"{md_text}\n"))

        if not skip_llm_check:
            res = _call_llm_compare(client=client, model_name=model_name, max_tokens_total=max_tokens_total, person_text=person_text, rebuild_text=md_text, system_prompt=system_prompt)
            check_markdown_tables.append(res)
            if not res["same"]:
                print((f"【原始个人数据文本 person_datas】【{i}】\n{person_text}\n"))
                print(f"\033[92mMarkdown 表格存在信息遗漏: {res['missing_info']}\033[0m")
                print(("【聚合结果 Markdown 表格】\n" f"{md_text}\n"))
                raise ValueError(f"【{i}】Markdown 表格存在信息遗漏")
            else:
                print(f"\033[92m【{i}】Markdown 表格没有信息遗漏\033[0m")

    if skip_llm_check:
        return patterns_all, markdown_tables, check_markdown_tables, markdown_tables
    return patterns_all, markdown_tables, check_markdown_tables


def check_aggregate_time_jsons_by_llm(
    person_datas: list[str],
    client: OpenAI,
    model_name: str,
    max_tokens_total: int,
    skip_llm_check: bool = False,
) -> tuple[
    list[list[Any]],
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """
    对 aggregate_time 输出的 time_jsons 做 LLM 一致性校验（检查 person_datas 与 time_jsons 是否存在信息遗漏）。

    返回：
    - patterns_all: list[list[PersonalDataPattern]]
    - time_jsons: list[list[dict[str, Any]]]（每条样本一个 list[bucket]）
    - check_time_jsons: list[dict]（每条样本一份，来自 LLM 的 JSON 解析结果）
    """
    if not person_datas:
        raise ValueError("person_datas 不能为空")

    strict_uncovered_to_unparsed = True
    include_unknown_time = True
    add_summary_text = True

    patterns_all = [get_patterns_all(t, strict_uncovered_to_unparsed=strict_uncovered_to_unparsed) for t in person_datas]
    time_jsons = [
        aggregate_patterns_by_timejson(
            pats,
            include_unknown_time=include_unknown_time,
            add_summary_text=add_summary_text,
        )
        for pats in patterns_all
    ]

    system_prompt = (
        "你是一个严谨的“数据一致性审查员”。你的任务是比较：\n"
        "- 原始个人数据文本\n"
        "- 重构结果文本\n"
        "判断重构结果文本是否遗漏了原文中出现的指标记录。\n\n"
        "输出必须是严格 JSON（不要包含任何多余文字）。JSON 结构要求：\n"
        "{\n"
        '  "same": true|false,\n'
        '  "missing_info": ["指标名称 数据 单位", "..."]\n'
        "}\n\n"
        "判定规则：\n"
        "- 若未发现遗漏：same=true，missing_info=[]\n"
        "- 若发现遗漏：same=false，missing_info 只需给出少量样例（例如 1~5 条），不要求穷尽。\n"
        "- missing_info 每条按“指标名称 数据 单位”拼接；若原文缺单位则用“无单位”。\n"
        "- 指标名称可按原始个人数据照写，不要因为格式差异误判。\n"
        "- 重构结果文本里若有同一指标但单位不同，不算信息遗漏。\n"
    )

    check_time_jsons: list[dict[str, Any]] = []
    rebuild_time_texts: list[str] = []
    for i, person_text in enumerate(person_datas):
        tj = time_jsons[i] if i < len(time_jsons) else []
        rebuild_text = _time_jsons_to_text(tj)
        rebuild_time_texts.append(rebuild_text)

        # if i not in [108, 47, 105, 25]:
        #     continue
        # if i in [108, 47, 105, 25]:
        #     print((f"【原始个人数据文本 person_datas】【{i}】\n{person_text}\n"))
        #     print(("【聚合结果 rebuild_text】\n" f"{rebuild_text}\n"))
        if not skip_llm_check:
            res = _call_llm_compare(client=client, model_name=model_name, max_tokens_total=max_tokens_total, person_text=person_text, rebuild_text=rebuild_text, system_prompt=system_prompt)
            check_time_jsons.append(res)
            if not res["same"]:
                print((f"【原始个人数据文本 person_datas】【{i}】\n{person_text}\n"))
                print(f"\033[92mtime_jsons 存在信息遗漏: {res['missing_info']}\033[0m")
                print(("【聚合结果 time_jsons（文本化）】\n" f"{rebuild_text}\n"))
                raise ValueError(f"【{i}】time_jsons 存在信息遗漏")
            else:
                print(f"\033[92m【{i}】time_jsons 没有信息遗漏\033[0m")

    if skip_llm_check:
        return patterns_all, time_jsons, check_time_jsons, rebuild_time_texts
    return patterns_all, time_jsons, check_time_jsons


def check_aggregate_dataline_texts_by_llm(
    person_datas: list[str],
    client: OpenAI,
    model_name: str,
    max_tokens_total: int,
    skip_llm_check: bool = False,
) -> tuple[
    list[list[Any]],
    list[str],
    list[dict[str, Any]],
]:
    """
    对 aggregate_dataline 输出的 dataline_texts 做 LLM 一致性校验（检查 person_datas 与 dataline_texts 是否存在信息遗漏）。

    返回：
    - patterns_all: list[list[PersonalDataPattern]]
    - dataline_texts: list[str]
    - check_dataline_texts: list[dict]（每条样本一份，来自 LLM 的 JSON 解析结果）
    """
    if not person_datas:
        raise ValueError("person_datas 不能为空")

    patterns_all = [get_patterns_all(t) for t in person_datas]
    dataline_texts = [
        aggregate_patterns_to_dataline_text(pats)
        for pats in patterns_all
    ]

    system_prompt = (
        "你是一个严谨的“数据一致性审查员”。你的任务是比较：\n"
        "- 原始个人数据文本\n"
        "- 重构结果文本\n"
        "判断重构结果文本是否遗漏了原文中出现的指标记录。\n\n"
        "输出必须是严格 JSON（不要包含任何多余文字）。JSON 结构要求：\n"
        "{\n"
        '  "same": true|false,\n'
        '  "missing_info": ["指标名称 数据 单位", "..."]\n'
        "}\n\n"
        "判定规则：\n"
        "- 若未发现遗漏：same=true，missing_info=[]\n"
        "- 若发现遗漏：same=false，missing_info 只需给出少量样例（例如 1~5 条），不要求穷尽。\n"
        "- missing_info 每条按“指标名称 数据 单位”拼接；若原文缺单位则用“无单位”。\n"
        "- 指标名称可按原始个人数据照写，不要因为格式差异误判。\n"
        "- 重构结果文本里若有同一指标但单位不同，不算信息遗漏。\n"
    )

    check_dataline_texts: list[dict[str, Any]] = []
    for i, person_text in enumerate(person_datas):
        dl_text = dataline_texts[i] if i < len(dataline_texts) else ""
        dl_text = dl_text.replace("数据类型：未定义，", "其他数据：")
        if not dl_text:
            dl_text = ""
        
        # if i not in [108, 47, 105, 25]:
        #     continue
        # if i in [108, 47, 105, 25]:
        #     print((f"【原始个人数据文本 person_datas】【{i}】\n{person_text}\n"))
        #     print(("【聚合结果 dataline_texts】\n" f"{dl_text}\n"))
        if not skip_llm_check:
            res = _call_llm_compare(client=client, model_name=model_name, max_tokens_total=max_tokens_total, person_text=person_text, rebuild_text=dl_text, system_prompt=system_prompt)
            check_dataline_texts.append(res)
            if not res["same"]:
                print((f"【原始个人数据文本 person_datas】【{i}】\n{person_text}\n"))
                print(f"\033[92mdataline_texts 存在信息遗漏: {res['missing_info']}\033[0m")
                print(("【聚合结果 dataline_texts】\n" f"{dl_text}\n"))
                raise ValueError(f"【{i}】dataline_texts 存在信息遗漏")
            else:
                print(f"\033[92m【{i}】dataline_texts 没有信息遗漏\033[0m")

    if skip_llm_check:
        return patterns_all, dataline_texts, check_dataline_texts, dataline_texts
    return patterns_all, dataline_texts, check_dataline_texts


if __name__ == "__main__":

    # xlsx_path = "summary_eval_diff_data.xlsx"
    xlsx_path = "sport_health_log2data_agent_result.with_last_answer_personal.xlsx"
    sheet_name = 0
    data_col = "data"

    _xlsx_path = Path(xlsx_path)
    df = pd.read_excel(_xlsx_path, sheet_name=sheet_name, dtype=object)
    cols = [str(c) for c in df.columns]

    # 构造person_datas的同时，构造剔除无效data的新df
    person_datas: list[str] = []
    kept_indices = []

    for idx, x in enumerate(df[data_col].tolist()):
        try:
            if x is None or bool(pd.isna(x)):
                continue
        except Exception:
            pass
        t = str(x).strip()
        if t.lower() in {"nan", "none", "null", "n/a", "na", "-"}:
            continue
        if t:
            person_datas.append(t)
            kept_indices.append(idx)

    if not person_datas:
        raise ValueError(f"列 {data_col!r} 中没有可用的非空文本：{_xlsx_path}")


    # person_datas 为原始 excel 表格中清洗后的 data 列的值，list[str]
    # df_valid 为剔除无效data后的新df，与person_datas一一对应。
    df_valid = df.iloc[kept_indices].reset_index(drop=True)

    
    model_name = (os.environ.get("LLM_MODEL_JUDGE_NAME") or "").strip()
    base_url = (os.environ.get("LLM_MODEL_JUDGE_URL") or "").strip()
    api_key = (os.environ.get("LLM_MODEL_JUDGE_API_KEY") or "").strip()
    context_window = int(os.environ.get("LLM_MODEL_CONTEXT_WINDOW") or 400000)
    max_tokens_total = int(context_window * 0.85)
    api_key = ""
    base_url = ""
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=DefaultHttpxClient(proxy="http://127.0.0.1:7890"))

    (
        _patterns_all_df,
        _dataframes,
        _wide_tables,
        _check_dataframes,
        _check_wide_tables,
        rebuild_dataframes_texts,
        rebuild_wide_tables_texts,
    ) = check_aggregate_dataframes_by_llm(
        person_datas=person_datas,
        client=client,
        model_name=model_name,
        max_tokens_total=max_tokens_total,
        skip_llm_check=True,
    )
    (
        _patterns_all_md,
        markdown_tables,
        _check_markdown_tables,
        rebuild_markdown_texts,
    ) = check_aggregate_markdown_tables_by_llm(
        person_datas=person_datas,
        client=client,
        model_name=model_name,
        max_tokens_total=max_tokens_total,
        skip_llm_check=True,
    )
    (
        _patterns_all_tj,
        time_jsons,
        _check_time_jsons,
        rebuild_timejson_texts,
    ) = check_aggregate_time_jsons_by_llm(
        person_datas=person_datas,
        client=client,
        model_name=model_name,
        max_tokens_total=max_tokens_total,
        skip_llm_check=True,
    )
    (
        _patterns_all_dl,
        dataline_texts,
        _check_dataline_texts,
        rebuild_dataline_texts,
    ) = check_aggregate_dataline_texts_by_llm(
        person_datas=person_datas,
        client=client,
        model_name=model_name,
        max_tokens_total=max_tokens_total,
        skip_llm_check=True,
    )

    # 把重构文本写回“原始输入表”（df），对无效 data 的行保持为空
    df_out = df.copy()
    new_cols = [
        "data_dataframes",
        "data_widetable",
        "data_markdown",
        "data_timejson",
        "data_dataline",
    ]
    for c in new_cols:
        if c not in df_out.columns:
            df_out[c] = None

    # kept_indices 与 person_datas 一一对应
    for j, orig_idx in enumerate(kept_indices):
        if j < len(rebuild_dataframes_texts):
            df_out.at[orig_idx, "data_dataframes"] = rebuild_dataframes_texts[j]
        if j < len(rebuild_wide_tables_texts):
            df_out.at[orig_idx, "data_widetable"] = rebuild_wide_tables_texts[j]
        if j < len(rebuild_markdown_texts):
            df_out.at[orig_idx, "data_markdown"] = rebuild_markdown_texts[j]
        if j < len(rebuild_timejson_texts):
            df_out.at[orig_idx, "data_timejson"] = rebuild_timejson_texts[j]
        if j < len(rebuild_dataline_texts):
            df_out.at[orig_idx, "data_dataline"] = rebuild_dataline_texts[j]

    out_path = _xlsx_path.with_name(f"{_xlsx_path.stem}.llm_rebuild{_xlsx_path.suffix}")
    df_out.to_excel(out_path, index=False)
    print(f"\033[92m已保存带重构文本的新文件: {out_path}\033[0m")

"""
export file='summary_eval_diff_data' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate
export file='summary_eval_diff_data_dataframes' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate &> $file.log
export file='summary_eval_diff_data_widetable' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate &> $file.log
export file='summary_eval_diff_data_markdown' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate &> $file.log
export file='summary_eval_diff_data_timejson' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate &> $file.log
export file='summary_eval_diff_data_dataline' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate &> $file.log

export file='summary_eval_diff_data' && python scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py --workers 2 --inner_workers 12 --num_repeat 3 --raw-data $file.xlsx
export file='summary_eval_diff_data_dataframes' && python scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py --workers 2 --inner_workers 12 --num_repeat 3 --raw-data $file.xlsx
export file='summary_eval_diff_data_widetable' && python scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py --workers 2 --inner_workers 12 --num_repeat 3 --raw-data $file.xlsx
export file='summary_eval_diff_data_markdown' && python scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py --workers 2 --inner_workers 12 --num_repeat 3 --raw-data $file.xlsx
export file='summary_eval_diff_data_timejson' && python scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py --workers 2 --inner_workers 12 --num_repeat 3 --raw-data $file.xlsx
export file='summary_eval_diff_data_dataline' && python scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py --workers 2 --inner_workers 12 --num_repeat 3 --raw-data $file.xlsx
"""