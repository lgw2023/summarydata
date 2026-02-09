from __future__ import annotations

import sys
import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_clean import *  # noqa: F403
from src.data_clean import __all__  # noqa: F401
from src.utils.env import load_env

if TYPE_CHECKING:  # pragma: no cover
    from src.data_clean.models import PersonalDataPattern


def _normalize_personal_data_text(text: Any) -> str:
    """
    将外部输入的“个人数据长字符串”做最小清洗：
    - 去首尾空白
    - 去掉常见的成对引号包裹（避免复制粘贴时带入 “" ... "” 这类外层符号）
    """
    t = "" if text is None else str(text)
    t = t.strip()
    if not t:
        return ""

    # 轻量去掉最外层的引号包裹（重复剥离若干次，防御 “"xxx"” 这类混用）
    quote_chars = {"'", '"', "“", "”", "‘", "’"}
    for _ in range(3):
        if len(t) >= 2 and (t[0] in quote_chars) and (t[-1] in quote_chars):
            t2 = t[1:-1].strip()
            if t2 == t:
                break
            t = t2
        else:
            break
    return t


def get_patterns_all(
    personal_data_text: str,
    *,
    strict_uncovered_to_unparsed: bool = True,
) -> list["PersonalDataPattern"]:
    """
    输入：一段“个人数据”长字符串
    输出：解析后的 patterns（list[PersonalDataPattern]）
    """
    t = _normalize_personal_data_text(personal_data_text)
    return explode_newlines_and_route_to_dataclasses(t, strict_uncovered_to_unparsed=strict_uncovered_to_unparsed)


def get_dataframes(
    personal_data_text: str,
    *,
    include_loose_lines: bool = True,
    strict_uncovered_to_unparsed: bool = True,
) -> list[pd.DataFrame]:
    """
    输入：一段“个人数据”长字符串
    输出：DataFrame 列表（list[pd.DataFrame]，每个 df 带 attrs 元信息）
    """
    patterns = get_patterns_all(personal_data_text, strict_uncovered_to_unparsed=strict_uncovered_to_unparsed)
    return aggregate_patterns_to_dataframes(patterns, include_loose_lines=include_loose_lines)


def get_wide_tables(
    personal_data_text: str,
    *,
    include_loose_lines: bool = True,
    strict_uncovered_to_unparsed: bool = True,
) -> pd.DataFrame:
    """
    输入：一段“个人数据”长字符串
    输出：合并后的宽表（pd.DataFrame；对应旧脚本里的 wide_tables 的单条结果）
    """
    dfs = get_dataframes(
        personal_data_text,
        include_loose_lines=include_loose_lines,
        strict_uncovered_to_unparsed=strict_uncovered_to_unparsed,
    )
    return aggregate_dataframes_to_table(dfs, include_loose_lines=include_loose_lines)


def get_markdown_tables(
    personal_data_text: str,
    *,
    include_loose_lines: bool = True,
    strict_uncovered_to_unparsed: bool = True,
) -> str:
    """
    输入：一段“个人数据”长字符串
    输出：Markdown 表格文本（str）
    """
    patterns = get_patterns_all(personal_data_text, strict_uncovered_to_unparsed=strict_uncovered_to_unparsed)
    return aggregate_patterns_to_formatted_text(patterns, include_loose_lines=include_loose_lines)


def get_time_jsons(
    personal_data_text: str,
    *,
    include_unknown_time: bool = True,
    add_summary_text: bool = True,
    strict_uncovered_to_unparsed: bool = True,
) -> list[dict[str, Any]]:
    """
    输入：一段“个人数据”长字符串
    输出：按时间桶聚合的结构化列表（list[dict[str, Any]]）
    """
    patterns = get_patterns_all(personal_data_text, strict_uncovered_to_unparsed=strict_uncovered_to_unparsed)
    return aggregate_patterns_by_timejson(
        patterns,
        include_unknown_time=include_unknown_time,
        add_summary_text=add_summary_text,
    )


def get_dataline_texts(
    personal_data_text: str,
    *,
    include_unconstructable_types: bool = True,
    unconstructable_prefix_type: bool = True,
    strict_uncovered_to_unparsed: bool = True,
) -> str:
    """
    输入：一段“个人数据”长字符串
    输出：dataline 风格训练文本（str）
    """
    patterns = get_patterns_all(personal_data_text, strict_uncovered_to_unparsed=strict_uncovered_to_unparsed)
    return aggregate_patterns_to_dataline_text(
        patterns,
        include_unconstructable_types=include_unconstructable_types,
        unconstructable_prefix_type=unconstructable_prefix_type,
    )


def aggregate(
    person_datas: list[str],
) -> tuple[
    list[list[Any]],
    list[list[pd.DataFrame]],
    list[pd.DataFrame],
    list[str],
    list[list[dict[str, Any]]],
    list[str],
]:
    """
    示例：演示 `src/data_clean` 几个聚合模块的用法。

    - `aggregate_format.aggregate_patterns_to_formatted_text`：输出 Markdown 表格文本
    - `aggregate_dataframe.aggregate_patterns_to_dataframes`：输出 DataFrame 列表（带 df.attrs 元信息）
    - `aggregate_time.aggregate_patterns_by_timejson`：按时间桶聚合为结构化 JSON/JSONL
    - `aggregate_dataline.aggregate_patterns_to_dataline_text`：输出 dataline 风格训练文本（逐行描述）

    返回：
    - patterns_all, dataframes, wide_tables, markdown_tables, time_jsons, dataline_texts
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
    total_patterns = 0
    total_patterns = sum(len(p) for p in patterns_all)
    print(f"[info] 输入样例条数={len(person_datas)}，总 patterns 数量={total_patterns}")
    print("\n" + "=" * 23 + " patterns 第一个 示例 " + "=" * 23)
    print(f"原始数据:\n{person_datas[0]}\n")
    # 模拟打印 patterns_all[0] 的原始真实格式，数据类列表
    print(f"\033[92m[\033[0m")
    for i in patterns_all[0]:
        print(f"\033[92m{i},\033[0m")
    print(f"\033[92m]\033[0m")


    #########################################################################################################
    # 1) aggregate_dataframe：输出 DataFrame 列表 + 元信息（df.attrs）
    # 解析：若干 PersonalDataPattern 列表 -> DataFrame 列表
    dataframes = [
        aggregate_patterns_to_dataframes(pats, include_loose_lines=include_loose_lines)
        for pats in patterns_all
    ]
    print(f"[info] 输出 DataFrame 数量={len(dataframes)}")
    print("\n" + "=" * 23 + " aggregate_dataframe 示例 patterns[0] 聚合结果 " + "=" * 23)
    print(f"原始数据:\n{person_datas[0]}\n")
    for df in dataframes[0]:
        print(f"\033[92m{df.to_string(index=False)}\033[0m")
    # 1) aggregate_dataframe -> 单表（宽 schema）
    # 说明：dataframes 的结构是 list[list[DataFrame]]，其中 dataframes 的长度 == 输入样本条数；
    # 对每条样本，把其分表 dataframes[i] 合并为一张宽表（列并集），因此输出宽表数量也应与 dataframes 长度相同。
    wide_tables = [
        aggregate_dataframes_to_table(dfs, include_loose_lines=include_loose_lines)
        for dfs in dataframes
    ]
    print(f"[info] 输出宽表数量={len(wide_tables)}（应等于 dataframes 长度={len(dataframes)}）")
    print("\n" + "=" * 23 + " aggregate_dataframe 示例 patterns[0] 大宽表结果 " + "=" * 23)
    print(f"原始数据:\n{person_datas[0]}\n")
    print(f"\033[92m{wide_tables[0].to_string(index=False)}\033[0m")

    #########################################################################################################
    # 2) aggregate_format：输出 Markdown 表格文本
    # 解析：若干 PersonalDataPattern 列表 -> Markdown 表格文本
    print("\n" + "=" * 24 + " aggregate_format 示例 patterns[0] 聚合结果 " + "=" * 24)
    print(f"原始数据:\n{person_datas[0]}\n")
    markdown_tables = [
        aggregate_patterns_to_formatted_text(pats, include_loose_lines=include_loose_lines)
        for pats in patterns_all
    ]
    print(f"\033[92m{markdown_tables[0]}\033[0m")


    #########################################################################################################
    # 3) aggregate_time：按“时间桶”聚合，输出结构化列表 / JSONL
    # 解析：若干 PersonalDataPattern 列表 -> 结构化列表 / JSONL
    print("\n" + "=" * 26 + " aggregate_time_json 示例 patterns[0] 聚合结果 " + "=" * 26)
    print(f"原始数据:\n{person_datas[0]}\n")
    time_jsons = [
        aggregate_patterns_by_timejson(
            pats,
            include_unknown_time=True,
            add_summary_text=True,
        )
        for pats in patterns_all
    ]
    for b in time_jsons[:1]:
        print(f"\033[92m[\033[0m")
        for i in range(len(b)):

            # 整个json dict打印
            # print(f"\033[92m{b[i]}\033[0m")

            # 手动做个模拟dict的格式化打印
            time_obj = b[i].get("time")
            events = b[i].get("events")
            summary = b[i].get("summary")
            fallback = b[i].get("fallback")
            print(f"\033[92m  {'{'}\033[0m")
            print(f"\033[92m    'time': {time_obj},\033[0m")
            print(f"\033[92m    'events': {events},\033[0m")
            print(f"\033[92m    'summary': '{summary}',\033[0m")
            print(f"\033[92m    'fallback': {fallback},\033[0m")
            print(f"\033[92m  {'},'}\033[0m")

        print(f"\033[92m]\033[0m")

    #########################################################################################################
    # 4) aggregate_dataline：输出 dataline 风格训练文本（逐行描述）
    # 解析：若干 PersonalDataPattern 列表 -> dataline 文本
    print("\n" + "=" * 25 + " aggregate_dataline 示例 patterns[0] 聚合结果 " + "=" * 25)
    print(f"原始数据:\n{person_datas[0]}\n")
    dataline_texts = [
        aggregate_patterns_to_dataline_text(
            pats,
            include_unconstructable_types=True,
            unconstructable_prefix_type=True,
        )
        for pats in patterns_all
    ]
    print(f"\033[92m{dataline_texts[0]}\033[0m")

    return patterns_all, dataframes, wide_tables, markdown_tables, time_jsons, dataline_texts


if __name__ == "__main__":

    # xlsx_path = "summary_eval_diff.xlsx"
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

    data_patterns, dataframes, wide_tables, markdown_tables, time_jsons, dataline_texts = aggregate(
        person_datas=person_datas
    )
