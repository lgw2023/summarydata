from __future__ import annotations

import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_clean import *  # noqa: F403
from src.data_clean import __all__  # noqa: F401


def aggregate(person_datas: list[str]) -> None:
    """
    示例：演示 `src/data_clean` 三个聚合模块的用法。

    - `aggregate_format.aggregate_patterns_to_formatted_text`：输出 Markdown 表格文本
    - `aggregate_dataframe.aggregate_patterns_to_dataframes`：输出 DataFrame 列表（带 df.attrs 元信息）
    - `aggregate_time.aggregate_patterns_by_timejson`：按时间桶聚合为结构化 JSON/JSONL
    """

    #########################################################################################################
    # 解析：一串单行或者多行的个人数据文本（包括多指标数据） -> 若干 PersonalDataPattern 列表
    # 入口在 src/data_clean/parse.py：explode_newlines_and_route_to_dataclasses
    patterns_all = []
    for t in person_datas:
        patterns_all.append(explode_newlines_and_route_to_dataclasses(t, strict_uncovered_to_unparsed=True))
    print(f"[info] 输入样例条数={len(person_datas)}，解析得到 patterns 数量={len(patterns_all)}")
    print("\n" + "=" * 23 + " patterns 第一个 示例 " + "=" * 23)
    print(f"原始数据:\n{person_datas[0]}\n")
    print(f"\033[92m{patterns_all[0]}\033[0m")


    #########################################################################################################
    # 1) aggregate_dataframe：输出 DataFrame 列表 + 元信息（df.attrs）
    # 解析：若干 PersonalDataPattern 列表 -> DataFrame 列表
    dataframes = []
    for i in range(len(patterns_all)):
        dataframes.append(aggregate_patterns_to_dataframes(patterns_all[i], include_loose_lines=True))
    print(f"[info] 输出 DataFrame 数量={len(dataframes)}")
    print("\n" + "=" * 23 + " aggregate_dataframe 示例 patterns[0] 聚合结果 " + "=" * 23)
    print(f"原始数据:\n{person_datas[0]}\n")
    for i in range(len(dataframes[0])):
        print(f"\033[92m{dataframes[0][i]}\033[0m")
    # 1) aggregate_dataframe -> 单表（宽 schema）
    # 说明：dataframes 的结构是 list[list[DataFrame]]，其中 dataframes 的长度 == 输入样本条数；
    # 对每条样本，把其分表 dataframes[i] 合并为一张宽表（列并集），因此输出宽表数量也应与 dataframes 长度相同。
    wide_tables = []
    for i in range(len(dataframes)):
        wide_tables.append(aggregate_dataframes_to_table(dataframes[i], include_loose_lines=True))
    print(f"[info] 输出宽表数量={len(wide_tables)}（应等于 dataframes 长度={len(dataframes)}）")
    print("\n" + "=" * 23 + " aggregate_dataframe 示例 patterns[0] 大宽表结果 " + "=" * 23)
    print(f"原始数据:\n{person_datas[0]}\n")
    print(f"\033[92m{wide_tables[0].to_string(index=False)}\033[0m")


    #########################################################################################################
    # 2) aggregate_format：输出 Markdown 表格文本
    # 解析：若干 PersonalDataPattern 列表 -> Markdown 表格文本
    print("\n" + "=" * 24 + " aggregate_format 示例 patterns[0] 聚合结果 " + "=" * 24)
    print(f"原始数据:\n{person_datas[0]}\n")
    markdown_tables = []
    for i in range(len(patterns_all)):
        markdown_tables.append(aggregate_patterns_to_formatted_text(patterns_all[i], include_loose_lines=True))
    print(f"\033[92m{markdown_tables[0]}\033[0m")


    #########################################################################################################
    # 3) aggregate_time：按“时间桶”聚合，输出结构化列表 / JSONL
    # 解析：若干 PersonalDataPattern 列表 -> 结构化列表 / JSONL
    print("\n" + "=" * 26 + " aggregate_time_json 示例 patterns[0] 聚合结果 " + "=" * 26)
    print(f"原始数据:\n{person_datas[0]}\n")
    buckets = []
    for i in range(len(patterns_all)):
        buckets.append(aggregate_patterns_by_timejson(patterns_all[i], include_unknown_time=True, add_summary_text=True))
    for b in [buckets[0]]:
        print(f"\033[92m[\033[0m")
        for i in range(len(b)):
            # print(f"\033[92m{b[i]}\033[0m")
            time_obj = b[i].get("time")
            events = b[i].get("events")
            summary = b[i].get("summary")
            fallback = b[i].get("fallback")
            print(f"\033[92m  {'{'}\033[0m")
            print(f"\033[92m    'time': {time_obj},\033[0m")
            print(f"\033[92m    'events': {events},\033[0m")
            print(f"\033[92m    'summary': {summary},\033[0m")
            print(f"\033[92m    'fallback': {fallback},\033[0m")
            print(f"\033[92m  {'}'}\033[0m")
        print(f"\033[92m]\033[0m")

    return dataframes, wide_tables, markdown_tables, buckets


if __name__ == "__main__":

    xlsx_path = "summary_eval_diff.xlsx"
    sheet_name = 0
    data_col = "data"


    _xlsx_path = Path(xlsx_path)
    df = pd.read_excel(_xlsx_path, sheet_name=sheet_name, dtype=object)
    cols = [str(c) for c in df.columns]
    
    # 取 N 条样例：把每条 data 解析为 dataclass patterns，再做聚合。
    person_datas: list[str] = []
    for x in df[data_col].tolist():
        # pandas 的空值通常是 NaN（float），str() 会变成 "nan"；这里统一过滤掉
        try:
            if x is None or bool(pd.isna(x)):  # type: ignore[attr-defined]
                continue
        except Exception:
            pass
        t = str(x).strip()
        if t.lower() in {"nan", "none", "null", "n/a", "na", "-"}:
            continue
        if t:
            person_datas.append(t)

    if not person_datas:
        raise ValueError(f"列 {data_col!r} 中没有可用的非空文本：{_xlsx_path}")


    dataframes, wide_tables, markdown_tables, buckets = aggregate(person_datas=person_datas)