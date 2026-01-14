from __future__ import annotations

"""兼容入口（薄封装）。

历史上 `scripts/classify_personal_data.py` 是一个 6k+ 行的单文件模块；为提升可维护性，
核心实现已拆分到 `src/data_clean/`，本文件仅负责：
- 在直接运行 scripts 下的脚本时补齐 sys.path
- re-export 旧版对外 API（保持 `from classify_personal_data import *` 可用）
"""

import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_clean import *  # noqa: F403
from src.data_clean import __all__  # noqa: F401

def main(
    file,
    *,
    xlsx_path: str | Path | None = None,
    sheet_name: str | int | None = 0,
    data_col: str = "data",
    max_rows: int = 5,
    max_rows_per_table: int = 200,
) -> None:
    """
    示例：演示 `src/data_clean` 三个聚合模块的用法。

    - `aggregate_format.aggregate_patterns_to_formatted_text`：输出 Markdown 表格文本
    - `aggregate_dataframe.aggregate_patterns_to_dataframes`：输出 DataFrame 列表（带 df.attrs 元信息）
    - `aggregate_time.aggregate_patterns_by_time / aggregate_patterns_to_time_jsonl`：按时间桶聚合为结构化 JSON/JSONL
    """
    _xlsx_path = Path(xlsx_path) if xlsx_path is not None else (PROJECT_ROOT / file)
    if not _xlsx_path.exists():
        raise FileNotFoundError(f"未找到 Excel 文件：{_xlsx_path}")

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("读取 Excel 需要依赖 pandas + openpyxl。请先安装：pip install pandas openpyxl") from exc

    df = pd.read_excel(_xlsx_path, sheet_name=sheet_name, dtype=object)
    cols = [str(c) for c in df.columns]
    if data_col not in set(cols):
        raise KeyError(f"Excel 中未找到 {data_col!r} 列：{_xlsx_path}；现有列={cols}")

    # 取 N 条样例：把每条 data 解析为 dataclass patterns，再做聚合。
    texts: list[str] = []
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
            texts.append(t)
        if len(texts) >= max(1, int(max_rows)):
            break

    if not texts:
        raise ValueError(f"列 {data_col!r} 中没有可用的非空文本：{_xlsx_path}")

    # 解析：一条文本 -> 若干 PersonalDataPattern
    # 入口在 src/data_clean/parse.py：explode_newlines_and_route_to_dataclasses
    patterns_all = []
    for t in texts:
        patterns_all.append(explode_newlines_and_route_to_dataclasses(t, strict_uncovered_to_unparsed=True))  # noqa: F405

    print(f"[info] 输入样例条数={len(texts)}，解析得到 patterns 数量={len(patterns_all)}")
    print("\n" + "=" * 23 + " patterns 第一个 示例 " + "=" * 23)
    print(f"patterns {0}: {patterns_all[0]}")

    # 1) aggregate_format：输出 Markdown 表格文本
    print("\n" + "=" * 24 + " aggregate_format 示例 patterns[0] 聚合结果 " + "=" * 24)
    formatted_texts = []
    for i in range(len(patterns_all)):
        formatted_texts.append(aggregate_patterns_to_formatted_text(patterns_all[i], max_rows_per_table=max_rows_per_table, include_loose_lines=True))  # noqa: F405
    print(formatted_texts[0])

    # 2) aggregate_dataframe：输出 DataFrame 列表 + 元信息（df.attrs）
    dfs = []
    for i in range(len(patterns_all)):
        dfs.append(aggregate_patterns_to_dataframes(patterns_all[i], max_rows_per_table=max_rows_per_table, include_loose_lines=True))  # noqa: F405
    print(f"[info] 输出 DataFrame 数量={len(dfs)}")
    print("\n" + "=" * 23 + " aggregate_dataframe 示例 patterns[0] 聚合结果 " + "=" * 23)
    for i in range(len(dfs[0])):
        print(dfs[0][i])

    # 3) aggregate_time：按“时间桶”聚合，输出结构化列表 / JSONL
    print("\n" + "=" * 26 + " aggregate_time 示例 patterns[0] 聚合结果 " + "=" * 26)
    buckets = []
    for i in range(len(patterns_all)):
        buckets.append(aggregate_patterns_by_time(patterns_all[i], include_unknown_time=True, add_summary_text=True))  # noqa: F405
    print(f"[info] 时间桶数量={len(buckets)}；下面打印第1个桶的 summary：")
    for b in [buckets[0]]:
        for i in range(len(b)):
            t = b[i].get("time") or {}
            print(f"\t- time={t} summary={str(b[i].get('summary') or '')}")

    jsonls = []
    for i in range(len(patterns_all)):
        jsonls.append(aggregate_patterns_to_time_jsonl(patterns_all[i], include_unknown_time=True, add_summary_text=True, ensure_ascii=False))  # noqa: F405
    print("\n" + "=" * 23 + " aggregate_time 示例 patterns[0] 聚合结果 " + "=" * 23)
    print("[info] 同样也可以直接得到 JSONL 输出 jsonl 数量={len(jsonls)}：")
    print(jsonls[0])


if __name__ == "__main__":
    main(file="summary_eval_diff.xlsx")