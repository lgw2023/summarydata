from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_clean import *  # type: ignore  # noqa: F403
else:
    from . import *  # type: ignore  # noqa: F403


def _self_test_aggregate_patterns_to_dataframes(
    lines: Sequence[str],
    *,
    max_cases: int | None = 30,
    print_preview: bool = True,
) -> None:
    """
    self-test：验证
    - explode_newlines_and_route_to_dataclasses()
    - aggregate_patterns_to_dataframes()
    的端到端行为（从原始多行文本 -> 数据类列表 -> 聚合为 DataFrame 列表）。

    风格：对齐 `test_aggregate_format.py` / `test_aggregate_time.py`：
    - 逐条样例做断言
    - 输出结构做“不过度脆弱”的检查（避免因表头/列顺序微调导致测试不稳定）
    """
    import pandas as pd  # type: ignore

    xs = list(lines or [])
    if max_cases is not None:
        xs = xs[: max(0, int(max_cases))]

    def _short(s: Any, n: int = 220) -> str:
        t = str(s if s is not None else "").replace("\n", " ").strip()
        if n <= 0:
            return ""
        return t if len(t) <= n else (t[: n - 1] + "…")

    def _assert_dfs(dfs: object, *, case_idx: int, raw: str) -> None:
        if not isinstance(dfs, list) or not dfs:
            raise AssertionError(
                f"[self-test][df-agg] 输出为空/非 list：case#{case_idx} raw={_short(raw)}"
            )

        for j, df in enumerate(dfs):
            if not isinstance(df, pd.DataFrame):
                raise AssertionError(
                    f"[self-test][df-agg] 结果项不是 DataFrame：case#{case_idx} df#{j} raw={_short(raw)} type={type(df).__name__}"
                )

            attrs = getattr(df, "attrs", None)
            if not isinstance(attrs, dict):
                raise AssertionError(
                    f"[self-test][df-agg] df.attrs 不是 dict：case#{case_idx} df#{j} raw={_short(raw)}"
                )

            et = str(attrs.get("entity_type") or "").strip()
            title = str(attrs.get("title") or "").strip()
            if not et:
                raise AssertionError(
                    f"[self-test][df-agg] 缺少 attrs.entity_type：case#{case_idx} df#{j} raw={_short(raw)} attrs={attrs!r}"
                )
            if not title:
                raise AssertionError(
                    f"[self-test][df-agg] 缺少 attrs.title：case#{case_idx} df#{j} raw={_short(raw)} attrs={attrs!r}"
                )
            if et not in title:
                raise AssertionError(
                    f"[self-test][df-agg] title 未包含 entity_type：case#{case_idx} df#{j} raw={_short(raw)} entity_type={et!r} title={title!r}"
                )

            # 不强制每个 df 必须非空（比如某些极端输入只产生 loose_lines），但非 loose 表通常应有列
            if et != "零散或无法聚合" and df.columns.size == 0:
                raise AssertionError(
                    f"[self-test][df-agg] 非 loose 表没有任何列：case#{case_idx} df#{j} raw={_short(raw)} entity_type={et!r}"
                )

            if et == "零散或无法聚合":
                if "raw" not in df.columns:
                    raise AssertionError(
                        f"[self-test][df-agg] loose 表缺少 raw 列：case#{case_idx} df#{j} raw={_short(raw)} cols={list(df.columns)!r}"
                    )

            # 对比表：若 attrs 提供了 date_range1/date_range2，则列中应包含（可能被缩写后的）范围列
            if et == "周期数值对比记录":
                # 新版：对比表保持长表结构（date_range1/value1/date_range2/value2/logic/diff/metric）
                for need in ("metric", "diff", "date_range1", "date_range2"):
                    if need not in df.columns:
                        raise AssertionError(
                            f"[self-test][df-agg] 对比表缺少列 {need!r}：case#{case_idx} df#{j} raw={_short(raw)} cols={list(df.columns)!r}"
                        )

    print(f"[self-test] aggregate_patterns_to_dataframes 样本数={len(xs)} format=dataframes")

    ok = 0
    skipped_empty = 0
    for i, text in enumerate(xs):
        raw = (text or "").strip()
        if not raw:
            skipped_empty += 1
            continue

        patterns = explode_newlines_and_route_to_dataclasses(raw)  # type: ignore[name-defined]  # noqa: F405
        dfs = aggregate_patterns_to_dataframes(patterns, include_loose_lines=True)  # type: ignore[name-defined]  # noqa: F405
        _assert_dfs(dfs, case_idx=i, raw=raw)

        ok += 1
        if print_preview:
            print(f"  - case#{i} 解析对象数={len(patterns)} 输出表数={len(dfs)}")
            print(f"    原始=\"\"\"\n{raw}\n\"\"\"")
            for j, df in enumerate(dfs):
                et = str(getattr(df, "attrs", {}).get("entity_type") or "")
                title = str(getattr(df, "attrs", {}).get("title") or "")
                print(f"    [df#{j}] entity_type={et!r} title={title!r} shape={df.shape!r} cols={list(df.columns)!r}")
                try:
                    print(f"\033[92m{df.to_string(index=False)}\033[0m")
                except Exception:
                    pass

    if skipped_empty:
        print(f"[self-test] aggregate_patterns_to_dataframes 跳过空样本：{skipped_empty}")
    print(f"[self-test] aggregate_patterns_to_dataframes 通过：{ok}/{len(xs) - skipped_empty}")


def _self_test_aggregate_dataframes_to_table(
    lines: Sequence[str],
    *,
    max_cases: int | None = 30,
    print_preview: bool = True,
) -> None:
    """
    self-test：验证
    - aggregate_patterns_to_dataframes()
    - aggregate_dataframes_to_table()
    的端到端行为（从原始多行文本 -> DataFrame 列表 -> 合并为单表）。
    """
    import pandas as pd  # type: ignore

    xs = list(lines or [])
    if max_cases is not None:
        xs = xs[: max(0, int(max_cases))]

    def _short(s: Any, n: int = 220) -> str:
        t = str(s if s is not None else "").replace("\n", " ").strip()
        if n <= 0:
            return ""
        return t if len(t) <= n else (t[: n - 1] + "…")

    print(f"[self-test] aggregate_dataframes_to_table 样本数={len(xs)} format=table")

    ok = 0
    skipped_empty = 0
    for i, text in enumerate(xs):
        raw = (text or "").strip()
        if not raw:
            skipped_empty += 1
            continue

        patterns = explode_newlines_and_route_to_dataclasses(raw)  # type: ignore[name-defined]  # noqa: F405
        dfs = aggregate_patterns_to_dataframes(patterns, include_loose_lines=True)  # type: ignore[name-defined]  # noqa: F405
        table = aggregate_dataframes_to_table(dfs, include_loose_lines=True)  # type: ignore[name-defined]  # noqa: F405

        if not isinstance(table, pd.DataFrame):
            raise AssertionError(f"[self-test][df-table] 输出不是 DataFrame：case#{i} raw={_short(raw)} type={type(table).__name__}")

        # 核心断言：行数应等于所有子表行数之和
        expect_n = sum(int(getattr(d, "shape", (0, 0))[0]) for d in dfs)
        if int(table.shape[0]) != int(expect_n):
            raise AssertionError(
                f"[self-test][df-table] 行数不一致：case#{i} raw={_short(raw)} expect={expect_n} got={int(table.shape[0])}"
            )

        # 核心断言：输出必须包含“固定完整宽表表头”（无论输入包含哪些类型）
        fixed_header = [
            "entity_type",
            "data_type",
            "title",
            "table_idx",
            "row_idx",
            "date",
            "time",
            "start_date",
            "end_date",
            "date_range1",
            "date_range2",
            "category",
            "metric",
            "value",
            "unit",
            "status",
            "value1",
            "value2",
            "logic",
            "diff",
            "raw",
        ]
        for need in fixed_header:
            if need not in table.columns:
                raise AssertionError(
                    f"[self-test][df-table] 缺少列 {need!r}：case#{i} raw={_short(raw)} cols={list(table.columns)!r}"
                )

        # include_loose_lines=False 应能剔除 loose 行（如果存在的话）
        table2 = aggregate_dataframes_to_table(dfs, include_loose_lines=False)  # type: ignore[name-defined]  # noqa: F405
        if "entity_type" in table2.columns:
            if (table2["entity_type"] == "零散或无法聚合").any():
                raise AssertionError(
                    f"[self-test][df-table] include_loose_lines=False 未剔除 loose 行：case#{i} raw={_short(raw)}"
                )

        ok += 1
        if print_preview:
            print(f"  - case#{i} 解析对象数={len(patterns)} 输出分表数={len(dfs)} 合并表shape={table.shape!r}")
            try:
                print(f"\033[92m{table.to_string(index=False)}\033[0m")
            except Exception:
                pass

    if skipped_empty:
        print(f"[self-test] aggregate_dataframes_to_table 跳过空样本：{skipped_empty}")
    print(f"[self-test] aggregate_dataframes_to_table 通过：{ok}/{len(xs) - skipped_empty}")


__all__ = ["_self_test_aggregate_patterns_to_dataframes", "_self_test_aggregate_dataframes_to_table"]

