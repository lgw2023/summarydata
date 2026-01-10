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
            if et != "零散/无法聚合" and df.columns.size == 0:
                raise AssertionError(
                    f"[self-test][df-agg] 非 loose 表没有任何列：case#{case_idx} df#{j} raw={_short(raw)} entity_type={et!r}"
                )

            if et == "零散/无法聚合":
                if "raw" not in df.columns:
                    raise AssertionError(
                        f"[self-test][df-agg] loose 表缺少 raw 列：case#{case_idx} df#{j} raw={_short(raw)} cols={list(df.columns)!r}"
                    )

            # 对比表：若 attrs 提供了 range1/range2，则列中应包含（可能被缩写后的）范围列
            if et == "周期数值对比记录":
                # 新版：对比表保持长表结构（range1/value1/range2/value2/logic/diff/metric）
                for need in ("metric", "diff", "range1", "range2"):
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
                # 只打印前几行避免日志过大
                try:
                    print(df.head(6).to_string(index=False))
                except Exception:
                    pass

    if skipped_empty:
        print(f"[self-test] aggregate_patterns_to_dataframes 跳过空样本：{skipped_empty}")
    print(f"[self-test] aggregate_patterns_to_dataframes 通过：{ok}/{len(xs) - skipped_empty}")


__all__ = ["_self_test_aggregate_patterns_to_dataframes"]

