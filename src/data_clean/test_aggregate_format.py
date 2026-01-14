from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence, Union

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_clean import *  # type: ignore  # noqa: F403
    from src.data_clean.models import _indent_lines  # noqa: F401
else:
    from . import *  # type: ignore  # noqa: F403
    from .models import _indent_lines  # noqa: F401

def _self_test_print_objs(
    objs: Sequence[PersonalDataPatternBase],
    *,
    max_items: int = 8,
    max_len: int = 140,
) -> None:
    """
    self-test 专用：把解析出来的数据类列表以"可读格式"打印出来。
    - 优先使用各数据类的 format_print()
    - 兜底使用 base.format_print()
    """
    total = len(objs)
    if total == 0:
        print("  [警告] 对象列表为空")
        return
    
    print(f"  [开始打印 {total} 个对象]")
    for k, obj in enumerate(objs):
        print(f"  - [#{k}/{total-1}]")
        if hasattr(obj, "format_print"):
            try:
                txt = obj.format_print(max_items=max_items, max_len=max_len)  # type: ignore[misc]
            except Exception as e:
                txt = f"【format_print 异常】{type(e).__name__}: {e}\n{str(obj)!r}"
        else:
            txt = str(obj)
        print(_indent_lines(txt, 4))

        # 新增：recover_to_raw_data() 预览（便于人工快速比对）
        if hasattr(obj, "recover_to_raw_data"):
            try:
                rec = obj.recover_to_raw_data()  # type: ignore[misc]
            except Exception as e:
                rec = f"[recover_to_raw_data 异常]{type(e).__name__}: {e}"
            print(_indent_lines(f"- recover_to_raw_data：{rec}", 4))
    
    print(f"  [完成打印 {total} 个对象]")


def _self_test_single_metric_detail_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `SingleMetricDetailRecord.from_raw_personal_data()` 是否能解析“一行多条记录”。
    """
    print(f"[self-test] SingleMetricDetailRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = SingleMetricDetailRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if all(isinstance(x, UnparsedRawPersonalData) for x in parsed_list):
            parsed0 = parsed_list[0]
            assert isinstance(parsed0, UnparsedRawPersonalData)
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed0.原因}，原文前100={parsed0.个人数据[:100]!r}")
        if any(isinstance(x, UnparsedRawPersonalData) for x in parsed_list):
            raise AssertionError(f"[self-test] 第{i}行解析结果混入 UnparsedRawPersonalData：{parsed_list!r}")

        # 每个指标一个 record；逐个做一致性检查
        for k, parsed in enumerate(parsed_list):
            assert isinstance(parsed, SingleMetricDetailRecord)
            n_d, n_t, n_v = len(parsed.日期列表), len(parsed.时间列表), len(parsed.数值列表)
            if not (n_d == n_t == n_v):
                raise AssertionError(f"[self-test] 第{i}行#{k}列表长度不一致：日期={n_d}, 时间={n_t}, 数值={n_v}")
            if n_d == 0:
                raise AssertionError(f"[self-test] 第{i}行#{k}解析得到 0 条记录")

            # 防御性断言：规范化后的中文日期若形如 “YYYY年MM月DD日 / MM月DD日”，则月份必须在 1..12
            for d in parsed.日期列表:
                m = re.fullmatch(r"(?:(?P<y>\d{4})年)?(?P<m>\d{2})月(?P<dd>\d{2})日", str(d or "").strip())
                if not m:
                    continue
                mo = int(m.group("m"))
                if not (1 <= mo <= 12):
                    raise AssertionError(f"[self-test] 第{i}行#{k}出现非法月份：date={d!r} raw_line前80={str(line)[:80]!r}")

        # 打印：如果返回的是“数据类列表”，则列表中每个值都要打印（而不是只打印第一个）
        print(f"[self-test] #{i} 指标数={len(parsed_list)} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=8, max_len=140)


def _self_test_period_value_single_summary_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `PeriodValueSingleSummaryRecord.from_raw_personal_data()` 是否能解析“周期数值单项总结”一行文本。
    """
    print(f"[self-test] PeriodValueSingleSummaryRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = PeriodValueSingleSummaryRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前100={parsed.个人数据[:100]!r}")
        assert isinstance(parsed, PeriodValueSingleSummaryRecord)

        n_s, n_e, n_n, n_t, n_u, n_v = (
            len(parsed.开始日期列表),
            len(parsed.结束日期列表),
            len(parsed.指标名称列表),
            len(parsed.数值类型列表),
            len(parsed.单位列表),
            len(parsed.数值列表),
        )
        if not (n_s == n_e == n_n == n_t == n_u == n_v):
            raise AssertionError(
                f"[self-test] 第{i}行列表长度不一致：开始={n_s}, 结束={n_e}, 指标={n_n}, 类型={n_t}, 单位={n_u}, 数值={n_v}"
            )
        if n_s == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=8, max_len=140)


def _self_test_period_text_summary_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `PeriodTextSummaryRecord.from_raw_personal_data()` 是否能解析“周期文本总结”一行文本。
    """
    print(f"[self-test] PeriodTextSummaryRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = PeriodTextSummaryRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前100={parsed.个人数据[:100]!r}")
        assert isinstance(parsed, PeriodTextSummaryRecord)

        n_s, n_e, n_n, n_d = (
            len(parsed.开始日期列表),
            len(parsed.结束日期列表),
            len(parsed.指标名称列表),
            len(parsed.状态描述列表),
        )
        if not (n_s == n_e == n_n == n_d):
            raise AssertionError(
                f"[self-test] 第{i}行列表长度不一致：开始={n_s}, 结束={n_e}, 指标={n_n}, 状态描述={n_d}"
            )
        if n_s == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=8, max_len=160)


def _self_test_period_value_compare_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `PeriodValueCompareRecord.from_raw_personal_data()` 是否能解析“周期数值对比记录”一行文本。
    """
    print(f"[self-test] PeriodValueCompareRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = PeriodValueCompareRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前100={parsed.个人数据[:100]!r}")
        assert isinstance(parsed, PeriodValueCompareRecord)

        n1, n2, n3, n4, n5 = (
            len(parsed.日期范围1列表),
            len(parsed.数值1列表),
            len(parsed.日期范围2列表),
            len(parsed.数值2列表),
            len(parsed.差异数值列表),
        )
        if not (n1 == n2 == n3 == n4 == n5):
            raise AssertionError(
                f"[self-test] 第{i}行列表长度不一致：r1={n1}, v1={n2}, r2={n3}, v2={n4}, diff={n5}"
            )
        if n1 == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=8, max_len=160)


def _self_test_period_value_multi_summary_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `PeriodValuemMultiSummaryRecord.from_raw_personal_data()` 是否能解析“周期数值多项总结”一行文本。
    """
    print(f"[self-test] PeriodValuemMultiSummaryRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = PeriodValuemMultiSummaryRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前100={parsed.个人数据[:100]!r}")
        assert isinstance(parsed, PeriodValuemMultiSummaryRecord)

        n_s, n_e, n_n, n_t, n_u, n_v, n_d = (
            len(parsed.开始日期列表),
            len(parsed.结束日期列表),
            len(parsed.指标名称列表),
            len(parsed.数值类型列表),
            len(parsed.单位列表),
            len(parsed.数值列表),
            len(parsed.状态描述列表),
        )
        if not (n_s == n_e == n_n == n_t == n_u == n_v == n_d):
            raise AssertionError(
                f"[self-test] 第{i}行列表长度不一致：开始={n_s}, 结束={n_e}, 指标={n_n}, 类型={n_t}, 单位={n_u}, 数值={n_v}, 状态描述={n_d}"
            )
        if n_s == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=8, max_len=160)


def _self_test_no_timestamp_text_summary_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `NoTimestampTextSummaryRecord.from_raw_personal_data()` 是否能解析“无时间日期的文本总结”一行文本。
    """
    print(f"[self-test] NoTimestampTextSummaryRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = NoTimestampTextSummaryRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前100={parsed.个人数据[:100]!r}")
        assert isinstance(parsed, NoTimestampTextSummaryRecord)

        n_m, n_s = len(parsed.指标名称列表), len(parsed.状态描述列表)
        if not (n_m == n_s):
            raise AssertionError(f"[self-test] 第{i}行列表长度不一致：指标名称={n_m}, 状态描述={n_s}")
        if n_m == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=12, max_len=160)


def _self_test_no_date_value_summary_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `NoDateValueSummaryRecord.from_raw_personal_data()` 是否能解析“无时间日期的数值总结”一行文本。
    """
    print(f"[self-test] NoDateValueSummaryRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = NoDateValueSummaryRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前100={parsed.个人数据[:100]!r}")
        assert isinstance(parsed, NoDateValueSummaryRecord)

        n_m = len(parsed.指标名称列表)
        n_t = len(parsed.数值类型列表)
        n_u = len(parsed.单位列表)
        n_v = len(parsed.数值列表)
        n_s = len(parsed.状态描述列表)
        if not (n_m == n_t == n_u == n_v == n_s):
            raise AssertionError(
                f"[self-test] 第{i}行列表长度不一致：指标={n_m}, 类型={n_t}, 单位={n_u}, 数值={n_v}, 状态描述={n_s}"
            )
        if n_m == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=12, max_len=160)


def _self_test_single_metric_stats_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `SingleMetricStatsRecord.from_raw_personal_data()` 是否能解析“单指标的明细汇总记录”一行文本。
    """
    print(f"[self-test] SingleMetricStatsRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = SingleMetricStatsRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前100={parsed.个人数据[:100]!r}")
        assert isinstance(parsed, SingleMetricStatsRecord)

        n_d, n_v = len(parsed.日期列表), len(parsed.数值列表)
        if n_d != n_v:
            raise AssertionError(f"[self-test] 第{i}行明细列表长度不一致：日期={n_d}, 数值={n_v}")
        if n_d == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条明细记录")

        n_sn, n_sv, n_ss = (
            len(parsed.统计指标名称列表),
            len(parsed.统计数值列表),
            len(parsed.统计状态描述列表),
        )
        if not (n_sn == n_sv == n_ss):
            raise AssertionError(f"[self-test] 第{i}行汇总列表长度不一致：指标={n_sn}, 数值={n_sv}, 状态={n_ss}")
        # 允许汇总列表为空（有些数据只有明细，没有汇总）

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=8, max_len=160)


def _self_test_single_date_value_multi_summary_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `SingleDateValueMultiSummaryRecord.from_raw_personal_data()` 是否能解析"单日期数值多项总结"一行文本。

    约定：该类型应返回 **1 个对象**，对象内部用列表字段承载多个逗号片段的明细。
    """
    print(f"[self-test] SingleDateValueMultiSummaryRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = SingleDateValueMultiSummaryRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")

        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前100={parsed.个人数据[:100]!r}")
        if not isinstance(parsed, SingleDateValueMultiSummaryRecord):
            raise AssertionError(f"[self-test] 第{i}行解析结果不是 SingleDateValueMultiSummaryRecord：实际={type(parsed).__name__}")

        n_d, n_m, n_t, n_u, n_v, n_s = (
            len(parsed.日期列表),
            len(parsed.指标名称列表),
            len(parsed.数值类型列表),
            len(parsed.单位列表),
            len(parsed.数值列表),
            len(parsed.状态描述列表),
        )
        if not (n_d == n_m == n_t == n_u == n_v == n_s):
            raise AssertionError(
                f"[self-test] 第{i}行列表长度不一致：日期={n_d}, 指标={n_m}, 类型={n_t}, 单位={n_u}, 数值={n_v}, 状态描述={n_s}"
            )
        if n_d == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=10, max_len=160)


def _self_test_single_date_value_single_summary_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `SingleDateValueSingleSummaryRecord.from_raw_personal_data()` 是否能解析“单日期数值单项总结”一行文本。
    """
    print(f"[self-test] SingleDateValueSingleSummaryRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = SingleDateValueSingleSummaryRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前150={parsed.个人数据[:150]!r}")
        assert isinstance(parsed, SingleDateValueSingleSummaryRecord)

        n_d, n_m, n_t, n_u, n_v, n_s = (
            len(parsed.日期列表),
            len(parsed.指标名称列表),
            len(parsed.数值类型列表),
            len(parsed.单位列表),
            len(parsed.数值列表),
            len(parsed.状态描述列表),
        )
        if not (n_d == n_m == n_t == n_u == n_v == n_s):
            raise AssertionError(
                f"[self-test] 第{i}行列表长度不一致：日期={n_d}, 指标={n_m}, 类型={n_t}, 单位={n_u}, 数值={n_v}, 状态描述={n_s}"
            )
        if n_d == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=10, max_len=160)


def _self_test_single_date_text_summary_record(lines: Sequence[str]) -> None:
    """
    用于快速验证 `SingleDateTextSummaryRecord.from_raw_personal_data()` 是否能解析“单日期文本总结”一行文本。
    """
    print(f"[self-test] SingleDateTextSummaryRecord 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = SingleDateTextSummaryRecord.from_raw_personal_data(line)
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行解析失败：原因={parsed.原因}，原文前150={parsed.个人数据[:150]!r}")
        assert isinstance(parsed, SingleDateTextSummaryRecord)

        n_d, n_m, n_s = len(parsed.日期列表), len(parsed.指标名称列表), len(parsed.状态描述列表)
        if not (n_d == n_m == n_s):
            raise AssertionError(
                f"[self-test] 第{i}行列表长度不一致：日期={n_d}, 指标={n_m}, 状态描述={n_s}"
            )
        if n_d == 0:
            raise AssertionError(f"[self-test] 第{i}行解析得到 0 条记录")

        # 打印：如果解析结果是数据类列表，则列表中每个值都要打印
        print(f"[self-test] #{i} 原文前50={str(line)[:50]!r}")
        _self_test_print_objs(parsed_list, max_items=10, max_len=180)


def _self_test_route_raw_personal_data_to_dataclass(
    *,
    test_SingleMetricDetailRecord: Sequence[str],
    test_PeriodValueSingleSummaryRecord: Sequence[str],
    test_PeriodValuemMultiSummaryRecord: Sequence[str],
    test_PeriodTextSummaryRecord: Sequence[str],
    test_PeriodValueCompareRecord: Sequence[str],
    test_SingleDateValueSingleSummaryRecord: Sequence[str],
    test_SingleDateValueMultiSummaryRecord: Sequence[str],
    test_SingleDateTextSummaryRecord: Sequence[str],
    test_NoDateValueSummaryRecord: Sequence[str],
    test_NoTimestampTextSummaryRecord: Sequence[str],
    test_SingleMetricStatsRecord: Sequence[str],
    test_UnparsedRawPersonalData: Sequence[str],
) -> None:
    """
    用“金标准样例”测试 router：`route_raw_personal_data_to_dataclass()`。

    设计：
    - 假装“不知情”（不提供实体类型/指标名称等任何 hint），只调用 router。
    - 逐条断言 router 返回的 dataclass 类型与金标准期望一致。
    - 对于期望非兜底类的样例，若 router 返回 `UnparsedRawPersonalData`，会把 `原因` 打印出来便于定位。
    """
    cases: list[tuple[str, type[PersonalDataPatternBase], Sequence[str]]] = [
        ("无时间日期的数值总结", NoDateValueSummaryRecord, test_NoDateValueSummaryRecord),
        ("无时间日期的文本总结", NoTimestampTextSummaryRecord, test_NoTimestampTextSummaryRecord),
        ("单日期数值单项总结", SingleDateValueSingleSummaryRecord, test_SingleDateValueSingleSummaryRecord),
        ("单日期数值多项总结", SingleDateValueMultiSummaryRecord, test_SingleDateValueMultiSummaryRecord),
        ("单日期文本总结", SingleDateTextSummaryRecord, test_SingleDateTextSummaryRecord),
        ("周期数值单项总结", PeriodValueSingleSummaryRecord, test_PeriodValueSingleSummaryRecord),
        ("周期数值多项总结", PeriodValuemMultiSummaryRecord, test_PeriodValuemMultiSummaryRecord),
        ("周期文本总结", PeriodTextSummaryRecord, test_PeriodTextSummaryRecord),
        ("周期数值对比记录", PeriodValueCompareRecord, test_PeriodValueCompareRecord),
        ("单指标的明细记录", SingleMetricDetailRecord, test_SingleMetricDetailRecord),
        ("单指标的明细汇总记录", SingleMetricStatsRecord, test_SingleMetricStatsRecord),
        ("未定义", UnparsedRawPersonalData, test_UnparsedRawPersonalData),
    ]

    total = 0
    passed = 0
    print("[self-test] route_raw_personal_data_to_dataclass")
    for exp_entity, exp_cls, lines in cases:
        print(f"  - 期望={exp_entity} 样例数={len(list(lines))}")
        for i, line in enumerate(lines):
            total += 1
            objs = route_raw_personal_data_to_dataclass(line)
            if not objs:
                raise AssertionError(f"[self-test][router] 失败：router 返回空列表 原文={str(line)!r}")

            # 1) 类型断言：最核心的“router 准确性”指标
            if not all(isinstance(x, exp_cls) for x in objs):
                reason = None
                for x in objs:
                    if isinstance(x, UnparsedRawPersonalData):
                        reason = x.原因
                        break
                raise AssertionError(
                    f"[self-test][router] 失败：期望={exp_entity}({exp_cls.__name__}) "
                    f"但实际={[type(x).__name__ for x in objs]!r} 实体类型={[getattr(x, '实体类型', None) for x in objs]!r} "
                    f"原因={reason!r} 原文={str(line)!r}"
                )

            # 2) 实体类型字符串：对非兜底类做更严格的一致性检查
            if exp_cls is not UnparsedRawPersonalData:
                # 新约定：除“单指标的明细记录”外，其它类型 router 期望返回长度为 1 的列表
                if exp_entity != "单指标的明细记录" and len(objs) != 1:
                    raise AssertionError(
                        f"[self-test][router] 失败：期望={exp_entity} 返回1个对象，但实际返回{len(objs)}个 "
                        f"原文={str(line)!r}"
                    )
                if any(getattr(x, "实体类型", None) != exp_entity for x in objs):
                    raise AssertionError(
                        f"[self-test][router] 失败：类型正确但实体类型字符串不一致："
                        f"期望实体类型={exp_entity!r} 实际实体类型={[getattr(x, '实体类型', None) for x in objs]!r} "
                        f"原文={str(line)!r}"
                    )
            else:
                # 兜底类：router 期望只返回 1 个 UnparsedRawPersonalData
                if len(objs) != 1:
                    raise AssertionError(
                        f"[self-test][router] 失败：期望兜底返回1个对象，但实际返回{len(objs)}个 "
                        f"原文={str(line)!r}"
                    )

            # 不打印具体解析结果内容（避免日志过大）；只打印必要的 case/条目与返回数
            # print(f"    [router] case={exp_entity} #{i} 原文前50={str(line)[:50]!r} 返回数={len(objs)}")

            passed += 1

    print(f"[self-test] route_raw_personal_data_to_dataclass 通过：{passed}/{total}")


def _self_test_recover_to_raw_data_roundtrip(
    *,
    test_SingleMetricDetailRecord: Sequence[str],
    test_PeriodValueSingleSummaryRecord: Sequence[str],
    test_PeriodValuemMultiSummaryRecord: Sequence[str],
    test_PeriodTextSummaryRecord: Sequence[str],
    test_PeriodValueCompareRecord: Sequence[str],
    test_SingleDateValueSingleSummaryRecord: Sequence[str],
    test_SingleDateValueMultiSummaryRecord: Sequence[str],
    test_SingleDateTextSummaryRecord: Sequence[str],
    test_NoDateValueSummaryRecord: Sequence[str],
    test_NoTimestampTextSummaryRecord: Sequence[str],
    test_SingleMetricStatsRecord: Sequence[str],
    test_UnparsedRawPersonalData: Sequence[str],
) -> None:
    """
    测试各数据类的 recover_to_raw_data()：
    - 覆盖每个实体类型与对应测试数据集
    - 对每条样例做 round-trip：raw -> dataclass -> recover -> router -> dataclass
    - 断言：
      1) recover_to_raw_data() 不抛异常
      2) recover 返回非空字符串
      3) recover 后再次 router 的类型与原类型一致
         - “单指标的明细记录”：允许返回多个对象，但每个对象都应为 SingleMetricDetailRecord
         - 其他类型：期望返回 1 个对象且类型匹配
    """
    def _short(s: Any, n: int = 220) -> str:
        t = str(s if s is not None else "").replace("\n", " ").strip()
        if n <= 0:
            return ""
        return t if len(t) <= n else (t[: n - 1] + "…")

    cases: list[tuple[str, type[PersonalDataPatternBase], Sequence[str]]] = [
        ("无时间日期的数值总结", NoDateValueSummaryRecord, test_NoDateValueSummaryRecord),
        ("无时间日期的文本总结", NoTimestampTextSummaryRecord, test_NoTimestampTextSummaryRecord),
        ("单日期数值单项总结", SingleDateValueSingleSummaryRecord, test_SingleDateValueSingleSummaryRecord),
        ("单日期数值多项总结", SingleDateValueMultiSummaryRecord, test_SingleDateValueMultiSummaryRecord),
        ("单日期文本总结", SingleDateTextSummaryRecord, test_SingleDateTextSummaryRecord),
        ("周期数值单项总结", PeriodValueSingleSummaryRecord, test_PeriodValueSingleSummaryRecord),
        ("周期数值多项总结", PeriodValuemMultiSummaryRecord, test_PeriodValuemMultiSummaryRecord),
        ("周期文本总结", PeriodTextSummaryRecord, test_PeriodTextSummaryRecord),
        ("周期数值对比记录", PeriodValueCompareRecord, test_PeriodValueCompareRecord),
        ("单指标的明细记录", SingleMetricDetailRecord, test_SingleMetricDetailRecord),
        ("单指标的明细汇总记录", SingleMetricStatsRecord, test_SingleMetricStatsRecord),
        ("未定义", UnparsedRawPersonalData, test_UnparsedRawPersonalData),
    ]

    total_lines = 0
    total_objs = 0
    passed = 0
    print("[self-test] recover_to_raw_data round-trip")

    for exp_entity, exp_cls, lines in cases:
        print(f"  - case={exp_entity} 样例数={len(list(lines))}")
        for i, raw in enumerate(lines):
            total_lines += 1
            print(f"    [#{i}]")
            print(f"      -  恢复：{_short(raw)}")
            objs = route_raw_personal_data_to_dataclass(raw)
            if not objs:
                raise AssertionError(f"[self-test][recover] router 返回空列表 case={exp_entity} #{i} 原文={str(raw)!r}")
            if not all(isinstance(x, exp_cls) for x in objs):
                raise AssertionError(
                    f"[self-test][recover] router 类型不匹配 case={exp_entity} #{i} "
                    f"期望={exp_cls.__name__} 实际={[type(x).__name__ for x in objs]!r} 原文={str(raw)!r}"
                )

            for k, obj in enumerate(objs):
                total_objs += 1
                try:
                    rec = obj.recover_to_raw_data()
                except Exception as e:
                    raise AssertionError(
                        f"[self-test][recover] recover_to_raw_data 异常 case={exp_entity} #{i} obj#{k} "
                        f"{type(e).__name__}: {e} 原文={str(raw)!r}"
                    )
                # print(f"      - obj#{k}({type(obj).__name__}) recover：{_short(rec)}")
                print(f"      -  恢复：{_short(rec)}")
                if not isinstance(rec, str) or not rec.strip():
                    raise AssertionError(
                        f"[self-test][recover] recover_to_raw_data 返回空/非字符串 case={exp_entity} #{i} obj#{k} "
                        f"返回={rec!r} 原文={str(raw)!r}"
                    )

                objs2 = route_raw_personal_data_to_dataclass(rec)
                if not objs2:
                    raise AssertionError(
                        f"[self-test][recover] recover 后 router 返回空列表 case={exp_entity} #{i} obj#{k} "
                        f"recover={rec!r}"
                    )

                # 类型断言：恢复后仍应归入相同实体类型（允许明细记录多返回）
                if exp_entity == "单指标的明细记录":
                    if not all(isinstance(x, SingleMetricDetailRecord) for x in objs2):
                        raise AssertionError(
                            f"[self-test][recover] recover 后类型不匹配(明细) case={exp_entity} #{i} obj#{k} "
                            f"实际={[type(x).__name__ for x in objs2]!r} recover={rec!r}"
                        )
                else:
                    if len(objs2) != 1 or not isinstance(objs2[0], exp_cls):
                        raise AssertionError(
                            f"[self-test][recover] recover 后类型不匹配 case={exp_entity} #{i} obj#{k} "
                            f"期望=1x{exp_cls.__name__} 实际={[type(x).__name__ for x in objs2]!r} recover={rec!r}"
                        )

                passed += 1

    print(f"[self-test] recover_to_raw_data round-trip 通过：{passed}/{total_objs}（行数={total_lines}）")


def _self_test_unparsed_raw_personal_data(lines: Sequence[str]) -> None:
    """
    用于快速验证 `UnparsedRawPersonalData.from_raw_personal_data()` 的兜底封装逻辑。
    """
    print(f"[self-test] UnparsedRawPersonalData 行数={len(list(lines))}")
    for i, line in enumerate(lines):
        parsed_list = UnparsedRawPersonalData.from_raw_personal_data(line, 原因="self-test: 强制走兜底")
        if not parsed_list:
            raise AssertionError(f"[self-test] 第{i}行解析返回空列表")
        if len(parsed_list) != 1:
            raise AssertionError(f"[self-test] 第{i}行期望返回1个对象，但得到{len(parsed_list)}个")
        parsed = parsed_list[0]
        if not isinstance(parsed, UnparsedRawPersonalData):
            raise AssertionError(f"[self-test] 第{i}行返回类型错误：实际类型={type(parsed).__name__}")
        if not parsed.个人数据:
            raise AssertionError(f"[self-test] 第{i}行兜底对象的个人数据为空")
        print(f"[self-test] #{i}")
        print(_indent_lines(parsed.format_print(max_items=6, max_len=220), 2))


def _self_test_aggregate_patterns_to_formatted_text(
    lines: Sequence[str],
    *,
    max_cases: Optional[int] = 30,
    print_preview: bool = True,
) -> None:
    """
    self-test：验证
    - explode_newlines_and_route_to_dataclasses()
    - aggregate_patterns_to_formatted_text()
    的端到端行为（从原始多行文本 -> 数据类列表 -> 聚合格式化输出）。
    """
    xs = list(lines or [])
    if max_cases is not None:
        xs = xs[: max(0, int(max_cases))]

    print(f"[self-test] aggregate_patterns_to_formatted_text 样本数={len(xs)} format=markdown")

    ok = 0
    skipped_empty = 0
    for i, text in enumerate(xs):
        raw = (text or "").strip()
        if not raw:
            skipped_empty += 1
            continue
        patterns = explode_newlines_and_route_to_dataclasses(raw)
        out = aggregate_patterns_to_formatted_text(patterns)

        # 关键断言：输出不应为空（即使全是兜底类，也应产生 loose_lines 或 markdown 的零散段）
        if not isinstance(out, str) or not out.strip():
            raise AssertionError(f"[self-test][agg] 输出为空/非字符串：case#{i} raw={raw}")

        # 更具体的结构断言（不过度严格，避免因表头细节调整导致测试脆弱）
        if ("### " not in out) and ("| " not in out) and ("零散或无法聚合" not in out):
            raise AssertionError(f"[self-test][agg] markdown 输出结构异常：case#{i} raw={raw}")

        ok += 1
        if print_preview:
            print(f"  - case#{i} 解析对象数={len(patterns)}")
            print(f"    原始=\"\"\"\n{raw}\n\"\"\"")
            print(f"    输出=\"\"\"\n{out}\n\"\"\"")
            # print(f"    对象=\"\"\"\n{patterns}\n\"\"\"")

    if skipped_empty:
        print(f"[self-test] aggregate_patterns_to_formatted_text 跳过空样本：{skipped_empty}")
    print(f"[self-test] aggregate_patterns_to_formatted_text 通过：{ok}/{len(xs) - skipped_empty}")

