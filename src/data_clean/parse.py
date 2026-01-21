from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from .models import (
    NoDateValueSummaryCore,
    NoDateValueSummaryRecord,
    NoTimestampTextSummaryCore,
    NoTimestampTextSummaryRecord,
    PeriodSummaryCore,
    PeriodTextSummaryCore,
    PeriodTextSummaryRecord,
    PeriodValueCompareCore,
    PeriodValueCompareRecord,
    PeriodValueSingleSummaryRecord,
    PeriodValueSummaryCore,
    PeriodValuemMultiSummaryRecord,
    PersonalDataPattern,
    SingleDateTextSummaryCore,
    SingleDateTextSummaryRecord,
    SingleDateValueMultiSummaryCore,
    SingleDateValueMultiSummaryRecord,
    SingleDateValueSummaryCore,
    SingleDateValueSingleSummaryRecord,
    SingleMetricDetailRecord,
    SingleMetricStatsRecord,
    SingleValueCore,
    StatsCompositeCore,
    StatsCompositeDataItem,
    StatsCompositeSummaryItem,
    UnparsedRawPersonalData,
    _PERIOD_SUMMARY_SEG_RE_1,
    _PERIOD_SUMMARY_SEG_RE_3,
    _PERIOD_TEXT_SUMMARY_SEG_RE,
    _PERIOD_VALUE_SUMMARY_SEG_RE,
    _PVC_CLAUSE_RE,
    _SINGLE_DATE_HEAD_RE,
    _SINGLE_DATE_VALUE_SUMMARY_PREFIX_SEG_RE,
    _SINGLE_DATE_VALUE_SUMMARY_SEG_RE,
    _STATS_COMP_HEAD_RE,
    _find_bracket_span,
    _has_time_unit,
)


_SPLIT_RE = re.compile(r"[，,]\s*")
_FIRST_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

# 特化脏数据拦截（router 入口级别）：
# - 形如：
#   活动心率：[4月12日87-194, 平均145次/分钟, 4月13日100-177, 平均132次/分钟, ...]，
#   平均活动心率128次/分钟，最低活动心率72次/分钟，最高活动心率194次/分钟
# - 该类数据会被“周期数值多项总结”等解析器误抢解析；按用户约定，直接归入 UnparsedRawPersonalData。
_DIRTY_ACTIVITY_HEART_RATE_RANGE_AVG_RE = re.compile(
    r"^\s*活动心率\s*[:：]\s*[\[【][\s\S]*?"
    r"(?:\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2})\s*\d+\s*-\s*\d+[\s\S]*?"
    # 明细列表内混入“平均xx(次/分钟)?”，单位可能缺失
    r"平均\s*\d+(?:\s*(?:次\s*/\s*分钟|bpm|BPM))?[\s\S]*?[\]】][\s\S]*?"
    # 末尾汇总必须包含“平均活动心率xx(次/分钟)?”，单位也可能缺失
    r"平均\s*活动心率\s*\d+(?:\s*(?:次\s*/\s*分钟|bpm|BPM))?",
    re.DOTALL,
)

# 特化脏数据拦截（router 入口级别）：
# - 形如：
#   2月17日22时07分体重70.4千克
# - 问题：缺少“的/是/为/：”等连接词，容易被“单日期数值单项总结”等解析器误判为
#   (指标="", 数值="22", 单位="时07分体重70.4千克") 这类脏结构，导致 Markdown 表格信息遗漏/空指标行。
# - 按用户约定：该类数据直接归入 UnparsedRawPersonalData，但在 `原因` 中保留可读的关键信息片段，避免信息丢失。
_DIRTY_WEIGHT_DETAIL_NO_DELIM_RE = re.compile(
    r"^\s*"
    r"(?P<date>\d{1,2}月\d{1,2}日)\s*"
    r"(?P<time>(?:\d{1,2}\s*[:：]\s*\d{2})|(?:\d{1,2}\s*时\s*\d{1,2}\s*分?))\s*"
    r"体重\s*"
    r"(?P<val>[-+]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>千克|kg|KG)\b"
    r"(?P<tail>[\s\S]*)$",
    re.DOTALL,
)

_FALLBACK_PARSERS: list[tuple[str, str, Callable[[str], list[PersonalDataPattern]]]] = [
    ("单指标的明细汇总记录", "单指标的明细汇总记录(全量兜底)", lambda s: SingleMetricStatsRecord.from_raw_personal_data(s)),
    ("周期数值对比记录", "周期数值对比记录(全量兜底)", lambda s: PeriodValueCompareRecord.from_raw_personal_data(s)),
    ("单日期数值多项总结", "单日期数值多项总结(全量兜底)", lambda s: SingleDateValueMultiSummaryRecord.from_raw_personal_data(s)),
    ("单日期数值单项总结", "单日期数值单项总结(全量兜底)", lambda s: SingleDateValueSingleSummaryRecord.from_raw_personal_data(s)),
    ("单日期文本总结", "单日期文本总结(全量兜底)", lambda s: SingleDateTextSummaryRecord.from_raw_personal_data(s)),
    ("单指标的明细记录", "单指标的明细记录(全量兜底)", lambda s: SingleMetricDetailRecord.from_raw_personal_data(s)),
    ("周期数值单项总结", "周期数值单项总结(全量兜底)", lambda s: PeriodValueSingleSummaryRecord.from_raw_personal_data(s)),
    ("周期数值多项总结", "周期数值多项总结(全量兜底)", lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s)),
    ("周期文本总结", "周期文本总结(全量兜底)", lambda s: PeriodTextSummaryRecord.from_raw_personal_data(s)),
    ("无时间日期的数值总结", "无时间日期的数值总结(全量兜底)", lambda s: NoDateValueSummaryRecord.from_raw_personal_data(s)),
    ("无时间日期的文本总结", "无时间日期的文本总结(全量兜底)", lambda s: NoTimestampTextSummaryRecord.from_raw_personal_data(s)),
]


def _parse_style_item_single_metric_detail(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return SingleMetricDetailRecord(
        核心字段=SingleValueCore(
            日期=str(core.get("日期", "")),
            时间=str(core.get("时间", "")),
            指标名称=str(core.get("指标名称", "")),
            数值类型=core.get("数值类型"),  # type: ignore[assignment]
            单位=str(core.get("单位", "")),
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_period_value_single(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return PeriodValueSingleSummaryRecord(
        核心字段=PeriodSummaryCore(
            开始日期=str(core.get("开始日期", "")),
            结束日期=str(core.get("结束日期", "")),
            指标名称=str(core.get("指标名称", "")),
            数值类型=core.get("数值类型"),  # type: ignore[assignment]
            单位=str(core.get("单位", "")),
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_period_text(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return PeriodTextSummaryRecord(
        核心字段=PeriodTextSummaryCore(
            开始日期=str(core.get("开始日期", "")),
            结束日期=str(core.get("结束日期", "")),
            指标名称=str(core.get("指标名称", "")),
            状态描述=str(core.get("状态描述", "")),
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_period_compare(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return PeriodValueCompareRecord(
        核心字段=PeriodValueCompareCore(
            日期范围1=str(core.get("日期范围1", "")),
            日期范围2=str(core.get("日期范围2", "")),
            指标名称=str(core.get("指标名称", "")),
            数值类型=core.get("数值类型"),  # type: ignore[assignment]
            单位=str(core.get("单位", "")),
            对比逻辑类型=str(core.get("对比逻辑类型", "")),
            差异数值类型=core.get("差异数值类型"),  # type: ignore[assignment]
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_period_value_multi(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return PeriodValuemMultiSummaryRecord(
        核心字段=PeriodValueSummaryCore(
            开始日期=str(core.get("开始日期", "")),
            结束日期=str(core.get("结束日期", "")),
            指标名称=str(core.get("指标名称", "")),
            数值类型=core.get("数值类型"),  # type: ignore[assignment]
            单位=str(core.get("单位", "")),
            状态描述=str(core.get("状态描述", "")),
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_single_date_value_single(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return SingleDateValueSingleSummaryRecord(
        核心字段=SingleDateValueSummaryCore(
            指标名称=str(core.get("指标名称", "")),
            日期=str(core.get("日期", "")),
            数值类型=core.get("数值类型"),  # type: ignore[assignment]
            单位=str(core.get("单位", "")),
            状态描述=str(core.get("状态描述", "")),
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_single_date_text(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return SingleDateTextSummaryRecord(
        核心字段=SingleDateTextSummaryCore(
            指标名称=str(core.get("指标名称", "")),
            时间=str(core.get("时间", "")),
            状态描述=str(core.get("状态描述", "")),
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_no_timestamp_text(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return NoTimestampTextSummaryRecord(
        核心字段=NoTimestampTextSummaryCore(
            指标名称=str(core.get("指标名称", "")),
            状态描述=str(core.get("状态描述", "")),
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_no_date_value(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return NoDateValueSummaryRecord(
        核心字段=NoDateValueSummaryCore(
            指标名称=str(core.get("指标名称", "")),
            数值类型=core.get("数值类型"),  # type: ignore[assignment]
            单位=str(core.get("单位", "")),
            状态描述=str(core.get("状态描述", "")),
        ),
        原始个人数据=raw_personal_data,
    )


def _parse_style_item_stats_composite(core: Mapping[str, Any], _raw_personal_data: str | None) -> PersonalDataPattern:
    data_list_raw = core.get("数据列表")
    sum_list_raw = core.get("统计汇总描述")
    data_list: list[StatsCompositeDataItem] = []
    sum_list: list[StatsCompositeSummaryItem] = []

    if isinstance(data_list_raw, Sequence):
        for x in data_list_raw:
            if isinstance(x, Mapping):
                data_list.append(
                    StatsCompositeDataItem(
                        日期=str(x.get("日期", "")),
                        数值类型=x.get("数值类型"),  # type: ignore[assignment]
                        单位=str(x.get("单位", "")),
                    )
                )
    if isinstance(sum_list_raw, Sequence):
        for x in sum_list_raw:
            if isinstance(x, Mapping):
                sum_list.append(
                    StatsCompositeSummaryItem(
                        指标名称=str(x.get("指标名称", "")),
                        数值类型=x.get("数值类型"),  # type: ignore[assignment]
                        单位=str(x.get("单位", "")),
                        状态描述=str(x.get("状态描述", "")),
                    )
                )

    # 兼容历史 style：core 里可能没有“数值类型/单位”，则从列表里兜底推断
    core_value_type = core.get("数值类型")
    if core_value_type is None:
        core_value_type = data_list[0].数值类型 if data_list else (sum_list[0].数值类型 if sum_list else "String")
    core_unit = core.get("单位")
    if core_unit is None:
        core_unit = data_list[0].单位 if data_list else (sum_list[0].单位 if sum_list else "无")

    return SingleMetricStatsRecord(
        核心字段=StatsCompositeCore(
            指标名称=str(core.get("指标名称", "")),
            数值类型=core_value_type,  # type: ignore[arg-type]
            单位=str(core_unit),
            数据列表=data_list,
            统计汇总描述=sum_list,
        )
    )


def _parse_style_item_single_date_value_multi(core: Mapping[str, Any], raw_personal_data: str | None) -> PersonalDataPattern:
    return SingleDateValueMultiSummaryRecord(
        核心字段=SingleDateValueMultiSummaryCore(
            指标名称=str(core.get("指标名称", "")),
            日期=str(core.get("日期", "")),
            数值类型=core.get("数值类型"),  # type: ignore[assignment]
            单位=str(core.get("单位", "")),
            状态描述=str(core.get("状态描述", "")),
        ),
        原始个人数据=raw_personal_data,
    )


_STYLE_ITEM_PARSERS: dict[str, Callable[[Mapping[str, Any], str | None], PersonalDataPattern]] = {
    "单指标的明细记录": _parse_style_item_single_metric_detail,
    "周期数值单项总结": _parse_style_item_period_value_single,
    "周期文本总结": _parse_style_item_period_text,
    "周期数值对比记录": _parse_style_item_period_compare,
    "周期数值多项总结": _parse_style_item_period_value_multi,
    "单日期数值单项总结": _parse_style_item_single_date_value_single,
    "单日期文本总结": _parse_style_item_single_date_text,
    "无时间日期的文本总结": _parse_style_item_no_timestamp_text,
    "无时间日期的数值总结": _parse_style_item_no_date_value,
    "单指标的明细汇总记录": _parse_style_item_stats_composite,
    "单日期数值多项总结": _parse_style_item_single_date_value_multi,
}


def parse_style_item_to_dataclass(
    item: Mapping[str, Any],
    *,
    raw_personal_data: str | None = None,
) -> PersonalDataPattern:
    """
    将单个 style item(dict) 转为对应的数据类。
    解析失败时返回 `UnparsedRawPersonalData`（优先使用 raw_personal_data 作为原始文本）。
    """
    et = str(item.get("实体类型", "")).strip()
    core: Any = item.get("核心字段")
    if not isinstance(core, Mapping):
        # 容错：用户/日志样例里可能把 core 直接平铺到了 item 顶层（缺少 "核心字段" 包裹）。
        # 由于 `predict_personal_data.py` 的强校验要求 item 只能有 {"实体类型","核心字段"}，
        # 这种结构一般不会来自模型真实输出，但这里做兼容以方便调试/迁移。
        if et == "单日期数值多项总结" and all(
            k in item for k in ("指标名称", "日期", "数值类型", "单位", "状态描述")
        ):
            core = item
        else:
            return UnparsedRawPersonalData(
                个人数据=raw_personal_data or "",
                原因="核心字段 缺失或不是对象(dict)",
                原始样式输出=dict(item),
            )

    try:
        fn = _STYLE_ITEM_PARSERS.get(et)
        if fn is None:
            return UnparsedRawPersonalData(
                个人数据=raw_personal_data or "",
                原因=f"未定义实体类型：{et!r}",
                原始样式输出=dict(item),
            )
        return fn(core, raw_personal_data)
    except Exception as e:
        return UnparsedRawPersonalData(
            个人数据=raw_personal_data or "",
            原因=f"解析 style item 异常：{type(e).__name__}: {e}",
            原始样式输出=dict(item),
        )


def parse_style_to_dataclasses(
    style: Any,
    *,
    raw_personal_data: str,
) -> list[PersonalDataPattern]:
    """
    将一行个人数据对应的 style(list) 转为数据类列表。
    - style 必须为 list[dict]；否则返回仅包含一个兜底类。
    """
    if not isinstance(style, list):
        return [
            UnparsedRawPersonalData(
                个人数据=raw_personal_data,
                原因=f"style 不是 list：实际类型={type(style).__name__}",
                原始样式输出=style,
            )
        ]
    out: list[PersonalDataPattern] = []
    for it in style:
        if isinstance(it, Mapping):
            out.append(parse_style_item_to_dataclass(it, raw_personal_data=raw_personal_data))
        else:
            out.append(
                UnparsedRawPersonalData(
                    个人数据=raw_personal_data,
                    原因=f"style item 不是 dict：实际类型={type(it).__name__}",
                    原始样式输出=it,
                )
            )
    return out


def route_raw_personal_data_to_dataclass(
    raw_line: str,
    *,
    strict_uncovered_to_unparsed: bool = True,
) -> list[PersonalDataPattern]:
    """
    Router：输入“一条个人数据长字符串”，自动判定数据类型并分发到对应的数据类解析器。

    设计目标：
    - 用户不提供实体类型时，尽量自动识别（基于特征 + 多候选打分）。
    - **统一返回 list[PersonalDataPattern]**：
      - 对于常规情况返回长度为 1 的列表
      - 对于“单指标的明细记录”的多指标场景，可能返回多个记录对象
      - 解析失败时，返回仅包含一个 `UnparsedRawPersonalData` 的列表，保留原始文本与失败原因。
    """

    raw = str(raw_line or "").strip()
    if not raw:
        return [UnparsedRawPersonalData(个人数据=raw, 原因="空行，router 无法判定数据类型")]

    # 用户允许的特化规则：活动心率“数字范围 + 平均”脏数据直接进入 Unparsed（避免被其它解析器误判）。
    if _DIRTY_ACTIVITY_HEART_RATE_RANGE_AVG_RE.search(raw):
        return [UnparsedRawPersonalData(个人数据=raw, 原因="活动心率(范围+平均)脏数据：按特化规则不解析")]

    # 用户允许的特化规则：体重“日期+时间+体重+数值”但缺少连接词的脏数据直接进入 Unparsed（避免误判为空指标行）。
    m_w = _DIRTY_WEIGHT_DETAIL_NO_DELIM_RE.match(raw)
    if m_w:
        val = (m_w.group("val") or "").strip()
        unit = (m_w.group("unit") or "").strip()
        # 统一展示单位（让下游“信息覆盖检查”更稳定）
        unit_norm = "千克" if unit.lower() == "kg" else (unit or "千克")
        hint = f"体重 {val} {unit_norm}".strip()
        return [
            UnparsedRawPersonalData(
                个人数据=raw,
                原因=(
                    "体重明细脏数据：日期+时间后直接拼接体重数值，缺少“的/是/为/：”等连接词，"
                    "会造成解析歧义；按特化规则归入未解析。"
                    f"抽取信息：{hint}"
                ),
            )
        ]

    # 注意：这里有一些“当前体系不覆盖”的句式（来自金标准 self-test），
    # 这些句式即使能被某些解析器“形式上解析出数值/单位”，语义也会被误归类。
    #
    # 因此提供一个开关：
    # - strict_uncovered_to_unparsed=True（默认）：命中这些句式就直接 Unparsed（保持金标准一致）
    # - strict_uncovered_to_unparsed=False：不强制兜底，继续尝试其它数据类（用于探索/扩展覆盖）
    _uncovered_hints: list[str] = []

    # 常见“状态/评价”后缀：用于 router 判别
    # 说明：
    # - 如果一句话带有“正常/偏晚/偏低/欠规律...”等状态词，通常更像“周期数值多项总结/单日期数值单项总结”
    #   而不是“周期数值单项总结”（后者往往是“累计/总计/合计/平均...为XXX单位”的纯汇总值）。
    _STATUS_SUFFIX_RE = re.compile(
        r"(欠规律|不规律|偏晚|偏早|偏低|偏高|偏少|偏多|偏长|偏短|过少|过多|过长|过短|正常|中等|一般|良好|标准|达标|优秀|较低|较高|不足|过低|过高|不佳|较差|偏重|超重|肥胖|偏瘦|偏胖|偏轻)\s*$"
    )

    # 先处理已知“当前体系不覆盖/应兜底”的句式（来自金标准 test_UnparsedRawPersonalData）
    # - 目标差距类：锻炼时长XX分钟，距离目标XX分钟还差XX分钟
    # - 达到目标类：活动热量424千卡，达到目标270千卡
    # - 情绪占比类：8/8占比最高的情绪是不愉悦
    # - 经期/预测经期类：经期为4/7~4/19，共13天；1/25是经期第2天
    if ("距离目标" in raw and "还差" in raw) or ("占比最高的情绪" in raw and "是" in raw):
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为未覆盖的句式，按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：距离目标/情绪占比最高（历史上按金标准走兜底）")
    if "达到目标" in raw:
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为达到目标类句式（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：达到目标（历史上按金标准走兜底）")
    # - 标签/称号类：4/1~4/15 活力达人（非指标总结，按金标准走兜底）
    if "活力达人" in raw:
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为标签/称号类句式（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：标签/称号（活力达人）")
    if re.search(r"(预测)?经期", raw):
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为经期/预测经期相关句式（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：经期/预测经期")
    # - 单独的“时间跨度阈值”短句：超过两年（缺少指标语义，按金标准走兜底）
    if re.fullmatch(r"\s*超过\s*[一二两三四五六七八九十0-9]+(?:\.\d+)?\s*年\s*", raw):
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为孤立的时间跨度阈值短句（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：孤立时间跨度阈值（例如“超过两年”）")
    # - 目标完成度/KPI 类：步数目标完成天数10天，完成率32%（当前未覆盖，按金标准走兜底）
    if ("目标完成天数" in raw) or ("完成率" in raw and "目标" in raw):
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为目标完成度/完成率类句式（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：目标完成度/完成率")

    def _has_date_token(s: str) -> bool:
        # 仅做“是否含日期线索”的粗判定（不要过重正则）
        return bool(
            re.search(
                # 只扩展“带年份的日期”支持 . / - 分隔，避免把 79.0 这类小数误判为日期
                r"(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2})|(\d{1,2}/\d{1,2})|(\d{1,2}月\d{1,2}(?:日)?)",
                s,
            )
        )

    def _has_time_token(s: str) -> bool:
        # 注意：Python 的 \w 会把中文也算作“单词字符”，导致 \b 在 "06:07的" 这类场景失效；
        # 因此这里用更稳的“直接搜索时间形态”。
        return bool(re.search(r"(\d{1,2}:\d{2})|(\d{1,2}\s*(?:时|点)\s*\d{1,2}\s*(?:分)?)", s))

    def _is_duration_like_value(s: str) -> bool:
        """
        判断字符串 s 是否更像“时长值”而不是“比值/速率”。

        背景：
        - router 在区分“周期数值单项总结”与“周期数值多项总结”时，会把“平均/最长/最短...”+时长
          的组合偏向解释为“周期数值多项总结”（例如：平均入睡时间23:20 / 最长睡眠时长7小时）。
        - 但像“平均配速 7.80分钟/公里”“平均心率 136次/分钟”这类 **含分母(/)** 的比值，
          虽然包含“分钟”等时间单位，但语义并不是“时长”，更符合“周期数值单项总结”的写法。
        """
        t = (s or "").strip()
        if not t:
            return False
        # 含分母(/)通常代表比值/速率，而非“时长”
        if "/" in t:
            return False
        return _has_time_unit(t)

    def _looks_like_stats_composite(s: str) -> bool:
        # 形如：指标名称：[...]
        if not _STATS_COMP_HEAD_RE.match(s):
            return False
        m = _STATS_COMP_HEAD_RE.match(s)
        if not m:
            return False
        rest = str(m.group("rest") or "")
        return _find_bracket_span(rest) is not None

    def _looks_like_single_date_stats_composite(s: str) -> bool:
        # 形如：8月8日压力15分-15分, 平均..., 最高..., 最低...
        if not _SINGLE_DATE_HEAD_RE.match(s):
            return False
        # 如果是“日期范围”（如 6/16~6/22），不要误判为单日期统计复合
        if re.match(
            # 兼容 “2025/2/1日到...”：slash 日期后可能带“日”
            r"^\s*((?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}(?:日)?|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
            s,
        ):
            return False
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        if len(segs) < 2:
            return False
        # 若整句更像“周期数值对比记录”（两段为/：子句 + 差异段），不要误判为统计复合
        # 例如：2025/3/6的平均浅睡比例为56，2025/2/27~2025/3/5平均浅睡比例为49.7，多6.3%
        if _looks_like_period_value_compare(s):
            return False
        # 有明显统计汇总词更可信
        return any(
            any(key in seg for key in ("平均", "最高", "最低", "最短", "最长", "最大", "最小"))
            for seg in segs[1:]
        )

    def _looks_like_single_date_value_summary(s: str) -> bool:
        # 形如：7/21锻炼时长17分钟偏低 / 4/23入睡时间01:40偏晚
        # 也支持“指标在前、日期在后”的常见写法：
        # - 皮肤温度：4月10日31摄氏度
        # - 体温：4月10日37摄氏度正常
        # - 必须是“单日期”而不是“日期范围”
        # - 必须包含数值（数字），否则更可能是“单日期文本总结”/“周期文本总结”
        # - 允许数值本身为时间点（如 01:40）；但要避免把“单指标的明细记录(YYYY/MM/DD HH:mm ...)”误判进来
        if re.search(r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}\s+\d{1,2}:\d{2}", s):
            return False
        # 同理：避免把“明细记录(YYYY年MM月DD日HH时mm分...)”误判为单日期总结
        if re.search(
            r"^\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*"
            r"(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:时|点)\s*\d{1,2}\s*(?:分)?)",
            s,
        ):
            return False
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        if not segs:
            return False
        first = segs[0]
        # 兼容两种头部形态：
        # 1) 日期开头：4/23锻炼时长17分钟偏低
        # 2) 指标开头：皮肤温度：4月10日31摄氏度
        rest: str | None = None
        m1 = _SINGLE_DATE_VALUE_SUMMARY_SEG_RE.match(first)
        if m1:
            # 防止把“日期范围”误判为单日期
            if re.match(
                # 兼容 “2025/2/1日到...”：slash 日期后可能带“日”
                r"^\s*((?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}(?:日)?|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
                first,
            ):
                return False
            rest = str(m1.group("rest") or "")
        else:
            m2 = _SINGLE_DATE_VALUE_SUMMARY_PREFIX_SEG_RE.match(first)
            if not m2:
                return False
            # 防止把“指标：日期范围 ...”误判为单日期（例如：体重：4/1~4/7平均...）
            if re.match(
                # 兼容 “指标：2025/2/1日到...” 这种写法：slash 日期后可能带“日”
                r"^\s*.+?[:：]\s*(?:\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}(?:日)?|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
                first,
            ):
                return False
            prefix = str(m2.group("prefix") or "").strip()
            rest_after_date = str(m2.group("rest") or "").strip()
            # 让后续“指标+数值+状态”拆解逻辑能拿到指标名：把 prefix 拼回 rest
            if prefix and not rest_after_date.startswith(prefix):
                rest = prefix + rest_after_date
            else:
                rest = rest_after_date
        # 关键：像 “2月13日20时33分的体重是61.1千克” 这种句式属于“明细记录”，
        # 若不排除，会被误判为“单日期数值单项总结”（因为 rest 里有数字）。
        if rest is None:
            return False
        if re.match(r"^\s*(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:时|点)\s*\d{1,2}\s*(?:分)?)\s*的", rest):
            return False
        return bool(_FIRST_NUMBER_RE.search(rest))

    def _looks_like_single_date_text_summary(s: str) -> bool:
        # 形如：8/2 睡眠得分中等，睡眠质量良好
        # - 必须是“单日期”而不是“日期范围”
        # - 不应包含数值（数字）；否则更可能是“单日期数值单项总结”
        # - 不应包含 HH:mm 时间点；否则更可能是“单日期数值单项总结/周期数值多项总结”
        if _has_time_token(s):
            return False
        if re.search(r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}\s+\d{1,2}:\d{2}", s):
            return False
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        if not segs:
            return False
        first = segs[0]
        m = _SINGLE_DATE_HEAD_RE.match(first)
        if not m:
            return False
        # 防止把“日期范围”误判为单日期
        if re.match(
            # 兼容 “2025/2/1日到...”：slash 日期后可能带“日”
            r"^\s*((?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}(?:日)?|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
            first,
        ):
            return False
        rest = str(m.group("rest") or "").strip()
        if not rest:
            return False
        return not bool(_FIRST_NUMBER_RE.search(rest))

    def _looks_like_period_value_compare(s: str) -> bool:
        # 形如：A的...为..., B的...为..., 少/多...
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        if len(segs) < 3:
            return False

        m1 = _PVC_CLAUSE_RE.match(segs[0])
        m2 = _PVC_CLAUSE_RE.match(segs[1])
        if not (m1 and m2):
            return False

        # 关键：对比记录的日期段，允许“单日 vs 周期”或“周期 vs 周期”，因此不强制两段都有范围分隔符。
        # 但至少应出现一次范围分隔符，或者两段日期本身不同（避免把单日统计复合误判为对比）。
        r1 = str(m1.group("range") or "")
        r2 = str(m2.group("range") or "")
        range_sep_re = re.compile(r"(到|至|~|～|-|—)")
        if not (range_sep_re.search(r1) or range_sep_re.search(r2) or (r1.strip() and r2.strip() and r1.strip() != r2.strip())):
            return False

        # 第三段应当是“差异描述”，而不是 또 一个 “...的...为...” 子句
        third = segs[2]
        if _PVC_CLAUSE_RE.match(third):
            return False

        # 差异段需要包含对比逻辑词（少/多/高/低/增加/减少/提升/下降... 或 早/晚/提前/延后 等）
        if not re.search(r"(少|多|高|低|增加|减少|提升|下降|降低|升高|早|晚|提前|延后|推迟|延迟)", third):
            return False

        # 且差异段通常包含数字（如 3.0% / 6分钟）
        if not _FIRST_NUMBER_RE.search(third):
            return False

        return True

    def _looks_like_single_value(s: str) -> bool:
        # 形如：
        # - 2025/2/1 06:07的户外跑步距离为：5.11千米
        # - 2月13日20时33分的体重是61.1 千克
        # 关键：必须出现“起始即 日期 + 时间点”，并且整句包含“为/是”这类赋值语义，
        # 以避免把 “4/23入睡时间01:40偏晚” 这种“单日期总结(时间点是数值本身)”误判成明细记录。
        return bool(
            re.search(
                r"^\s*(?P<date>(?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*"
                r"(?P<time>\d{1,2}:\d{2}|\d{1,2}\s*(?:时|点)\s*\d{1,2}\s*(?:分)?)\s*(?:的)?",
                s,
            )
        ) and (("为" in s) or ("是" in s))

    def _is_all_unparsed(objs: list[PersonalDataPattern]) -> bool:
        return all(isinstance(x, UnparsedRawPersonalData) for x in objs)

    def _safe_call(
        name: str,
        fn: Callable[[str], list[PersonalDataPattern]],
    ) -> list[PersonalDataPattern]:
        try:
            out = fn(raw)
            if not out:
                return [UnparsedRawPersonalData(个人数据=raw, 原因=f"router 调用解析器异常：{name}: 返回空列表")]
            return out
        except Exception as e:
            return [UnparsedRawPersonalData(个人数据=raw, 原因=f"router 调用解析器异常：{name}: {type(e).__name__}: {e}")]

    # 统一“尝试某个解析器”的入口：
    # - 记录已尝试的解析器，避免在不同分支重复调用
    # - 解析成功（返回不全是 Unparsed）则立即返回
    _tried: set[str] = set()

    def _try_parser(
        key: str,
        display_name: str,
        fn: Callable[[str], list[PersonalDataPattern]],
    ) -> list[PersonalDataPattern] | None:
        if key in _tried:
            return None
        _tried.add(key)
        objs = _safe_call(display_name, fn)
        return None if _is_all_unparsed(objs) else objs

    def _looks_like_period_summary(s: str) -> bool:
        # 周期汇总：通常存在“日期范围 + (的)? + 指标 + 为/： + 数值”
        #
        # 但真实数据/金标准里也大量出现“不带 为/：分隔符”的写法，例如：
        # - 12月1日到12月31日总计清醒次数31次
        # - 6/1~6/30累计跑步距离201.27千米
        #
        # 这类句式如果不特殊处理，会被 _looks_like_period_value_summary 误判为“周期数值多项总结”。
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        summary_hint_keywords = (
            "总计",
            "总共",
            "合计",
            "累计",
            "总和",
            "共计",
            "共",
            "平均",
            "最大",
            "最小",
            "最高",
            "最低",
        )
        # “总量/累计”更像周期汇总；而“平均/最长/最短/最高/最低...”在没有总量信号时，
        # 很多是“周期数值多项总结”的多条统计项（尤其当值是时长/时间点）。
        total_hint_keywords = ("总计", "总共", "合计", "累计", "总和", "共计", "总")
        stats_only_keywords = ("平均", "最长", "最短", "最大", "最小", "最高", "最低")
        for seg in segs:
            if _PERIOD_SUMMARY_SEG_RE_1.match(seg):
                # 若“为/：”后面带明显状态词（如 "23:20正常" / "15分钟偏低"），更像“周期数值多项总结”，避免误判为汇总。
                m1 = _PERIOD_SUMMARY_SEG_RE_1.match(seg)
                if m1:
                    val = str(m1.group("val") or "").strip()
                    nm = str(m1.group("name") or "").strip()
                    # 若 val 本身是“时间点”(HH:mm)，通常是睡眠/入睡时间这类统计，更像“周期数值多项总结”
                    # 例如：平均入睡时间23:20 / 最晚入睡时间01:30
                    if val and _has_time_token(val):
                        continue
                    if val and _STATUS_SUFFIX_RE.search(val):
                        continue
                return True
            m3 = _PERIOD_SUMMARY_SEG_RE_3.match(seg)
            if not m3:
                continue
            rest = str(m3.group("rest") or "").strip()
            if not rest:
                continue
            # 必须含数字，否则更像“周期文本总结”
            if not _FIRST_NUMBER_RE.search(rest):
                continue
            # 若包含“时间点”(HH:mm)，更像“周期数值多项总结”，避免被误判成周期汇总
            # 例如：平均零星小睡入睡时间13:00
            if _has_time_token(rest):
                continue
            # 若 rest 结尾出现明显“状态/评价词”，更像“周期数值多项总结”（例如：平均入睡时间23:20正常）
            if _STATUS_SUFFIX_RE.search(rest):
                continue
            # 有“汇总信号词”更强烈指向“周期汇总”
            if any(k in rest for k in summary_hint_keywords):
                return True
        return False

    def _looks_like_period_value_summary(s: str) -> bool:
        # 周期数值多项总结：日期(或日期范围)+rest，且 rest 内有数值；并且不应带 HH:mm 时间点
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        for seg in segs:
            m = _PERIOD_VALUE_SUMMARY_SEG_RE.match(seg)
            if not m:
                continue
            rest = str(m.group("rest") or "")
            if _FIRST_NUMBER_RE.search(rest):
                return True
        return False

    def _looks_like_period_text_summary(s: str) -> bool:
        # 周期文本总结：日期(或日期范围)+rest，且 rest 大概率不含数值（更偏文本状态）
        if _has_time_token(s):
            return False
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        for seg in segs:
            m = _PERIOD_TEXT_SUMMARY_SEG_RE.match(seg)
            if not m:
                continue
            rest = str(m.group("rest") or "")
            if rest and (not _FIRST_NUMBER_RE.search(rest)):
                return True
        return False

    has_date = _has_date_token(raw)

    # 1) 强特征：多日期统计复合（带“指标: [..]”）
    if _looks_like_stats_composite(raw):
        got = _try_parser(
            "单指标的明细汇总记录",
            "单指标的明细汇总记录",
            lambda s: SingleMetricStatsRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 2) 强特征：周期对比（三段：A、B、差异）
    if _looks_like_period_value_compare(raw):
        got = _try_parser(
            "周期数值对比记录",
            "周期数值对比记录",
            lambda s: PeriodValueCompareRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 3) 强特征：单日期统计复合（日期开头 + 统计汇总词）
    if _looks_like_single_date_stats_composite(raw):
        got = _try_parser(
            "单日期数值多项总结",
            "单日期数值多项总结",
            lambda s: SingleDateValueMultiSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 3.5) 强特征：单日期数值单项总结（单日期 + 数值 + 状态）
    if _looks_like_single_date_value_summary(raw):
        got = _try_parser(
            "单日期数值单项总结",
            "单日期数值单项总结",
            lambda s: SingleDateValueSingleSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 3.6) 强特征：单日期文本总结（单日期 + 文本状态；无数值）
    if _looks_like_single_date_text_summary(raw):
        got = _try_parser(
            "单日期文本总结",
            "单日期文本总结",
            lambda s: SingleDateTextSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 4) 强特征：单次记录（含 YYYY/MM/DD + HH:mm）
    if _looks_like_single_value(raw):
        got = _try_parser(
            "单指标的明细记录",
            "单指标的明细记录",
            lambda s: SingleMetricDetailRecord.from_raw_personal_data(s),
        )
        if got is not None:
            # 关键变更：多指标情况下，直接返回全部记录（不再只取第一个）
            return got

    # 5) 周期汇总（日期范围 + 为 + 数值）
    if _looks_like_period_summary(raw):
        # 关键定义（按你的描述）：
        # - 周期数值多项总结：一条记录里 >=1 个指标记录；可能带状态词；且常见为“逗号分隔多条指标”，后段可继承日期范围
        # - 周期数值单项总结：一条记录结构中只有 1 个指标+数字，且没有状态词
        segs = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
        seg_has_status = any(_STATUS_SUFFIX_RE.search(x.strip()) for x in segs)
        seg_with_num_cnt = sum(1 for x in segs if _FIRST_NUMBER_RE.search(x))

        # 多指标 or 有状态：优先走“多项总结”
        if seg_has_status or seg_with_num_cnt >= 2:
            got = _try_parser(
                "周期数值多项总结",
                "周期数值多项总结",
                lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s),
            )
            if got is not None:
                return got

        # 否则才尝试“单项总结”（严格单指标、无状态）
        got_single = _try_parser(
            "周期数值单项总结",
            "周期数值单项总结",
            lambda s: PeriodValueSingleSummaryRecord.from_raw_personal_data(s),
        )
        if got_single is not None:
            obj = got_single[0]
            # 若解析出多条指标，按定义应归入“多项总结”
            try:
                if getattr(obj, "记录条数", 1) > 1:
                    got2 = _try_parser(
                        "周期数值多项总结",
                        "周期数值多项总结(由单项降级)",
                        lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s),
                    )
                    if got2 is not None:
                        return got2
            except Exception:
                pass
            return got_single

    # 6) 仍然有日期线索：在 周期数值多项总结 vs 周期文本总结 之间做更谨慎的选择
    if has_date:
        pv_ok = _looks_like_period_value_summary(raw)
        pt_ok = _looks_like_period_text_summary(raw)

        if pv_ok and not pt_ok:
            got = _try_parser(
                "周期数值多项总结",
                "周期数值多项总结",
                lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s),
            )
            if got is not None:
                return got
        if pt_ok and not pv_ok:
            got = _try_parser(
                "周期文本总结",
                "周期文本总结",
                lambda s: PeriodTextSummaryRecord.from_raw_personal_data(s),
            )
            if got is not None:
                return got

        # 两者都可能：都试一遍，做二选一（尽量避免把“文本总结”误判成“数值总结”）
        # 这里故意不走 _try_parser：需要拿到两边结果做比较（但仍会把 key 记为 tried，避免后面重复扫描）
        _tried.add("周期数值多项总结")
        _tried.add("周期文本总结")
        objs_v = _safe_call("周期数值多项总结", lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s))
        objs_t = _safe_call("周期文本总结", lambda s: PeriodTextSummaryRecord.from_raw_personal_data(s))
        v_bad, t_bad = _is_all_unparsed(objs_v), _is_all_unparsed(objs_t)
        if not v_bad and t_bad:
            return objs_v
        if not t_bad and v_bad:
            return objs_t
        if not v_bad and not t_bad:
            obj_v = objs_v[0]
            obj_t = objs_t[0]
            # 以“是否解析到了数值列表且包含数字”作为更强信号
            v_vals = getattr(obj_v, "数值列表", [])
            t_descs = getattr(obj_t, "状态描述列表", [])
            v_has_num = any(_FIRST_NUMBER_RE.search(str(x or "")) for x in (v_vals or []))
            t_has_num = any(_FIRST_NUMBER_RE.search(str(x or "")) for x in (t_descs or []))
            if v_has_num and not t_has_num:
                return objs_v
            if t_has_num and not v_has_num:
                return objs_t
            # 默认：更偏向文本总结（更保守）
            return objs_t

        # 日期线索但都失败：给一个最后的兜底尝试（可能是边界格式）
        got = _try_parser(
            "周期文本总结",
            "周期文本总结(兜底)",
            lambda s: PeriodTextSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got
        got = _try_parser(
            "周期数值多项总结",
            "周期数值多项总结(兜底)",
            lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 7) 无日期线索：NoDateValueSummary vs NoTimestampTextSummary
    has_digit = bool(_FIRST_NUMBER_RE.search(raw))
    if has_digit:
        got = _try_parser(
            "无时间日期的数值总结",
            "无时间日期的数值总结",
            lambda s: NoDateValueSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got
        got = _try_parser(
            "无时间日期的文本总结",
            "无时间日期的文本总结",
            lambda s: NoTimestampTextSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got
    else:
        got = _try_parser(
            "无时间日期的文本总结",
            "无时间日期的文本总结",
            lambda s: NoTimestampTextSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got
        got = _try_parser(
            "无时间日期的数值总结",
            "无时间日期的数值总结",
            lambda s: NoDateValueSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 8) 兜底策略：把剩余未尝试的解析器全部跑一遍
    # 目的：确保“只要存在一个可兼容的数据类，就不会过早 Unparsed”。
    for key, disp, fn in _FALLBACK_PARSERS:
        got = _try_parser(key, disp, fn)
        if got is not None:
            return got

    reason = "router 未能判定：所有数据类解析器均不兼容/解析失败"
    if _uncovered_hints:
        reason = reason + "；提示：" + "；".join(_uncovered_hints)
    return [UnparsedRawPersonalData(个人数据=raw, 原因=reason)]


def explode_newlines_and_route_to_dataclasses(
    text: str,
    *,
    strict_uncovered_to_unparsed: bool = True,
    keep_empty: bool = False,
    normalize_escaped_newlines: bool = True,
    max_rounds: int = 50,
) -> list[PersonalDataPattern]:
    """
    将一段文本按“换行裂变”拆成多行，然后对每一行做数据类分发与加载，最终返回总的数据类列表。

    设计意图：
    - 支持输入是一整段多行文本（例如复制粘贴的日志/对话/多条个人数据拼在一起）
    - 对 `\\n` / `\\r\\n` 这类“字面量转义换行”做可选归一化
    - 反复执行 splitlines() 直到不再产生新的换行（防御极端/脏数据）

    参数：
    - strict_uncovered_to_unparsed: 透传给 `route_raw_personal_data_to_dataclass`，默认 True（保持金标准策略）
    - keep_empty: 是否保留空行（默认 False，空行直接丢弃）
    - normalize_escaped_newlines: 是否把字面量 "\\n"/"\\r\\n"/"\\r" 转成真实换行（默认 True）
    - max_rounds: 最大裂变轮数（防御性参数，避免意外死循环）
    """
    raw_text = "" if text is None else str(text)
    if normalize_escaped_newlines and raw_text:
        # 注意顺序：先处理 \r\n，再处理单独的 \r 与 \n
        raw_text = raw_text.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")

    def _looks_like_external_suggestion_noise_block(s: str) -> bool:
        """
        外部环节引入的意外脏数据：典型特征为
        - 多行“时间+指标+...为+数值(可带单位)”；
        - 同一段中混入“，建议...”/“,建议...” 这类建议句（常见为被错误换行切断成单独一行）。
        该类数据不应进入结构化解析链路，直接整段兜底即可。
        """
        t = (s or "").strip()
        if not t:
            return False
        if "建议" not in t:
            return False
        # 既可能是同一行 “...,建议...”，也可能是换行后单独一行 “，建议...”
        has_suggest = bool(re.search(r"(^|[\n\r])\s*[，,]\s*建议", t)) or bool(re.search(r"[，,]\s*建议", t))
        if not has_suggest:
            return False
        # 时间线索：样例中是 “2025年02月16日19时17分”；也兼容 “YYYY/MM/DD HH:mm”
        has_cn_datetime = bool(re.search(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*\d{1,2}\s*时\s*\d{1,2}\s*分", t))
        has_std_datetime = bool(re.search(r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}\s+\d{1,2}:\d{2}", t))
        if not (has_cn_datetime or has_std_datetime):
            return False
        # 指标/数值线索：大多数行会包含 “为”
        if "为" not in t:
            return False
        return True

    if _looks_like_external_suggestion_noise_block(raw_text):
        reason = "检测到外部意外数据：包含“逗号+建议”段落（可能被错误换行切断），按整段兜底"
        return [UnparsedRawPersonalData(个人数据=raw_text.strip(), 原因=reason)]

    # “裂变”：反复 splitlines() + flatten，直到稳定
    chunks: list[str] = [raw_text]
    rounds = max(1, int(max_rounds))
    for _ in range(rounds):
        changed = False
        new_chunks: list[str] = []
        for ch in chunks:
            s = "" if ch is None else str(ch)
            parts = s.splitlines()
            if len(parts) <= 1:
                new_chunks.append(s)
                continue
            changed = True
            new_chunks.extend(parts)
        chunks = new_chunks
        if not changed:
            break

    # 清洗：strip + 去空
    lines: list[str] = []
    for ch in chunks:
        t = (ch or "").strip()
        if not t and not keep_empty:
            continue
        lines.append(t)

    # 对每一行路由并聚合
    out: list[PersonalDataPattern] = []
    for ln in lines:
        if (not ln) and (not keep_empty):
            continue
        out.extend(
            route_raw_personal_data_to_dataclass(
                ln,
                strict_uncovered_to_unparsed=strict_uncovered_to_unparsed,
            )
        )
    return out


__all__ = [
    "parse_style_item_to_dataclass",
    "parse_style_to_dataclasses",
    "route_raw_personal_data_to_dataclass",
    "explode_newlines_and_route_to_dataclasses",
]


