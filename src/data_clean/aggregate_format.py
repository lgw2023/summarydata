from __future__ import annotations

import re
from typing import Any, Sequence

from .models import (
    NoDateValueSummaryRecord,
    NoTimestampTextSummaryRecord,
    PeriodTextSummaryRecord,
    PeriodValueCompareRecord,
    PeriodValueSingleSummaryRecord,
    PeriodValuemMultiSummaryRecord,
    PersonalDataPattern,
    SingleDateTextSummaryRecord,
    SingleDateValueMultiSummaryRecord,
    SingleDateValueSingleSummaryRecord,
    SingleMetricDetailRecord,
    SingleMetricStatsRecord,
    UnparsedRawPersonalData,
)
from .normalize import _attach_unit_to_value, _is_missing_token

_PREFER_COLS_BASE: list[str] = [
    "类别",
    "日期",
    "时间",
    "运动类型",
    "开始",
    "结束",
    "范围1",
    "值1",
    "范围2",
    "值2",
    "指标",
    "数值",
    "单位",
    "状态",
    "逻辑",
    "差异",
]

_METRIC_ORDER_SINGLE_METRIC_DETAIL: list[str] = [
    # 这里匹配“指标名后缀”，因为列名现在保留完整形式（如“登山速度”“跑步距离”等）
    "距离",
    "用时",
    "时长",
    "运动时间",
    "配速",
    "最快配速",
    "速度",
    "热量",
    "心率",
    "最大心率",
    "平均心率",
    "步频",
    "最大步频",
    "步幅",
    "无氧训练压力",
    "有氧训练压力",
    "训练压力",
]

_METRIC_ORDER_STATS_COMPOSITE: list[str] = [
    "运动次数",
    "运动时长",
    "运动热量",
    "活动小时数",
    "活动热量",
    "锻炼时长",
    "距离",
    "用时",
    "时长",
    "配速",
    "最快配速",
    "速度",
    "热量",
    "心率",
    "最大心率",
    "步频",
    "最大步频",
    "步幅",
]


def aggregate_patterns_to_formatted_text(
    patterns: Sequence[PersonalDataPattern],
    *,
    include_loose_lines: bool = True,
) -> str:
    """
    聚合/汇总一个数据类列表，并输出为可读文本（Markdown 表格）。

    目标：
    - “同类数据”尽量按 `实体类型` 聚合为表格/rows
    - 对无法抽取为表格行的对象，输出一个精简单行，并追加在主体输出之后，保证信息不丢失

    参数：
    - include_loose_lines: 是否在主体输出后追加“零散或无法聚合”的单行列表
    """
    objs = list(patterns or [])
    if not objs:
        return ""

    def _safe_str(x: Any) -> str:
        return str(x if x is not None else "").strip()

    def _cell(v: Any) -> str:
        s = _safe_str(v)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        # markdown 表格需要转义管道符
        s = s.replace("|", r"\|")
        return s

    def _pick_first_non_empty(*xs: Any) -> str:
        for x in xs:
            s = _safe_str(x)
            if s:
                return s
        return ""

    def _make_md_section_title(entity_type: str, *, date_or_range: str | None = None) -> str:
        """
        Markdown 分节标题生成规则：
        - 需要把日期/日期范围放在前面
        - 实体类型放在后面
        - 不使用括号

        例如：
        - "01月30日 单日期数值单项总结"
        - "01月19日~01月25日 周期数值单项总结"
        """
        et = _safe_str(entity_type) or "未定义"
        suf = _safe_str(date_or_range) if date_or_range is not None else ""
        if suf:
            return f"{suf} {et}".strip()
        return et

    def _rows_from_obj(obj: PersonalDataPattern) -> list[dict[str, str]] | None:
        """
        把一个数据类对象尽量展开成 rows（用于 Markdown 表格聚合）。
        返回：
        - list[dict]：可聚合
        - None：不可聚合（走 loose line）
        """
        et = _safe_str(getattr(obj, "实体类型", ""))
        if not et:
            return None

        # 兜底类：不聚合
        if isinstance(obj, UnparsedRawPersonalData):
            return None

        # 单指标的明细记录：按(日期/时间/数值)展开
        if isinstance(obj, SingleMetricDetailRecord):
            core = getattr(obj, "核心字段", None)
            metric = _safe_str(getattr(core, "指标名称", "")) if core else ""
            unit = _safe_str(getattr(core, "单位", "")) if core else ""
            ds = list(getattr(obj, "日期列表", []) or [])
            ts = list(getattr(obj, "时间列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
            n = max(len(ds), len(ts), len(vs))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                d = _safe_str(ds[i] if i < len(ds) else "")
                t = _safe_str(ts[i] if i < len(ts) else "")
                v = _safe_str(vs[i] if i < len(vs) else "")
                v2 = _attach_unit_to_value(v, unit) if (v and unit and unit != "无") else v
                rows.append(
                    {
                        "日期": d,
                        "时间": t,
                        "指标": metric,
                        "数值": v2,
                        "单位": unit or "无",
                    }
                )
            return rows

        # 周期数值单项总结
        if isinstance(obj, PeriodValueSingleSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            n = max(len(starts), len(ends), len(names), len(vals), len(units))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                st = _safe_str(starts[i] if i < len(starts) else "")
                ed = _safe_str(ends[i] if i < len(ends) else "")
                nm = _safe_str(names[i] if i < len(names) else "")
                v = _safe_str(vals[i] if i < len(vals) else "")
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                # 数值列保持“纯数值”，单位单独放在“单位”列（便于后续再聚合/透视）
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                v = _trim_number_like(v)
                rows.append(
                    {
                        "开始": st,
                        "结束": ed,
                        "指标": nm,
                        "数值": v,
                        "单位": u,
                    }
                )
            return rows

        # 周期文本总结
        if isinstance(obj, PeriodTextSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(starts), len(ends), len(names), len(descs))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                rows.append(
                    {
                        "开始": _safe_str(starts[i] if i < len(starts) else ""),
                        "结束": _safe_str(ends[i] if i < len(ends) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "状态": _safe_str(descs[i] if i < len(descs) else ""),
                    }
                )
            return rows

        # 周期数值对比记录
        if isinstance(obj, PeriodValueCompareRecord):
            r1s = list(getattr(obj, "日期范围1列表", []) or [])
            v1s = list(getattr(obj, "数值1列表", []) or [])
            r2s = list(getattr(obj, "日期范围2列表", []) or [])
            v2s = list(getattr(obj, "数值2列表", []) or [])
            diffs = list(getattr(obj, "差异数值列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            logics = list(getattr(obj, "对比逻辑类型列表", []) or [])
            n = max(len(r1s), len(v1s), len(r2s), len(v2s), len(diffs), len(names), len(logics))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                rows.append(
                    {
                        "范围1": _safe_str(r1s[i] if i < len(r1s) else ""),
                        "值1": _safe_str(v1s[i] if i < len(v1s) else ""),
                        "范围2": _safe_str(r2s[i] if i < len(r2s) else ""),
                        "值2": _safe_str(v2s[i] if i < len(v2s) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "逻辑": _safe_str(logics[i] if i < len(logics) else ""),
                        "差异": _safe_str(diffs[i] if i < len(diffs) else ""),
                    }
                )
            return rows

        # 周期数值多项总结
        if isinstance(obj, PeriodValuemMultiSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(starts), len(ends), len(names), len(vals), len(units), len(sts))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []

            def _strip_unit_from_value_for_table(v: str, unit: str) -> str:
                """
                周期数值多项总结的 Markdown 表格展示：
                - “单位”已单列展示，因此“数值”列应尽量保持为纯数值/纯区间
                - 兼容区间值：如 "96%-96%" -> "96-96"
                """
                t = _safe_str(v)
                u = _safe_str(unit) or "无"
                if (not t) or (not u) or u == "无":
                    return t
                # 区间：两侧分别剥离单位（避免只剥离尾部导致残留）
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
                # 普通：剥离尾部单位
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                return t2 if t2 else t

            for i in range(n):
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                v = _safe_str(vals[i] if i < len(vals) else "")
                # 数值列保持“纯数值”，单位单独放在“单位”列（避免出现“数值=0.6次 + 单位=次”的重复展示）
                v = _trim_number_like(_strip_unit_from_value_for_table(v, u))
                rows.append(
                    {
                        "开始": _safe_str(starts[i] if i < len(starts) else ""),
                        "结束": _safe_str(ends[i] if i < len(ends) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "数值": v,
                        "单位": u,
                        "状态": _safe_str(sts[i] if i < len(sts) else ""),
                    }
                )
            return rows

        # 单日期数值单项总结
        if isinstance(obj, SingleDateValueSingleSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(ds), len(names), len(vals), len(units), len(sts))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                v = _safe_str(vals[i] if i < len(vals) else "")
                # 数值列保持“纯数值”，单位单独放在“单位”列（避免出现“数值=233千卡 + 单位=千卡”的重复展示）
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                v = _trim_number_like(v)
                rows.append(
                    {
                        "日期": _safe_str(ds[i] if i < len(ds) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "数值": v,
                        "单位": u,
                        "状态": _safe_str(sts[i] if i < len(sts) else ""),
                    }
                )
            return rows

        # 单日期文本总结
        if isinstance(obj, SingleDateTextSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(ds), len(names), len(descs))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                rows.append(
                    {
                        "日期": _safe_str(ds[i] if i < len(ds) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "状态": _safe_str(descs[i] if i < len(descs) else ""),
                    }
                )
            return rows

        # 无时间日期的文本总结
        if isinstance(obj, NoTimestampTextSummaryRecord):
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(names), len(descs))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                rows.append({"指标": _safe_str(names[i] if i < len(names) else ""), "状态": _safe_str(descs[i] if i < len(descs) else "")})
            return rows

        # 无时间日期的数值总结
        if isinstance(obj, NoDateValueSummaryRecord):
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(names), len(vals), len(units), len(sts))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                v = _safe_str(vals[i] if i < len(vals) else "")
                # 数值列保持“纯数值”，单位单独放在“单位”列
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                v = _trim_number_like(v)
                rows.append(
                    {
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "数值": v,
                        "单位": u,
                        "状态": _safe_str(sts[i] if i < len(sts) else ""),
                    }
                )
            return rows

        # 单指标的明细汇总记录：输出两类行（明细/汇总）
        if isinstance(obj, SingleMetricStatsRecord):
            core = getattr(obj, "核心字段", None)
            metric = _safe_str(getattr(core, "指标名称", "")) if core else ""
            unit = _safe_str(getattr(core, "单位", "")) if core else "无"
            # 明细
            ds = list(getattr(obj, "日期列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
            # 汇总
            sn = list(getattr(obj, "统计指标名称列表", []) or [])
            sv = list(getattr(obj, "统计数值列表", []) or [])
            ss = list(getattr(obj, "统计状态描述列表", []) or [])

            rows: list[dict[str, str]] = []
            n_d = max(len(ds), len(vs))
            for i in range(n_d):
                d = _safe_str(ds[i] if i < len(ds) else "")
                v = _safe_str(vs[i] if i < len(vs) else "")
                if not (d or v):
                    continue
                # 统计复合记录的明细 value 往往已经包含“分钟/小时/秒”等单位词。
                # 这里不要再二次拼接单位，否则当 unit 被推断为 "分钟" 时，
                # "1小时5分钟" 会在透视表阶段被剥离成 "1小时5"（只去掉尾部“分钟”）。
                # 交给透视表逻辑按 unit 做“可剥离则剥离”的展示即可。
                rows.append({"类别": "明细", "日期": d, "指标": metric, "数值": v, "单位": unit})

            n_s = max(len(sn), len(sv), len(ss))
            for i in range(n_s):
                nm = _safe_str(sn[i] if i < len(sn) else "")
                v = _safe_str(sv[i] if i < len(sv) else "")
                st = _safe_str(ss[i] if i < len(ss) else "")
                if not (nm or v or st):
                    continue
                rows.append({"类别": "汇总", "日期": "", "指标": nm or metric, "数值": v, "单位": unit, "状态": st})

            return rows if rows else None

        # 单日期数值多项总结（新版样式）：每个对象本身就是“一条指标记录”
        if isinstance(obj, SingleDateValueMultiSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])

            n = max(len(ds), len(names), len(vs), len(units), len(sts))
            if n <= 0:
                return None

            def _strip_unit_from_value_for_table(v: str, unit: str) -> str:
                """
                单日期多项总结的 Markdown 表格展示：
                - “单位”已单列展示，因此“数值”列应尽量保持为纯数值/纯区间
                - 修复区间值：如 "37°C-37°C" / "96%-96%" -> "37-37" / "96-96"
                """
                t = _safe_str(v)
                u = _safe_str(unit) or "无"
                if (not t) or (not u) or u == "无":
                    return t
                # 区间：两侧分别剥离单位（避免只剥离尾部导致残留）
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
                # 普通：剥离尾部单位
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                return t2 if t2 else t

            rows: list[dict[str, str]] = []
            for i in range(n):
                d = _safe_str(ds[i] if i < len(ds) else "")
                nm = _safe_str(names[i] if i < len(names) else "")
                v = _safe_str(vs[i] if i < len(vs) else "")
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                st = _safe_str(sts[i] if i < len(sts) else "")
                rows.append(
                    {
                        "日期": d,
                        "指标": nm,
                        # 单位列已存在，数值列避免重复拼接单位
                        "数值": _trim_number_like(_strip_unit_from_value_for_table(v, u)),
                        "单位": u,
                        "状态": st,
                    }
                )
            return rows

        # 其它已知类型：当前不做强行表格化（走 loose line）
        return None

    def _loose_line(obj: PersonalDataPattern) -> str:
        try:
            s = obj.recover_to_raw_data()
        except Exception:
            try:
                s = obj.format_print()
            except Exception:
                s = str(obj)
        s2 = _safe_str(s).replace("\n", " ").replace("\r", " ").strip()
        return s2

    def _trim_number_like(s: str) -> str:
        """
        把看起来像数字的字符串做轻量格式化：
        - "7.70" -> "7.7"
        - "10.00" -> "10.0"（保留至少 1 位小数，尽量贴近你现有示例）
        - "5" -> "5"
        其它内容原样返回（如 "-" / "01:40" / "2小时49分钟"）。
        """
        t = (s or "").strip()
        if not t:
            return t
        if t in ("-", "—", "无", "None", "null", "NULL", "N/A", "NA"):
            return t
        # 只处理纯数字（含可选小数点）
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t):
            return t
        if "." not in t:
            return t
        a, b = t.split(".", 1)
        b2 = b.rstrip("0")
        if b2 == "":
            # 10.00 -> 10.0
            return f"{a}.0"
        return f"{a}.{b2}"

    def _pivot_single_metric_detail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        将 “单指标的明细记录” 的长表 rows：
          日期/时间/指标/数值/单位
        透视成宽表：
          日期/时间 + 各指标列（保留完整指标名，不再拆分“运动类型/后缀”）
        """
        if not rows:
            return rows

        def _col_name(metric_name: str, unit: str) -> str:
            u = (unit or "").strip() or "无"
            if not metric_name:
                return "数值"
            if (not u) or u == "无":
                return metric_name
            return f"{metric_name} ({u})"

        def _strip_unit_from_value(v: str, unit: str) -> str:
            t = (v or "").strip()
            u = (unit or "").strip() or "无"
            if not t:
                return t
            # 特例：配速/比率类（单位形如 "每公里/每米" 或历史上的 "分钟/公里"、"小时/公里"、"秒/公里"）
            # 原始值常写作 "7分52秒/公里" 或 "7.8分钟/公里"：
            # - 表头已显示单位（每公里/分钟/公里），单元格里再带 "/公里" 会显得“单位重复”
            # - 同时，解析阶段可能会把单位推断为 "分钟/公里"，但值里只包含 "/公里"（不完全等于单位串），
            #   仅靠“尾部匹配单位”无法剥离。
            if u != "无" and u.startswith("每") and "/" in t:
                return t.split("/", 1)[0].strip()
            if u != "无" and "/" in u and "/" in t and u.startswith(("分钟/", "小时/", "秒/")):
                left = t.split("/", 1)[0].strip()
                # 若左侧包含“分钟/小时/秒”字样，去掉以保持与表头一致（表头已经体现单位）
                if u.startswith("分钟/"):
                    left2 = re.sub(r"\s*分钟\s*$", "", left).strip()
                    left = left2 if left2 else left
                elif u.startswith("小时/"):
                    left2 = re.sub(r"\s*小时\s*$", "", left).strip()
                    left = left2 if left2 else left
                elif u.startswith("秒/"):
                    left2 = re.sub(r"\s*秒\s*$", "", left).strip()
                    left = left2 if left2 else left
                return left
            # 区间值：形如 "37°C-37°C" / "97%-97%" / "17分-17分"
            # 之前仅做“尾部单位剥离”，会导致 "37°C-37°C" -> "37°C-37"（只去掉后半段单位）。
            # 这里对分隔符两侧分别剥离单位，避免表头已带单位时单元格仍残留单位。
            if u and u != "无":
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
            if u and u != "无":
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                t = t2 if t2 else t
            return t

        # group by (日期, 时间)
        groups: dict[tuple[str, str], dict[str, str]] = {}
        # 记录列名出现顺序（稳定）
        col_seen: list[str] = []
        col_seen_set: set[str] = set()

        for r in rows:
            d = _safe_str(r.get("日期", ""))
            t = _safe_str(r.get("时间", ""))
            full_metric = _safe_str(r.get("指标", ""))
            unit = _safe_str(r.get("单位", "")) or "无"
            val_with_unit = _safe_str(r.get("数值", ""))

            # 直接使用“完整指标名”作为列名（必要时带单位），不做后缀拆分。
            col = _col_name(full_metric, unit)

            v0 = _strip_unit_from_value(val_with_unit, unit)
            v0 = _trim_number_like(v0)
            if _is_missing_token(v0):
                v0 = "-"

            key = (d, t)
            row = groups.get(key)
            if row is None:
                row = {"日期": d, "时间": t}
                groups[key] = row
            # 同一 cell 多次出现：后写覆盖（更接近“最新值”直觉）
            row[col] = v0

            if col not in col_seen_set:
                col_seen.append(col)
                col_seen_set.add(col)

        def _parse_date_for_sort(s: str) -> tuple[int, int, int, str]:
            """
            将日期字符串解析成可排序的 (Y, M, D, raw)。
            支持：
            - YYYY年MM月DD日
            - MM月DD日
            - YYYY/M/D, YYYY-MM-DD, YYYY.MM.DD
            - M/D, M-D, M.D
            - 兜底返回 (9999, 99, 99, raw)
            """
            raw = (s or "").strip()
            if not raw:
                return (9999, 99, 99, raw)
            m_cn_y = re.fullmatch(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m_cn_y:
                return (int(m_cn_y.group("y")), int(m_cn_y.group("m")), int(m_cn_y.group("d")), raw)
            m_cn_md = re.fullmatch(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m_cn_md:
                # 无年份：用 9999 占位，但保留 M/D 以便同类内部排序正确
                return (9999, int(m_cn_md.group("m")), int(m_cn_md.group("d")), raw)
            m = re.fullmatch(r"(?P<y>\d{4})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m:
                return (int(m.group("y")), int(m.group("m")), int(m.group("d")), raw)
            m_md = re.fullmatch(r"(?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m_md:
                return (9999, int(m_md.group("m")), int(m_md.group("d")), raw)
            return (9999, 99, 99, raw)

        def _parse_time_for_sort(s: str) -> tuple[int, int, str]:
            raw = (s or "").strip()
            if not raw:
                return (99, 99, raw)
            m = re.fullmatch(r"(?P<h>\d{1,2}):(?P<mi>\d{2})", raw)
            if m:
                return (int(m.group("h")), int(m.group("mi")), raw)
            return (99, 99, raw)

        # 输出按 “真实日期/时间” 排序（修复 2/10 排在 2/2 前的问题）
        def _group_sort_key(k: tuple[str, str]) -> tuple[int, int, int, int, int, str, str]:
            d, t = k
            y, mo, da, d_raw = _parse_date_for_sort(d)
            hh, mm, t_raw = _parse_time_for_sort(t)
            # y/mo/da/hh/mm 为主排序；raw 保证同类场景稳定
            return (y, mo, da, hh, mm, d_raw, t_raw)

        out = [groups[k] for k in sorted(groups.keys(), key=_group_sort_key)]
        # 确保每行都有所有列（缺失填 "-"，更接近你示例里的 "-"）
        for row in out:
            for col in col_seen:
                if col not in row:
                    row[col] = "-"
        return out

    def _compact_period_value_single_summary_rows(rows: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        """
        周期数值单项总结：若全部行的(开始,结束)一致，则把日期范围挪到标题里，并去掉重复列。
        返回：(title_suffix, new_rows)
        """
        if not rows:
            return None, rows
        starts = {(_safe_str(r.get("开始", ""))) for r in rows if _safe_str(r.get("开始", ""))}
        ends = {(_safe_str(r.get("结束", ""))) for r in rows if _safe_str(r.get("结束", ""))}
        if len(starts) == 1 and len(ends) == 1:
            st = next(iter(starts))
            ed = next(iter(ends))
            new_rows: list[dict[str, str]] = []
            for r in rows:
                r2 = dict(r)
                r2.pop("开始", None)
                r2.pop("结束", None)
                new_rows.append(r2)
            return f"{st}~{ed}", new_rows
        return None, rows

    def _compact_rows_by_same_date_range(
        rows: list[dict[str, str]],
        *,
        start_key: str = "开始",
        end_key: str = "结束",
        drop_keys: tuple[str, str] | None = None,
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        通用"日期范围上移标题"压缩：
        - 若 rows 中所有非空的 (start_key, end_key) 仅有一组取值，则：
          - 返回 title_suffix = "start~end"（若 start==end，则返回 start）
          - 并从每行里删除 start_key/end_key（避免表格重复列）
        - 否则原样返回。
        """
        if not rows:
            return None, rows

        sk = str(start_key or "").strip() or "开始"
        ek = str(end_key or "").strip() or "结束"
        dk = drop_keys if drop_keys is not None else (sk, ek)

        starts = {(_safe_str(r.get(sk, ""))) for r in rows if _safe_str(r.get(sk, ""))}
        ends = {(_safe_str(r.get(ek, ""))) for r in rows if _safe_str(r.get(ek, ""))}
        if len(starts) == 1 and len(ends) == 1:
            st = next(iter(starts))
            ed = next(iter(ends))
            suffix = st if (st and st == ed) else f"{st}~{ed}"
            new_rows: list[dict[str, str]] = []
            for r in rows:
                r2 = dict(r)
                for k in dk:
                    r2.pop(k, None)
                new_rows.append(r2)
            return suffix, new_rows
        return None, rows

    def _pivot_stats_composite_detail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        将"单指标的明细汇总记录"的明细部分透视成宽表：
          类别/日期/指标/数值/单位
        透视成：
          日期 + 各指标列
        """
        if not rows:
            return rows
        
        # 只处理明细行
        detail_rows = [r for r in rows if _safe_str(r.get("类别", "")) == "明细"]
        if not detail_rows:
            return rows
        
        def _col_name(metric_name: str, unit: str) -> str:
            u = (unit or "").strip() or "无"
            if not metric_name:
                return "数值"
            if (not u) or u == "无":
                return metric_name
            return f"{metric_name} ({u})"
        
        def _strip_unit_from_value(v: str, unit: str) -> str:
            t = (v or "").strip()
            u = (unit or "").strip() or "无"
            if not t:
                return t
            # 配速/比率类：单位为 "每X" 时，值里若仍带 "/X" 则剥离，避免与表头单位重复
            if u != "无" and u.startswith("每") and "/" in t:
                return t.split("/", 1)[0].strip()
            # 区间值：形如 "37°C-37°C" / "97%-97%" / "17分-17分"
            # 之前仅做“尾部单位剥离”，会导致 "37°C-37°C" -> "37°C-37"（只去掉后半段单位）。
            if u and u != "无":
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
            if u and u != "无":
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                t = t2 if t2 else t
            return t
        
        # group by 日期
        groups: dict[str, dict[str, str]] = {}
        # 记录列名出现顺序（稳定）
        col_seen: list[str] = []
        col_seen_set: set[str] = set()
        
        for r in detail_rows:
            d = _safe_str(r.get("日期", ""))
            metric = _safe_str(r.get("指标", ""))
            unit = _safe_str(r.get("单位", "")) or "无"
            val_with_unit = _safe_str(r.get("数值", ""))
            
            col = _col_name(metric, unit)
            v0 = _strip_unit_from_value(val_with_unit, unit)
            v0 = _trim_number_like(v0)
            if _is_missing_token(v0):
                v0 = "-"
            
            row = groups.get(d)
            if row is None:
                row = {"日期": d}
                groups[d] = row
            # 同一 cell 多次出现：后写覆盖（更接近"最新值"直觉）
            row[col] = v0
            
            if col not in col_seen_set:
                col_seen.append(col)
                col_seen_set.add(col)
        
        def _parse_date_for_sort(s: str) -> tuple[int, int, int, str]:
            """
            将日期字符串解析成可排序的 (Y, M, D, raw)。
            支持：
            - YYYY年MM月DD日
            - MM月DD日
            - YYYY/M/D, YYYY-MM-DD, YYYY.MM.DD
            - M/D, MM/DD
            - 兜底返回 (9999, 99, 99, raw)
            """
            raw = (s or "").strip()
            if not raw:
                return (9999, 99, 99, raw)
            # YYYY年MM月DD日
            m = re.fullmatch(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m:
                return (int(m.group("y")), int(m.group("m")), int(m.group("d")), raw)
            # YYYY/M/D
            m = re.fullmatch(r"(?P<y>\d{4})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m:
                return (int(m.group("y")), int(m.group("m")), int(m.group("d")), raw)
            # M/D
            m = re.fullmatch(r"(?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m:
                return (9999, int(m.group("m")), int(m.group("d")), raw)
            # MM月DD日
            m = re.fullmatch(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m:
                return (9999, int(m.group("m")), int(m.group("d")), raw)
            return (9999, 99, 99, raw)
        
        # 输出按日期排序
        def _date_sort_key(d: str) -> tuple[int, int, int, str]:
            y, mo, da, d_raw = _parse_date_for_sort(d)
            return (y, mo, da, d_raw)
        
        out = [groups[k] for k in sorted(groups.keys(), key=_date_sort_key)]
        # 确保每行都有所有列（缺失填 "-"）
        for row in out:
            for col in col_seen:
                if col not in row:
                    row[col] = "-"
        
        # 只返回透视后的明细行（汇总行由调用方单独处理）
        return out
    
    def _format_stats_composite_summary_as_sentences(rows: list[dict[str, str]]) -> list[str]:
        """
        将"单指标的明细汇总记录"的汇总部分转换为句子列表。
        """
        summary_rows = [r for r in rows if _safe_str(r.get("类别", "")) == "汇总"]
        if not summary_rows:
            return []
        
        sentences: list[str] = []
        for r in summary_rows:
            metric = _safe_str(r.get("指标", ""))
            value = _safe_str(r.get("数值", ""))
            status = _safe_str(r.get("状态", ""))
            
            if not metric and not value:
                continue
            
            parts = []
            if metric:
                parts.append(metric)
            if value:
                parts.append(value)
            if status and status not in ("无", "-", ""):
                parts.append(status)
            
            if parts:
                sentences.append("".join(parts))
        
        return sentences
    
    def _compact_rows_by_same_date(
        rows: list[dict[str, str]],
        *,
        date_key: str = "日期",
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        单日期类型"日期上移标题"压缩：
        - 若 rows 中所有非空的 date_key 仅有一个取值，则：
          - 返回 title_suffix = 该日期值
          - 并从每行里删除 date_key（避免表格重复列）
        - 否则原样返回。
        """
        if not rows:
            return None, rows

        dk = str(date_key or "").strip() or "日期"
        dates = {(_safe_str(r.get(dk, ""))) for r in rows if _safe_str(r.get(dk, ""))}
        if len(dates) == 1:
            date_val = next(iter(dates))
            new_rows: list[dict[str, str]] = []
            for r in rows:
                r2 = dict(r)
                r2.pop(dk, None)
                new_rows.append(r2)
            return date_val, new_rows
        return None, rows

    def _group_rows_by_date(
        rows: list[dict[str, str]],
        *,
        date_key: str = "日期",
    ) -> list[tuple[str, list[dict[str, str]]]]:
        """
        按日期分组行：
        - 返回 [(date1, rows1), (date2, rows2), ...]
        - 每个日期组的行中，date_key 列会被删除
        """
        if not rows:
            return []

        dk = str(date_key or "").strip() or "日期"
        groups: dict[str, list[dict[str, str]]] = {}
        empty_date_rows: list[dict[str, str]] = []

        for r in rows:
            date_val = _safe_str(r.get(dk, ""))
            if date_val:
                if date_val not in groups:
                    groups[date_val] = []
                r2 = dict(r)
                r2.pop(dk, None)
                groups[date_val].append(r2)
            else:
                r2 = dict(r)
                r2.pop(dk, None)
                empty_date_rows.append(r2)

        result: list[tuple[str, list[dict[str, str]]]] = []
        # 按日期排序（保持稳定顺序）
        for date_val in sorted(groups.keys()):
            result.append((date_val, groups[date_val]))
        # 空日期的行放在最后
        if empty_date_rows:
            result.append(("", empty_date_rows))

        return result

    # 1) 聚合为 tables
    tables: dict[str, list[dict[str, str]]] = {}
    loose: list[str] = []
    # 单独处理"单指标的明细汇总记录"，保留对象边界信息
    stats_composite_objs: list[PersonalDataPattern] = []
    for obj in objs:
        et = _safe_str(getattr(obj, "实体类型", "")) or "未定义"
        if et == "单指标的明细汇总记录":
            stats_composite_objs.append(obj)
            continue
        rows = _rows_from_obj(obj)
        if rows is None:
            if include_loose_lines:
                loose.append(_loose_line(obj))
            continue
        tables.setdefault(et, []).extend(rows)

    # 1.5) 二次聚合（更高级的表格）
    # - 单指标明细：长表 -> 宽表（按 日期/时间 pivot；保留完整指标名，不输出“运动类型”列）
    # - 周期数值单项总结：若日期范围一致，去掉重复列
    # - 周期文本总结 / 周期数值多项总结：若日期范围一致，去掉重复列（与单项总结一致）
    # - 单日期文本总结 / 单日期数值多项总结 / 单日期数值单项总结：若日期一致，去掉重复列
    # - 单指标的明细汇总记录：明细部分透视成宽表，汇总部分按对象分组单独处理
    table_title_suffix: dict[str, str] = {}
    stats_composite_summary_sentences_by_obj: dict[str, list[list[str]]] = {}
    
    # 处理"单指标的明细汇总记录"：分别处理每个对象
    if stats_composite_objs:
        all_detail_rows: list[dict[str, str]] = []
        summary_sentences_by_obj: list[list[str]] = []
        
        for obj in stats_composite_objs:
            rows = _rows_from_obj(obj)
            if rows is None:
                if include_loose_lines:
                    loose.append(_loose_line(obj))
                continue
            # 提取该对象的汇总句子
            obj_summary_sentences = _format_stats_composite_summary_as_sentences(rows)
            summary_sentences_by_obj.append(obj_summary_sentences)
            # 收集明细行（用于后续透视）
            detail_rows = [r for r in rows if _safe_str(r.get("类别", "")) == "明细"]
            all_detail_rows.extend(detail_rows)
        
        # 透视所有明细行
        if all_detail_rows:
            tables["单指标的明细汇总记录"] = _pivot_stats_composite_detail_rows(all_detail_rows)
            stats_composite_summary_sentences_by_obj["单指标的明细汇总记录"] = summary_sentences_by_obj
    
    if "单指标的明细记录" in tables:
        tables["单指标的明细记录"] = _pivot_single_metric_detail_rows(tables["单指标的明细记录"])
    if "周期数值单项总结" in tables:
        suf, new_rows = _compact_period_value_single_summary_rows(tables["周期数值单项总结"])
        tables["周期数值单项总结"] = new_rows
        if suf:
            table_title_suffix["周期数值单项总结"] = suf
    if "周期文本总结" in tables:
        suf, new_rows = _compact_rows_by_same_date_range(tables["周期文本总结"])
        tables["周期文本总结"] = new_rows
        if suf:
            table_title_suffix["周期文本总结"] = suf
    if "周期数值多项总结" in tables:
        suf, new_rows = _compact_rows_by_same_date_range(tables["周期数值多项总结"])
        tables["周期数值多项总结"] = new_rows
        if suf:
            table_title_suffix["周期数值多项总结"] = suf
    # 单日期文本总结、单日期数值多项总结和单日期数值单项总结：按日期分组，每个日期一个表格
    # 这三个类型不在这里处理，而是在输出时按日期分组

    # 输出：markdown
    parts: list[str] = []
    for et, rows in tables.items():
        if not rows:
            continue

        # 单日期文本总结、单日期数值多项总结和单日期数值单项总结：按日期分组输出
        if et in ("单日期文本总结", "单日期数值多项总结", "单日期数值单项总结"):
            date_groups = _group_rows_by_date(rows)
            for date_val, group_rows in date_groups:
                if not group_rows:
                    continue
                # 列顺序：优先常见字段
                prefer_cols = list(_PREFER_COLS_BASE)

                all_cols: list[str] = []
                seen: set[str] = set()
                # 先按 prefer_cols 收集
                for c in prefer_cols:
                    if any(c in r for r in group_rows):
                        all_cols.append(c)
                        seen.add(c)
                # 再补齐剩余列（稳定排序）
                more = sorted({c for r in group_rows for c in r.keys()} - seen)
                all_cols.extend(more)

                parts.append(f"### {_make_md_section_title(et, date_or_range=date_val)}")
                header = "| " + " | ".join(all_cols) + " |"
                sep = "| " + " | ".join(["---"] * len(all_cols)) + " |"
                parts.append(header)
                parts.append(sep)
                for r in group_rows:
                    parts.append("| " + " | ".join(_cell(r.get(c, "")) for c in all_cols) + " |")
                parts.append("")  # 空行分隔
            continue

        # 周期数值对比记录：输出为“指标 + 两个周期列 + 差异”的宽表样式
        # 目标（示例）：
        # | 指标 | 12月31日 | 11月11日~20日 | 差异 |
        # | --- | --- | --- | --- |
        # | 平均深睡时长 | 4.23 | 1.58 | 多2.66小时 |
        if et == "周期数值对比记录":
            def _shorten_date_range_label(s: str) -> str:
                """
                将日期范围在表头做简写：
                - "11月11日~11月20日" -> "11月11日~20日"（同月范围）
                其它保持原样。
                """
                t = _safe_str(s)
                if not t:
                    return t
                # 已经是简写或不是范围：直接返回
                if ("~" not in t and "～" not in t) or ("月" not in t):
                    return t
                m = re.fullmatch(
                    r"\s*(?P<m1>\d{1,2})月(?P<d1>\d{1,2})日\s*(?P<sep>[~～\-—−])\s*(?P<m2>\d{1,2})月(?P<d2>\d{1,2})日\s*",
                    t,
                )
                if not m:
                    return t
                m1 = m.group("m1")
                d1 = m.group("d1")
                sep = m.group("sep")
                m2 = m.group("m2")
                d2 = m.group("d2")
                if m1 == m2:
                    # 统一用 "~" 作为展示分隔符（与现有示例一致）
                    return f"{m1}月{d1}日~{d2}日"
                return f"{m1}月{d1}日{sep}{m2}月{d2}日"

            def _format_compare_diff(logic: str, diff: str) -> str:
                lg = _safe_str(logic)
                df = _safe_str(diff)
                # 轻量统一展示：纯“X小时”差异改成“Xh”（更紧凑）
                if df:
                    df = re.sub(r"(?P<num>[-+]?\d+(?:\.\d+)?)\s*小时\s*$", r"\g<num>h", df)
                if not (lg or df):
                    return ""
                if not lg:
                    return df
                if not df:
                    return lg
                # “早/晚”语义更像短语：中间加空格；其它保持紧凑拼接（如：多2.66小时）
                if lg in ("早", "晚"):
                    return f"{lg} {df}".strip()
                return f"{lg}{df}".strip()

            # 按(范围1, 范围2)分组，避免不同对比周期混在同一张宽表里
            groups: dict[tuple[str, str], list[dict[str, str]]] = {}
            group_order: list[tuple[str, str]] = []
            for r in rows:
                r1 = _safe_str(r.get("范围1", ""))
                r2 = _safe_str(r.get("范围2", ""))
                key = (r1, r2)
                if key not in groups:
                    groups[key] = []
                    group_order.append(key)
                groups[key].append(r)

            for (r1, r2) in group_order:
                group_rows = groups.get((r1, r2), [])
                if not group_rows:
                    continue

                col_r1 = _shorten_date_range_label(r1) or "范围1"
                col_r2 = _shorten_date_range_label(r2) or "范围2"
                all_cols = ["指标", col_r1, col_r2, "差异"]

                parts.append("### 周期睡眠数据对比")
                header = "| " + " | ".join(all_cols) + " |"
                sep = "| " + " | ".join(["---"] * len(all_cols)) + " |"
                parts.append(header)
                parts.append(sep)
                for rr in group_rows:
                    metric = _safe_str(rr.get("指标", ""))
                    v1 = _safe_str(rr.get("值1", ""))
                    v2 = _safe_str(rr.get("值2", ""))
                    diff_cell = _format_compare_diff(rr.get("逻辑", ""), rr.get("差异", ""))
                    parts.append("| " + " | ".join(_cell(x) for x in [metric, v1, v2, diff_cell]) + " |")
                parts.append("")  # 空行分隔
            continue

        # 列顺序：优先常见字段
        prefer_cols = list(_PREFER_COLS_BASE)

        # 预先计算“实际出现过的列”，避免后续多次扫描 rows
        cols_present: set[str] = {c for r in rows for c in r.keys()}

        # "单指标的明细记录(宽表)"：把动态指标列按更自然的顺序输出
        if et == "单指标的明细记录":
            metric_order = _METRIC_ORDER_SINGLE_METRIC_DETAIL

            def _metric_col_sort_key(c: str) -> tuple[int, int, str]:
                # c 形如 "登山距离 (千米)" / "登山无氧训练压力 (分)" 等
                name = c.split(" (", 1)[0].strip()
                if name in ("日期", "时间"):
                    return (-1, 0, name)
                for i, suf in enumerate(metric_order):
                    if name.endswith(suf) and len(name) > len(suf):
                        # 同一后缀下按全名稳定排序
                        return (i, 1, c)
                    if name == suf:
                        return (i, 0, c)
                return (9999, 9, c)

            # 把所有"非基础列"视作指标列并按“后缀偏好”排序
            base_cols = {"日期", "时间"}
            metric_cols = sorted(
                cols_present - set(prefer_cols) - base_cols,
                key=_metric_col_sort_key,
            )
            # 保证基础列优先，其次指标列（动态），最后其它列
            prefer_cols = ["日期", "时间"] + metric_cols
            # 兼容：若还有其它列（非常规），后面仍会走 more 补齐
        
        # "单指标的明细汇总记录(宽表)"：把动态指标列按更自然的顺序输出
        if et == "单指标的明细汇总记录":
            metric_order = _METRIC_ORDER_STATS_COMPOSITE
            order_map = {name: i for i, name in enumerate(metric_order)}

            def _metric_col_sort_key(c: str) -> tuple[int, str]:
                # c 形如 "运动次数 (次)" / "运动时长" 等
                name = c.split(" (", 1)[0].strip()
                if name == "日期":
                    return (-1, name)
                idx = order_map.get(name)
                if idx is not None:
                    return (idx, c)
                return (9999, c)

            # 把所有"非基础列"视作指标列并按 metric_order 排序
            base_cols = {"日期"}
            metric_cols = sorted(
                cols_present - set(prefer_cols) - base_cols,
                key=_metric_col_sort_key,
            )
            # 保证基础列优先，其次指标列（动态），最后其它列
            prefer_cols = ["日期"] + metric_cols
            # 兼容：若还有其它列（非常规），后面仍会走 more 补齐

        all_cols: list[str] = []
        seen: set[str] = set()
        # 先按 prefer_cols 收集
        for c in prefer_cols:
            if c in cols_present:
                all_cols.append(c)
                seen.add(c)
        # 再补齐剩余列（稳定排序）
        more = sorted(cols_present - seen)
        all_cols.extend(more)

        parts.append(f"### {_make_md_section_title(et, date_or_range=table_title_suffix.get(et))}")
        header = "| " + " | ".join(all_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(all_cols)) + " |"
        parts.append(header)
        parts.append(sep)
        for r in rows:
            parts.append("| " + " | ".join(_cell(r.get(c, "")) for c in all_cols) + " |")
        parts.append("")  # 空行分隔
        
        # 单指标的明细汇总记录：在表格后输出汇总句子（按对象分组）
        if et == "单指标的明细汇总记录" and et in stats_composite_summary_sentences_by_obj:
            obj_summary_groups = stats_composite_summary_sentences_by_obj[et]
            for obj_summary_sentences in obj_summary_groups:
                if obj_summary_sentences:
                    # 每个对象的汇总句子用逗号连接，以句号结尾
                    parts.append("，".join(obj_summary_sentences) + "。")
            if any(obj_summary_groups):
                parts.append("")  # 空行分隔

    if include_loose_lines and loose:
        parts.append("### 零散或无法聚合")
        for ln in loose:
            parts.append(f"- {ln}")

    return "\n".join(parts).rstrip() + "\n"


__all__ = ["aggregate_patterns_to_formatted_text"]


