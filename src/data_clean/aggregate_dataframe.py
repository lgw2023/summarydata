from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

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
from .normalize import _is_missing_token

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore


def aggregate_patterns_to_dataframes(
    patterns: Sequence[PersonalDataPattern],
    *,
    max_rows_per_table: int = 300,
    include_loose_lines: bool = True,
    data_type_col: str | None = "data_type",
) -> list["pd.DataFrame"]:
    """
    聚合/汇总一个数据类列表，并输出为 pandas DataFrame 列表（每个表一个 DataFrame）。

    设计目标：复用/对齐 `aggregate_format.aggregate_patterns_to_formatted_text` 的“聚合/透视/压缩”规则，
    区别仅在于输出介质从 Markdown 表格变为 DataFrame。

    约定：
    - 每个 DataFrame 都会在 `df.attrs` 写入元信息：
      - entity_type: 实体类型（表类型）
      - title: 建议展示标题（可能包含日期/日期范围）
      - date_or_range: 若该表是“单日期/同日期范围”压缩产生，会填充该值
      - date_range1/date_range2: 对比表（周期数值对比记录）会填充
      - summary_sentences_by_obj: 单指标的明细汇总记录的汇总句（按对象分组）
    - 另外会在 DataFrame 里额外写入一列 “数据类型”（默认列名可配），便于导出 Excel / 直接筛选。
    - 若 include_loose_lines=True，会额外返回一个 entity_type="零散或无法聚合" 的 DataFrame（单列 raw）。
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("需要安装 pandas 才能使用 aggregate_patterns_to_dataframes（requirements.txt 已包含 pandas）。") from e

    objs = list(patterns or [])
    if not objs:
        return []

    def _safe_str(x: Any) -> str:
        return str(x if x is not None else "").strip()

    def _normalize_date_to_iso(s: str) -> str:
        """
        把常见日期表达式尽量转成 ISO 8601（YYYY-MM-DD）。

        说明：
        - 上游解析（models.py）通常已将日期规范化为 "YYYY年MM月DD日" 或 "MM月DD日"。
        - 若缺少年份（如 "01月24日"），这里不强行补全，避免误改；将原样返回。
        """
        raw = _safe_str(s)
        if not raw:
            return ""
        m_cn = re.fullmatch(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
        if m_cn:
            y, mo, da = int(m_cn.group("y")), int(m_cn.group("m")), int(m_cn.group("d"))
            return f"{y:04d}-{mo:02d}-{da:02d}"
        m_ymd = re.fullmatch(r"(?P<y>\d{4})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
        if m_ymd:
            y, mo, da = int(m_ymd.group("y")), int(m_ymd.group("m")), int(m_ymd.group("d"))
            return f"{y:04d}-{mo:02d}-{da:02d}"
        m_ymd2 = re.fullmatch(r"(?P<y>\d{2})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
        if m_ymd2:
            y2, mo, da = int(m_ymd2.group("y")), int(m_ymd2.group("m")), int(m_ymd2.group("d"))
            # 兜底：两位年份默认按 20xx 处理（与模型侧默认逻辑一致）
            y = 2000 + y2
            return f"{y:04d}-{mo:02d}-{da:02d}"
        return raw

    def _normalize_date_range_like_to_iso(s: str) -> str:
        """
        尽量把形如 "A~B" 的日期范围两端转 ISO；若无法识别则原样返回。
        """
        raw = _safe_str(s)
        if not raw:
            return ""
        m = re.fullmatch(r"\s*(?P<a>.+?)\s*(?P<sep>[~～\\-—−])\s*(?P<b>.+?)\s*$", raw)
        if not m:
            return raw
        a_raw = (m.group("a") or "").strip()
        b_raw = (m.group("b") or "").strip()
        a = _normalize_date_to_iso(a_raw)
        b = _normalize_date_to_iso(b_raw)
        if a == a_raw and b == b_raw:
            return raw
        return f"{a}{m.group('sep')}{b}".strip()

    _COL_RENAME: dict[str, str] = {
        "日期": "date",
        "时间": "time",
        "指标": "metric",
        "数值": "value",
        "单位": "unit",
        "状态": "status",
        "开始": "start_date",
        "结束": "end_date",
        "范围1": "date_range1",
        "值1": "value1",
        "范围2": "date_range2",
        "值2": "value2",
        "逻辑": "logic",
        "差异": "diff",
        "类别": "category",
    }

    def _normalize_rows_for_long_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        将不同实体类型的“表行”统一为更 tidy 的长表风格：
        - 列名统一为英文字段（date/time/metric/value/unit/...）
        - 日期类字段尽量转为 ISO
        """
        if not rows:
            return rows
        out_rows: list[dict[str, str]] = []
        for r in rows:
            r2: dict[str, str] = {}
            for k, v in (r or {}).items():
                kk = _COL_RENAME.get(str(k).strip(), str(k).strip())
                r2[kk] = _safe_str(v)

            if "date" in r2:
                r2["date"] = _normalize_date_to_iso(r2.get("date", ""))
            if "start_date" in r2:
                r2["start_date"] = _normalize_date_to_iso(r2.get("start_date", ""))
            if "end_date" in r2:
                r2["end_date"] = _normalize_date_to_iso(r2.get("end_date", ""))
            if "date_range1" in r2:
                r2["date_range1"] = _normalize_date_range_like_to_iso(r2.get("date_range1", ""))
            if "date_range2" in r2:
                r2["date_range2"] = _normalize_date_range_like_to_iso(r2.get("date_range2", ""))

            # 补齐常用字段（便于下游稳定消费；不会强行覆盖已有值）
            if "time" in r2 and not r2.get("time"):
                r2["time"] = ""
            if "unit" in r2 and not r2.get("unit"):
                r2["unit"] = "无"

            out_rows.append(r2)

        # 尝试对 time-series 友好的排序（仅当存在 date 列时）
        #
        # 额外规则：若存在 category（如“明细/汇总”），希望“明细”排在前、“汇总”排在后，
        # 避免汇总行 date 为空导致排序时跑到最前面。
        if any(("date" in rr) for rr in out_rows):
            def _cat_rank(cat: str) -> int:
                c = (cat or "").strip()
                if c in ("明细", "detail"):
                    return 0
                if c in ("汇总", "summary"):
                    return 1
                return 2

            out_rows = sorted(
                out_rows,
                key=lambda rr: (
                    _cat_rank(rr.get("category", "")),
                    rr.get("date", ""),
                    rr.get("time", ""),
                    rr.get("metric", ""),
                ),
            )

        return out_rows

    def _make_title(entity_type: str, *, date_or_range: str | None = None) -> str:
        et = _safe_str(entity_type) or "未定义"
        suf = _safe_str(date_or_range) if date_or_range is not None else ""
        if suf:
            return f"{suf} {et}".strip()
        return et

    def _trim_number_like(s: str) -> str:
        """
        把看起来像数字的字符串做轻量格式化：
        - "7.70" -> "7.7"
        - "10.00" -> "10.0"
        - "5" -> "5"
        其它内容原样返回（如 "-" / "01:40" / "2小时49分钟"）。
        """
        t = (s or "").strip()
        if not t:
            return t
        if t in ("-", "—", "无", "None", "null", "NULL", "N/A", "NA"):
            return t
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t):
            return t
        if "." not in t:
            return t
        a, b = t.split(".", 1)
        b2 = b.rstrip("0")
        if b2 == "":
            return f"{a}.0"
        return f"{a}.{b2}"

    def _rows_from_obj(obj: PersonalDataPattern) -> list[dict[str, str]] | None:
        et = _safe_str(getattr(obj, "实体类型", ""))
        if not et:
            return None

        if isinstance(obj, UnparsedRawPersonalData):
            return None

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
                u = unit or "无"
                # 单位单独作为列：数值列尽量保持“纯数值/纯文本”，不拼接单位
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                rows.append({"日期": d, "时间": t, "指标": metric, "数值": v, "单位": u})
            return rows

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
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                v = _trim_number_like(v)
                rows.append({"开始": st, "结束": ed, "指标": nm, "数值": v, "单位": u})
            return rows

        if isinstance(obj, PeriodTextSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(starts), len(ends), len(names), len(descs))
            if n <= 0:
                return None
            return [
                {
                    "开始": _safe_str(starts[i] if i < len(starts) else ""),
                    "结束": _safe_str(ends[i] if i < len(ends) else ""),
                    "指标": _safe_str(names[i] if i < len(names) else ""),
                    "状态": _safe_str(descs[i] if i < len(descs) else ""),
                }
                for i in range(n)
            ]

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
            return [
                {
                    "范围1": _safe_str(r1s[i] if i < len(r1s) else ""),
                    "值1": _safe_str(v1s[i] if i < len(v1s) else ""),
                    "范围2": _safe_str(r2s[i] if i < len(r2s) else ""),
                    "值2": _safe_str(v2s[i] if i < len(v2s) else ""),
                    "指标": _safe_str(names[i] if i < len(names) else ""),
                    "逻辑": _safe_str(logics[i] if i < len(logics) else ""),
                    "差异": _safe_str(diffs[i] if i < len(diffs) else ""),
                }
                for i in range(n)
            ]

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

            def _strip_unit_from_value_for_table(v: str, unit: str) -> str:
                t = _safe_str(v)
                u = _safe_str(unit) or "无"
                if (not t) or (not u) or u == "无":
                    return t
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                return t2 if t2 else t

            rows: list[dict[str, str]] = []
            for i in range(n):
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                v = _safe_str(vals[i] if i < len(vals) else "")
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

        if isinstance(obj, SingleDateTextSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(ds), len(names), len(descs))
            if n <= 0:
                return None
            return [
                {"日期": _safe_str(ds[i] if i < len(ds) else ""), "指标": _safe_str(names[i] if i < len(names) else ""), "状态": _safe_str(descs[i] if i < len(descs) else "")}
                for i in range(n)
            ]

        if isinstance(obj, NoTimestampTextSummaryRecord):
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(names), len(descs))
            if n <= 0:
                return None
            return [{"指标": _safe_str(names[i] if i < len(names) else ""), "状态": _safe_str(descs[i] if i < len(descs) else "")} for i in range(n)]

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
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                v = _trim_number_like(v)
                rows.append({"指标": _safe_str(names[i] if i < len(names) else ""), "数值": v, "单位": u, "状态": _safe_str(sts[i] if i < len(sts) else "")})
            return rows

        if isinstance(obj, SingleMetricStatsRecord):
            core = getattr(obj, "核心字段", None)
            metric = _safe_str(getattr(core, "指标名称", "")) if core else ""
            unit = _safe_str(getattr(core, "单位", "")) if core else "无"
            ds = list(getattr(obj, "日期列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
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
                t = _safe_str(v)
                u = _safe_str(unit) or "无"
                if (not t) or (not u) or u == "无":
                    return t
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                return t2 if t2 else t

            rows: list[dict[str, str]] = []
            for i in range(n):
                d = _safe_str(ds[i] if i < len(ds) else "")
                nm = _safe_str(names[i] if i < len(names) else "")
                v = _safe_str(vs[i] if i < len(vs) else "")
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                st = _safe_str(sts[i] if i < len(sts) else "")
                rows.append({"日期": d, "指标": nm, "数值": _trim_number_like(_strip_unit_from_value_for_table(v, u)), "单位": u, "状态": st})
            return rows

        return None

    def _loose_line(obj: PersonalDataPattern) -> str:
        try:
            s = obj.recover_to_raw_data()
        except Exception:
            try:
                s = obj.format_print(max_items=4, max_len=200)
            except Exception:
                s = str(obj)
        return _safe_str(s).replace("\n", " ").replace("\r", " ").strip()

    def _pivot_single_metric_detail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        if not rows:
            return rows

        # “单位”需要作为独立列：这里输出为宽表，但每个指标列旁边配一个“指标单位”列
        # - 值列：指标名（若同名指标存在不同单位，则退化为“指标名(单位)”保证唯一性）
        # - 单位列：值列名 + "单位"
        metric_units: dict[str, set[str]] = {}
        for r in rows:
            m = _safe_str(r.get("指标", ""))
            u = _safe_str(r.get("单位", "")) or "无"
            if not m:
                continue
            metric_units.setdefault(m, set()).add(u)
        single_metric_only = len(metric_units) == 1

        def _value_col_name(metric_name: str, unit: str) -> str:
            m = (metric_name or "").strip()
            u = (unit or "").strip() or "无"
            if not m:
                return "数值"
            # 同名指标出现多单位时，必须做列名去重
            if len(metric_units.get(m, set())) > 1:
                return f"{m}({u})" if u else m
            return m

        def _unit_col_name(value_col: str) -> str:
            vc = (value_col or "").strip() or "数值"
            if vc == "数值" or single_metric_only:
                return "单位"
            return f"{vc}单位"

        def _strip_unit_from_value(v: str, unit: str) -> str:
            t = (v or "").strip()
            u = (unit or "").strip() or "无"
            if not t:
                return t
            if u != "无" and u.startswith("每") and "/" in t:
                return t.split("/", 1)[0].strip()
            if u != "无" and "/" in u and "/" in t and u.startswith(("分钟/", "小时/", "秒/")):
                left = t.split("/", 1)[0].strip()
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

        groups: dict[tuple[str, str], dict[str, str]] = {}
        col_seen: list[str] = []
        col_seen_set: set[str] = set()

        for r in rows:
            d = _safe_str(r.get("日期", ""))
            t = _safe_str(r.get("时间", ""))
            full_metric = _safe_str(r.get("指标", ""))
            unit = _safe_str(r.get("单位", "")) or "无"
            val_with_unit = _safe_str(r.get("数值", ""))
            col = _value_col_name(full_metric, unit)
            unit_col = _unit_col_name(col)
            v0 = _strip_unit_from_value(val_with_unit, unit)
            v0 = _trim_number_like(v0)
            if _is_missing_token(v0):
                v0 = "-"
            key = (d, t)
            row = groups.get(key)
            if row is None:
                row = {"日期": d, "时间": t}
                groups[key] = row
            row[col] = v0
            row[unit_col] = unit
            for c in (col, unit_col):
                if c not in col_seen_set:
                    col_seen.append(c)
                    col_seen_set.add(c)

        def _parse_date_for_sort(s: str) -> tuple[int, int, int, str]:
            raw = (s or "").strip()
            if not raw:
                return (9999, 99, 99, raw)
            m_cn_y = re.fullmatch(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m_cn_y:
                return (int(m_cn_y.group("y")), int(m_cn_y.group("m")), int(m_cn_y.group("d")), raw)
            m_cn_md = re.fullmatch(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m_cn_md:
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

        def _group_sort_key(k: tuple[str, str]) -> tuple[int, int, int, int, int, str, str]:
            d, t = k
            y, mo, da, d_raw = _parse_date_for_sort(d)
            hh, mm, t_raw = _parse_time_for_sort(t)
            return (y, mo, da, hh, mm, d_raw, t_raw)

        out = [groups[k] for k in sorted(groups.keys(), key=_group_sort_key)]
        for row in out:
            for col in col_seen:
                row.setdefault(col, "-")
        return out

    def _compact_rows_by_same_date_range(
        rows: list[dict[str, str]],
        *,
        start_key: str = "开始",
        end_key: str = "结束",
        drop_keys: tuple[str, str] | None = None,
    ) -> tuple[str | None, list[dict[str, str]]]:
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

    def _group_rows_by_date(rows: list[dict[str, str]], *, date_key: str = "日期") -> list[tuple[str, list[dict[str, str]]]]:
        if not rows:
            return []
        dk = str(date_key or "").strip() or "日期"
        groups: dict[str, list[dict[str, str]]] = {}
        empty_date_rows: list[dict[str, str]] = []
        for r in rows:
            date_val = _safe_str(r.get(dk, ""))
            r2 = dict(r)
            r2.pop(dk, None)
            if date_val:
                groups.setdefault(date_val, []).append(r2)
            else:
                empty_date_rows.append(r2)
        result: list[tuple[str, list[dict[str, str]]]] = [(d, groups[d]) for d in sorted(groups.keys())]
        if empty_date_rows:
            result.append(("", empty_date_rows))
        return result

    def _pivot_stats_composite_detail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        if not rows:
            return rows
        detail_rows = [r for r in rows if _safe_str(r.get("类别", "")) == "明细"]
        if not detail_rows:
            return rows

        # “单位”需要作为独立列：同 _pivot_single_metric_detail_rows
        metric_units: dict[str, set[str]] = {}
        for r in detail_rows:
            m = _safe_str(r.get("指标", ""))
            u = _safe_str(r.get("单位", "")) or "无"
            if not m:
                continue
            metric_units.setdefault(m, set()).add(u)
        single_metric_only = len(metric_units) == 1

        def _value_col_name(metric_name: str, unit: str) -> str:
            m = (metric_name or "").strip()
            u = (unit or "").strip() or "无"
            if not m:
                return "数值"
            if len(metric_units.get(m, set())) > 1:
                return f"{m}({u})" if u else m
            return m

        def _unit_col_name(value_col: str) -> str:
            vc = (value_col or "").strip() or "数值"
            if vc == "数值" or single_metric_only:
                return "单位"
            return f"{vc}单位"

        def _strip_unit_from_value(v: str, unit: str) -> str:
            t = (v or "").strip()
            u = (unit or "").strip() or "无"
            if not t:
                return t
            if u != "无" and u.startswith("每") and "/" in t:
                return t.split("/", 1)[0].strip()
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

        groups: dict[str, dict[str, str]] = {}
        col_seen: list[str] = []
        col_seen_set: set[str] = set()
        for r in detail_rows:
            d = _safe_str(r.get("日期", ""))
            metric = _safe_str(r.get("指标", ""))
            unit = _safe_str(r.get("单位", "")) or "无"
            val_with_unit = _safe_str(r.get("数值", ""))
            col = _value_col_name(metric, unit)
            unit_col = _unit_col_name(col)
            v0 = _strip_unit_from_value(val_with_unit, unit)
            v0 = _trim_number_like(v0)
            if _is_missing_token(v0):
                v0 = "-"
            row = groups.get(d)
            if row is None:
                row = {"日期": d}
                groups[d] = row
            row[col] = v0
            row[unit_col] = unit
            for c in (col, unit_col):
                if c not in col_seen_set:
                    col_seen.append(c)
                    col_seen_set.add(c)

        def _parse_date_for_sort(s: str) -> tuple[int, int, int, str]:
            raw = (s or "").strip()
            if not raw:
                return (9999, 99, 99, raw)
            m = re.fullmatch(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m:
                return (int(m.group("y")), int(m.group("m")), int(m.group("d")), raw)
            m = re.fullmatch(r"(?P<y>\d{4})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m:
                return (int(m.group("y")), int(m.group("m")), int(m.group("d")), raw)
            m = re.fullmatch(r"(?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m:
                return (9999, int(m.group("m")), int(m.group("d")), raw)
            m = re.fullmatch(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m:
                return (9999, int(m.group("m")), int(m.group("d")), raw)
            return (9999, 99, 99, raw)

        out = [groups[k] for k in sorted(groups.keys(), key=lambda d: _parse_date_for_sort(d))]
        for row in out:
            for col in col_seen:
                row.setdefault(col, "-")
        return out

    def _format_stats_composite_summary_as_sentences(rows: list[dict[str, str]]) -> list[str]:
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
            parts: list[str] = []
            if metric:
                parts.append(metric)
            if value:
                parts.append(value)
            if status and status not in ("无", "-", ""):
                parts.append(status)
            if parts:
                sentences.append("".join(parts))
        return sentences

    # 1) 聚合为 tables（entity_type -> rows）
    tables: dict[str, list[dict[str, str]]] = {}
    loose: list[str] = []
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

    # 1.5) 二次聚合
    # 旧版会把一些表“压缩/透视”成宽表并把日期范围挪到 title；
    # 现在统一输出为长表（tidy data），不再依赖 table_title_suffix。
    table_title_suffix: dict[str, str] = {}
    stats_composite_summary_sentences_by_obj: dict[str, list[list[str]]] = {}

    if stats_composite_objs:
        all_rows: list[dict[str, str]] = []
        summary_sentences_by_obj: list[list[str]] = []
        for obj in stats_composite_objs:
            rows = _rows_from_obj(obj)
            if rows is None:
                if include_loose_lines:
                    loose.append(_loose_line(obj))
                continue
            summary_sentences_by_obj.append(_format_stats_composite_summary_as_sentences(rows))
            # 之前只保留“明细”会导致 DataFrame 丢失“汇总”信息；
            # 这里保留明细+汇总两类行，由 category 列区分。
            all_rows.extend(rows)
        if all_rows:
            # 长表：保留明细+汇总行，不再做宽表透视
            tables["单指标的明细汇总记录"] = all_rows
            stats_composite_summary_sentences_by_obj["单指标的明细汇总记录"] = summary_sentences_by_obj

    if "单指标的明细记录" in tables:
        # 长表：不再做宽表透视（原先会把“不同指标” pivot 成多列）
        tables["单指标的明细记录"] = tables["单指标的明细记录"]

    # 周期类：保持 start/end 字段在表内（长表），不再把日期范围“折叠”到 title

    # 2) 截断行数
    max_n = max(0, int(max_rows_per_table))
    if max_n > 0:
        for k in list(tables.keys()):
            if len(tables[k]) > max_n:
                tables[k] = tables[k][:max_n]

    # 3) 输出 DataFrame 列表
    out: list[pd.DataFrame] = []

    for et, rows in tables.items():
        if not rows:
            continue

        # 统一：长表 + 列名规范化 + 日期 ISO 化
        rows2 = _normalize_rows_for_long_table(rows)
        df = pd.DataFrame(rows2)
        # 显式列：数据类型（= entity_type），便于导出后直接筛选；不覆盖已有同名列
        if data_type_col is not None:
            col = str(data_type_col).strip()
            if col and col not in set(df.columns):
                try:
                    df.insert(0, col, et)
                except Exception:
                    df[col] = et
        df.attrs["entity_type"] = et
        df.attrs["title"] = _make_title(et)
        if et == "单指标的明细汇总记录" and et in stats_composite_summary_sentences_by_obj:
            df.attrs["summary_sentences_by_obj"] = stats_composite_summary_sentences_by_obj[et]
        out.append(df)

    if include_loose_lines and loose:
        df_loose = pd.DataFrame({"raw": loose})
        df_loose.attrs["entity_type"] = "零散或无法聚合"
        df_loose.attrs["title"] = "零散或无法聚合"
        out.append(df_loose)

    return out


def aggregate_dataframes_to_table(
    dfs: Sequence["pd.DataFrame"],
    *,
    include_loose_lines: bool = True,
    fillna: str | None = "",
    entity_type_col: str = "entity_type",
    data_type_col: str | None = "data_type",
    title_col: str = "title",
    table_idx_col: str = "table_idx",
    row_idx_col: str = "row_idx",
    unavailable_token: str = "不适用",
    empty_token: object | None = None,
) -> "pd.DataFrame":
    """
    将 `aggregate_patterns_to_dataframes()` 的“多个 DataFrame（按实体类型分表）”合并成一个统一表。

    目标：
    - **把 11 种数据类型 + 1 种无法解析（零散或无法聚合）**统一放到一个 DataFrame 里；
    - 不强行把不同类型“硬 pivot 成同一套业务列”，而是采用“列的并集”形成一个**宽 schema**：
      同一行只会填充该类型相关列，其它列为空（可用 fillna 控制）。

    参数：
    - include_loose_lines: 是否保留 entity_type="零散或无法聚合" 的行
    - fillna: 统一填充缺失值；设为 None 则保留 NaN
    - entity_type_col/title_col: 将 df.attrs 中的元信息写入列（便于下游单表消费/筛选）
    - data_type_col: 额外写入“数据类型”列（默认列名为中文），其值与 entity_type 相同；设为 None 关闭
    - table_idx_col/row_idx_col: 记录来源表序号与行序号，便于回溯
    - unavailable_token: 对该实体类型“不适用”的列填充值（用于区分“不适用” vs “空值”）
    - empty_token: 对该实体类型“适用但该条记录为空/缺失”的列填充值；设为 None 表示使用 pandas 标准缺失值 `pd.NA`
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("需要安装 pandas 才能使用 aggregate_dataframes_to_table（requirements.txt 已包含 pandas）。") from e

    # 统一宽表 schema（业务列固定集合；元信息列名可配置）
    _BUSINESS_COLS: list[str] = [
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

    def _full_schema_cols() -> list[str]:
        cols: list[str] = [entity_type_col]
        dtc = None if data_type_col is None else str(data_type_col).strip()
        if dtc and dtc != entity_type_col:
            cols.append(dtc)
        cols.extend([title_col, table_idx_col, row_idx_col, *_BUSINESS_COLS])
        return cols

    def _applicable_business_cols(et: str) -> set[str]:
        """
        定义：对某个实体类型来说，哪些业务列“可用”（应该填真实值或 empty_token），
        其余业务列视为“不适用”（填 unavailable_token）。
        """
        t = (et or "").strip()
        if t == "零散或无法聚合":
            return {"raw"}
        if t == "单指标的明细记录":
            return {"date", "time", "metric", "value", "unit"}
        if t == "周期数值单项总结":
            return {"start_date", "end_date", "metric", "value", "unit"}
        if t == "周期文本总结":
            return {"start_date", "end_date", "metric", "status"}
        if t == "周期数值对比记录":
            return {"date_range1", "value1", "date_range2", "value2", "logic", "diff", "metric"}
        if t == "周期数值多项总结":
            return {"start_date", "end_date", "metric", "value", "unit", "status"}
        if t == "单日期数值单项总结":
            return {"date", "metric", "value", "unit", "status"}
        if t == "单日期文本总结":
            return {"date", "metric", "status"}
        if t == "无时间日期的文本总结":
            return {"metric", "status"}
        if t == "无时间日期的数值总结":
            return {"metric", "value", "unit", "status"}
        if t == "单指标的明细汇总记录":
            # 新版 df：保留“明细+汇总”行，以 category 区分
            return {"category", "date", "metric", "value", "unit", "status"}
        if t == "单日期数值多项总结":
            return {"date", "metric", "value", "unit", "status"}
        # 未知类型：保守策略 —— 仅把该 df 里出现过的业务列视为“可用”
        return set()

    def _is_empty_cell(v: object) -> bool:
        if v is None:
            return True
        try:
            if pd.isna(v):  # type: ignore[arg-type]
                return True
        except Exception:
            pass
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    _EMPTY = pd.NA if empty_token is None else empty_token

    xs = list(dfs or [])
    if not xs:
        out = pd.DataFrame()
        out.attrs["entity_type"] = "ALL"
        out.attrs["title"] = "ALL"
        # 即使为空也输出固定 schema
        out = out.reindex(columns=_full_schema_cols())
        return out

    frames: list[pd.DataFrame] = []
    for table_idx, df in enumerate(xs):
        if df is None:
            continue
        df2 = df.copy()
        df2 = df2.reset_index(drop=True)

        attrs = getattr(df, "attrs", {}) or {}
        et = str(attrs.get("entity_type") or "").strip() or "未定义"
        title = str(attrs.get("title") or "").strip() or et

        # 元信息列（写入每一行，便于合并后筛选）
        df2[entity_type_col] = et
        dtc = None if data_type_col is None else str(data_type_col).strip()
        if dtc and dtc != entity_type_col:
            df2[dtc] = et
        df2[title_col] = title
        df2[table_idx_col] = int(table_idx)
        df2[row_idx_col] = list(range(len(df2)))

        # 业务列全量对齐 + 不适用/空值区分
        applicable = _applicable_business_cols(et)
        # 未知类型：用 df 中已出现的业务列作为“可用列”
        if not applicable and et not in (
            "单指标的明细记录",
            "单指标的明细汇总记录",
            "周期数值对比记录",
            "周期数值单项总结",
            "周期数值多项总结",
            "周期文本总结",
            "单日期数值单项总结",
            "单日期数值多项总结",
            "单日期文本总结",
            "无时间日期的文本总结",
            "无时间日期的数值总结",
            "零散或无法聚合",
        ):
            applicable = {c for c in _BUSINESS_COLS if c in df2.columns}

        for col in _BUSINESS_COLS:
            if col not in df2.columns:
                df2[col] = pd.NA
            if col in applicable:
                # 适用列：空/缺失 -> empty_token；其余值保留
                df2[col] = df2[col].apply(lambda x: _EMPTY if _is_empty_cell(x) else x)
            else:
                # 不适用列：统一标记
                df2[col] = unavailable_token

        frames.append(df2)

    if not frames:
        out = pd.DataFrame()
        out.attrs["entity_type"] = "ALL"
        out.attrs["title"] = "ALL"
        out = out.reindex(columns=_full_schema_cols())
        return out

    out = pd.concat(frames, ignore_index=True, sort=False)

    # 过滤 loose 表（无法解析/零散行）
    if not include_loose_lines and entity_type_col in out.columns:
        out = out[out[entity_type_col] != "零散或无法聚合"].reset_index(drop=True)

    if fillna is not None:
        out = out.fillna(fillna)

    # 固定 schema 列永远放在最前面，其余列保持原顺序追加（兼容未来扩展）
    schema_cols = _full_schema_cols()
    extra_cols = [c for c in list(out.columns) if c not in set(schema_cols)]
    cols = [c for c in schema_cols if c in out.columns] + extra_cols
    out = out.loc[:, cols]

    out.attrs["entity_type"] = "ALL"
    out.attrs["title"] = "ALL"
    return out


__all__ = ["aggregate_patterns_to_dataframes", "aggregate_dataframes_to_table"]

