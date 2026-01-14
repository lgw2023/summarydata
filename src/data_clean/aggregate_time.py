from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
import re
from typing import Any, Mapping, Sequence

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
from .normalize import (
    _format_date_range,
    _is_missing_token,
    _normalize_date_cn,
    _normalize_date_or_range_cn,
    _normalize_time_cn_token,
)


def aggregate_patterns_by_timejson(
    patterns: Sequence[PersonalDataPattern],
    *,
    include_unknown_time: bool = True,
    add_summary_text: bool = True,
) -> list[dict[str, Any]]:
    """
    将数据类列表按“时间桶”聚合为结构化 JSON（list[dict]）。

    时间桶类型（time.type）：
    - datetime: 单日 + 具体时间点（例：2025年02月01日 06:07）
    - date: 单日日期（例：02月01日）
    - period: 日期范围（例：02月01日~02月07日）
    - DateRange: 两段“日期范围”的对比（例：2023年03月04日~2023年12月31日 vs 2024年01月01日~2024年12月31日）
    - unknown: 无法抽取任何时间信息

    输出每个 bucket 为一条 dict：
      {
        "time": {...},
        "events": [ {name,value,unit,status,entity_type,...}, ... ],
        "fallback": [ <raw/dict> ... ],
        "summary": "xx时间，事件1...，事件2..."
      }

    约定：
    - 若某个对象/片段无法抽取成“事件三元组(名/值/单位)+可选状态”，则写入 fallback；
      仍会尽量把同一时间桶内其它可抽取事件合并输出。
    """
    objs = list(patterns or [])
    if not objs:
        return []

    exploded = []
    for obj in objs:
        exploded.extend(_explode_pattern_to_time_items(obj))

    # group by time.bucket_key
    groups: dict[str, list[dict[str, Any]]] = {}
    order_keys: list[str] = []
    for it in exploded:
        t = it.get("time") or {}
        key = str(t.get("bucket_key") or "")
        if (not key) and (not include_unknown_time):
            continue
        if not key:
            key = "unknown"
            t["bucket_key"] = key
            t.setdefault("type", "unknown")
        if key not in groups:
            groups[key] = []
            order_keys.append(key)
        groups[key].append(it)

    # stable sort by time semantics (unknown at end)
    order_keys_sorted = sorted(order_keys, key=lambda k: _bucket_sort_key(groups[k][0].get("time") or {}))

    out: list[dict[str, Any]] = []
    for key in order_keys_sorted:
        items = groups.get(key, [])
        if not items:
            continue

        time_obj = dict(items[0].get("time") or {})
        # remove internal key
        time_obj.pop("bucket_key", None)

        events: list[dict[str, Any]] = []
        fallback: list[Any] = []
        for it in items:
            if it.get("event") is not None:
                events.append(it["event"])
            if it.get("fallback") is not None:
                fb = it["fallback"]
                if isinstance(fb, list):
                    fallback.extend(fb)
                else:
                    fallback.append(fb)

        row: dict[str, Any] = {"time": time_obj, "events": events}
        if fallback:
            row["fallback"] = fallback

        if add_summary_text:
            # 若上游为某些 pattern 显式提供了“原始摘要”，则优先使用它
            custom_summaries = [
                str(it.get("summary") or "").strip()
                for it in items
                if isinstance(it, dict) and isinstance(it.get("summary"), str) and str(it.get("summary") or "").strip()
            ]
            if custom_summaries:
                row["summary"] = custom_summaries[0]
                out.append(row)
                continue
            label = str(time_obj.get("label") or "")
            row["summary"] = _format_bucket_summary(label, events, fallback)

        out.append(row)

    return out


def _explode_pattern_to_time_items(obj: PersonalDataPattern) -> list[dict[str, Any]]:
    """
    将一个 pattern 对象尽量展开成若干条“带时间桶的事件 item”：
      { time: {...bucket_key...}, event: {...} } 或 { time: {...}, fallback: ... }
    """
    # 兜底类：直接丢入 unknown
    if isinstance(obj, UnparsedRawPersonalData):
        return [
            {
                "time": {"type": "unknown", "label": "", "bucket_key": "unknown"},
                "fallback": _fallback_payload(obj),
            }
        ]

    et = str(getattr(obj, "实体类型", "") or "").strip() or "未定义"

    # 周期对比：按“两段日期范围”聚合；事件结构与其它类型不同（value=[v1,v2] + diff/logic）
    if isinstance(obj, PeriodValueCompareRecord):
        ranges1 = list(getattr(obj, "日期范围1列表", []) or [])
        ranges2 = list(getattr(obj, "日期范围2列表", []) or [])
        v1s = list(getattr(obj, "数值1列表", []) or [])
        v2s = list(getattr(obj, "数值2列表", []) or [])
        diffs = list(getattr(obj, "差异数值列表", []) or [])
        names = list(getattr(obj, "指标名称列表", []) or [])
        units = list(getattr(obj, "单位列表", []) or [])
        logics = list(getattr(obj, "对比逻辑类型列表", []) or [])
        diff_types = list(getattr(obj, "差异数值类型列表", []) or [])
        n = max(
            len(ranges1),
            len(ranges2),
            len(v1s),
            len(v2s),
            len(diffs),
            len(names),
            len(units),
            len(logics),
            len(diff_types),
        )
        if n <= 0:
            return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]

        summary_text = ""
        try:
            summary_text = str(getattr(obj, "原始个人数据", None) or "").strip()
        except Exception:
            summary_text = ""
        if not summary_text:
            try:
                summary_text = str(obj.recover_to_raw_data() or "").strip()
            except Exception:
                summary_text = ""

        items: list[dict[str, Any]] = []
        for i in range(n):
            r1 = str(ranges1[i] if i < len(ranges1) else "").strip()
            r2 = str(ranges2[i] if i < len(ranges2) else "").strip()
            name = str(names[i] if i < len(names) else "").strip()
            v1 = str(v1s[i] if i < len(v1s) else "").strip()
            v2 = str(v2s[i] if i < len(v2s) else "").strip()
            diff = str(diffs[i] if i < len(diffs) else "").strip()
            unit_raw = str(units[i] if i < len(units) else "").strip() or "无"
            logic = str(logics[i] if i < len(logics) else "").strip()
            dv_type = diff_types[i] if i < len(diff_types) else None

            time_info = _time_bucket_from_date_range_compare(r1, r2)
            if not time_info or not name:
                items.append(
                    {
                        "time": time_info or {"type": "unknown", "label": "", "bucket_key": "unknown"},
                        "fallback": _fallback_payload(obj),
                        "summary": summary_text,
                    }
                )
                continue

            # compare event：保持字段更贴近原语义（不复用 _event_obj）
            unit = unit_raw if unit_raw and unit_raw != "无" and unit_raw != "-" else None
            ev: dict[str, Any] = {
                "entity_type": et,
                "name": name,
                "value": [v1, v2],
                "logic": logic if logic else None,
                "diff_value": diff if diff else None,
                "diff_value_type": str(dv_type) if dv_type is not None else None,
                # 与其它事件不同：这里显式输出 unit（可能为 null），便于下游 schema 稳定
                "unit": unit,
            }
            items.append({"time": time_info, "event": ev, "summary": summary_text})
        return items

    # 1) 单指标明细：按 (日期, 时间) 或 日期 聚合
    if isinstance(obj, SingleMetricDetailRecord):
        core = getattr(obj, "核心字段", None)
        metric = (getattr(core, "指标名称", "") if core else "") or ""
        unit = (getattr(core, "单位", "") if core else "") or "无"
        vt = getattr(core, "数值类型", None) if core else None

        ds = list(getattr(obj, "日期列表", []) or [])
        ts = list(getattr(obj, "时间列表", []) or [])
        vs = list(getattr(obj, "数值列表", []) or [])
        n = max(len(ds), len(ts), len(vs))
        if n <= 0:
            return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]

        items: list[dict[str, Any]] = []
        for i in range(n):
            d_raw = str(ds[i] if i < len(ds) else "").strip()
            t_raw = str(ts[i] if i < len(ts) else "").strip()
            v_raw = str(vs[i] if i < len(vs) else "").strip()

            time_info = _time_bucket_from_date_time(d_raw, t_raw)
            if not time_info:
                items.append({"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)})
                continue

            if not metric:
                items.append({"time": time_info, "fallback": _fallback_payload(obj)})
                continue

            v2 = _normalize_value_for_json(v_raw, unit, vt)
            ev = _event_obj(
                entity_type=et,
                name=metric,
                value=v2 if v2 else None,
                unit=unit if unit and unit != "无" else None,
                status=None,
                value_type=str(vt) if vt is not None else None,
            )
            items.append({"time": time_info, "event": ev})
        return items

    # 2) 单日期数值单项总结
    if isinstance(obj, SingleDateValueSingleSummaryRecord):
        ds = list(getattr(obj, "日期列表", []) or [])
        names = list(getattr(obj, "指标名称列表", []) or [])
        vals = list(getattr(obj, "数值列表", []) or [])
        units = list(getattr(obj, "单位列表", []) or [])
        sts = list(getattr(obj, "状态描述列表", []) or [])
        vts = list(getattr(obj, "数值类型列表", []) or [])
        n = max(len(ds), len(names), len(vals), len(units), len(sts), len(vts))
        if n <= 0:
            return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]

        items: list[dict[str, Any]] = []
        for i in range(n):
            d = str(ds[i] if i < len(ds) else "").strip()
            name = str(names[i] if i < len(names) else "").strip()
            val = str(vals[i] if i < len(vals) else "").strip()
            unit = str(units[i] if i < len(units) else "").strip() or "无"
            st = str(sts[i] if i < len(sts) else "").strip()
            vt = vts[i] if i < len(vts) else None

            time_info = _time_bucket_from_date_time(d, "")
            if not time_info or not name:
                items.append({"time": time_info or {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)})
                continue

            val2 = _normalize_value_for_json(val, unit, vt)
            ev = _event_obj(
                entity_type=et,
                name=name,
                value=val2 if val2 else None,
                unit=unit if unit and unit != "无" else None,
                status=st if (st and st != "无" and st != "-") else None,
                value_type=str(vt) if vt is not None else None,
            )
            items.append({"time": time_info, "event": ev})
        return items

    # 3) 单日期数值多项总结（新版）
    if isinstance(obj, SingleDateValueMultiSummaryRecord):
        ds = list(getattr(obj, "日期列表", []) or [])
        names = list(getattr(obj, "指标名称列表", []) or [])
        vals = list(getattr(obj, "数值列表", []) or [])
        units = list(getattr(obj, "单位列表", []) or [])
        sts = list(getattr(obj, "状态描述列表", []) or [])
        vts = list(getattr(obj, "数值类型列表", []) or [])
        n = max(len(ds), len(names), len(vals), len(units), len(sts), len(vts))
        if n <= 0:
            return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]

        items: list[dict[str, Any]] = []
        for i in range(n):
            d = str(ds[i] if i < len(ds) else "").strip()
            name = str(names[i] if i < len(names) else "").strip()
            val = str(vals[i] if i < len(vals) else "").strip()
            unit = str(units[i] if i < len(units) else "").strip() or "无"
            st = str(sts[i] if i < len(sts) else "").strip()
            vt = vts[i] if i < len(vts) else None

            time_info = _time_bucket_from_date_time(d, "")
            if not time_info or not name:
                items.append({"time": time_info or {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)})
                continue

            val2 = _normalize_value_for_json(val, unit, vt)
            ev = _event_obj(
                entity_type=et,
                name=name,
                value=val2 if val2 else None,
                unit=unit if unit and unit != "无" else None,
                status=st if (st and st != "无" and st != "-") else None,
                value_type=str(vt) if vt is not None else None,
            )
            items.append({"time": time_info, "event": ev})
        return items

    # 4) 单日期文本总结
    if isinstance(obj, SingleDateTextSummaryRecord):
        ds = list(getattr(obj, "日期列表", []) or [])
        names = list(getattr(obj, "指标名称列表", []) or [])
        sts = list(getattr(obj, "状态描述列表", []) or [])
        n = max(len(ds), len(names), len(sts))
        if n <= 0:
            return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]

        items: list[dict[str, Any]] = []
        for i in range(n):
            d = str(ds[i] if i < len(ds) else "").strip()
            name = str(names[i] if i < len(names) else "").strip()
            st = str(sts[i] if i < len(sts) else "").strip()
            time_info = _time_bucket_from_date_time(d, "")
            if not time_info or not name:
                items.append({"time": time_info or {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)})
                continue

            ev = _event_obj(
                entity_type=et,
                name=name,
                value=None,
                unit=None,
                status=st if (st and st != "无" and st != "-") else None,
                value_type=None,
            )
            items.append({"time": time_info, "event": ev})
        return items

    # 5) 周期数值单项总结 / 周期文本总结 / 周期数值多项总结：按日期范围聚合
    if isinstance(obj, (PeriodValueSingleSummaryRecord, PeriodTextSummaryRecord, PeriodValuemMultiSummaryRecord)):
        starts = list(getattr(obj, "开始日期列表", []) or [])
        ends = list(getattr(obj, "结束日期列表", []) or [])
        names = list(getattr(obj, "指标名称列表", []) or [])
        n = max(len(starts), len(ends), len(names))

        vals = list(getattr(obj, "数值列表", []) or [])
        units = list(getattr(obj, "单位列表", []) or [])
        sts = list(getattr(obj, "状态描述列表", []) or [])
        vts = list(getattr(obj, "数值类型列表", []) or [])
        n = max(n, len(vals), len(units), len(sts), len(vts))
        if n <= 0:
            return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]

        items: list[dict[str, Any]] = []
        for i in range(n):
            st = str(starts[i] if i < len(starts) else "").strip()
            ed = str(ends[i] if i < len(ends) else "").strip()
            name = str(names[i] if i < len(names) else "").strip()
            val = str(vals[i] if i < len(vals) else "").strip()
            unit = str(units[i] if i < len(units) else "").strip() or "无"
            status = str(sts[i] if i < len(sts) else "").strip()
            vt = vts[i] if i < len(vts) else None

            time_info = _time_bucket_from_period(st, ed)
            if not time_info or not name:
                items.append({"time": time_info or {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)})
                continue

            val2 = _normalize_value_for_json(val, unit, vt)
            ev = _event_obj(
                entity_type=et,
                name=name,
                value=val2 if val2 else None,
                unit=unit if unit and unit != "无" else None,
                status=status if (status and status != "无" and status != "-") else None,
                value_type=str(vt) if vt is not None else None,
            )
            items.append({"time": time_info, "event": ev})
        return items

    # 6) 单指标统计复合记录：明细按日期聚合；汇总无日期 -> unknown
    if isinstance(obj, SingleMetricStatsRecord):
        core = getattr(obj, "核心字段", None)
        metric = (getattr(core, "指标名称", "") if core else "") or ""
        unit = (getattr(core, "单位", "") if core else "") or "无"
        vt = getattr(core, "数值类型", None) if core else None

        ds = list(getattr(obj, "日期列表", []) or [])
        vs = list(getattr(obj, "数值列表", []) or [])
        n_d = max(len(ds), len(vs))

        items: list[dict[str, Any]] = []
        for i in range(n_d):
            d = str(ds[i] if i < len(ds) else "").strip()
            v = str(vs[i] if i < len(vs) else "").strip()
            time_info = _time_bucket_from_date_time(d, "")
            if not time_info or not metric:
                items.append({"time": time_info or {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)})
                continue
            v2 = _normalize_value_for_json(v, unit, vt)
            ev = _event_obj(
                entity_type=et,
                name=metric,
                value=v2 if v2 else None,
                unit=unit if unit and unit != "无" else None,
                status=None,
                value_type=str(vt) if vt is not None else None,
            )
            items.append({"time": time_info, "event": ev})

        # 汇总：原始文本的末尾统计项通常不带日期，但可由明细日期列表推断周期 [min_date, max_date]。
        # 需求：不要落到 unknown/fallback，而是按 period 时间桶输出为 events（对齐 PeriodValuemMultiSummaryRecord 的语义）。
        sum_names = list(getattr(obj, "统计指标名称列表", []) or [])
        sum_vals = list(getattr(obj, "统计数值列表", []) or [])
        sum_sts = list(getattr(obj, "统计状态描述列表", []) or [])
        n_s = max(len(sum_names), len(sum_vals), len(sum_sts))
        if n_s > 0:
            period_time = _infer_period_bucket_from_dates(ds)
            if period_time is None:
                # 极端兜底：确实无法推断日期范围时，也按 unknown 输出事件（仍避免 fallback）
                period_time = {"type": "unknown", "label": "", "bucket_key": "unknown"}
            for i in range(n_s):
                nm = str(sum_names[i] if i < len(sum_names) else "").strip()
                sv = str(sum_vals[i] if i < len(sum_vals) else "").strip()
                st = str(sum_sts[i] if i < len(sum_sts) else "").strip()
                if not (nm or sv or st):
                    continue
                sv2 = _normalize_value_for_json(sv, unit, vt)
                ev = _event_obj(
                    entity_type=et,
                    name=nm or None,
                    value=sv2 if sv2 else None,
                    unit=unit if unit and unit != "无" else None,
                    status=st if (st and st != "无" and st != "-") else None,
                    value_type=str(vt) if vt is not None else None,
                )
                items.append({"time": period_time, "event": ev})

        return items if items else [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]

    # 7) 无时间日期的两类总结：无法按时间聚合 -> unknown
    if isinstance(obj, NoTimestampTextSummaryRecord):
        # 需求：这类数据不要整条塞进 fallback，而是按其它类型一样输出 events；
        # 仅时间桶信息为空（unknown/label=""）。
        names = list(getattr(obj, "指标名称列表", []) or [])
        sts = list(getattr(obj, "状态描述列表", []) or [])
        n = max(len(names), len(sts))
        if n <= 0:
            # 极端异常：没有任何可用字段时，仍走兜底（避免丢数据）
            return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]
        items: list[dict[str, Any]] = []
        time_info = {"type": "unknown", "label": "", "bucket_key": "unknown"}
        for i in range(n):
            name = str(names[i] if i < len(names) else "").strip()
            st = str(sts[i] if i < len(sts) else "").strip()
            # 与其它“文本总结”一致：value 为空，状态在 status
            ev = _event_obj(
                entity_type=et,
                name=name or None,
                value=None,
                unit=None,
                status=st if (st and st != "无" and st != "-") else None,
                value_type=None,
            )
            items.append({"time": time_info, "event": ev})
        return items

    if isinstance(obj, NoDateValueSummaryRecord):
        # 同上：按 events 输出；时间为空（unknown/label=""）。
        names = list(getattr(obj, "指标名称列表", []) or [])
        vals = list(getattr(obj, "数值列表", []) or [])
        units = list(getattr(obj, "单位列表", []) or [])
        sts = list(getattr(obj, "状态描述列表", []) or [])
        vts = list(getattr(obj, "数值类型列表", []) or [])
        n = max(len(names), len(vals), len(units), len(sts), len(vts))
        if n <= 0:
            return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]
        items: list[dict[str, Any]] = []
        time_info = {"type": "unknown", "label": "", "bucket_key": "unknown"}
        for i in range(n):
            name = str(names[i] if i < len(names) else "").strip()
            val = str(vals[i] if i < len(vals) else "").strip()
            unit = str(units[i] if i < len(units) else "").strip() or "无"
            st = str(sts[i] if i < len(sts) else "").strip()
            vt = vts[i] if i < len(vts) else None
            val2 = _normalize_value_for_json(val, unit, vt)
            ev = _event_obj(
                entity_type=et,
                name=name or None,
                value=val2 if val2 else None,
                unit=unit if unit and unit != "无" else None,
                status=st if (st and st != "无" and st != "-") else None,
                value_type=str(vt) if vt is not None else None,
            )
            items.append({"time": time_info, "event": ev})
        return items

    # 其它未知类型：兜底
    return [{"time": {"type": "unknown", "label": "", "bucket_key": "unknown"}, "fallback": _fallback_payload(obj)}]


def _time_bucket_from_date_time(date_raw: str, time_raw: str) -> dict[str, Any] | None:
    d0 = str(date_raw or "").strip()
    t0 = str(time_raw or "").strip()
    if not d0 and not t0:
        return None
    d = _normalize_date_cn(d0) if d0 else ""
    t = _normalize_time_cn_token(t0) if t0 else ""
    if t and _is_missing_token(t):
        t = ""

    if d and t:
        label = f"{d} {t}"
        return {"type": "datetime", "date": d, "time": t, "label": label, "bucket_key": f"dt:{label}"}
    if d:
        return {"type": "date", "date": d, "label": d, "bucket_key": f"d:{d}"}
    # 只有时间点但无日期：不做强行归并
    return {"type": "unknown", "label": t, "bucket_key": "unknown"}


def _time_bucket_from_period(start_raw: str, end_raw: str) -> dict[str, Any] | None:
    st = str(start_raw or "").strip()
    ed = str(end_raw or "").strip()
    if not st and not ed:
        return None
    label = _format_date_range(st, ed)
    label = label.strip()
    if not label:
        # 兜底：尝试单端日期
        if st:
            label = _normalize_date_cn(st)
        elif ed:
            label = _normalize_date_cn(ed)
    if not label:
        return None
    return {"type": "period", "start": _normalize_date_cn(st) if st else "", "end": _normalize_date_cn(ed) if ed else "", "label": label, "bucket_key": f"p:{label}"}


def _time_bucket_from_date_range_compare(range1_raw: str, range2_raw: str) -> dict[str, Any] | None:
    """
    周期对比：用“两段日期范围”作为时间桶。
    - range1_raw / range2_raw 可以是单日期或日期范围；内部统一标准化为中文日期/范围格式。
    """
    r1 = str(range1_raw or "").strip()
    r2 = str(range2_raw or "").strip()
    if not r1 and not r2:
        return None
    r1n = _normalize_date_or_range_cn(r1) if r1 else ""
    r2n = _normalize_date_or_range_cn(r2) if r2 else ""
    if not (r1n or r2n):
        return None
    label = f"{r1n} vs {r2n}".strip()
    return {"type": "DateRange", "start": r1n, "end": r2n, "label": label, "bucket_key": f"c:{r1n}|{r2n}"}


def _event_obj(
    *,
    entity_type: str,
    name: str | None,
    value: str | None,
    unit: str | None,
    status: str | None,
    value_type: str | None,
) -> dict[str, Any]:
    ev: dict[str, Any] = {"entity_type": entity_type}
    if name:
        ev["name"] = name
    if value is not None and str(value).strip():
        ev["value"] = str(value).strip()
    if unit is not None and str(unit).strip() and str(unit).strip() != "无":
        ev["unit"] = str(unit).strip()
    if status is not None and str(status).strip() and str(status).strip() != "无":
        ev["status"] = str(status).strip()
    if value_type is not None and str(value_type).strip():
        ev["value_type"] = str(value_type).strip()
    return ev


def _strip_trailing_unit(value: str, unit: str) -> str:
    """
    尽量把“尾部单位”从 value 中剥离，使 JSON 的 value 更接近“纯值”。
    仅做非常保守的处理，避免破坏复合表达（如 "2小时49分钟"）：
    - unit == "无"：不处理
    - value 以 unit 结尾：直接剥离
    - unit 形如 "每公里" / "每米" 且 value 以 "/公里" / "/米" 结尾：剥离斜杠后缀
    """
    v = str(value or "").strip()
    u = str(unit or "").strip()
    if not v or (not u) or u == "无":
        return v
    if v.endswith(u):
        v2 = v[: -len(u)].strip()
        return v2 if v2 else v
    # 配速类：unit=每公里，而 value 可能写作 "7分39秒/公里"
    if u.startswith("每") and "/" in v:
        denom = u[1:].strip()
        if denom in ("公里", "千米", "km", "KM", "Km") and v.endswith("/公里"):
            v2 = v[: -len("/公里")].strip()
            return v2 if v2 else v
        if denom == "米" and v.endswith("/米"):
            v2 = v[: -len("/米")].strip()
            return v2 if v2 else v
    return v


def _normalize_float_range_value_for_json(value: str, unit: str) -> str:
    """
    FloatRange 专用：把区间两侧的单位都剥离，使 JSON 的 value 更接近“纯区间值”。

    目标：
    - "31.3°C-31.3°C" -> "31.3-31.3"
    - "96%-98%" -> "96-98"
    - "-5°C--2°C" -> "-5--2"

    约束：
    - unit 为空/为“无”：不处理（保持原样）
    - 仅在“看起来像数值区间”时处理，避免误伤其它复合表达
    """
    v = str(value or "").strip()
    u = str(unit or "").strip()
    if not v or (not u) or u == "无":
        return v

    # 先用较严格的 regex（支持负号区间，如 -5--2）
    seps = r"[-～~—−]"
    pat = re.compile(
        rf"^\s*(?P<a>[-+]?\d+(?:\.\d+)?)\s*(?:{re.escape(u)})?\s*"
        rf"(?P<sep>{seps})\s*(?P<b>[-+]?\d+(?:\.\d+)?)\s*(?:{re.escape(u)})?\s*$"
    )
    m = pat.fullmatch(v)
    if m:
        a = m.group("a")
        b = m.group("b")
        return f"{a}-{b}"

    # 兜底：宽松 split（仅在确认为“两段都含数字”时才做两侧剥离）
    for sep in ("-", "～", "~", "—", "−"):
        if sep in v and not v.startswith(sep):
            parts = [p.strip() for p in v.split(sep)]
            if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                return f"{p0}-{p1}".strip()

    # 仍然无法识别为区间时，保持旧逻辑（最多剥离尾部单位）
    return _strip_trailing_unit(v, u)


def _normalize_value_for_json(value: str, unit: str, value_type: Any) -> str:
    """
    JSON 事件 value 规范化入口：
    - FloatRange：区间两侧剥离单位
    - 其它：延续原有“仅剥离尾部单位”的保守策略
    """
    vt = str(value_type or "").strip()
    if vt == "FloatRange":
        return _normalize_float_range_value_for_json(value, unit)
    return _strip_trailing_unit(value, unit)


def _attach_unit_to_float_range_for_summary(value: str, unit: str) -> str:
    """
    summary 文本用：将 "a-b" + unit 还原成更自然的 "a单位-b单位"。
    若 value 已含 unit 或无法识别为区间，则退化为“必要时追加到末尾”。
    """
    v = str(value or "").strip()
    u = str(unit or "").strip()
    if not v or (not u) or u == "无":
        return v
    if u in v:
        return v

    m = re.fullmatch(r"\s*(?P<a>[-+]?\d+(?:\.\d+)?)\s*(?P<sep>[-～~—−])\s*(?P<b>[-+]?\d+(?:\.\d+)?)\s*", v)
    if m:
        a = m.group("a")
        b = m.group("b")
        return f"{a}{u}-{b}{u}"

    return v if v.endswith(u) else f"{v}{u}"


def _fallback_payload(obj: Any) -> Any:
    """
    将无法结构化抽取的对象，转为 JSON 可序列化的兜底载荷。
    优先级：
    - to_full_item()（最完整、结构化）
    - recover_to_raw_data()（最接近原始文本）
    - dataclass -> asdict
    - str(obj)
    """
    try:
        fn = getattr(obj, "to_full_item", None)
        if callable(fn):
            return fn()
    except Exception:
        pass
    try:
        fn = getattr(obj, "recover_to_raw_data", None)
        if callable(fn):
            s = fn()
            if isinstance(s, str) and s.strip():
                return s.strip()
    except Exception:
        pass
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    return str(obj)


_CN_YMD_RE = re.compile(r"(?P<y>\d{4})年(?P<m>\d{2})月(?P<d>\d{2})日")
_CN_MD_RE = re.compile(r"(?P<m>\d{2})月(?P<d>\d{2})日")


def _parse_cn_date_for_sort(s: str) -> tuple[int, int, int, str]:
    raw = str(s or "").strip()
    if not raw:
        return (9999, 99, 99, raw)
    t = _normalize_date_cn(raw)
    m = _CN_YMD_RE.fullmatch(t)
    if m:
        return (int(m.group("y")), int(m.group("m")), int(m.group("d")), raw)
    m2 = _CN_MD_RE.fullmatch(t)
    if m2:
        # 无年份：放到后面，但保持同月日内部排序
        return (9999, int(m2.group("m")), int(m2.group("d")), raw)
    return (9999, 99, 99, raw)


def _bucket_sort_key(time_obj: Mapping[str, Any]) -> tuple[int, int, int, int, int, int, str]:
    """
    排序规则：
    - datetime/date/period 先按日期排序，unknown 放最后
    - period 按 start 作为主排序
    """
    tp = str(time_obj.get("type") or "")
    if tp == "datetime":
        d = str(time_obj.get("date") or "")
        t = str(time_obj.get("time") or "")
        y, mo, da, _ = _parse_cn_date_for_sort(d)
        hh, mm = _parse_hhmm_for_sort(t)
        return (0, y, mo, da, hh, mm, str(time_obj.get("label") or ""))
    if tp == "date":
        d = str(time_obj.get("date") or "")
        y, mo, da, _ = _parse_cn_date_for_sort(d)
        return (1, y, mo, da, 0, 0, str(time_obj.get("label") or ""))
    if tp == "period":
        st = str(time_obj.get("start") or "")
        y, mo, da, _ = _parse_cn_date_for_sort(st)
        return (2, y, mo, da, 0, 0, str(time_obj.get("label") or ""))
    if tp == "DateRange":
        # 以第一个范围的左端日期作为主排序键
        r1 = str(time_obj.get("start") or "").strip()
        left = r1.split("~", 1)[0].strip() if "~" in r1 else r1
        y, mo, da, _ = _parse_cn_date_for_sort(left)
        return (2, y, mo, da, 0, 0, str(time_obj.get("label") or ""))
    return (9, 9999, 99, 99, 99, 99, str(time_obj.get("label") or ""))


def _parse_hhmm_for_sort(s: str) -> tuple[int, int]:
    raw = str(s or "").strip()
    m = re.fullmatch(r"(?P<h>\d{2}):(?P<m>\d{2})", raw)
    if not m:
        return (99, 99)
    try:
        return (int(m.group("h")), int(m.group("m")))
    except Exception:
        return (99, 99)


def _format_bucket_summary(label: str, events: Sequence[Mapping[str, Any]], fallback: Sequence[Any]) -> str:
    t = str(label or "").strip()
    parts: list[str] = []
    for ev in events or []:
        name = str(ev.get("name") or "").strip()
        value = str(ev.get("value") or "").strip()
        unit = str(ev.get("unit") or "").strip()
        status = str(ev.get("status") or "").strip()
        vt = str(ev.get("value_type") or "").strip()
        if not (name or value or status):
            continue
        seg = ""
        if name:
            seg += name
        if value:
            # value 可能已包含单位；unit 仅在必要时追加
            if vt == "FloatRange" and unit:
                seg += _attach_unit_to_float_range_for_summary(value, unit)
            else:
                if unit and (not value.endswith(unit)):
                    seg += f"{value}{unit}"
                else:
                    seg += value
        if status and status not in ("无", "-"):
            seg += status
        if seg:
            parts.append(seg)

    if not parts:
        # 没法形成摘要：给个最低信息量兜底
        if t:
            return f"{t}（无可结构化事件，fallback={len(list(fallback or []))}）"
        return f"无可结构化事件（fallback={len(list(fallback or []))}）"

    # label 为空（例如 unknown/label=""）时，不输出“无时间，”前缀，直接给事件摘要
    if not t:
        return "，".join(parts)
    return f"{t}，" + "，".join(parts)


def _infer_period_bucket_from_dates(dates: Sequence[str]) -> dict[str, Any] | None:
    """
    从一组“中文日期 token”（可能含年份或不含）推断一个 period 时间桶：
    - start = 最小日期
    - end = 最大日期
    返回值结构对齐 `_time_bucket_from_period`。

    说明：这里不尝试补年份，只做排序与范围格式化。
    """
    ds = [str(d or "").strip() for d in list(dates or [])]
    # 过滤明显的缺失占位
    cleaned: list[str] = []
    for d in ds:
        if not d:
            continue
        dn = _normalize_date_cn(d)
        if not dn:
            continue
        if _is_missing_token(dn):
            continue
        cleaned.append(dn)
    if not cleaned:
        return None

    cleaned_sorted = sorted(cleaned, key=_parse_cn_date_for_sort)
    st = cleaned_sorted[0]
    ed = cleaned_sorted[-1]
    return _time_bucket_from_period(st, ed)


__all__ = ["aggregate_patterns_by_timejson"]

