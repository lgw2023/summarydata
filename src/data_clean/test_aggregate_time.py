from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_clean import *  # type: ignore  # noqa: F403
else:
    from . import *  # type: ignore  # noqa: F403


def _self_test_aggregate_patterns_to_time_json(
    lines,
    *,
    max_cases: int | None = 30,
    print_preview: bool = True,
) -> None:
    """
    self-test：验证
    - explode_newlines_and_route_to_dataclasses()
    - aggregate_patterns_by_timejson()
    的端到端行为（从原始多行文本 -> 数据类列表 -> 按时间桶聚合 JSON/JSONL）。

    风格：对齐 `test_aggregate_format.py` 的 `_self_test_aggregate_patterns_to_formatted_text`：
    - 逐条样例做断言
    - 输出结构做“不过度脆弱”的检查
    """
    import json

    xs = list(lines or [])
    if max_cases is not None:
        xs = xs[: max(0, int(max_cases))]

    def _short(s: object, n: int = 220) -> str:
        t = str(s if s is not None else "").replace("\n", " ").strip()
        if n <= 0:
            return ""
        return t if len(t) <= n else (t[: n - 1] + "…")

    def _assert_items(items: object, *, case_idx: int, raw: str) -> None:
        if not isinstance(items, list) or not items:
            raise AssertionError(
                f"[self-test][time-agg] 输出为空/非 list：case#{case_idx} raw={_short(raw)}"
            )

        seen: set[tuple[str, str, str, str, str, str]] = set()
        types: list[str] = []
        for j, row in enumerate(items):
            if not isinstance(row, dict):
                raise AssertionError(
                    f"[self-test][time-agg] 行不是 dict：case#{case_idx} row#{j} raw={_short(raw)}"
                )
            if "time" not in row or "events" not in row:
                raise AssertionError(
                    f"[self-test][time-agg] 缺少 time/events：case#{case_idx} row#{j} raw={_short(raw)} row={row!r}"
                )

            time_obj = row.get("time")
            events = row.get("events")
            if not isinstance(time_obj, dict):
                raise AssertionError(
                    f"[self-test][time-agg] time 不是 dict：case#{case_idx} row#{j} raw={_short(raw)}"
                )
            if "bucket_key" in time_obj:
                raise AssertionError(
                    f"[self-test][time-agg] time 泄漏内部字段 bucket_key：case#{case_idx} row#{j} raw={_short(raw)}"
                )

            tp = str(time_obj.get("type") or "")
            if tp not in {"datetime", "date", "period", "DateRange", "unknown"}:
                raise AssertionError(
                    f"[self-test][time-agg] 非法 time.type：{tp!r} case#{case_idx} row#{j} raw={_short(raw)}"
                )
            types.append(tp)

            label = str(time_obj.get("label") or "").strip()
            if tp != "unknown" and not label:
                raise AssertionError(
                    f"[self-test][time-agg] 非 unknown 时间桶缺少 label：case#{case_idx} row#{j} raw={_short(raw)} time={time_obj!r}"
                )

            if not isinstance(events, list):
                raise AssertionError(
                    f"[self-test][time-agg] events 不是 list：case#{case_idx} row#{j} raw={_short(raw)}"
                )
            for k, ev in enumerate(events):
                if not isinstance(ev, dict):
                    raise AssertionError(
                        f"[self-test][time-agg] event 不是 dict：case#{case_idx} row#{j} ev#{k} raw={_short(raw)}"
                    )
                if "entity_type" not in ev:
                    raise AssertionError(
                        f"[self-test][time-agg] event 缺少 entity_type：case#{case_idx} row#{j} ev#{k} raw={_short(raw)} ev={ev!r}"
                    )

            if (
                "summary" not in row
                or not isinstance(row.get("summary"), str)
                or not str(row.get("summary") or "").strip()
            ):
                raise AssertionError(
                    f"[self-test][time-agg] 缺少/空 summary：case#{case_idx} row#{j} raw={_short(raw)}"
                )

            key = (
                tp,
                label,
                str(time_obj.get("date") or ""),
                str(time_obj.get("time") or ""),
                str(time_obj.get("start") or ""),
                str(time_obj.get("end") or ""),
            )
            if key in seen:
                raise AssertionError(
                    f"[self-test][time-agg] 重复时间桶（未聚合）：case#{case_idx} raw={_short(raw)} key={key!r}"
                )
            seen.add(key)

        if "unknown" in types and (types[-1] != "unknown"):
            raise AssertionError(
                f"[self-test][time-agg] unknown 时间桶未排在最后：case#{case_idx} raw={_short(raw)} types={types!r}"
            )

    print(f"[self-test] aggregate_patterns_to_time_json 样本数={len(xs)} format=jsonl")

    ok = 0
    skipped_empty = 0
    for i, text in enumerate(xs):
        raw = (text or "").strip()
        if not raw:
            skipped_empty += 1
            continue

        patterns = explode_newlines_and_route_to_dataclasses(raw)  # type: ignore[name-defined]  # noqa: F405
        items = aggregate_patterns_by_timejson(  # type: ignore[name-defined]  # noqa: F405
            patterns, include_unknown_time=True, add_summary_text=True
        )
        _assert_items(items, case_idx=i, raw=raw)

        ok += 1
        if print_preview:
            print(
                f"  - case#{i} 解析对象数={len(patterns)} 时间桶数={len(items)}"
            )
            print(f"    原始=\"\"\"\n{raw}\n\"\"\"")
            print(f"    输出(jsonl)=\"\"\"\n[")
            for i in items:
                print(f"\t{i}")
            print("]\n\"\"\"")


    if skipped_empty:
        print(f"[self-test] aggregate_patterns_to_time_json 跳过空样本：{skipped_empty}")
    print(
        f"[self-test] aggregate_patterns_to_time_json 通过：{ok}/{len(xs) - skipped_empty}"
    )


def _self_test_time_bucket_merge_effect(*, lines) -> None:
    """
    self-test：验证“按日期/时间桶聚合”的核心效果：
    - 支持输入为“长文本字符串列表”（例如从 xlsx 的 data 列提取，每个元素是一条多行个人数据）
    - 在**单条长文本内部**寻找“至少两条事件落到同一个非 unknown 时间桶”的情况
    - 对同一条文本做整体聚合后，该时间桶只能出现 1 次（真正发生 merge）

    说明：
    - 不能依赖 time.bucket_key（它是内部字段，输出会被清理；见 `_self_test_aggregate_patterns_to_time_json` 的断言）
    - 这里用“对单条 pattern 单独聚合”得到时间桶签名，再统计重复桶，从而验证 merge 发生
    """
    from typing import Any

    def _normalize_text_item(x: Any) -> str:
        """
        将输入列表的元素归一化为“长文本字符串”。
        - str: 原样
        - list/tuple: 用换行拼接（常见于某些上游把多行拆成 list 的情况）
        - 其他: str() 兜底
        """
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)):
            return "\n".join(str(v) for v in x if v is not None)
        return str(x)

    xs = [
        _normalize_text_item(x).strip()
        for x in list(lines or [])
        if _normalize_text_item(x).strip()
    ]
    if not xs:
        print("[self-test] time_bucket merge：空样本，跳过")
        return

    def _bucket_sig(time_obj: dict) -> tuple[str, str, str, str, str, str]:
        tp = str(time_obj.get("type") or "")
        return (
            tp,
            str(time_obj.get("label") or ""),
            str(time_obj.get("date") or ""),
            str(time_obj.get("time") or ""),
            str(time_obj.get("start") or ""),
            str(time_obj.get("end") or ""),
        )

    # 在每条长文本内部寻找重复非 unknown 时间桶：
    # 对每个 pattern 单独聚合 -> 统计桶签名 -> 找到计数>=2 的桶，再对全量 patterns 聚合做最终断言。
    for case_idx, raw in enumerate(xs):
        patterns = explode_newlines_and_route_to_dataclasses(raw)  # type: ignore[name-defined]  # noqa: F405
        if not patterns or len(patterns) < 2:
            continue

        sig_to_count: dict[tuple[str, str, str, str, str, str], int] = {}
        for pat in patterns:
            one = aggregate_patterns_by_timejson(  # type: ignore[name-defined]  # noqa: F405
                [pat], include_unknown_time=True, add_summary_text=False
            )
            if not isinstance(one, list) or not one:
                continue
            time_obj = (one[0] or {}).get("time") or {}
            if not isinstance(time_obj, dict):
                continue
            sig = _bucket_sig(time_obj)
            if sig[0] == "unknown":
                continue
            sig_to_count[sig] = sig_to_count.get(sig, 0) + 1

        picked_sig = None
        for sig, c in sig_to_count.items():
            if c >= 2:
                picked_sig = sig
                break

        if picked_sig is None:
            continue

        merged = aggregate_patterns_by_timejson(  # type: ignore[name-defined]  # noqa: F405
            patterns, include_unknown_time=True, add_summary_text=True
        )
        cnt = 0
        for row in merged:
            time_obj = row.get("time") or {}
            if isinstance(time_obj, dict) and _bucket_sig(time_obj) == picked_sig:
                cnt += 1
        if cnt != 1:
            raise AssertionError(
                f"[self-test][time-agg] merge 效果失败：case#{case_idx} 同一时间桶出现次数={cnt} sig={picked_sig!r}"
            )

        print(f"[self-test] time_bucket merge：通过 case#{case_idx} sig={picked_sig!r}")
        # 打印这条个人数据的“完整时间聚合 JSON”，方便人工核对
        import json

        print(f"  原始=\"\"\"\n{raw}\n\"\"\"")
        print("  输出(json)=\"\"\"")
        print(f"  时间桶数: {len(merged)}")
        for idx, m in enumerate(merged):
            print(f"[时间桶 {idx}] \033[92m{m}\033[0m")

        return

    print("[self-test] time_bucket merge：未找到可复现的“同文本内重复非 unknown 时间桶”（跳过）")


def test_single_metric_stats_record_infer_period_bucket_for_summary_stats() -> None:
    """
    回归：单指标明细汇总记录（SingleMetricStatsRecord）
    - 明细有日期列表
    - 末尾统计项（平均/最高/最低）无日期，但应推断为 [min_date, max_date] 的 period 桶并输出为 events
    """
    raw = (
        "零星小睡时长：[5月27日33分钟,5月28日25分钟,5月29日31分钟,5月30日20分钟,5月31日37分钟,"
        "6月1日36分钟,6月2日25分钟,6月3日23分钟,6月4日32分钟,6月5日1小时14分钟,6月6日1小时16分钟,"
        "6月7日21分钟,6月8日37分钟,6月9日33分钟,6月10日38分钟,6月11日32分钟,6月12日24分钟,"
        "6月13日38分钟,6月14日21分钟,6月15日36分钟,6月16日24分钟,6月17日43分钟,6月18日24分钟,"
        "6月19日38分钟,6月20日21分钟,6月21日36分钟,6月22日24分钟,6月23日43分钟,6月24日41分钟,"
        "6月25日33分钟,6月26日32分钟] , 平均零星小睡时长34分钟正常, 最高零星小睡时长1小时16分钟偏长, 最低零星小睡时长20分钟正常"
    )

    patterns = explode_newlines_and_route_to_dataclasses(raw)  # type: ignore[name-defined]  # noqa: F405
    assert isinstance(patterns, list) and len(patterns) == 1

    items = aggregate_patterns_by_timejson(patterns, include_unknown_time=True, add_summary_text=True)  # type: ignore[name-defined]  # noqa: F405
    assert isinstance(items, list) and len(items) >= 1

    # 需要出现一个 period 桶：label 覆盖首末日期
    period_rows = [
        row
        for row in items
        if isinstance(row, dict)
        and isinstance(row.get("time"), dict)
        and str((row.get("time") or {}).get("type") or "") == "period"
    ]
    assert period_rows, "应输出 period 时间桶承载统计汇总（平均/最高/最低）"
    period = period_rows[0]
    t = period.get("time") or {}
    label = str(t.get("label") or "")
    assert "05月27日" in label and "06月26日" in label

    evs = period.get("events") or []
    assert isinstance(evs, list) and len(evs) >= 3
    names = {str(ev.get("name") or "") for ev in evs if isinstance(ev, dict)}
    assert "平均零星小睡时长" in names
    assert "最高零星小睡时长" in names
    assert "最低零星小睡时长" in names

    # 不应再出现“整条对象塞进 unknown/fallback”的尾巴
    for row in items:
        if not isinstance(row, dict):
            continue
        time_obj = row.get("time") or {}
        if isinstance(time_obj, dict) and str(time_obj.get("type") or "") == "unknown":
            fb = row.get("fallback")
            if fb is None:
                continue
            # 若仍有 fallback，里面不应包含统计列表字段（否则说明汇总仍在走旧逻辑）
            txt = str(fb)
            assert "统计指标名称列表" not in txt

