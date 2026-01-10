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


def _self_test_aggregate_patterns_to_time_jsonl(
    lines,
    *,
    max_cases: int | None = 30,
    print_preview: bool = True,
) -> None:
    """
    self-test：验证
    - explode_newlines_and_route_to_dataclasses()
    - aggregate_patterns_by_time() / aggregate_patterns_to_time_jsonl()
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

    print(f"[self-test] aggregate_patterns_to_time_jsonl 样本数={len(xs)} format=jsonl")

    ok = 0
    skipped_empty = 0
    for i, text in enumerate(xs):
        raw = (text or "").strip()
        if not raw:
            skipped_empty += 1
            continue

        patterns = explode_newlines_and_route_to_dataclasses(raw)  # type: ignore[name-defined]  # noqa: F405
        items = aggregate_patterns_by_time(  # type: ignore[name-defined]  # noqa: F405
            patterns, include_unknown_time=True, add_summary_text=True
        )
        _assert_items(items, case_idx=i, raw=raw)

        out_jsonl = aggregate_patterns_to_time_jsonl(  # type: ignore[name-defined]  # noqa: F405
            patterns,
            include_unknown_time=True,
            add_summary_text=True,
            ensure_ascii=False,
        )
        if not isinstance(out_jsonl, str) or not out_jsonl.strip():
            raise AssertionError(
                f"[self-test][time-agg] JSONL 输出为空/非字符串：case#{i} raw={_short(raw)}"
            )

        parsed_items = [
            json.loads(line) for line in out_jsonl.splitlines() if line.strip()
        ]
        if parsed_items != items:
            raise AssertionError(
                f"[self-test][time-agg] JSONL 与 items 不一致：case#{i} raw={_short(raw)}"
            )

        ok += 1
        if print_preview:
            print(
                f"  - case#{i} 解析对象数={len(patterns)} 时间桶数={len(items)}"
            )
            print(f"    原始=\"\"\"\n{raw}\n\"\"\"")
            print(f"    输出(jsonl)=\"\"\"\n{out_jsonl}\n\"\"\"")
            # print(patterns)

    if skipped_empty:
        print(f"[self-test] aggregate_patterns_to_time_jsonl 跳过空样本：{skipped_empty}")
    print(
        f"[self-test] aggregate_patterns_to_time_jsonl 通过：{ok}/{len(xs) - skipped_empty}"
    )


def _self_test_time_bucket_merge_effect(*, lines) -> None:
    """
    self-test：验证“按日期/时间桶聚合”的核心效果：
    - 找到至少两个样例会落到同一个非 unknown 时间桶
    - 将它们合并后，聚合输出中该桶只能出现 1 次（真正发生 merge）
    """
    xs = [str(x or "").strip() for x in list(lines or []) if str(x or "").strip()]
    if len(xs) < 2:
        print("[self-test] time_bucket merge：样例不足，跳过")
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

    bucket_to_raws: dict[tuple[str, str, str, str, str, str], list[str]] = {}
    for raw in xs:
        patterns = explode_newlines_and_route_to_dataclasses(raw)  # type: ignore[name-defined]  # noqa: F405
        items = aggregate_patterns_by_time(  # type: ignore[name-defined]  # noqa: F405
            patterns, include_unknown_time=True, add_summary_text=True
        )
        for row in items:
            time_obj = row.get("time") or {}
            if not isinstance(time_obj, dict):
                continue
            sig = _bucket_sig(time_obj)
            if sig[0] == "unknown":
                continue
            bucket_to_raws.setdefault(sig, [])
            if raw not in bucket_to_raws[sig]:
                bucket_to_raws[sig].append(raw)

    picked_sig = None
    for sig, raws in bucket_to_raws.items():
        if len(raws) >= 2:
            picked_sig = sig
            break

    if picked_sig is None:
        print("[self-test] time_bucket merge：未找到可复现的重复非 unknown 时间桶（跳过）")
        return

    raw1, raw2 = bucket_to_raws[picked_sig][0], bucket_to_raws[picked_sig][1]
    pats1 = explode_newlines_and_route_to_dataclasses(raw1)  # type: ignore[name-defined]  # noqa: F405
    pats2 = explode_newlines_and_route_to_dataclasses(raw2)  # type: ignore[name-defined]  # noqa: F405
    merged = aggregate_patterns_by_time(  # type: ignore[name-defined]  # noqa: F405
        [*pats1, *pats2], include_unknown_time=True, add_summary_text=True
    )

    cnt = 0
    for row in merged:
        time_obj = row.get("time") or {}
        if isinstance(time_obj, dict) and _bucket_sig(time_obj) == picked_sig:
            cnt += 1
    if cnt != 1:
        raise AssertionError(
            f"[self-test][time-agg] merge 效果失败：同一时间桶出现次数={cnt} sig={picked_sig!r}"
        )

    print(f"[self-test] time_bucket merge：通过 sig={picked_sig!r}")

