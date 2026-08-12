from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping, Sequence

from src.experiments.summary_synthesis import (
    root_context_id,
    sha256_text,
    source_record_hash,
    stable_trace_candidate_id,
)


def _root_representatives(
    rows: Sequence[Mapping[str, str]],
) -> list[tuple[str, int]]:
    representatives: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        representatives.setdefault(root_context_id(row), row_index)
    return sorted(representatives.items())


def build_production_schedule(
    rows: Sequence[Mapping[str, str]],
    *,
    pipeline_epoch: str,
    candidate_count: int,
    start_ordinal: int = 0,
    strategy: str = "trace_relevance_safe_positive_v28",
) -> dict[str, Any]:
    """Create a deterministic, balanced, source-text-free variant schedule."""
    if not pipeline_epoch.strip():
        raise ValueError("pipeline_epoch must be non-empty")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if start_ordinal < 0:
        raise ValueError("start_ordinal must be non-negative")
    representatives = _root_representatives(rows)
    if not representatives:
        raise ValueError("rows must contain at least one root")

    records: list[dict[str, Any]] = []
    root_count = len(representatives)
    for ordinal in range(start_ordinal, start_ordinal + candidate_count):
        cycle, position = divmod(ordinal, root_count)
        cycle_order = sorted(
            representatives,
            key=lambda item: sha256_text(
                f"{pipeline_epoch}:{cycle}:{item[0]}"
            ),
        )
        root_id, row_index = cycle_order[position]
        variant_id = f"production-{ordinal:08d}"
        records.append(
            {
                "ordinal": ordinal,
                "row_index": row_index,
                "root_context_id": root_id,
                "source_hash": source_record_hash(rows[row_index]),
                "variant_id": variant_id,
                "candidate_id": stable_trace_candidate_id(
                    root_id, pipeline_epoch, variant_id
                ),
            }
        )

    root_frequencies = Counter(record["root_context_id"] for record in records)
    rendered_records = json.dumps(records, separators=(",", ":"), sort_keys=True)
    return {
        "schema_version": "summary-synthesis-production-schedule-v1",
        "pipeline_epoch": pipeline_epoch,
        "strategy": strategy,
        "start_ordinal": start_ordinal,
        "candidate_count": candidate_count,
        "source_unique_root_count": root_count,
        "root_variant_count_min": min(root_frequencies.values()),
        "root_variant_count_max": max(root_frequencies.values()),
        "selection_sha256": sha256_text(rendered_records),
        "records": records,
        "contains_source_text": False,
    }


def validate_production_schedule(
    schedule: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    *,
    pipeline_epoch: str,
) -> list[dict[str, Any]]:
    """Validate schedule integrity against the frozen, hash-verified source rows."""
    if schedule.get("schema_version") != "summary-synthesis-production-schedule-v1":
        raise ValueError("unsupported production schedule schema")
    if schedule.get("contains_source_text") is not False:
        raise ValueError("production schedule must explicitly exclude source text")
    if schedule.get("pipeline_epoch") != pipeline_epoch:
        raise ValueError("production schedule pipeline epoch mismatch")
    raw_records = schedule.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("production schedule records must be a non-empty list")
    records = [dict(record) for record in raw_records if isinstance(record, Mapping)]
    if len(records) != len(raw_records):
        raise ValueError("production schedule contains a non-object record")
    if int(schedule.get("candidate_count", -1)) != len(records):
        raise ValueError("production schedule candidate count mismatch")
    rendered = json.dumps(records, separators=(",", ":"), sort_keys=True)
    if schedule.get("selection_sha256") != sha256_text(rendered):
        raise ValueError("production schedule selection hash mismatch")

    expected_ordinals = list(
        range(
            int(schedule.get("start_ordinal", -1)),
            int(schedule.get("start_ordinal", -1)) + len(records),
        )
    )
    if [int(record.get("ordinal", -1)) for record in records] != expected_ordinals:
        raise ValueError("production schedule ordinals are not contiguous")
    candidate_ids: set[str] = set()
    for record in records:
        row_index = int(record.get("row_index", -1))
        if not 0 <= row_index < len(rows):
            raise ValueError("production schedule row index is out of range")
        row = rows[row_index]
        root_id = root_context_id(row)
        if record.get("root_context_id") != root_id:
            raise ValueError("production schedule root fingerprint mismatch")
        if record.get("source_hash") != source_record_hash(row):
            raise ValueError("production schedule source fingerprint mismatch")
        ordinal = int(record["ordinal"])
        variant_id = f"production-{ordinal:08d}"
        if record.get("variant_id") != variant_id:
            raise ValueError("production schedule variant ID mismatch")
        candidate_id = stable_trace_candidate_id(
            root_id, pipeline_epoch, variant_id
        )
        if record.get("candidate_id") != candidate_id:
            raise ValueError("production schedule candidate ID mismatch")
        if candidate_id in candidate_ids:
            raise ValueError("production schedule candidate IDs are not unique")
        candidate_ids.add(candidate_id)
    return records


def select_production_schedule_shard(
    records: Sequence[Mapping[str, Any]],
    *,
    start_ordinal: int,
    candidate_count: int,
) -> list[dict[str, Any]]:
    if start_ordinal < 0:
        raise ValueError("start_ordinal must be non-negative")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    end_ordinal = start_ordinal + candidate_count
    shard = [
        dict(record)
        for record in records
        if start_ordinal <= int(record["ordinal"]) < end_ordinal
    ]
    if len(shard) != candidate_count:
        raise ValueError("requested production shard is outside the schedule")
    expected = list(range(start_ordinal, end_ordinal))
    if [int(record["ordinal"]) for record in shard] != expected:
        raise ValueError("production shard ordinals are not contiguous")
    return shard
