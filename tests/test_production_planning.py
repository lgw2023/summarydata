from __future__ import annotations

import pytest

from src.experiments.production_planning import (
    build_production_schedule,
    select_production_schedule_shard,
    validate_production_schedule,
)


def _rows(count: int = 7) -> list[dict[str, str]]:
    return [
        {"query": f"synthetic-{index}", "data": "redacted", "domain": "test"}
        for index in range(count)
    ]


def test_production_schedule_is_deterministic_unique_and_balanced() -> None:
    first = build_production_schedule(
        _rows(), pipeline_epoch="frozen-v28", candidate_count=25
    )
    second = build_production_schedule(
        _rows(), pipeline_epoch="frozen-v28", candidate_count=25
    )
    assert first == second
    assert len({record["candidate_id"] for record in first["records"]}) == 25
    assert first["root_variant_count_max"] - first["root_variant_count_min"] <= 1
    assert first["contains_source_text"] is False


def test_production_schedule_shards_do_not_collide() -> None:
    first = build_production_schedule(
        _rows(), pipeline_epoch="frozen-v28", candidate_count=10, start_ordinal=0
    )
    second = build_production_schedule(
        _rows(), pipeline_epoch="frozen-v28", candidate_count=10, start_ordinal=10
    )
    first_ids = {record["candidate_id"] for record in first["records"]}
    second_ids = {record["candidate_id"] for record in second["records"]}
    assert first_ids.isdisjoint(second_ids)
    assert [record["ordinal"] for record in second["records"]] == list(range(10, 20))


def test_duplicate_source_rows_share_one_root_representative() -> None:
    rows = _rows(3)
    rows.append(dict(rows[0]))
    schedule = build_production_schedule(
        rows, pipeline_epoch="frozen-v28", candidate_count=6
    )
    assert schedule["source_unique_root_count"] == 3


def test_schedule_validation_rejects_tampering() -> None:
    rows = _rows()
    schedule = build_production_schedule(
        rows, pipeline_epoch="frozen-v28", candidate_count=10
    )
    records = validate_production_schedule(
        schedule, rows, pipeline_epoch="frozen-v28"
    )
    assert len(records) == 10
    schedule["records"][0]["candidate_id"] = "tampered"
    with pytest.raises(ValueError, match="selection hash mismatch"):
        validate_production_schedule(
            schedule, rows, pipeline_epoch="frozen-v28"
        )


def test_schedule_shard_requires_a_complete_contiguous_range() -> None:
    records = build_production_schedule(
        _rows(), pipeline_epoch="frozen-v28", candidate_count=25
    )["records"]
    shard = select_production_schedule_shard(
        records, start_ordinal=10, candidate_count=5
    )
    assert [record["ordinal"] for record in shard] == list(range(10, 15))
    with pytest.raises(ValueError, match="outside the schedule"):
        select_production_schedule_shard(
            records, start_ordinal=23, candidate_count=5
        )
