from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.production_planning import (
    select_production_schedule_shard,
    validate_production_schedule,
)
from src.experiments.summary_synthesis import (
    ExperimentConfig,
    HardStop,
    SynthesisExperiment,
    nonaccept_stop_count_for_target,
    read_jsonl,
    replay_run_metrics,
    trace_v14_infrastructure_stop_reason,
    validate_context_compiler_v7_protocol,
    write_json,
)


STRATEGY = "trace_relevance_safe_positive_v28"


def build_production_report(
    run_dir: Path,
    *,
    schedule: Mapping[str, Any],
    shard: Sequence[Mapping[str, Any]],
    attempted_count: int,
    stop_reason: str,
) -> dict[str, Any]:
    metrics = replay_run_metrics(run_dir)
    candidates = read_jsonl(run_dir / "processed" / "candidates.jsonl")
    judges = read_jsonl(run_dir / "processed" / "judge_results.jsonl")
    accepted_ids = {
        str(result.get("candidate_id"))
        for result in judges
        if result.get("status") == "ok" and result.get("accepted") is True
    }
    accepted_hashes = {
        str(candidate.get("exact_hash"))
        for candidate in candidates
        if str(candidate.get("candidate_id")) in accepted_ids
        and candidate.get("exact_hash")
    }
    attempted_records = [dict(record) for record in shard[:attempted_count]]
    return {
        "schema_version": "summary-synthesis-production-run-v1",
        "schedule_selection_sha256": schedule["selection_sha256"],
        "pipeline_epoch": schedule["pipeline_epoch"],
        "strategy": STRATEGY,
        "planned_shard_count": len(shard),
        "attempted_schedule_count": attempted_count,
        "attempted_ordinal_start": (
            attempted_records[0]["ordinal"] if attempted_records else None
        ),
        "attempted_ordinal_end_inclusive": (
            attempted_records[-1]["ordinal"] if attempted_records else None
        ),
        "stop_reason": stop_reason,
        "request_metrics": metrics,
        "dedup_metrics": {
            "accepted_before_exact_dedup": len(accepted_ids),
            "accepted_unique_exact_hashes": len(accepted_hashes),
            "accepted_exact_duplicates": len(accepted_ids) - len(accepted_hashes),
        },
        "contains_source_text": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen source-text-safe v28 production schedule shard"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--start-ordinal", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config.resolve())
    validate_context_compiler_v7_protocol(config)
    if config.phase1_count != args.candidate_count:
        raise ValueError("config phase1_count must equal candidate-count")
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))

    qwen_key = os.environ.get("SUMMARY_SYNTH_QWEN_API_KEY", "")
    luna_key = os.environ.get("SUMMARY_SYNTH_LUNA_API_KEY", "")
    if not qwen_key or not luna_key:
        raise ValueError(
            "SUMMARY_SYNTH_QWEN_API_KEY and SUMMARY_SYNTH_LUNA_API_KEY are required"
        )

    run_dir = args.run_dir.resolve()
    experiment = SynthesisExperiment(config, run_dir, qwen_key, luna_key)
    try:
        records = validate_production_schedule(
            schedule,
            experiment.rows,
            pipeline_epoch=config.pipeline_epoch,
        )
        if schedule.get("strategy") != STRATEGY:
            raise ValueError("production schedule strategy mismatch")
        shard = select_production_schedule_shard(
            records,
            start_ordinal=args.start_ordinal,
            candidate_count=args.candidate_count,
        )
        experiment.write_manifest(args.config.resolve())
        checkpoints = set(config.phase1_checkpoints) | {len(shard)}
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(shard))
        stop_reason = "planned_candidates_completed"
        attempted_count = 0
        for position, record in enumerate(shard, start=1):
            attempted_count = position
            row_index = int(record["row_index"])
            candidate_id = str(record["candidate_id"])
            try:
                candidate = experiment.ensure_candidate(
                    row_index,
                    STRATEGY,
                    variant_id=str(record["variant_id"]),
                )
                if candidate.get("candidate_id") != candidate_id:
                    raise ValueError("runtime candidate ID differs from frozen schedule")
                result = experiment.ensure_judged(
                    row_index, STRATEGY, candidate, 0
                )
                status = "accepted" if result.get("accepted") is True else "rejected"
            except HardStop as exc:
                experiment._record_trace_terminal_failure(row_index, STRATEGY, exc)
                status = "failed:HardStop"
                stop_reason = "hard_stop"
            except Exception as exc:
                experiment._record_trace_terminal_failure(row_index, STRATEGY, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"production-v28 {position}/{len(shard)} "
                f"ordinal={record['ordinal']} candidate={candidate_id} "
                f"status={status} nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = build_production_report(
                    run_dir,
                    schedule=schedule,
                    shard=shard,
                    attempted_count=attempted_count,
                    stop_reason="checkpoint",
                )
                write_json(
                    run_dir
                    / "processed"
                    / f"production_v28_checkpoint_{position:05d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                experiment.gateway.events, candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if stop_reason == "hard_stop":
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break

        report = build_production_report(
            run_dir,
            schedule=schedule,
            shard=shard,
            attempted_count=attempted_count,
            stop_reason=stop_reason,
        )
        write_json(
            run_dir / "processed" / "production_v28_report.json",
            report,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "stop_reason": stop_reason,
                    "attempted": attempted_count,
                    "request_metrics": report["request_metrics"],
                    "dedup_metrics": report["dedup_metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
