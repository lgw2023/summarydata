from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.summary_synthesis import (
    ExperimentConfig,
    build_teacher_diagnosis,
    historical_local_validator_penalties,
    load_source_rows,
    read_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved dual-judge results with frozen local validators"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    rows = load_source_rows(config)
    candidates = {
        str(row["candidate_id"]): row
        for row in read_jsonl(args.run_dir / "processed" / "candidates.jsonl")
    }
    bands = {"positive": 0, "negative": 0, "ambiguous": 0, "unusable_zero": 0}
    replayed_rows: list[dict[str, object]] = []
    accepted = 0
    for result in read_jsonl(args.run_dir / "processed" / "judge_results.jsonl"):
        if result.get("status") != "ok":
            continue
        candidate = candidates[str(result["candidate_id"])]
        row_index = int(candidate["row_index"])
        answer = str(candidate["response"])
        local_penalties = historical_local_validator_penalties(rows[row_index], answer)
        diagnosis = build_teacher_diagnosis(
            result["raw_ground"],
            result["raw_structure"],
            answer,
            pass_threshold=config.pass_threshold,
            critical_dimension_floor=config.critical_dimension_floor,
            local_penalties=local_penalties,
        )
        band = str(diagnosis["kto_score_band"])
        bands[band] += 1
        accepted += int(bool(diagnosis["accepted"]))
        replayed_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "row_index": row_index,
                "original_score": result["total_score_20"],
                "original_band": result["kto_score_band"],
                "local_rule_ids": [
                    penalty["rule_id"] for penalty in local_penalties
                ],
                "replayed_score": diagnosis["total_score_20"],
                "replayed_band": band,
                "replayed_accepted": diagnosis["accepted"],
            }
        )
    qwen_requests = sum(
        event.get("provider") == "qwen"
        for event in read_jsonl(args.run_dir / "intermediate" / "api_events.jsonl")
    )
    report = {
        "schema_version": "historical-local-validator-replay-v1",
        "run_dir": str(args.run_dir.resolve()),
        "scored_outputs": len(replayed_rows),
        "score_band_counts": bands,
        "accepted_outputs": accepted,
        "qwen_requests": qwen_requests,
        "accepts_per_qwen_request": accepted / qwen_requests if qwen_requests else 0.0,
        "rows": replayed_rows,
        "contains_source_text": False,
    }
    output = args.output or (
        args.run_dir / "processed" / "historical_local_validator_replay.json"
    )
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
