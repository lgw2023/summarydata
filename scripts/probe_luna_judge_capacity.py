from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.summary_synthesis import (
    GatewayClient,
    PromptBundle,
    build_judge_inputs,
    judge_response_format,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe both frozen Luna judge dimensions with synthetic text"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        action="append",
        required=True,
        help="Luna base URL; repeat to configure an endpoint pool",
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--initial-request-delay-seconds", type=float, default=240)
    parser.add_argument("--min-request-interval-seconds", type=float, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = os.environ.get("SUMMARY_SYNTH_LUNA_API_KEY", "")
    if not key:
        raise ValueError("SUMMARY_SYNTH_LUNA_API_KEY is required")
    prompts = PromptBundle.from_snapshot(
        Path("/tmp/summarydata-viz-4341810.a57DOn/response_prompt_v2.py"),
        Path(
            "/tmp/summarydata-viz-4341810.a57DOn/src/scoring/"
            "kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py"
        ),
    )
    synthetic_row = {
        "domain": "其他",
        "query": "synthetic redacted capacity probe",
        "data": "synthetic redacted record",
        "suggest": "synthetic redacted suggestion",
        "rag": "",
        "services": "",
        "last_query": "",
        "last_answer_phone": "",
    }
    synthetic_answer = "## Synthetic redacted\n\n- Synthetic capacity probe only."
    prompt_by_dimension = build_judge_inputs(
        synthetic_row,
        synthetic_answer,
        corrected=True,
        prompts=prompts,
    )
    run_dir = args.run_dir.resolve()
    client = GatewayClient(
        run_dir=run_dir,
        qwen_url="http://127.0.0.1:1/v1",
        qwen_key="unused",
        qwen_model="unused",
        luna_url=str(args.base_url[0]).rstrip("/"),
        luna_url_pool=tuple(str(value).rstrip("/") for value in args.base_url),
        luna_key=key,
        luna_model=args.model,
        luna_min_request_interval_seconds=args.min_request_interval_seconds,
        qwen_cap=0,
        luna_cap=2,
        max_attempts=1,
        stop_after_failures=2,
        luna_initial_request_delay_seconds=args.initial_request_delay_seconds,
    )
    statuses: dict[str, dict[str, Any]] = {}
    try:
        for dimension in ("ground", "structure"):
            system, user = prompt_by_dimension[dimension]
            try:
                client.call(
                    provider="luna",
                    operation_id=f"capacity-probe:synthetic:0:{dimension}",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format=judge_response_format(dimension),
                    reasoning_effort=args.reasoning_effort,
                    expect_json_dimension=dimension,
                )
                event = client.events[-1]
                statuses[dimension] = {
                    "status": "ok",
                    "http_status": event.get("http_status"),
                    "endpoint_url": event.get("endpoint_url"),
                }
            except Exception:
                event = client.events[-1] if client.events else {}
                statuses[dimension] = {
                    "status": "failed",
                    "http_status": event.get("http_status"),
                    "endpoint_url": event.get("endpoint_url"),
                    "failure_code": (event.get("gateway_error") or {}).get("code"),
                }
    finally:
        client.close()
    summary = {
        "synthetic_only": True,
        "judge_prompt_source": "frozen historical snapshot",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "dimensions": statuses,
        "ready": all(value.get("status") == "ok" for value in statuses.values())
        and set(statuses) == {"ground", "structure"},
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
