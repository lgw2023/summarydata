from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.scale_planning import build_scale_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan source-text-free request and teacher capacity for synthesis"
    )
    parser.add_argument("--sampling-requests", type=int, default=25_000)
    parser.add_argument("--target-rate", type=float, default=0.80)
    parser.add_argument("--teacher-interval-seconds", type=float, default=240.0)
    parser.add_argument(
        "--deadline-days", type=float, nargs="+", default=(30.0, 14.0, 7.0)
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; existing files are not overwritten",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_scale_plan(
        sampling_requests=args.sampling_requests,
        target_rate=args.target_rate,
        teacher_interval_seconds=args.teacher_interval_seconds,
        deadline_days=args.deadline_days,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
