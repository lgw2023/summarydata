from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.production_planning import build_production_schedule
from src.experiments.summary_synthesis import ExperimentConfig, load_source_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-register a source-text-free synthesis production schedule"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=25_000)
    parser.add_argument("--start-ordinal", type=int, default=0)
    parser.add_argument("--pipeline-epoch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing schedule: {output}")
    config = ExperimentConfig.from_yaml(args.config.resolve())
    rows = load_source_rows(config)
    schedule = build_production_schedule(
        rows,
        pipeline_epoch=args.pipeline_epoch,
        candidate_count=args.candidate_count,
        start_ordinal=args.start_ordinal,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in schedule.items() if key != "records"}
    summary["output"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
