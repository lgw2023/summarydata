from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.summary_synthesis import replay_run_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay KTO acceptance metrics from an existing API event ledger"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; existing files are not overwritten",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = replay_run_metrics(args.run_dir.resolve())
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
