from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.summary_synthesis import ExperimentConfig, SynthesisExperiment


def report_request_usage(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return report.get(
        "request_usage", report.get("actual_experiment_request_usage", {})
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen summary_train_v3 synthesis/judge experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "summary_train_v3_qwen37_luna56.yaml",
    )
    parser.add_argument(
        "--phase",
        choices=(
            "canary",
            "compare",
            "trace-phase1",
            "trace-prompt-dev",
            "trace-prompt-search-v2",
            "trace-prompt-gate-v3",
            "trace-evidence-renderer-v4",
            "trace-evidence-renderer-v4-1-smoke",
            "trace-evidence-renderer-v4-4",
            "trace-packed-bon-v5-smoke",
            "trace-context-compiler-v7",
            "trace-context-compiler-v8",
            "trace-silent-jury-v9",
            "trace-silent-jury-v9-validation",
            "trace-evidence-packet-v10",
            "trace-full-context-index-v11",
            "trace-full-context-index-v11-audit",
            "trace-fact-cards-v12",
            "trace-source-priority-v13",
            "trace-visual-contract-v14",
            "trace-visual-contract-v14-validation",
            "trace-proof-carrying-v15",
            "trace-evidence-compiler-v16",
            "trace-grounded-composer-v17",
            "trace-contract-jury-v18",
            "trace-numeric-shield-jury-v19",
            "trace-packed-contract-jury-v20",
            "trace-packed-contract-jury-v20-validation",
            "trace-guarded-visual-contract-v21",
            "trace-controlled-negative-v22",
            "trace-complete-plaintext-negative-v23",
            "trace-full-answer-plaintext-negative-v24",
            "trace-full-answer-plaintext-negative-v24-validation",
            "trace-malformed-mechanical-negative-v25",
            "trace-visual-budget-negative-v26",
            "trace-visual-budget-negative-v26-audit",
            "trace-markdown-preserving-negative-v27",
            "trace-relevance-safe-positive-v28",
            "trace-relevance-safe-positive-v28-audit",
        ),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    run_dir = args.run_dir or (
        PROJECT_ROOT
        / "data"
        / f"summary_train_v3_qwen37_luna56_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    qwen_key = os.environ.get("SUMMARY_SYNTH_QWEN_API_KEY", "")
    luna_key = os.environ.get("SUMMARY_SYNTH_LUNA_API_KEY", "")
    if not qwen_key or not luna_key:
        raise ValueError(
            "SUMMARY_SYNTH_QWEN_API_KEY and SUMMARY_SYNTH_LUNA_API_KEY are required"
        )
    experiment = SynthesisExperiment(config, run_dir.resolve(), qwen_key, luna_key)
    try:
        experiment.write_manifest(args.config.resolve())
        if args.phase == "canary":
            report = experiment.run_canary()
        elif args.phase == "compare":
            report = experiment.run_compare()
        elif args.phase == "trace-phase1":
            report = experiment.run_trace_phase1()
        elif args.phase == "trace-prompt-dev":
            report = experiment.run_trace_prompt_dev()
        elif args.phase == "trace-prompt-search-v2":
            report = experiment.run_trace_prompt_search_v2()
        elif args.phase == "trace-prompt-gate-v3":
            report = experiment.run_trace_prompt_gate_v3()
        elif args.phase == "trace-evidence-renderer-v4":
            report = experiment.run_trace_evidence_renderer_v4()
        elif args.phase == "trace-evidence-renderer-v4-1-smoke":
            report = experiment.run_trace_evidence_renderer_smoke_v4_1()
        elif args.phase == "trace-evidence-renderer-v4-4":
            report = experiment.run_trace_evidence_renderer_canary_v4_4()
        elif args.phase == "trace-packed-bon-v5-smoke":
            report = experiment.run_trace_packed_bon_smoke_v5()
        elif args.phase == "trace-context-compiler-v7":
            report = experiment.run_trace_context_compiler_v7()
        elif args.phase == "trace-context-compiler-v8":
            report = experiment.run_trace_context_compiler_v8()
        elif args.phase == "trace-silent-jury-v9":
            report = experiment.run_trace_silent_jury_v9()
        elif args.phase == "trace-silent-jury-v9-validation":
            report = experiment.run_trace_silent_jury_v9_validation()
        elif args.phase == "trace-evidence-packet-v10":
            report = experiment.run_trace_evidence_packet_v10()
        elif args.phase == "trace-full-context-index-v11":
            report = experiment.run_trace_full_context_index_v11()
        elif args.phase == "trace-full-context-index-v11-audit":
            report = experiment.run_trace_full_context_index_v11_audit()
        elif args.phase == "trace-fact-cards-v12":
            report = experiment.run_trace_fact_cards_v12()
        elif args.phase == "trace-source-priority-v13":
            report = experiment.run_trace_source_priority_v13()
        elif args.phase == "trace-visual-contract-v14":
            report = experiment.run_trace_visual_contract_v14()
        elif args.phase == "trace-visual-contract-v14-validation":
            report = experiment.run_trace_visual_contract_v14_validation()
        elif args.phase == "trace-proof-carrying-v15":
            report = experiment.run_trace_proof_carrying_v15()
        elif args.phase == "trace-evidence-compiler-v16":
            report = experiment.run_trace_evidence_compiler_v16()
        elif args.phase == "trace-grounded-composer-v17":
            report = experiment.run_trace_grounded_composer_v17()
        elif args.phase == "trace-contract-jury-v18":
            report = experiment.run_trace_contract_jury_v18()
        elif args.phase == "trace-numeric-shield-jury-v19":
            report = experiment.run_trace_numeric_shield_jury_v19()
        elif args.phase == "trace-packed-contract-jury-v20":
            report = experiment.run_trace_packed_contract_jury_v20()
        elif args.phase == "trace-packed-contract-jury-v20-validation":
            report = experiment.run_trace_packed_contract_jury_v20_validation()
        elif args.phase == "trace-guarded-visual-contract-v21":
            report = experiment.run_trace_guarded_visual_contract_v21()
        elif args.phase == "trace-controlled-negative-v22":
            report = experiment.run_trace_controlled_negative_v22()
        elif args.phase == "trace-complete-plaintext-negative-v23":
            report = experiment.run_trace_complete_plaintext_negative_v23()
        elif args.phase == "trace-full-answer-plaintext-negative-v24":
            report = experiment.run_trace_full_answer_plaintext_negative_v24()
        elif args.phase == "trace-full-answer-plaintext-negative-v24-validation":
            report = experiment.run_trace_full_answer_plaintext_negative_v24_validation()
        elif args.phase == "trace-malformed-mechanical-negative-v25":
            report = experiment.run_trace_malformed_mechanical_negative_v25()
        elif args.phase == "trace-visual-budget-negative-v26":
            report = experiment.run_trace_visual_budget_negative_v26()
        elif args.phase == "trace-visual-budget-negative-v26-audit":
            report = experiment.run_trace_visual_budget_negative_v26_audit()
        elif args.phase == "trace-markdown-preserving-negative-v27":
            report = experiment.run_trace_markdown_preserving_negative_v27()
        elif args.phase == "trace-relevance-safe-positive-v28":
            report = experiment.run_trace_relevance_safe_positive_v28()
        elif args.phase == "trace-relevance-safe-positive-v28-audit":
            report = experiment.run_trace_relevance_safe_positive_v28_audit()
        else:
            raise AssertionError(f"unhandled phase: {args.phase}")
        request_usage = report_request_usage(report)
        print(
            f"phase={args.phase} complete run_dir={run_dir.resolve()} "
            f"requests={request_usage}",
            flush=True,
        )
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
