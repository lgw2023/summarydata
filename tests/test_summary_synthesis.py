from dataclasses import replace
import hashlib
import json
from pathlib import Path
from string import Template

import pandas as pd
import pytest
import httpx

from prompts.system_prompt_v5_yixuan import build_phone_personal_prompt
from scripts.run_summary_synthesis_experiment import parse_args, report_request_usage
from scripts.probe_luna_judge_capacity import parse_args as parse_probe_args
from src.experiments.summary_synthesis import (
    PromptBundle,
    ExperimentConfig,
    GatewayClient,
    SynthesisExperiment,
    build_generation_messages,
    build_claim_gated_dual_messages,
    build_context_compiler_v7_messages,
    build_context_compiler_v8_messages,
    build_silent_jury_v9_messages,
    build_evidence_packet_v10_messages,
    build_full_context_index_v11_messages,
    build_fact_cards_v12_messages,
    build_source_priority_v13_messages,
    build_visual_contract_v14_messages,
    build_guarded_visual_contract_v21_messages,
    build_relevance_safe_direct_v24_messages,
    build_proof_carrying_v15_messages,
    build_evidence_compiler_v16_messages,
    build_grounded_composer_v17_messages,
    build_contract_jury_v18_messages,
    build_numeric_shield_jury_v19_messages,
    build_packed_contract_jury_v20_messages,
    build_evidence_plan_messages,
    build_packed_bon_messages,
    build_slim_evidence_system,
    build_slim_generation_messages,
    build_root_splits,
    build_teacher_diagnosis,
    build_teacher_patch_messages,
    build_context,
    apply_label_free_output_guard,
    apply_guarded_visual_contract_v21,
    render_controlled_negative_v22,
    render_complete_plaintext_negative_v23,
    render_full_answer_plaintext_negative_v24,
    render_malformed_mechanical_negative_v25,
    render_visual_budget_negative_v26,
    render_markdown_preserving_negative_v27,
    historical_local_validator_penalties,
    apply_proof_citation_firewall_v15,
    render_evidence_compiler_v16,
    render_grounded_composer_v17,
    compile_context_profile,
    compile_evidence_packet,
    compile_personal_fact_cards_v12,
    compile_source_priority_blueprint_v13,
    compile_visual_contract_v14,
    compile_proof_ledger_v15,
    checks_to_penalties,
    exact_mcnemar_p_value,
    extract_packed_candidates,
    extract_packed_candidates_v20,
    extract_json_object,
    load_source_rows,
    minimum_accepts_for_target,
    nonaccept_stop_count_for_target,
    classify_kto_score,
    is_kto_accepted_score,
    judge_response_format,
    one_sided_exact_lower_bound,
    parse_persisted_judge_event,
    replay_run_metrics,
    render_evidence_plan,
    render_evidence_plan_conservative,
    render_evidence_plan_balanced,
    render_evidence_plan_natural,
    score_packed_candidate_risk,
    score_packed_contract_candidate_v20,
    root_context_id,
    score_judges,
    select_compare_indices,
    select_trace_phase1_indices,
    stable_candidate_id,
    trace_v14_infrastructure_stop_reason,
    validate_context_compiler_v7_protocol,
    validate_judge_output,
    wilson_interval,
)


def _prompts() -> PromptBundle:
    return PromptBundle("legacy", "ground", Template("$answer"), "structure", Template("$answer"))


def test_context_preserves_legacy_services_bug_and_corrects_it() -> None:
    row = {"query": "redacted", "services": "course-redacted"}
    assert "course-redacted" not in build_context(row, corrected=False)
    assert "course-redacted" in build_context(row, corrected=True)


def test_hash_pinned_xlsx_source_loads_without_fake_git_ref(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.xlsx"
    pd.DataFrame(
        [{"query": "合成问题", "data": "合成数据", "services": "合成课程"}]
    ).to_excel(source, index=False)
    base = ExperimentConfig.from_yaml(
        Path("configs/summary_train_v3_trace_source_priority_v13_6.yaml")
    )
    config = replace(
        base,
        source_path=source,
        source_git_path="",
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    rows = load_source_rows(config)
    assert rows == [
        {"query": "合成问题", "data": "合成数据", "services": "合成课程"}
    ]


def test_config_parses_separate_validation_exclusions() -> None:
    config = ExperimentConfig.from_yaml(
        Path("configs/summary_train_v36_trace_packed_contract_jury_v20_5.yaml")
    )
    assert config.validation_excluded_indices == ()


def test_frozen_confirmation_config_excludes_consumed_audit_roots() -> None:
    config = ExperimentConfig.from_yaml(
        Path(
            "configs/summary_train_v3_trace_relevance_safe_positive_v28_"
            "qwen38max_confirm50.yaml"
        )
    )
    assert config.phase1_count == 50
    assert config.qwen_request_cap == 50
    assert config.luna_request_cap == 100
    assert config.audit_excluded_indices == (176, 254, 193, 44, 184)


def test_selection_is_deterministic_and_keeps_required_rows() -> None:
    rows = [
        {"domain": str(index % 4), "一级分类": str(index % 3), "query": "redacted"}
        for index in range(30)
    ]
    first = select_compare_indices(rows, 12, [0, 1, 2])
    second = select_compare_indices(rows, 12, [0, 1, 2])
    assert first == second
    assert first[:3] == [0, 1, 2]
    assert len(first) == len(set(first)) == 12


def test_candidate_id_is_stable_and_strategy_specific() -> None:
    assert stable_candidate_id("hash", 3, "legacy") == stable_candidate_id(
        "hash", 3, "legacy"
    )
    assert stable_candidate_id("hash", 3, "legacy") != stable_candidate_id(
        "hash", 3, "corrected_v5"
    )


def test_candidate_cache_distinguishes_production_variants() -> None:
    experiment = SynthesisExperiment.__new__(SynthesisExperiment)
    experiment.candidates = [
        {
            "row_index": 7,
            "strategy": "trace_relevance_safe_positive_v28",
            "variant_id": "production-00000000",
            "status": "ok",
        },
        {
            "row_index": 7,
            "strategy": "trace_relevance_safe_positive_v28",
            "variant_id": "production-00000001",
            "status": "ok",
        },
    ]
    first = experiment._find_candidate(
        7,
        "trace_relevance_safe_positive_v28",
        "production-00000000",
    )
    second = experiment._find_candidate(
        7,
        "trace_relevance_safe_positive_v28",
        "production-00000001",
    )
    assert first is experiment.candidates[0]
    assert second is experiment.candidates[1]


def test_json_extraction_handles_fenced_extra_text() -> None:
    assert extract_json_object('prefix ```json\n{"checks": [], "confidence": 1}\n``` suffix')[
        "confidence"
    ] == 1


def test_historical_penalty_mapping_and_strict_pass_boundary() -> None:
    ground = {
        "checks": [
            {
                "rule_id": "FACT_LOGIC_ISSUE",
                "hit": True,
                "severity": "minor",
                "reason": "redacted",
                "excerpt": "",
            }
        ],
        "confidence": 0.8,
    }
    structure = {"checks": [], "confidence": 0.8}
    result = score_judges(ground, structure, "## redacted\n- item")
    assert result["total_score_20"] == 15.0
    assert result["total_score_20"] > 14
    strict = checks_to_penalties(
        [
            {
                "rule_id": "PERSONAL_DATA_MISMATCH",
                "hit": True,
                "severity": "strict",
            }
        ]
    )
    assert strict[0]["score"] == 20.0


def test_unknown_judge_checks_are_ignored_like_historical_scorer() -> None:
    value = validate_judge_output(
        {
            "checks": [
                {"rule_id": "CONFIDENCE", "hit": False},
                {"rule_id": "FACT_LOGIC_ISSUE", "hit": False},
            ],
            "confidence": 0.9,
        },
        "ground",
    )
    assert [check["rule_id"] for check in value["checks"]] == ["FACT_LOGIC_ISSUE"]


def test_wilson_interval_keeps_small_pilot_claim_conservative() -> None:
    lower, upper = wilson_interval(11, 12)
    assert 0.64 < lower < 0.65
    assert 0.98 < upper < 0.99


def test_exact_lower_bound_requires_at_least_45_of_50_for_eighty_percent() -> None:
    assert one_sided_exact_lower_bound(44, 50) < 0.80
    assert one_sided_exact_lower_bound(45, 50) > 0.80


def test_root_split_keeps_duplicates_and_protected_pilot_in_development() -> None:
    rows = [
        {
            "query": f"synthetic-{index}",
            "data": "redacted",
            "domain": str(index % 4),
        }
        for index in range(18)
    ]
    rows[17] = dict(rows[0])
    splits = build_root_splits(
        rows,
        "source-hash",
        protected_development_indices=[0],
        validation_root_count=4,
        audit_root_count=4,
    )
    assert 0 in splits["development"]
    assert 17 in splits["development"]
    roots_by_split = {
        name: {root_context_id(rows[index]) for index in indices}
        for name, indices in splits.items()
    }
    assert roots_by_split["development"].isdisjoint(roots_by_split["validation"])
    assert roots_by_split["development"].isdisjoint(roots_by_split["audit"])
    assert roots_by_split["validation"].isdisjoint(roots_by_split["audit"])


def test_trace_selection_excludes_pilot_roots_and_is_deterministic() -> None:
    rows = [
        {
            "query": f"synthetic-{index}",
            "data": "redacted",
            "domain": str(index % 3),
            "一级分类": str(index % 5),
        }
        for index in range(30)
    ]
    first = select_trace_phase1_indices(
        rows,
        "source-hash",
        development_indices=list(range(30)),
        excluded_indices=[0, 1, 2],
        count=10,
    )
    second = select_trace_phase1_indices(
        rows,
        "source-hash",
        development_indices=list(range(30)),
        excluded_indices=[0, 1, 2],
        count=10,
    )
    assert first == second
    assert not set(first).intersection({0, 1, 2})
    assert len({root_context_id(rows[index]) for index in first}) == 10


def test_teacher_diagnosis_drives_minimal_patch_without_changing_pass_rule() -> None:
    ground = {
        "checks": [
            {
                "rule_id": "NUM_COMPARE_ERROR",
                "hit": True,
                "severity": "strict",
                "reason": "synthetic numeric issue",
                "excerpt": "redacted",
            }
        ],
        "confidence": 0.9,
    }
    structure = {"checks": [], "confidence": 0.8}
    diagnosis = build_teacher_diagnosis(
        ground,
        structure,
        "## redacted\n- item",
        pass_threshold=14,
        critical_dimension_floor=15,
    )
    assert diagnosis["accepted"] is False
    assert diagnosis["fatal_flags"] == ["NUM_COMPARE_ERROR"]
    assert diagnosis["repair_targets"] == ["numeric_units"]
    messages = build_teacher_patch_messages(
        {"query": "redacted", "data": "redacted"},
        "## redacted\n- item",
        diagnosis,
    )
    assert "numeric_units" in messages[1]["content"]
    assert "最小定向修补" in messages[1]["content"]


def test_teacher_diagnosis_accepts_historical_score_fifteen() -> None:
    ground = {
        "checks": [
            {
                "rule_id": "FACT_LOGIC_ISSUE",
                "hit": True,
                "severity": "minor",
                "reason": "synthetic",
                "excerpt": "",
            }
        ],
        "confidence": 0.8,
    }
    diagnosis = build_teacher_diagnosis(
        ground,
        {"checks": [], "confidence": 0.8},
        "## redacted\n- item",
        pass_threshold=14,
        critical_dimension_floor=15,
    )
    assert diagnosis["total_score_20"] == 15
    assert diagnosis["accepted"] is True


@pytest.mark.parametrize(
    ("score", "expected_band", "expected_accepted"),
    [
        (0, "unusable_zero", False),
        (1, "negative", True),
        (8, "negative", True),
        (9, "ambiguous", False),
        (13, "ambiguous", False),
        (14, "positive", True),
        (20, "positive", True),
    ],
)
def test_kto_acceptance_uses_the_user_defined_score_bands(
    score: float, expected_band: str, expected_accepted: bool
) -> None:
    assert classify_kto_score(score) == expected_band
    assert is_kto_accepted_score(score) is expected_accepted


def test_v21_guard_removes_unsupported_numbers_and_repairs_markdown() -> None:
    row = {
        "query": "请分析记录",
        "data": "本次记录为1000步。",
        "suggest": "保持规律活动。",
        "domain": "其他",
    }
    response = """本次比500步增加了500步。

## 建议

- 必须停药并立即治疗。"""
    guarded, stats = apply_guarded_visual_contract_v21(row, response)
    assert "500" not in guarded
    assert "停药" not in guarded
    assert "本次记录为1000步。" in guarded
    assert "## 记录依据" in guarded
    assert "## 建议" in guarded
    assert stats["v21_fallback_used"] == 1


def test_v21_prompt_freezes_one_shot_source_and_safety_policy() -> None:
    messages = build_guarded_visual_contract_v21_messages(
        {"query": "redacted", "data": "redacted", "domain": "其他"}
    )
    assert len(messages) == 2
    assert "v21 保守完成策略" in messages[0]["content"]
    assert "专家建议原文" in messages[0]["content"]
    assert "最终门禁" in messages[1]["content"]


def test_v22_controlled_negative_is_grounded_complete_and_has_no_markdown() -> None:
    row = {
        "query": "请分析记录",
        "data": "本次记录为1000步。",
        "suggest": "保持规律活动。",
        "domain": "其他",
    }
    response = """可以先核对现有记录。

## 记录依据

- 本次记录为1000步。

## 建议

- 保持规律活动。"""
    rendered, stats = render_controlled_negative_v22(row, response)
    assert "1000" in rendered
    assert "保持规律活动" in rendered
    assert "#" not in rendered
    assert "\n" not in rendered
    assert rendered.count("先把现有记录和建议分开来看") == 2
    assert stats["v22_output_lines"] == 1


def test_v23_negative_keeps_direct_complete_sections_without_markdown() -> None:
    row = {
        "query": "请分析记录",
        "data": "本次记录为1000步。",
        "suggest": "保持规律活动。",
        "domain": "其他",
    }
    response = """可以先核对现有记录。

## 记录依据

- 本次记录为1000步。

## 建议

- 保持规律活动。"""
    rendered, stats = render_complete_plaintext_negative_v23(row, response)
    assert rendered.startswith("可以先核对现有记录。")
    assert "记录依据：本次记录为1000步。" in rendered
    assert "建议：保持规律活动。" in rendered
    assert "建议建议" in rendered
    assert "#" not in rendered
    assert "- " not in rendered
    assert rendered.count("\n\n") >= 3
    assert stats["v23_direct_opening_preserved"] == 1


def test_v24_negative_preserves_full_answer_and_only_degrades_presentation() -> None:
    response = """直接回答当前问题。

## 原因

- 第一条完整解释。
- 第二条完整解释。

## 下一步

1. 采取合规做法。
2. 必要时咨询专业人士。"""
    rendered, stats = render_full_answer_plaintext_negative_v24(response)
    assert rendered.startswith("直接回答当前问题。")
    assert "原因：第一条完整解释。" in rendered
    assert "第二条完整解释。" in rendered
    assert "下一步：采取合规做法。" in rendered
    assert "必要时咨询专业人士。" in rendered
    assert "内容内容" in rendered
    assert "#" not in rendered
    assert "- " not in rendered
    assert stats["v24_full_answer_preserved"] == 1


def test_v24_prompt_prioritizes_query_relevance_and_safe_alternatives() -> None:
    messages = build_relevance_safe_direct_v24_messages(
        {"query": "当前问题", "data": "无关材料", "domain": "其他"}
    )
    combined = "\n".join(str(message["content"]) for message in messages)
    assert "当前用户问题" in combined
    assert "不要强行套用材料" in combined
    assert "合法、合规" in combined


def test_v25_preserves_v24_content_and_adds_bounded_visible_defects() -> None:
    rendered, stats = render_malformed_mechanical_negative_v25(
        "## 回答\n\n- 完整内容。\n- 后续建议。"
    )
    assert "回答：完整内容。" in rendered
    assert "后续建议。" in rendered
    assert rendered.endswith(
        "** 排版状态：答复模块模块已经处理完成。答复模块模块已经处理完成。"
    )
    assert rendered.count("答复模块模块已经处理完成") == 2
    assert stats["v25_unclosed_markdown_added"] == 1
    assert stats["v25_mechanical_module_tone_added"] == 1
    assert stats["v25_full_answer_preserved"] == 1


def test_v26_adds_visual_anchor_without_changing_the_defect_budget_text() -> None:
    rendered, stats = render_visual_budget_negative_v26(
        "## 回答\n\n- 完整内容。"
    )
    assert "回答：完整内容。" in rendered
    assert rendered.endswith(
        "📌 ** 排版状态：答复模块模块已经处理完成。答复模块模块已经处理完成。"
    )
    assert stats["v26_visual_anchor_added"] == 1
    assert stats["v26_unclosed_markdown_added"] == 1
    assert stats["v26_mechanical_module_tone_added"] == 1
    assert stats["v26_target_structure_deduction"] == 12


def test_v27_preserves_original_markdown_byte_for_byte_before_defect() -> None:
    response = "## 回答\n\n- 完整内容。\n- 后续建议。"
    rendered, stats = render_markdown_preserving_negative_v27(response)
    assert rendered.startswith(response + "\n\n")
    assert "📌 ** 排版状态" in rendered
    assert stats["v27_original_response_preserved"] == 1
    assert stats["v27_original_characters_retained"] == len(response)
    assert stats["v27_target_structure_deduction"] == 12


def test_historical_local_validators_reproduce_chart_and_service_penalties() -> None:
    penalties = historical_local_validator_penalties(
        {
            "data": json.dumps(
                {
                    "require_chart_or_table": True,
                    "services": ["合规咨询"],
                },
                ensure_ascii=False,
            )
        },
        "只有普通段落。",
    )
    assert {item["rule_id"]: item["score"] for item in penalties} == {
        "MISSING_CHART_TABLE": 10.0,
        "MISSING_SERVICE": 10.0,
    }


def test_historical_local_sleep_validator_detects_missing_duration_and_grade() -> None:
    penalties = historical_local_validator_penalties(
        {
            "data": json.dumps(
                {
                    "sleep": {
                        "duration_hours": 7.5,
                        "score": 70,
                        "score_thresholds": {"good": [60, 80]},
                    }
                }
            )
        },
        "建议保持规律作息。",
    )
    assert [item["rule_id"] for item in penalties] == [
        "PERSONAL_DATA_ANALYSIS_ISSUE"
    ]
    assert penalties[0]["score"] == 5.0


def test_persisted_successful_judge_event_is_reused_without_an_api_retry() -> None:
    event = {
        "status": "ok",
        "raw_response": {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"checks": [], "confidence": 0.9}
                        )
                    }
                }
            ]
        },
    }
    assert parse_persisted_judge_event(event, "ground") == {
        "checks": [],
        "confidence": 0.9,
    }
    assert parse_persisted_judge_event({**event, "status": "failed"}, "ground") is None


def test_judge_response_format_uses_strict_dimension_specific_schema() -> None:
    ground = judge_response_format("ground")
    structure = judge_response_format("structure")
    assert ground["type"] == "json_schema"
    assert ground["json_schema"]["strict"] is True
    ground_rules = set(
        ground["json_schema"]["schema"]["properties"]["checks"]["items"][
            "properties"
        ]["rule_id"]["enum"]
    )
    structure_rules = set(
        structure["json_schema"]["schema"]["properties"]["checks"]["items"][
            "properties"
        ]["rule_id"]["enum"]
    )
    assert "NUM_COMPARE_ERROR" in ground_rules
    assert "BAD_MARKDOWN_USAGE" in structure_rules
    assert ground_rules.isdisjoint(structure_rules)


def test_capacity_probe_cli_requires_explicit_run_dir_and_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe_luna_judge_capacity.py",
            "--run-dir",
            "/tmp/synthetic-probe",
            "--base-url",
            "http://127.0.0.1:18004/v1",
        ],
    )
    args = parse_probe_args()
    assert args.run_dir == Path("/tmp/synthetic-probe")
    assert args.base_url == ["http://127.0.0.1:18004/v1"]


def test_replay_uses_all_sampling_events_as_denominator(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "intermediate").mkdir(parents=True)
    (run_dir / "processed").mkdir()
    events = [
        {"provider": "qwen", "operation_id": "generate:a:x", "status": "ok"},
        {"provider": "qwen", "operation_id": "generate:b:x", "status": "ok"},
        {"provider": "qwen", "operation_id": "generate:c:x", "status": "failed"},
        {"provider": "luna", "operation_id": "judge:a:0:ground", "status": "ok"},
        {"provider": "luna", "operation_id": "judge:a:0:structure", "status": "ok"},
        {"provider": "luna", "operation_id": "judge:b:0:ground", "status": "ok"},
        {"provider": "luna", "operation_id": "judge:b:0:structure", "status": "ok"},
    ]
    candidates = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    judges = [
        {"candidate_id": "a", "status": "ok", "total_score_20": 14},
        {"candidate_id": "b", "status": "ok", "total_score_20": 8},
    ]
    for path, rows in (
        (run_dir / "intermediate" / "api_events.jsonl", events),
        (run_dir / "processed" / "candidates.jsonl", candidates),
        (run_dir / "processed" / "judge_results.jsonl", judges),
    ):
        path.write_text(
            "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
        )

    report = replay_run_metrics(run_dir)
    assert report["sampling_requests_denominator"] == 3
    assert report["accepted_outputs"] == 2
    assert report["score_band_counts"] == {
        "positive": 1,
        "negative": 1,
        "ambiguous": 0,
        "unusable_zero": 0,
    }
    assert report["accepts_per_sampling_request"] == pytest.approx(2 / 3)
    assert report["strict_protocol_satisfied"] is True


def test_evidence_contract_is_one_shot_and_contains_failure_prevention() -> None:
    messages = build_generation_messages(
        {"query": "redacted", "data": "redacted", "domain": "其他"},
        "evidence_contract_v1",
        _prompts(),
    )
    assert len(messages) == 2
    user = messages[1]["content"]
    assert "证据与输出契约" in user
    assert "不得互相替代" in user
    assert "Markdown" in user
    assert "最终只输出" in user


def test_request_metrics_include_failed_retry_attempts() -> None:
    experiment = SynthesisExperiment.__new__(SynthesisExperiment)
    experiment.gateway = type(
        "FakeGateway",
        (),
        {
            "events": [
                {
                    "provider": "qwen",
                    "operation_id": "generate:candidate-a:draft",
                    "status": "failed",
                },
                {
                    "provider": "qwen",
                    "operation_id": "generate:candidate-a:draft",
                    "status": "ok",
                    "usage": {"total_tokens": 10},
                },
            ]
        },
    )()
    metrics = experiment._request_metrics_for_candidates(
        "qwen", {"candidate-a"}
    )
    assert metrics["requests_including_retries"] == 2
    assert metrics["successful_requests"] == 1
    assert metrics["usage"]["total_tokens"] == 10


def test_slim_prompt_is_shorter_and_dual_draft_remains_one_call() -> None:
    row = {"query": "redacted", "data": "redacted", "domain": "睡眠"}
    slim = build_slim_generation_messages(row, internal_dual_draft=False)
    dual = build_slim_generation_messages(row, internal_dual_draft=True)
    historical = build_phone_personal_prompt(domain="睡眠")
    assert len(slim) == len(dual) == 2
    assert len(slim[0]["content"]) < len(historical) / 2
    assert "证据优先级" in slim[0]["content"]
    assert "静默择优" not in slim[1]["content"]
    assert "静默择优" in dual[1]["content"]
    assert "两个候选草案" in dual[1]["content"]
    assert "睡眠包括夜间睡眠" in build_slim_evidence_system("睡眠")


def test_claim_gated_dual_prompt_is_one_call_and_requires_source_gate() -> None:
    messages = build_claim_gated_dual_messages(
        {"query": "redacted", "data": "redacted", "domain": "其他"}
    )
    assert len(messages) == 2
    assert "来源账本" in messages[1]["content"]
    assert "任何主张若无明确来源" in messages[1]["content"]
    assert "双草案" in messages[1]["content"]
    assert "最终只输出" in messages[1]["content"]


def test_context_compiler_v7_routes_from_current_input_without_labels() -> None:
    row = {
        "query": "请比较最近两天的合成步数变化",
        "data": "第1天8000步，第2天9000步。",
        "suggest": "建议保持记录。",
        "rag": "活动量应循序渐进。",
        "services": "",
        "domain": "其他",
    }
    profile = compile_context_profile(row)
    assert profile["task_type"] == "trend_or_comparison"
    assert profile["comparison_requested"] is True
    assert profile["personal_number_count"] >= 2
    assert "query" not in profile
    assert "judge" not in profile
    messages = build_context_compiler_v7_messages(row)
    assert len(messages) == 2
    assert "v7 单输出执行协议" in messages[0]["content"]
    assert "只比较同一指标" in messages[1]["content"]
    assert "最终只输出一个" in messages[1]["content"]


def test_context_compiler_v7_protocol_rejects_retries_and_wrong_caps() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "summary_train_v3_trace_context_compiler_v7_10.yaml"
    )
    config = ExperimentConfig.from_yaml(config_path)
    validate_context_compiler_v7_protocol(config)
    with pytest.raises(ValueError, match="forbids retries"):
        validate_context_compiler_v7_protocol(
            replace(config, max_attempts_per_operation=2)
        )
    with pytest.raises(ValueError, match="luna cap"):
        validate_context_compiler_v7_protocol(
            replace(config, luna_request_cap=19)
        )


def test_luna_pool_and_rate_policy_parse_without_changing_call_caps() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "summary_train_v3_trace_full_context_index_v11_audit_pool4_10.yaml"
    )
    config = ExperimentConfig.from_yaml(config_path)
    validate_context_compiler_v7_protocol(config)
    assert config.luna_min_request_interval_seconds == 90
    assert config.luna_initial_request_delay_seconds == 0
    assert config.luna_url_pool == (
        "http://127.0.0.1:18002/v1",
        "http://127.0.0.1:18003/v1",
        "http://127.0.0.1:18004/v1",
        "http://127.0.0.1:18005/v1",
    )
    assert config.qwen_request_cap == 10
    assert config.luna_request_cap == 20


def test_context_compiler_v8_guard_removes_only_unsupported_claim_lines() -> None:
    row = {
        "query": "请给出合成建议",
        "data": "合成步数为8000步。",
        "suggest": "建议保持记录。",
        "rag": "",
        "services": "轻松步行",
        "domain": "其他",
    }
    response = """目前记录为8000步。

## 关键依据
- 合成步数为8000步。
- 推算后是9000步。

## 建议
1. 保持记录。
2. 可尝试课程 <轻松步行>。
3. 可尝试课程 <不存在课程>。"""
    guarded, stats = apply_label_free_output_guard(row, response)
    assert "8000步" in guarded
    assert "9000步" not in guarded
    assert "<轻松步行>" in guarded
    assert "<不存在课程>" not in guarded
    assert "1. 保持记录" in guarded
    assert stats["unsupported_numeric_lines_removed"] == 1
    assert stats["unsupported_course_lines_removed"] == 1
    messages = build_context_compiler_v8_messages(row)
    assert "禁止自行计算差值" in messages[0]["content"]
    assert "v8 最终静默门禁" in messages[1]["content"]


def test_silent_jury_v9_keeps_one_output_and_three_veto_roles() -> None:
    messages = build_silent_jury_v9_messages(
        {"query": "合成问题", "data": "合成记录", "domain": "其他"}
    )
    assert len(messages) == 2
    system = messages[0]["content"]
    assert "事实陪审员" in system
    assert "完整性陪审员" in system
    assert "表达陪审员" in system
    assert "三个候选" in system
    assert "最终只能输出一个" in system


def test_evidence_packet_v10_is_deterministic_source_isolated_and_query_ranked() -> None:
    unrelated = [f"合成无关指标{index}为{index}。" for index in range(9)]
    row = {
        "query": "我的合成心率怎么样？",
        "data": "".join(unrelated + ["合成心率为70次/分。"]),
        "suggest": "建议继续记录合成心率。",
        "rag": "心率需要结合场景解释。",
        "services": "合成舒缓课程；合成跑步课程。",
        "domain": "其他",
    }
    first = compile_evidence_packet(row)
    second = compile_evidence_packet(row)
    assert first == second
    personal = first["sources"]["personal"]
    assert len(personal) == 8
    assert any(item["text"] == "合成心率为70次/分。" for item in personal)
    assert all(item["id"].startswith("P") for item in personal)
    assert all(item["id"].startswith("E") for item in first["sources"]["expert"])
    assert "judge" not in first
    assert first["source_stats"]["personal"] == {
        "available_sentences": 10,
        "selected_sentences": 8,
        "selection_limit": 8,
    }
    messages = build_evidence_packet_v10_messages(row)
    assert len(messages) == 2
    assert "Evidence Packet v10" in messages[0]["content"]
    assert "合成心率为70次/分" in messages[1]["content"]
    assert "最终 Markdown" in messages[1]["content"]


def test_full_context_index_v11_keeps_raw_context_and_adds_exact_anchors() -> None:
    row = {
        "query": "合成步数趋势如何？",
        "data": "第一天合成步数8000步。第二天合成步数9000步。",
        "suggest": "建议继续记录。",
        "rag": "变化需要结合时间解释。",
        "services": "合成步行课程。",
        "domain": "其他",
    }
    messages = build_full_context_index_v11_messages(row)
    assert len(messages) == 2
    user = messages[1]["content"]
    assert "【个人数据】" in user
    assert "第一天合成步数8000步" in user
    assert "本地相关证据索引" in user
    assert "[P1]" in user
    assert "完整原文优先" in user
    assert "只输出一个" in user


def test_fact_cards_v12_preserve_exact_personal_sentence_and_bind_numbers() -> None:
    row = {
        "query": "合成步数怎么样？",
        "data": "第一天合成步数8000步。第二天合成步数9000步。",
        "suggest": "建议继续记录。",
        "rag": "变化需要结合时间解释。",
        "services": "合成步行课程。",
        "domain": "其他",
    }
    first = compile_personal_fact_cards_v12(row)
    second = compile_personal_fact_cards_v12(row)
    assert first == second
    assert first["schema_version"] == "personal-fact-cards-v12"
    assert [card["exact_sentence"] for card in first["cards"]] == [
        "第一天合成步数8000步。",
        "第二天合成步数9000步。",
    ]
    assert first["cards"][0]["numeric_bindings"][0]["token"] == "8000"
    assert first["cards"][0]["policy"] == "copy_exact_sentence_or_omit"
    messages = build_fact_cards_v12_messages(row)
    assert len(messages) == 2
    assert "【个人数据】" in messages[1]["content"]
    assert "第一天合成步数8000步。" in messages[1]["content"]
    assert "复制或省略" in messages[0]["content"]
    assert "不得把两张卡片" in messages[0]["content"]


def test_source_priority_v13_requires_table_only_for_comparison_cards() -> None:
    row = {
        "query": "合成步数趋势如何？",
        "data": "第一天合成步数8000步。第二天合成步数9000步。",
        "suggest": "建议继续记录。",
        "rag": "记录变化可结合时间解释。",
        "services": "合成步行课程。",
        "domain": "其他",
    }
    blueprint = compile_source_priority_blueprint_v13(row)
    assert blueprint["table_required"] is True
    assert blueprint["presentation"] == "compact_table"
    assert blueprint["advice_policy"] == "expert_source_only"
    messages = build_source_priority_v13_messages(row)
    assert "来源权限严格分层" in messages[0]["content"]
    assert "行动建议只能来自【专家建议】" in messages[0]["content"]
    assert "第一天合成步数8000步。" in messages[1]["content"]


def test_visual_contract_v14_honors_explicit_draw_label_and_no_repeat() -> None:
    row = {
        "query": "合成步数怎么样？",
        "data": "今日合成步数8000步。",
        "suggest": "建议继续记录。",
        "rag": "记录可用于后续观察。",
        "services": "合成步行课程。",
        "domain": "其他",
        "画图": "是",
    }
    contract = compile_visual_contract_v14(row)
    assert contract["table_required"] is True
    assert contract["presentation"] == "compact_evidence_table"
    assert contract["repeat_fact_after_table"] is False
    messages = build_visual_contract_v14_messages(row)
    assert "table_required=true" in messages[0]["content"]
    assert "表后不得" in messages[0]["content"]
    assert "今日合成步数8000步。" in messages[1]["content"]


def test_evidence_plan_renderer_keeps_only_exactly_supported_items() -> None:
    row = {
        "query": "redacted",
        "data": "每日步数为8000步。",
        "suggest": "建议循序渐进增加活动量。",
        "services": "轻松步行",
        "domain": "其他",
    }
    plan = {
        "direct": {
            "text": "目前记录为8000步。",
            "source": "personal",
            "support_quote": "每日步数为8000步",
        },
        "evidence": [
            {
                "label": "每日步数",
                "display_value": "8000步",
                "source": "personal",
                "support_quote": "每日步数为8000步",
                "interpretation": "这是本次可用记录。",
                "action": "保持记录。",
            },
            {
                "label": "unsupported",
                "display_value": "999",
                "source": "personal",
                "support_quote": "不存在的原文",
                "interpretation": "drop",
                "action": "drop",
            },
        ],
        "actions": [
            {
                "text": "循序渐进增加活动量。",
                "source": "expert",
                "support_quote": "建议循序渐进增加活动量",
            }
        ],
        "courses": [{"name": "轻松步行", "reason": "与步行相关"}],
        "safety_note": "",
    }
    rendered, stats = render_evidence_plan(row, plan)
    assert "8000步" in rendered
    assert "999" not in rendered
    assert "<轻松步行>" in rendered
    assert "## 关键依据" in rendered
    assert "## 建议" in rendered
    assert stats["evidence_proposed"] == 2
    assert stats["evidence_retained"] == 1
    assert len(build_evidence_plan_messages(row)) == 2


def test_gateway_records_operator_interrupt_in_request_denominator(tmp_path: Path) -> None:
    gateway = GatewayClient(
        run_dir=tmp_path,
        qwen_url="http://127.0.0.1:1/v1",
        qwen_key="redacted",
        qwen_model="synthetic-model",
        luna_url="http://127.0.0.1:2/v1",
        luna_key="redacted",
        luna_model="synthetic-judge",
        luna_min_request_interval_seconds=0,
        qwen_cap=3,
        luna_cap=3,
        max_attempts=3,
        stop_after_failures=3,
    )

    class InterruptingClient:
        def post(self, *args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt

        def close(self) -> None:
            return None

    gateway.client.close()
    gateway.client = InterruptingClient()
    with pytest.raises(KeyboardInterrupt):
        gateway.call(
            provider="qwen",
            operation_id="synthetic-interrupt",
            messages=[{"role": "user", "content": "redacted"}],
        )
    assert gateway._request_count("qwen") == 1
    assert gateway.events[0]["status"] == "interrupted"
    with pytest.raises(RuntimeError, match="retry forbidden"):
        gateway.call(
            provider="qwen",
            operation_id="synthetic-interrupt",
            messages=[{"role": "user", "content": "redacted"}],
        )
    assert gateway._request_count("qwen") == 1


def test_gateway_records_safe_upstream_error_code(tmp_path: Path) -> None:
    gateway = GatewayClient(
        run_dir=tmp_path,
        qwen_url="http://127.0.0.1:1/v1",
        qwen_key="redacted",
        qwen_model="synthetic-model",
        luna_url="http://127.0.0.1:2/v1",
        luna_key="redacted",
        luna_model="synthetic-judge",
        luna_min_request_interval_seconds=0,
        qwen_cap=1,
        luna_cap=1,
        max_attempts=1,
        stop_after_failures=3,
    )

    class FailingClient:
        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", "http://127.0.0.1:2/v1/chat/completions")
            return httpx.Response(
                502,
                request=request,
                json={
                    "error": {
                        "type": "server_error",
                        "code": "synthetic_upstream_failure",
                        "message": "synthetic safe detail",
                    }
                },
            )

        def close(self) -> None:
            return None

    gateway.client.close()
    gateway.client = FailingClient()
    with pytest.raises(RuntimeError):
        gateway.call(
            provider="luna",
            operation_id="synthetic-failure",
            messages=[{"role": "user", "content": "redacted"}],
        )
    assert gateway.events[0]["gateway_error"] == {
        "type": "server_error",
        "code": "synthetic_upstream_failure",
        "message": "synthetic safe detail",
    }
    assert gateway.events[0]["status"] == "failed"


def test_trace_v14_stops_before_next_qwen_on_known_infrastructure_failure() -> None:
    candidate_id = "synthetic-candidate"
    assert trace_v14_infrastructure_stop_reason(
        [
            {
                "provider": "qwen",
                "operation_id": f"generate:{candidate_id}:draft",
                "status": "failed",
                "gateway_error": {"code": "opencode_unreachable"},
            }
        ],
        candidate_id,
    ) == "qwen_opencode_unreachable"
    assert trace_v14_infrastructure_stop_reason(
        [
            {
                "provider": "luna",
                "operation_id": f"judge:{candidate_id}:ground:0",
                "status": "failed",
                "gateway_error": {"message": "Selected model is at capacity."},
            }
        ],
        candidate_id,
    ) == "luna_model_capacity"
    assert (
        trace_v14_infrastructure_stop_reason(
            [
                {
                    "provider": "qwen",
                    "operation_id": "generate:different-candidate:draft",
                    "status": "failed",
                    "gateway_error": {"code": "opencode_unreachable"},
                }
            ],
            candidate_id,
        )
        is None
    )


def test_conservative_renderer_drops_generated_numeric_interpretation() -> None:
    row = {
        "query": "redacted",
        "data": "每日步数为8000步。",
        "suggest": "建议循序渐进增加活动量。",
        "services": "轻松步行",
        "domain": "其他",
    }
    plan = {
        "direct": {
            "text": "目前是9000步并且偏高。",
            "source": "personal",
            "support_quote": "每日步数为8000步",
        },
        "evidence": [
            {
                "label": "每日步数",
                "display_value": "9000步",
                "source": "personal",
                "support_quote": "每日步数为8000步",
                "interpretation": "偏高",
                "action": "生成的建议不应保留",
            }
        ],
        "actions": [
            {
                "text": "模型改写",
                "source": "expert",
                "support_quote": "建议循序渐进增加活动量",
            }
        ],
        "courses": [{"name": "轻松步行", "reason": "生成原因"}],
    }
    rendered, stats = render_evidence_plan_conservative(row, plan)
    assert "8000步" in rendered
    assert "9000" not in rendered
    assert "偏高" not in rendered
    assert "生成的建议" not in rendered
    assert "建议循序渐进增加活动量" in rendered
    assert "生成原因" not in rendered
    assert stats["direct_generated_text_retained"] == 0


def test_balanced_renderer_uses_quote_for_invalid_direct_and_safe_action_text() -> None:
    row = {
        "query": "redacted",
        "data": "每日步数为8000步。",
        "suggest": "建议循序渐进增加活动量。",
        "services": "轻松步行",
        "domain": "其他",
    }
    plan = {
        "direct": {
            "text": "目前是9000步。",
            "source": "personal",
            "support_quote": "每日步数为8000步",
        },
        "evidence": [
            {
                "label": "每日步数",
                "source": "personal",
                "support_quote": "每日步数为8000步",
            }
        ],
        "actions": [
            {
                "text": "可以循序渐进增加活动量。",
                "source": "expert",
                "support_quote": "建议循序渐进增加活动量",
            }
        ],
        "courses": [],
    }
    rendered, stats = render_evidence_plan_balanced(row, plan)
    assert rendered.startswith("从现有记录看，每日步数为8000步")
    assert "9000" not in rendered
    assert "可以循序渐进增加活动量" in rendered
    assert stats["direct_used_first_evidence_quote"] == 1


def test_natural_renderer_deduplicates_quote_used_as_direct_answer() -> None:
    row = {
        "query": "redacted",
        "data": "每日步数为8000步。静息心率为70次/分。",
        "suggest": "建议循序渐进增加活动量。",
        "services": "",
        "domain": "其他",
    }
    plan = {
        "direct": {
            "text": "目前是9000步。",
            "source": "personal",
            "support_quote": "每日步数为8000步",
        },
        "evidence": [
            {
                "label": "每日步数",
                "source": "personal",
                "support_quote": "每日步数为8000步",
                "interpretation": "本次记录是8000步。",
            },
            {
                "label": "静息心率",
                "source": "personal",
                "support_quote": "静息心率为70次/分",
                "interpretation": "本次记录是70次/分。",
            },
        ],
        "actions": [],
        "courses": [],
    }
    rendered, stats = render_evidence_plan_natural(row, plan)
    assert rendered.count("每日步数为8000步") == 1
    assert "静息心率为70次/分" in rendered
    assert "9000" not in rendered
    assert stats["evidence_displayed_after_direct_dedup"] == 1


def test_packed_bon_parser_and_risk_selection_features() -> None:
    packed = """<candidate_a>
直接回答。\n\n## 建议\n- 保持记录。
</candidate_a>
<candidate_b>
## 先标题\n错误数字999，推荐<不存在课程>。
</candidate_b>"""
    candidate_a, candidate_b = extract_packed_candidates(packed)
    row = {
        "query": "redacted",
        "data": "每日步数8000步",
        "suggest": "保持记录",
        "services": "轻松步行",
        "domain": "其他",
    }
    risk_a = score_packed_candidate_risk(row, candidate_a)
    risk_b = score_packed_candidate_risk(row, candidate_b)
    assert risk_a["total"] < risk_b["total"]
    assert risk_b["unsupported_courses"] == 1
    assert risk_b["unsupported_numbers"] == 1
    assert risk_b["starts_with_heading"] == 1
    assert len(build_packed_bon_messages(row)) == 2


def test_packed_bon_parser_requires_both_marked_candidates() -> None:
    with pytest.raises(ValueError):
        extract_packed_candidates("<candidate_a>only one</candidate_a>")


def test_teacher_bootstrap_protocol_is_not_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-teacher-bootstrap-v6-smoke"]
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_context_compiler_v7_protocol_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-context-compiler-v7"]
    )
    assert parse_args().phase == "trace-context-compiler-v7"


def test_context_compiler_v8_protocol_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-context-compiler-v8"]
    )
    assert parse_args().phase == "trace-context-compiler-v8"


def test_silent_jury_v9_protocol_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["runner", "--phase", "trace-silent-jury-v9"])
    assert parse_args().phase == "trace-silent-jury-v9"


def test_silent_jury_v9_validation_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-silent-jury-v9-validation"]
    )
    assert parse_args().phase == "trace-silent-jury-v9-validation"


def test_evidence_packet_v10_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-evidence-packet-v10"]
    )
    assert parse_args().phase == "trace-evidence-packet-v10"


def test_full_context_index_v11_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-full-context-index-v11"]
    )
    assert parse_args().phase == "trace-full-context-index-v11"


def test_full_context_index_v11_audit_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-full-context-index-v11-audit"]
    )
    assert parse_args().phase == "trace-full-context-index-v11-audit"


def test_fact_cards_v12_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["runner", "--phase", "trace-fact-cards-v12"])
    assert parse_args().phase == "trace-fact-cards-v12"


def test_source_priority_v13_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-source-priority-v13"]
    )
    assert parse_args().phase == "trace-source-priority-v13"


def test_visual_contract_v14_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-visual-contract-v14"]
    )
    assert parse_args().phase == "trace-visual-contract-v14"


def test_visual_contract_v14_validation_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-visual-contract-v14-validation"]
    )
    assert parse_args().phase == "trace-visual-contract-v14-validation"


def test_proof_carrying_v15_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-proof-carrying-v15"]
    )
    assert parse_args().phase == "trace-proof-carrying-v15"


def test_evidence_compiler_v16_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-evidence-compiler-v16"]
    )
    assert parse_args().phase == "trace-evidence-compiler-v16"


def test_evidence_compiler_v16_renders_only_exact_selected_sources() -> None:
    row = {
        "query": "本周有哪些合成记录？",
        "data": "周一完成合成训练。周三完成恢复训练。",
        "suggest": "优先保持规律安排。记录训练后的感受。",
        "rag": "训练安排应循序渐进。",
        "services": "基础训练课程。",
        "domain": "运动",
    }
    raw_plan = (
        '{"personal_ids":["P2"],"expert_ids":["E1"],'
        '"knowledge_ids":[],"course_ids":[],"layout":"table"}'
    )
    rendered, stats = render_evidence_compiler_v16(row, raw_plan)
    assert "周三完成恢复训练。" in rendered
    assert "周一完成合成训练。" not in rendered
    assert "优先保持规律安排。" in rendered
    assert "| 记录 | 原始内容 |" in rendered
    assert stats["plan_parse_status"] == "model_json"
    assert stats["personal_selected"] == 1
    messages = build_evidence_compiler_v16_messages(row)
    assert "只返回一个 JSON 对象" in messages[0]["content"]


def test_evidence_compiler_v16_falls_back_without_retry_or_free_text() -> None:
    row = {
        "query": "请给出建议",
        "data": "仅有一条合成记录。",
        "suggest": "保留合成建议原文。",
        "rag": "通用合成知识。",
        "services": "",
        "domain": "运动",
    }
    rendered, stats = render_evidence_compiler_v16(row, "not json")
    assert "仅有一条合成记录。" in rendered
    assert "保留合成建议原文。" in rendered
    assert "not json" not in rendered
    assert stats["plan_parse_status"] == "deterministic_fallback"
    assert stats["fallback_source_groups"] == 2


def test_evidence_compiler_v16_rejects_unknown_ids_and_handles_missing_personal() -> None:
    row = {
        "query": "解释一般原则",
        "data": "",
        "suggest": "",
        "rag": "只使用这条合成知识。",
        "services": "",
        "domain": "运动",
    }
    rendered, stats = render_evidence_compiler_v16(
        row,
        '{"personal_ids":["P999"],"expert_ids":[],"knowledge_ids":[],"course_ids":[]}',
    )
    assert "不能判断你的具体情况" in rendered
    assert "只使用这条合成知识。" in rendered
    assert "P999" not in rendered
    assert stats["invalid_ids_removed"] == 1
    assert stats["knowledge_selected"] == 1


def test_evidence_compiler_v16_keeps_empty_input_judgeable() -> None:
    row = {"query": "请分析", "data": "", "suggest": "", "rag": "", "services": ""}
    rendered, stats = render_evidence_compiler_v16(row, "{}")
    assert "不能判断你的具体情况" in rendered
    assert "## 下一步" in rendered
    assert "原始指标、时间、单位" in rendered
    assert stats["rendered_lines"] >= 9


def test_grounded_composer_v17_retains_natural_source_addressed_fields() -> None:
    row = {
        "query": "本周有哪些记录？",
        "data": "周一完成合成训练。周三完成恢复训练。",
        "suggest": "建议保持规律安排。",
        "rag": "训练安排应循序渐进。",
        "services": "",
        "domain": "运动",
    }
    raw = (
        '{"opening":{"text":"记录显示周一完成合成训练。","source_ids":["P1"]},'
        '"personal_ids":["P1","P2"],'
        '"advice":[{"text":"建议保持规律安排。","source_ids":["E1"]}],'
        '"explanation":[{"text":"训练安排应循序渐进。","source_ids":["K1"]}],'
        '"course_ids":[]}'
    )
    rendered, stats = render_grounded_composer_v17(row, raw)
    assert rendered.startswith("记录显示周一完成合成训练。")
    assert "周三完成恢复训练。" in rendered
    assert "建议保持规律安排。" in rendered
    assert "训练安排应循序渐进。" in rendered
    assert stats["opening_generated_text_retained"] == 1
    assert stats["fallback_claims"] == 0
    messages = build_grounded_composer_v17_messages(row)
    assert "最终只输出一个 JSON 对象" in messages[0]["content"]


def test_grounded_composer_v17_removes_unbound_numbers_and_falls_back_exactly() -> None:
    row = {
        "query": "怎么安排？",
        "data": "合成记录存在。",
        "suggest": "建议保持规律安排。",
        "rag": "",
        "services": "",
        "domain": "运动",
    }
    raw = (
        '{"opening":{"text":"这是不存在的个人结论。","source_ids":["P1"]},'
        '"personal_ids":["P1"],'
        '"advice":[{"text":"每天安排30分钟。","source_ids":["E1"]}],'
        '"explanation":[],"course_ids":[]}'
    )
    rendered, stats = render_grounded_composer_v17(row, raw)
    assert "每天安排30分钟" not in rendered
    assert "建议保持规律安排。" in rendered
    assert "不存在的个人结论" not in rendered
    assert stats["opening_generated_text_retained"] == 0
    assert stats["advice_removed"] == 1
    assert stats["fallback_claims"] == 1


def test_grounded_composer_v17_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-grounded-composer-v17"]
    )
    assert parse_args().phase == "trace-grounded-composer-v17"


def test_contract_jury_v18_combines_visual_contract_and_single_output_jury() -> None:
    row = {
        "query": "比较两条合成记录",
        "data": "第一条合成记录。第二条合成记录。",
        "suggest": "保持合成建议。",
        "rag": "合成知识。",
        "services": "",
        "画图": "是",
        "domain": "运动",
    }
    messages = build_contract_jury_v18_messages(row)
    assert "v14 来源优先与可视化契约" in messages[0]["content"]
    assert "v18 单次调用合同陪审团" in messages[0]["content"]
    assert "只能输出一份最终 Markdown" in messages[0]["content"]
    assert '"table_required": true' in messages[1]["content"]
    assert "三稿陪审和五项否决" in messages[1]["content"]


def test_contract_jury_v18_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-contract-jury-v18"]
    )
    assert parse_args().phase == "trace-contract-jury-v18"


def test_numeric_shield_jury_v19_forbids_all_derived_arithmetic() -> None:
    row = {
        "query": "比较两条合成记录",
        "data": "第一条是10。第二条是12。",
        "suggest": "保留合成建议。",
        "rag": "合成知识。",
        "services": "",
        "domain": "运动",
    }
    messages = build_numeric_shield_jury_v19_messages(row)
    system = messages[0]["content"]
    assert "v18 单次调用合同陪审团" in system
    assert "v19 数字原句屏蔽层" in system
    assert "差值、总和、均值、比例、百分比" in system
    assert "单卡原句的连续逐字复制" in system


def test_numeric_shield_jury_v19_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-numeric-shield-jury-v19"]
    )
    assert parse_args().phase == "trace-numeric-shield-jury-v19"


def test_packed_contract_jury_v20_parses_three_complete_candidates() -> None:
    packed = """<candidate_a>
直接回答甲。\n\n## 建议\n- 甲建议。
</candidate_a>
<candidate_b>
直接回答乙。\n\n## 建议\n- 乙建议。
</candidate_b>
<candidate_c>
直接回答丙。\n\n## 建议\n- 丙建议。
</candidate_c>"""
    candidates = extract_packed_candidates_v20(packed)
    assert list(candidates) == ["a", "b", "c"]
    assert candidates["b"].startswith("直接回答乙")


def test_packed_contract_jury_v20_risk_penalizes_derived_and_ungrounded_numbers() -> None:
    row = {
        "query": "比较记录",
        "data": "第一条记录为10。第二条记录为12。",
        "suggest": "保持记录。",
        "rag": "",
        "services": "",
        "画图": "是",
        "domain": "运动",
    }
    grounded = """现有两条记录可分别核对。

## 记录依据

| 记录 | 原始内容 |
| --- | --- |
| 相关记录 | 第一条记录为10。 |
| 相关记录 | 第二条记录为12。 |

## 建议
- 保持记录。"""
    risky = """相比之下增加了2。

## 建议
- 保持记录。"""
    safe_score = score_packed_contract_candidate_v20(row, grounded)
    risky_score = score_packed_contract_candidate_v20(row, risky)
    assert safe_score["required_table_missing"] == 0
    assert risky_score["required_table_missing"] == 1
    assert risky_score["derived_relation_lines"] >= 1
    assert risky_score["ungrounded_numeric_lines"] >= 1
    assert safe_score["total"] < risky_score["total"]
    assert "三个彼此独立" in build_packed_contract_jury_v20_messages(row)[0]["content"]


def test_packed_contract_jury_v20_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["runner", "--phase", "trace-packed-contract-jury-v20"]
    )
    assert parse_args().phase == "trace-packed-contract-jury-v20"


def test_packed_contract_jury_v20_validation_is_exposed_by_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "--phase", "trace-packed-contract-jury-v20-validation"],
    )
    assert parse_args().phase == "trace-packed-contract-jury-v20-validation"


def test_proof_citation_firewall_keeps_only_locally_proved_lines() -> None:
    row = {
        "query": "合成步数怎么样？",
        "data": "今日合成步数8000步。",
        "suggest": "建议继续记录。",
        "rag": "持续记录有助于观察变化。",
        "services": "合成步行课程。",
        "domain": "其他",
        "画图": "否",
    }
    ledger = compile_proof_ledger_v15(row)
    assert {"P1", "E1", "K1", "C1"}.issubset(ledger["sources"])
    assert "[S:P1]" in build_proof_carrying_v15_messages(row)[0]["content"]
    response = """概括但没有逐字个人事实。 [S:P1]

## 记录依据

- 今日合成步数8000步。 [S:P1]
- 今日合成步数8000步。补充9000步。 [S:P1]

## 建议

- 建议继续记录。 [S:E1]
- 可尝试课程 <不存在课程>。 [S:C1]
"""
    rendered, stats = apply_proof_citation_firewall_v15(row, response)
    assert "今日合成步数8000步。" in rendered
    assert "建议继续记录。" in rendered
    assert "9000" not in rendered
    assert "不存在课程" not in rendered
    assert "[S:" not in rendered
    assert stats["personal_exactness_lines_removed"] == 1
    assert stats["numeric_binding_lines_removed"] == 1
    assert stats["course_binding_lines_removed"] == 1

    embedded, embedded_stats = apply_proof_citation_firewall_v15(
        row,
        "今日合成步数8000步。[S:P1] 建议继续记录。[S:E1]。",
    )
    assert "今日合成步数8000步。" in embedded
    assert "建议继续记录。" in embedded
    assert "[S:" not in embedded
    assert embedded_stats["substantive_lines_kept"] == 1


def test_proof_ledger_represents_missing_personal_source_without_model_inference() -> None:
    row = {
        "query": "合成比较问题",
        "data": "",
        "suggest": "",
        "rag": "",
        "services": "",
        "domain": "跑步",
    }
    ledger = compile_proof_ledger_v15(row)
    assert ledger["sources"]["M1"] == {
        "source": "metadata",
        "exact_text": "当前输入未提供可引用的个人数据。",
    }
    rendered, stats = apply_proof_citation_firewall_v15(
        row, "当前输入未提供可引用的个人数据。 [S:M1]"
    )
    assert rendered == "当前输入未提供可引用的个人数据。"
    assert stats["substantive_lines_kept"] == 1


def test_proof_firewall_allows_grounded_paraphrase_but_blocks_new_judgement() -> None:
    row = {
        "query": "合成步数怎么样？",
        "data": "今日合成步数8000步。",
        "suggest": "",
        "rag": "",
        "services": "",
        "domain": "其他",
    }
    rendered, stats = apply_proof_citation_firewall_v15(
        row,
        "- 记录显示，今日合成步数为8000步。[S:P1]\n"
        "- 今日合成步数8000步，说明已经达标。[S:P1]",
    )
    assert "记录显示" in rendered
    assert "达标" not in rendered
    assert stats["substantive_lines_kept"] == 1
    assert stats["personal_exactness_lines_removed"] == 1


def test_at_least_80_percent_batch_boundary() -> None:
    assert minimum_accepts_for_target(10) == 8
    assert nonaccept_stop_count_for_target(10) == 3
    assert minimum_accepts_for_target(20) == 16
    assert nonaccept_stop_count_for_target(20) == 5


def test_exact_mcnemar_handles_no_and_asymmetric_discordance() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(0, 6) == 0.03125


def test_cli_accepts_historical_and_trace_request_usage_fields() -> None:
    assert report_request_usage({"request_usage": {"qwen": 1}}) == {"qwen": 1}
    assert report_request_usage(
        {"actual_experiment_request_usage": {"qwen": 2}}
    ) == {"qwen": 2}
