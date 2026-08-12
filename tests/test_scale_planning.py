from __future__ import annotations

import pytest

from src.experiments.scale_planning import (
    build_scale_plan,
    minimum_successes_for_exact_lower_bound,
    minimum_successes_for_point_rate,
)


def test_point_rate_rounds_up_to_the_required_success_count() -> None:
    assert minimum_successes_for_point_rate(5, 0.80) == 4
    assert minimum_successes_for_point_rate(25_000, 0.80) == 20_000


def test_exact_lower_bound_gate_is_stricter_than_the_point_rate() -> None:
    assert minimum_successes_for_exact_lower_bound(5, 0.80) is None
    assert minimum_successes_for_exact_lower_bound(50, 0.80) == 45
    assert minimum_successes_for_exact_lower_bound(100, 0.80) == 87


def test_scale_plan_matches_fixed_one_qwen_two_luna_protocol() -> None:
    plan = build_scale_plan()
    assert plan["primary_budget"] == {
        "sampling_requests": 25_000,
        "maximum_judge_requests": 50_000,
        "target_acceptance_rate_at_least": 0.80,
        "minimum_accepted_outputs": 20_000,
    }
    assert plan["serial_teacher_capacity"]["conservative_elapsed_days"] == pytest.approx(
        138.8888888889
    )
    assert plan["deadline_capacity_targets"][0][
        "required_independent_teacher_lanes"
    ] == 5
    assert plan["alternate_goal_if_25k_means_accepted_outputs"][
        "minimum_sampling_requests_at_target_rate"
    ] == 31_250
