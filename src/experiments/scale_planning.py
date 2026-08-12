from __future__ import annotations

import math
from typing import Any, Sequence

from src.experiments.summary_synthesis import one_sided_exact_lower_bound


SECONDS_PER_DAY = 86_400


def minimum_successes_for_point_rate(total: int, target_rate: float) -> int:
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0.0 < target_rate <= 1.0:
        raise ValueError("target_rate must be in (0, 1]")
    return math.ceil(total * target_rate - 1e-12)


def minimum_successes_for_exact_lower_bound(
    total: int,
    target_rate: float,
    *,
    confidence: float = 0.95,
) -> int | None:
    """Return the first success count whose one-sided exact lower bound hits target."""
    first = minimum_successes_for_point_rate(total, target_rate)
    for successes in range(first, total + 1):
        if (
            one_sided_exact_lower_bound(
                successes, total, confidence=confidence
            )
            >= target_rate
        ):
            return successes
    return None


def build_scale_plan(
    *,
    sampling_requests: int = 25_000,
    target_rate: float = 0.80,
    judge_calls_per_successful_sample: int = 2,
    teacher_interval_seconds: float = 240.0,
    deadline_days: Sequence[float] = (30.0, 14.0, 7.0),
) -> dict[str, Any]:
    """Build a source-text-free request, yield, and teacher-capacity plan."""
    if sampling_requests <= 0:
        raise ValueError("sampling_requests must be positive")
    if judge_calls_per_successful_sample <= 0:
        raise ValueError("judge_calls_per_successful_sample must be positive")
    if teacher_interval_seconds <= 0:
        raise ValueError("teacher_interval_seconds must be positive")
    if any(days <= 0 for days in deadline_days):
        raise ValueError("deadline_days must contain only positive values")

    minimum_accepted = minimum_successes_for_point_rate(
        sampling_requests, target_rate
    )
    maximum_judge_requests = sampling_requests * judge_calls_per_successful_sample
    serial_seconds = maximum_judge_requests * teacher_interval_seconds
    serial_days = serial_seconds / SECONDS_PER_DAY

    capacity_targets = []
    for days in deadline_days:
        available_seconds = days * SECONDS_PER_DAY
        capacity_targets.append(
            {
                "deadline_days": days,
                "required_independent_teacher_lanes": math.ceil(
                    serial_seconds / available_seconds
                ),
                "required_global_teacher_start_interval_seconds": (
                    available_seconds / maximum_judge_requests
                ),
                "required_judge_requests_per_day": maximum_judge_requests / days,
            }
        )

    sampling_for_accepted_goal = math.ceil(
        sampling_requests / target_rate - 1e-12
    )
    judge_for_accepted_goal = (
        sampling_for_accepted_goal * judge_calls_per_successful_sample
    )

    validation_gates = []
    for total in (5, 50, 100):
        exact_required = minimum_successes_for_exact_lower_bound(total, target_rate)
        validation_gates.append(
            {
                "sampling_requests": total,
                "point_rate_minimum_accepted": minimum_successes_for_point_rate(
                    total, target_rate
                ),
                "one_sided_exact_95_lower_bound_minimum_accepted": exact_required,
                "maximum_judge_requests": total
                * judge_calls_per_successful_sample,
                "serial_teacher_hours": (
                    total
                    * judge_calls_per_successful_sample
                    * teacher_interval_seconds
                    / 3_600
                ),
            }
        )

    return {
        "schema_version": "summary-synthesis-scale-plan-v1",
        "primary_goal_interpretation": (
            "25k sampling-model API requests produce 25k candidate records"
        ),
        "primary_budget": {
            "sampling_requests": sampling_requests,
            "maximum_judge_requests": maximum_judge_requests,
            "target_acceptance_rate_at_least": target_rate,
            "minimum_accepted_outputs": minimum_accepted,
        },
        "serial_teacher_capacity": {
            "request_start_interval_seconds": teacher_interval_seconds,
            "conservative_elapsed_days": serial_days,
            "assumption": (
                "one shared teacher-capacity lane; two judge calls for every "
                "successful sampling response; no retries"
            ),
        },
        "deadline_capacity_targets": capacity_targets,
        "alternate_goal_if_25k_means_accepted_outputs": {
            "target_accepted_outputs": sampling_requests,
            "minimum_sampling_requests_at_target_rate": sampling_for_accepted_goal,
            "maximum_judge_requests": judge_for_accepted_goal,
            "conservative_elapsed_days": (
                judge_for_accepted_goal
                * teacher_interval_seconds
                / SECONDS_PER_DAY
            ),
        },
        "validation_gates": validation_gates,
        "claim_boundary": (
            "Capacity arithmetic is deterministic. Acceptance yield remains an "
            "empirical quantity and must be reported from immutable request ledgers."
        ),
    }
