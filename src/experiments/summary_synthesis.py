from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import random
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Iterable, Mapping, Sequence

import httpx
import pandas as pd
import yaml

from prompts.system_prompt_v5_yixuan import DOMAIN_DESCRIPTION, build_phone_personal_prompt


GROUND_RULES = {
    "PERSONAL_DATA_MISMATCH",
    "COURSE_LIB_MISSING",
    "NUM_COMPARE_ERROR",
    "ARITH_ERROR",
    "CONTRADICT_KB_OR_EXPERT",
    "FACT_LOGIC_ISSUE",
    "IRRELEVANT",
}
STRUCTURE_RULES = {
    "EMPTY_OR_INCOMPLETE",
    "ILLEGAL_CONTENT",
    "SENSITIVE_ADVICE",
    "NO_MARKDOWN",
    "BAD_MARKDOWN_USAGE",
    "BURIED_CORE_ANSWER",
    "UNNATURAL_TONE",
    "LACK_VISUAL_AID",
    "THIN_CONTENT",
    "PERSONAL_DATA_ANALYSIS_ISSUE",
    "REDUNDANT",
    "GRAMMAR",
}
STRICT_20 = {
    "EMPTY_OR_INCOMPLETE",
    "ILLEGAL_CONTENT",
    "SENSITIVE_ADVICE",
    "PERSONAL_DATA_MISMATCH",
    "COURSE_LIB_MISSING",
    "NUM_COMPARE_ERROR",
    "ARITH_ERROR",
    "IRRELEVANT",
}


def is_infrastructure_failure_event(event: Mapping[str, Any]) -> bool:
    """Identify service, network, container, auth, and capacity failures."""
    if event.get("status") != "failed":
        return False
    gateway_error = event.get("gateway_error", {})
    if not isinstance(gateway_error, Mapping):
        gateway_error = {}
    http_status = event.get("http_status")
    if isinstance(http_status, int) and (
        http_status >= 500 or http_status in {401, 403, 408, 425, 429}
    ):
        return True
    failure_class = str(event.get("failure_class", "")).lower()
    if any(
        marker in failure_class
        for marker in (
            "connecterror",
            "connecttimeout",
            "networkerror",
            "readerror",
            "readtimeout",
            "remoteprotocolerror",
            "writetimeout",
        )
    ):
        return True
    error_text = " ".join(
        (
            str(gateway_error.get("code", "")),
            str(gateway_error.get("type", "")),
            str(gateway_error.get("message", "")),
            str(event.get("failure_message", "")),
        )
    ).lower()
    return any(
        marker in error_text
        for marker in (
            "at capacity",
            "connection refused",
            "container",
            "network",
            "opencode_unreachable",
            "refresh token",
            "service unavailable",
            "timed out",
            "timeout",
        )
    )


def is_sampling_infrastructure_failure_event(event: Mapping[str, Any]) -> bool:
    """Identify sampler infrastructure failures excluded from the denominator."""
    return bool(
        event.get("provider") == "qwen"
        and is_infrastructure_failure_event(event)
    )


def sampling_quality_denominator(
    events: Sequence[Mapping[str, Any]],
    *,
    exclude_infrastructure_failures: bool,
) -> tuple[int, int]:
    """Return valid sampler calls and the separately reported infra exclusions."""
    excluded = (
        sum(is_sampling_infrastructure_failure_event(event) for event in events)
        if exclude_infrastructure_failures
        else 0
    )
    return len(events) - excluded, excluded


def trace_v14_infrastructure_stop_reason(
    events: Sequence[Mapping[str, Any]], candidate_id: str
) -> str | None:
    """Return a safe pre-next-Qwen stop reason for the current v14 candidate."""
    qwen_operation_prefix = f"generate:{candidate_id}:"
    luna_operation_prefix = f"judge:{candidate_id}:"
    for event in events:
        if event.get("status") != "failed":
            continue
        operation_id = str(event.get("operation_id", ""))
        gateway_error = event.get("gateway_error", {})
        if not isinstance(gateway_error, Mapping):
            gateway_error = {}
        if event.get("provider") == "qwen" and operation_id.startswith(
            qwen_operation_prefix
        ):
            if str(gateway_error.get("code", "")) == "opencode_unreachable":
                return "qwen_opencode_unreachable"
            if is_infrastructure_failure_event(event):
                return "qwen_infrastructure_failure"
        if event.get("provider") == "luna" and operation_id.startswith(
            luna_operation_prefix
        ):
            if "at capacity" in str(gateway_error.get("message", "")).lower():
                return "luna_model_capacity"
            if is_infrastructure_failure_event(event):
                return "luna_infrastructure_failure"
    return None


def trace_infrastructure_stop_reason_for_iteration(
    events: Sequence[Mapping[str, Any]],
    candidate_id: str,
    *,
    failure_preexisted: bool,
) -> str | None:
    """Stop for a new infrastructure failure, not one replayed during resume."""
    reason = trace_v14_infrastructure_stop_reason(events, candidate_id)
    return None if failure_preexisted else reason


ROOT_CONTEXT_FIELDS = (
    "domain",
    "一级分类",
    "二级分类",
    "三级分类",
    "query",
    "data",
    "suggest",
    "rag",
    "services",
    "last_query",
    "last_answer_phone",
)
REPAIR_TARGET_BY_RULE = {
    "PERSONAL_DATA_MISMATCH": "fact_alignment",
    "COURSE_LIB_MISSING": "course_grounding",
    "NUM_COMPARE_ERROR": "numeric_units",
    "ARITH_ERROR": "numeric_units",
    "CONTRADICT_KB_OR_EXPERT": "fact_alignment",
    "FACT_LOGIC_ISSUE": "fact_alignment",
    "IRRELEVANT": "conciseness_grammar",
    "EMPTY_OR_INCOMPLETE": "structure_markdown",
    "ILLEGAL_CONTENT": "safety_boundary",
    "SENSITIVE_ADVICE": "safety_boundary",
    "NO_MARKDOWN": "structure_markdown",
    "BAD_MARKDOWN_USAGE": "structure_markdown",
    "BURIED_CORE_ANSWER": "structure_markdown",
    "UNNATURAL_TONE": "conciseness_grammar",
    "LACK_VISUAL_AID": "structure_markdown",
    "THIN_CONTENT": "personalization",
    "PERSONAL_DATA_ANALYSIS_ISSUE": "personalization",
    "REDUNDANT": "conciseness_grammar",
    "GRAMMAR": "conciseness_grammar",
}
REPAIR_INSTRUCTION_BY_TARGET = {
    "fact_alignment": "逐项对齐输入证据，删除无依据的事实、因果和结论。",
    "numeric_units": "逐项复核数字、单位、比较关系和必要计算。",
    "course_grounding": "只使用课程库中实际出现的精确课程名。",
    "personalization": "使用已提供的个人数据，不用通用知识冒充个人事实。",
    "structure_markdown": "结论前置并修复 Markdown 可读性，不扩写新事实。",
    "safety_boundary": "移除越界诊断或承诺，并补充必要的安全边界。",
    "conciseness_grammar": "删除冗余、病句、跑题内容和重复建议。",
}

TARGET_ACCEPTS_PER_QWEN_REQUEST = 0.80
KTO_POSITIVE_MIN_SCORE = 14.0
KTO_NEGATIVE_MIN_EXCLUSIVE = 0.0
KTO_NEGATIVE_MAX_SCORE = 8.0


def classify_kto_score(score: float) -> str:
    """Classify one completed dual-judge score under the frozen KTO policy."""
    value = float(score)
    if value >= KTO_POSITIVE_MIN_SCORE:
        return "positive"
    if KTO_NEGATIVE_MIN_EXCLUSIVE < value <= KTO_NEGATIVE_MAX_SCORE:
        return "negative"
    if value == KTO_NEGATIVE_MIN_EXCLUSIVE:
        return "unusable_zero"
    return "ambiguous"


def is_kto_accepted_score(score: float) -> bool:
    """Return whether a score is usable as a positive or negative KTO sample."""
    return classify_kto_score(score) in {"positive", "negative"}


def normalize_judge_result_acceptance(result: dict[str, Any]) -> dict[str, Any]:
    """Apply the current score-band policy when replaying persisted results."""
    if result.get("status") != "ok":
        result["accepted"] = False
        result["kto_score_band"] = None
        result["kto_label"] = None
        return result
    score_band = classify_kto_score(float(result.get("total_score_20", 0.0)))
    result["accepted"] = score_band in {"positive", "negative"}
    result["kto_score_band"] = score_band
    result["kto_label"] = score_band if result["accepted"] else None
    diagnosis = result.get("diagnosis")
    if isinstance(diagnosis, dict):
        diagnosis["accepted"] = result["accepted"]
        diagnosis["kto_score_band"] = score_band
        diagnosis["kto_label"] = result["kto_label"]
    return result


def minimum_accepts_for_target(planned_root_count: int) -> int:
    """Return the fewest accepts whose planned-batch rate is at least 80%."""
    if planned_root_count < 0:
        raise ValueError("planned_root_count must be non-negative")
    return math.ceil(
        TARGET_ACCEPTS_PER_QWEN_REQUEST * planned_root_count - 1e-12
    )


def nonaccept_stop_count_for_target(planned_root_count: int) -> int:
    """Return the first nonaccept count that makes the >=80% target impossible."""
    return planned_root_count - minimum_accepts_for_target(planned_root_count) + 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(passed: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = passed / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _binomial_tail_at_least(successes: int, total: int, probability: float) -> float:
    if successes <= 0:
        return 1.0
    if successes > total or probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(value + 1)
        - math.lgamma(total - value + 1)
        + value * math.log(probability)
        + (total - value) * math.log1p(-probability)
        for value in range(successes, total + 1)
    ]
    maximum = max(logs)
    return min(1.0, math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))


def one_sided_exact_lower_bound(
    passed: int, total: int, confidence: float = 0.95
) -> float:
    """Clopper-Pearson lower bound for a binomial success probability."""
    if total <= 0 or passed <= 0:
        return 0.0
    alpha = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_tail_at_least(passed, total, midpoint) < alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def paired_difference_bootstrap_interval(
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    seed: int = 20260811,
    repeats: int = 10_000,
) -> tuple[float, float]:
    """Percentile interval for paired rate difference second minus first."""
    if len(first) != len(second):
        raise ValueError("Paired samples must have equal length")
    if not first:
        return 0.0, 0.0
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(repeats):
        sampled = [rng.randrange(len(first)) for _ in first]
        differences.append(
            sum(float(second[index]) - float(first[index]) for index in sampled)
            / len(sampled)
        )
    differences.sort()
    return (
        differences[int(0.025 * (repeats - 1))],
        differences[int(0.975 * (repeats - 1))],
    )


def exact_mcnemar_p_value(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    one_tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * one_tail)


def normalize_root_value(value: str) -> str:
    return " ".join(value.split())


def root_context_id(row: Mapping[str, str]) -> str:
    canonical = {
        field: normalize_root_value(str(row.get(field, "")))
        for field in ROOT_CONTEXT_FIELDS
    }
    return sha256_text(json.dumps(canonical, ensure_ascii=False, sort_keys=True))[:24]


def source_record_hash(row: Mapping[str, str]) -> str:
    canonical = {
        str(key): normalize_root_value(str(value))
        for key, value in sorted(row.items())
    }
    return sha256_text(json.dumps(canonical, ensure_ascii=False, sort_keys=True))


def build_root_splits(
    rows: Sequence[Mapping[str, str]],
    source_hash: str,
    *,
    protected_development_indices: Sequence[int],
    validation_root_count: int,
    audit_root_count: int,
) -> dict[str, list[int]]:
    roots: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        roots.setdefault(root_context_id(row), []).append(index)
    protected_roots = {
        root_context_id(rows[index]) for index in protected_development_indices
    }
    eligible_roots = sorted(
        (root for root in roots if root not in protected_roots),
        key=lambda root: sha256_text(f"{source_hash}:trace-split-v1:{root}"),
    )
    if len(eligible_roots) < validation_root_count + audit_root_count:
        raise ValueError("Not enough independent roots for validation and audit splits")
    audit_roots = set(eligible_roots[:audit_root_count])
    validation_roots = set(
        eligible_roots[audit_root_count : audit_root_count + validation_root_count]
    )
    split_by_root = {
        root: (
            "audit"
            if root in audit_roots
            else "validation"
            if root in validation_roots
            else "development"
        )
        for root in roots
    }
    output = {"development": [], "validation": [], "audit": []}
    for root, indices in roots.items():
        output[split_by_root[root]].extend(indices)
    for indices in output.values():
        indices.sort()
    return output


def select_trace_phase1_indices(
    rows: Sequence[Mapping[str, str]],
    source_hash: str,
    *,
    development_indices: Sequence[int],
    excluded_indices: Sequence[int],
    count: int,
) -> list[int]:
    excluded_roots = {root_context_id(rows[index]) for index in excluded_indices}
    representative_by_root: dict[str, int] = {}
    for index in development_indices:
        root = root_context_id(rows[index])
        if root not in excluded_roots:
            representative_by_root.setdefault(root, index)

    def signature(index: int) -> tuple[str, ...]:
        row = rows[index]
        return (
            row.get("domain", ""),
            row.get("一级分类", ""),
            row.get("二级分类", ""),
            row.get("三级分类", ""),
            "history" if row.get("last_answer_phone", "").strip() else "single",
        )

    available = sorted(
        representative_by_root.values(),
        key=lambda index: sha256_text(
            f"{source_hash}:trace-phase1-v1:{root_context_id(rows[index])}"
        ),
    )
    selected: list[int] = []
    seen: set[tuple[str, ...]] = set()
    for index in available:
        if len(selected) >= count:
            break
        if signature(index) not in seen:
            selected.append(index)
            seen.add(signature(index))
    for index in available:
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    if len(selected) != count:
        raise ValueError(f"Requested {count} phase-1 roots, found {len(selected)}")
    return selected


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _literal_assignment(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Template"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
        ):
            return value.args[0].value
    raise ValueError(f"Could not find literal assignment {name!r}")


@dataclass(frozen=True)
class PromptBundle:
    legacy_generation_system: str
    ground_system: str
    ground_template: Template
    structure_system: str
    structure_template: Template

    @classmethod
    def from_snapshot(cls, response_prompt: Path, scorer: Path) -> "PromptBundle":
        return cls.from_source_texts(
            response_prompt.read_text(encoding="utf-8"),
            scorer.read_text(encoding="utf-8"),
        )

    @classmethod
    def from_source_texts(cls, response_source: str, scorer_source: str) -> "PromptBundle":
        return cls(
            legacy_generation_system=_literal_assignment(
                response_source, "SYSTEMT_PROMPT_PHONE_GENERAL"
            ),
            ground_system=_literal_assignment(scorer_source, "GROUND_SYSTEM_PROMPT_TPL"),
            ground_template=Template(_literal_assignment(scorer_source, "GROUND_PROMPT_TPL")),
            structure_system=_literal_assignment(
                scorer_source, "STRUCT_SYSTEM_PROMPT_TPL"
            ),
            structure_template=Template(
                _literal_assignment(scorer_source, "STRUCT_PROMPT_TPL")
            ),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    expected_source_sha256: str
    snapshot_commit: str
    snapshot_repo: Path
    source_git_path: str
    legacy_response_prompt_git_path: str
    legacy_scorer_git_path: str
    legacy_response_prompt: Path
    legacy_scorer: Path
    qwen_url: str
    qwen_model: str
    luna_url: str
    luna_url_pool: tuple[str, ...]
    luna_model: str
    luna_reasoning_effort: str
    luna_min_request_interval_seconds: float
    luna_initial_request_delay_seconds: float
    canary_indices: tuple[int, ...]
    compare_count: int
    judge_repeats: int
    pass_threshold: float
    qwen_request_cap: int
    luna_request_cap: int
    max_attempts_per_operation: int
    stop_after_consecutive_failures: int
    pipeline_epoch: str
    phase1_count: int
    phase1_excluded_indices: tuple[int, ...]
    validation_excluded_indices: tuple[int, ...]
    audit_excluded_indices: tuple[int, ...]
    split_protected_indices: tuple[int, ...]
    phase1_checkpoints: tuple[int, ...]
    validation_root_count: int
    audit_root_count: int
    critical_dimension_floor: float
    stop_when_target_impossible: bool

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]
        trace = raw.get("trace", {})

        def configured_indices(
            inline_key: str, file_key: str, *extra_keys: str
        ) -> tuple[int, ...]:
            values = list(trace.get(inline_key, []))
            for extra_key in extra_keys:
                values.extend(trace.get(extra_key, []))
            configured_file = str(trace.get(file_key, "")).strip()
            if configured_file:
                index_path = Path(configured_file)
                if not index_path.is_absolute():
                    index_path = path.parent / index_path
                loaded = json.loads(index_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, list):
                    raise ValueError(f"{file_key} must point to a JSON array")
                values.extend(loaded)
            return tuple(dict.fromkeys(int(index) for index in values))

        return cls(
            source_path=Path(raw["source_path"]),
            expected_source_sha256=str(raw["expected_source_sha256"]),
            snapshot_commit=str(raw["snapshot_commit"]),
            snapshot_repo=Path(raw["snapshot_repo"]),
            source_git_path=str(raw.get("source_git_path", "")),
            legacy_response_prompt_git_path=str(raw["legacy_response_prompt_git_path"]),
            legacy_scorer_git_path=str(raw["legacy_scorer_git_path"]),
            legacy_response_prompt=Path(raw["legacy_response_prompt"]),
            legacy_scorer=Path(raw["legacy_scorer"]),
            qwen_url=str(raw["generation"]["base_url"]).rstrip("/"),
            qwen_model=str(raw["generation"]["model"]),
            luna_url=str(raw["judge"]["base_url"]).rstrip("/"),
            luna_url_pool=tuple(
                str(value).rstrip("/")
                for value in raw["judge"].get(
                    "base_urls", [raw["judge"]["base_url"]]
                )
            ),
            luna_model=str(raw["judge"]["model"]),
            luna_reasoning_effort=str(raw["judge"]["reasoning_effort"]),
            luna_min_request_interval_seconds=float(
                raw["judge"].get("min_request_interval_seconds", 0.0)
            ),
            luna_initial_request_delay_seconds=float(
                raw["judge"].get("initial_request_delay_seconds", 0.0)
            ),
            canary_indices=tuple(int(i) for i in raw["canary_indices"]),
            compare_count=int(raw["compare_count"]),
            judge_repeats=int(raw["judge_repeats"]),
            pass_threshold=float(raw["pass_threshold_strictly_greater_than"]),
            qwen_request_cap=int(raw["request_caps"]["qwen"]),
            luna_request_cap=int(raw["request_caps"]["luna"]),
            max_attempts_per_operation=int(raw["max_attempts_per_operation"]),
            stop_after_consecutive_failures=int(raw["stop_after_consecutive_failures"]),
            pipeline_epoch=str(trace.get("pipeline_epoch", "historical-pilot-v1")),
            phase1_count=int(trace.get("phase1_count", 0)),
            phase1_excluded_indices=configured_indices(
                "excluded_indices",
                "excluded_indices_file",
                "additional_excluded_indices",
            ),
            validation_excluded_indices=tuple(
                int(index) for index in trace.get("validation_excluded_indices", [])
            ),
            audit_excluded_indices=tuple(
                int(index) for index in trace.get("audit_excluded_indices", [])
            ),
            split_protected_indices=(
                configured_indices(
                    "split_protected_indices", "split_protected_indices_file"
                )
                if trace.get("split_protected_indices")
                or trace.get("split_protected_indices_file")
                else configured_indices("excluded_indices", "excluded_indices_file")
            ),
            phase1_checkpoints=tuple(
                int(value) for value in trace.get("checkpoints", [])
            ),
            validation_root_count=int(trace.get("validation_root_count", 0)),
            audit_root_count=int(trace.get("audit_root_count", 0)),
            critical_dimension_floor=float(
                trace.get("critical_dimension_floor", 15.0)
            ),
            stop_when_target_impossible=bool(
                trace.get("stop_when_target_impossible", True)
            ),
        )


def read_frozen_artifact(
    local_path: Path,
    *,
    repo: Path,
    commit: str,
    git_path: str,
) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{git_path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    git_value = completed.stdout
    if local_path.exists():
        local_value = local_path.read_bytes()
        if local_value != git_value:
            raise ValueError(
                f"Local snapshot differs from git blob {commit}:{git_path}"
            )
        return local_value
    return git_value


def load_source_rows(config: ExperimentConfig) -> list[dict[str, str]]:
    source_bytes = read_source_artifact(config)
    actual_hash = sha256_bytes(source_bytes)
    if actual_hash != config.expected_source_sha256:
        raise ValueError(
            "Frozen source hash mismatch: "
            f"expected={config.expected_source_sha256} actual={actual_hash}"
        )
    suffix = config.source_path.suffix.lower()
    if suffix == ".csv":
        return [
            dict(row)
            for row in csv.DictReader(io.StringIO(source_bytes.decode("utf-8")))
        ]
    if suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(
            io.BytesIO(source_bytes), dtype=str, keep_default_na=False
        )
        return [
            {str(key): str(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
    raise ValueError(f"Unsupported frozen source format: {suffix}")


def read_source_artifact(config: ExperimentConfig) -> bytes:
    """Read a Git-frozen source, or a local artifact verified by mandatory hash."""
    if config.source_git_path:
        return read_frozen_artifact(
            config.source_path,
            repo=config.snapshot_repo,
            commit=config.snapshot_commit,
            git_path=config.source_git_path,
        )
    if not config.source_path.is_file():
        raise ValueError("Local hash-pinned source_path does not exist")
    return config.source_path.read_bytes()


def stable_candidate_id(source_hash: str, row_index: int, strategy: str) -> str:
    return sha256_text(f"{source_hash}:{row_index}:{strategy}")[:24]


def stable_trace_candidate_id(
    root_id: str, pipeline_epoch: str, variant_id: str
) -> str:
    return sha256_text(f"{root_id}:{pipeline_epoch}:{variant_id}")[:24]


def select_compare_indices(
    rows: Sequence[Mapping[str, str]], count: int, required: Sequence[int]
) -> list[int]:
    """Choose deterministic, metadata-stratified rows without inspecting answer text."""
    selected = list(dict.fromkeys(int(index) for index in required))
    if len(selected) > count:
        return selected[:count]

    remaining = [index for index in range(len(rows)) if index not in selected]

    def signature(index: int) -> tuple[str, ...]:
        row = rows[index]
        return (
            row.get("domain", ""),
            row.get("一级分类", ""),
            row.get("二级分类", ""),
            row.get("三级分类", ""),
            "history" if row.get("last_answer_phone", "").strip() else "single",
        )

    remaining.sort(key=lambda index: sha256_text(json.dumps(signature(index), ensure_ascii=False) + f":{index}"))
    seen = {signature(index) for index in selected}
    for index in remaining:
        if len(selected) >= count:
            break
        if signature(index) not in seen:
            selected.append(index)
            seen.add(signature(index))
    if len(selected) < count:
        for index in remaining:
            if len(selected) >= count:
                break
            if index not in selected:
                selected.append(index)
    return selected


def build_context(row: Mapping[str, str], *, corrected: bool) -> str:
    open_marker, close_marker = ("【", "】") if corrected else ("[", "]")
    current_label = "当前用户提问" if corrected else "用户提问"
    services = row.get("services", "") if corrected else row.get("service", "")
    blocks: list[str] = []
    for label, value in (
        ("个人数据", row.get("data", "")),
        ("专家建议", row.get("suggest", "")),
        ("知识库知识", row.get("rag", "")),
        ("课程库", services),
    ):
        blocks.extend([f"{open_marker}{label}{close_marker}", value.strip() or "（无）", ""])
    history: list[str] = []
    if row.get("last_query", "").strip():
        history.append(f"user: {row['last_query'].strip()}")
    if row.get("last_answer_phone", "").strip():
        history.append(f"assistant: {row['last_answer_phone'].strip()}")
    blocks.extend(
        [
            f"{open_marker}对话历史{close_marker}",
            "\n".join(history) or "（无）",
            "",
            f"{open_marker}{current_label}{close_marker}",
            row.get("query", "").strip(),
        ]
    )
    return "\n".join(blocks)


def build_slim_evidence_system(domain: str | None) -> str:
    """Return a compact, evidence-priority system prompt for one-shot generation."""
    domain_rules = DOMAIN_DESCRIPTION.get((domain or "").strip(), "").replace(
        "\\n", "\n"
    )
    return f"""你是“小艺”，负责用中文回答运动健康问题。只输出给用户看的完整 Markdown 回答。

## 证据优先级与禁止项
1. 先回答【当前用户提问】，只分析与问题直接相关的内容。
2. 个人事实只能来自【个人数据】和明确针对该用户的【专家建议】。逐字核对指标名、数值、单位、日期、时间范围和运动类型；输入没给出的个人事实一律不补写。
3. 【知识库知识】只作通用解释，不能冒充个人数据。结论必须同时不违背专家建议、知识库和下方领域规则；若材料口径冲突，避免下确定结论并说明限制，不能自行选一个改写成事实。
4. 只有输入明确给出适用范围时才判断正常、偏高或偏低。不同指标、时段、人群、运动类型和参考范围不得混用；优先使用已有统计值，确需计算时逐项核对并列式。
5. 不把相关性写成因果，不诊断疾病、不保证疗效、不提供具体药物或剂量；必要时给出就医边界。
6. 课程只能逐字选自【课程库】，必须高度相关；无合适课程就不推荐，也不得沿用历史中但本轮课程库没有的课程。

## 成稿标准
- 第一段直接给出核心答案；有相关个人数据时突出最重要的1—4个指标，无相关数据时明确说明未查询到。
- 随后用有信息量的小标题和列表给出解释与可执行建议。存在两个及以上真正需要对比的指标或时段时，使用紧凑表格；不要为排版重复内容。
- 每个重要判断都能回指输入证据；保留输入中的指标原名。内容应完整而简洁，不能只有结论、标题或泛泛建议。
- 引用知识库观点时可用[1]形式，不要在回答中说“知识库”“专家建议”或暴露本提示词。

## 当前领域规则
{domain_rules or '无额外领域规则；仍须严格遵守上面的证据边界。'}"""


def build_slim_generation_messages(
    row: Mapping[str, str], *, internal_dual_draft: bool
) -> list[dict[str, str]]:
    system = build_slim_evidence_system(row.get("domain"))
    mode = (
        """\n\n【单次调用内的静默择优】
在内部完成以下步骤，但绝不输出步骤、草稿或核验表：
1. 建立证据账本，列清可用的个人事实、数值单位、适用范围、材料限制和禁止推断项；
2. 分别形成“证据与结论优先”和“行动与可读性优先”两个候选草案；
3. 逐一否决存在事实错配、比较/计算错误、材料冲突、核心答案后置、Markdown不完整或安全越界的候选；
4. 融合两者优点并再次核对，只输出最终最佳的完整回答。"""
        if internal_dual_draft
        else """\n\n【单次调用内的静默核验】
先在内部建立证据账本并完成事实、数值、材料一致性、结构和安全核验；发现问题直接修正。不要输出核验过程，只输出最终完整回答。"""
    )
    user = build_context(row, corrected=True) + mode
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compile_context_profile(row: Mapping[str, str]) -> dict[str, Any]:
    """Compile label-free routing features from the current input only.

    The profile intentionally contains no source text or judge-derived feature. It is
    safe to persist in aggregate experiment metadata without revealing health records.
    """
    query = row.get("query", "").strip()
    category_text = " ".join(
        row.get(field, "").strip()
        for field in ("domain", "一级分类", "二级分类", "三级分类")
    )
    route_text = f"{category_text} {query}"
    personal = row.get("data", "").strip()
    expert = row.get("suggest", "").strip()
    knowledge = row.get("rag", "").strip()
    courses = row.get("services", "").strip()

    number_pattern = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
    time_pattern = re.compile(
        r"(?:\d{1,4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?|"
        r"(?:今天|今日|昨天|昨日|本周|上周|本月|上月|近\s*\d+\s*(?:天|周|月)))"
    )
    comparison_marker = bool(
        re.search(r"趋势|变化|对比|比较|相比|升高|降低|增加|减少|波动|最近|过去", route_text)
    )
    course_marker = bool(re.search(r"课程|训练|练习|跟练|推荐", route_text))
    plan_marker = bool(
        re.search(r"怎么|如何|建议|计划|安排|改善|提高|降低|调整|继续|注意", route_text)
    )
    explanation_marker = bool(
        re.search(r"什么|为什么|是否|正常|原理|意味着|说明什么|原因", route_text)
    )

    if courses and course_marker:
        task_type = "course_recommendation"
    elif comparison_marker:
        task_type = "trend_or_comparison"
    elif plan_marker:
        task_type = "action_plan"
    elif explanation_marker or not personal:
        task_type = "knowledge_explanation"
    else:
        task_type = "personal_status"

    personal_number_count = len(number_pattern.findall(personal))
    personal_time_marker_count = len(time_pattern.findall(personal))
    return {
        "compiler_version": "context-compiler-v7",
        "task_type": task_type,
        "has_personal_data": bool(personal),
        "has_expert_guidance": bool(expert),
        "has_knowledge": bool(knowledge),
        "has_course_catalog": bool(courses),
        "has_dialog_history": bool(
            row.get("last_query", "").strip()
            or row.get("last_answer_phone", "").strip()
        ),
        "personal_number_count": personal_number_count,
        "personal_time_marker_count": personal_time_marker_count,
        "comparison_requested": comparison_marker,
        "comparison_evidence_likely": (
            comparison_marker
            and (personal_time_marker_count >= 2 or personal_number_count >= 2)
        ),
    }


def build_context_compiler_v7_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Build one Qwen request that returns exactly one final Markdown answer."""
    profile = compile_context_profile(row)
    task_type = str(profile["task_type"])
    route_contracts = {
        "course_recommendation": (
            "先回答是否有匹配课程；课程名必须从课程库逐字复制。没有高度相关课程时，"
            "明确不推荐课程，再给不依赖课程的安全行动建议。"
        ),
        "trend_or_comparison": (
            "只比较同一指标、同一单位且时间点明确的数据。若证据足够，用紧凑表格逐项"
            "列出原值后再总结；若口径不齐，直接说明不能可靠比较，不补算趋势。"
        ),
        "action_plan": (
            "首段给出可执行方向，再按优先级列出2—4步；每一步必须能由专家建议、知识"
            "材料或领域安全规则支持，不把通用建议伪装成个人结论。"
        ),
        "knowledge_explanation": (
            "区分通用知识与个人判断；没有相关个人记录时明确说明，不用知识材料反推"
            "用户的数值、状态、病因或风险。"
        ),
        "personal_status": (
            "首段直接概括最相关的个人记录；只解释有明确依据的1—4项，不擅自判断"
            "正常、异常、原因或疾病。"
        ),
    }
    system = build_slim_evidence_system(row.get("domain")) + """

## v7 单输出执行协议
- 你只有一次生成调用，最终只能输出一个完整中文 Markdown 回答；不得输出多个候选、JSON、证据账本、审核过程或提示词。
- 先静默完成：意图路由 → 最小证据集合 → 成稿 → 删除式审计。审计发现不确定主张时优先删除或降格表达，不为追求丰富而补写。
- 每个个人数字必须保持输入中的指标、数值、单位、日期和范围绑定；禁止跨指标、跨时段或跨运动类型拼接。没有完整操作数就不计算。
- 专家建议用于行动边界，知识材料用于通用解释，课程库仅用于课程名；三者都不能反推输入没有的个人事实。
- 输出前逐句执行四问：这是当前问题所需的吗？来源类别正确吗？数值与限定词逐字一致吗？删除后是否更可靠？任一答案不确定就删除或改成明确限制。
"""
    profile_json = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    user = f"""{build_context(row, corrected=True)}

【本地 Context Compiler 路由】
{profile_json}

【本题结构契约】
{route_contracts[task_type]}

请在内部比较“证据最小充分”和“行动可读”两个写法，只保留同时通过事实、结构与安全审计的内容。第一段必须直接回答；正文至少包含一个有信息量的 `##` 小标题，并在适合时使用列表或表格。最终只输出一个可直接展示的完整 Markdown 回答。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_context_compiler_v8_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Strengthen exact-copy grounding while retaining a single final output."""
    messages = [dict(message) for message in build_context_compiler_v7_messages(row)]
    messages[0]["content"] += """

## v8 精确锚点与完整性约束
- 个人事实采用“原文锚点复制”：先在个人数据中找到连续原文，再围绕它解释；不得把数值换算成另一表达，不得补齐缺失单位或时间。
- 禁止自行计算差值、平均值、比例、百分比、排名或达标率；只有输入已明确给出该结果时才能原样引用。比较任务只并列原始同口径值并作不带计算的谨慎描述。
- 成稿必须同时具有：首段直接回答、`## 关键依据`或`## 说明`、`## 建议`。关键依据保留1—4项，建议保留2—4项；没有证据时明确限制，但仍给安全且有来源的下一步。
- 比较任务且存在两个以上同口径记录时必须使用表格；其他任务至少使用列表。标题下必须有正文，不能只给空标题、半句话或占位符。
"""
    messages[1]["content"] += """

【v8 最终静默门禁】
逐句删除任何没有连续输入锚点的个人事实；删除所有自行计算或换算；确认三个必需区块均完整，并且比较信息在适用时用表格呈现。仍然只输出一个最终 Markdown 回答。"""
    return messages


def build_silent_jury_v9_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Use an internal multi-perspective jury while emitting one final datum."""
    profile = compile_context_profile(row)
    system = build_slim_evidence_system(row.get("domain")) + """

## 单次调用静默陪审团
你只有一次生成调用。内部可以形成多个候选，但最终只能输出一个完整中文 Markdown 回答，绝不输出草稿、评分、陪审意见、JSON或推理过程。

内部依次执行三个互相独立的否决审查：
1. 事实陪审员：逐句检查个人事实是否能在个人数据中找到连续依据；数字、单位、日期、指标与范围是否绑定；是否自行计算、混淆口径、违背专家/知识材料、虚构课程或把相关写成因果。任一问题均否决该句。
2. 完整性陪审员：检查是否首段直接回答、内容足够完成当前任务、Markdown完整；确有多时段/多指标比较时是否使用紧凑表格，否则是否至少使用清晰列表。不得为了形式强加无关区块。
3. 表达陪审员：检查语句是否自然、无病句、无重复、无自相矛盾；标题必须有信息量且下方有正文，建议要具体但不能超出材料。

内部至少比较“最小证据版”“均衡说明版”“行动优先版”三个候选。严格事实错误拥有最高否决权；在均无严格错误的候选中，选择最直接、完整、自然的一版。若证据不足，明确限制比猜测更优。
"""
    profile_json = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    user = f"""{build_context(row, corrected=True)}

【无标签任务画像】
{profile_json}

请按静默陪审团流程完成本题。第一段直接回答，随后只使用完成本题真正需要的小标题、列表或表格。个人事实尽量保留输入中的原名和原句，不做输入未给出的算术、换算、诊断或因果解释。最终只输出一个可直接展示的完整 Markdown 回答。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _split_evidence_sentences(value: str) -> list[str]:
    """Split source text deterministically while preserving exact source wording."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    pieces = re.split(r"(?<=[。！？!?；;])|\n+", normalized)
    return [piece.strip() for piece in pieces if piece.strip()]


def _text_ngrams(value: str, n: int = 2) -> set[str]:
    compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()
    if not compact:
        return set()
    if len(compact) < n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _rank_evidence_sentences(
    query: str,
    value: str,
    *,
    limit: int,
    prefer_numeric: bool,
) -> list[str]:
    sentences = _split_evidence_sentences(value)
    if len(sentences) <= limit:
        return sentences
    query_grams = _text_ngrams(query)
    number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_grams = _text_ngrams(sentence)
        overlap = len(query_grams & sentence_grams)
        coverage = overlap / max(1.0, math.sqrt(len(sentence_grams)))
        numeric_bonus = 0.35 if prefer_numeric and number_pattern.search(sentence) else 0.0
        scored.append((coverage + numeric_bonus, index, sentence))
    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    return [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]


def compile_evidence_packet(row: Mapping[str, str]) -> dict[str, Any]:
    """Build a source-isolated, query-relevant packet without labels or model calls."""
    query = row.get("query", "").strip()
    route_query = " ".join(
        value
        for value in (
            row.get("domain", "").strip(),
            row.get("一级分类", "").strip(),
            row.get("二级分类", "").strip(),
            row.get("三级分类", "").strip(),
            row.get("last_query", "").strip(),
            query,
        )
        if value
    )
    profile = compile_context_profile(row)
    personal_limit = 12 if profile["task_type"] == "trend_or_comparison" else 8
    source_specs = (
        ("personal", "P", row.get("data", ""), personal_limit, True),
        ("expert", "E", row.get("suggest", ""), 6, False),
        ("knowledge", "K", row.get("rag", ""), 6, False),
        ("courses", "C", row.get("services", ""), 12, False),
    )
    sources: dict[str, list[dict[str, str]]] = {}
    source_stats: dict[str, dict[str, int]] = {}
    for source_name, prefix, value, limit, prefer_numeric in source_specs:
        raw_sentences = _split_evidence_sentences(value)
        selected = _rank_evidence_sentences(
            route_query,
            value,
            limit=limit,
            prefer_numeric=prefer_numeric,
        )
        sources[source_name] = [
            {"id": f"{prefix}{index}", "text": sentence}
            for index, sentence in enumerate(selected, start=1)
        ]
        source_stats[source_name] = {
            "available_sentences": len(raw_sentences),
            "selected_sentences": len(selected),
            "selection_limit": limit,
        }
    packet = {
        "schema_version": "evidence-packet-v10",
        "task_profile": profile,
        "current_query": query,
        "previous_user_query": row.get("last_query", "").strip(),
        "sources": sources,
        "source_stats": source_stats,
    }
    packet["packet_sha256"] = sha256_text(
        json.dumps(packet, ensure_ascii=False, sort_keys=True)
    )
    return packet


def build_evidence_packet_v10_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Generate once from a deterministic source-isolated evidence packet."""
    packet = compile_evidence_packet(row)
    system = build_slim_evidence_system(row.get("domain")) + """

## Evidence Packet v10 使用协议
本次用户材料已由无标签本地检索器整理成证据包。你只有一次生成调用，最终只能输出一个完整中文 Markdown 回答，不输出证据 ID、JSON、草稿、评分或核验过程。

来源含义严格固定：
- P：个人数据，只能据此陈述该用户的事实；指标、数字、单位、日期、时段和运动类型必须保持绑定。
- E：针对性专家建议，用于行动边界和谨慎判断；不能改写成用户已经发生的事实。
- K：通用知识，只用于解释；不能反推个人状态、病因、风险或诊断。
- C：课程库；只有证据包中逐字存在的课程名才可推荐。

每个准备输出的个人主张，先在内部绑定一个 P 原文片段；没有 P 依据就删除或明确说当前没有相关个人记录。禁止自行补算差值、均值、比例、阈值或趋势。E 与 K 若口径不一致，不自行调和为确定结论，只保留共同且保守的部分并说明限制。

结构必须服务于问题：首段直接回答；存在两个以上同口径时段或指标且用户要求比较时用紧凑表格；否则使用有信息量的小标题与列表。答案需要完整、自然、无重复，但丰富度不能来自虚构事实。
"""
    model_packet = {
        key: packet[key]
        for key in (
            "schema_version",
            "task_profile",
            "current_query",
            "previous_user_query",
            "sources",
        )
    }
    packet_json = json.dumps(model_packet, ensure_ascii=False, sort_keys=True)
    user = f"""【Evidence Packet】
{packet_json}

请只依据证据包回答 current_query。先静默完成来源绑定、事实一致性、材料冲突、结构和语言检查，再输出唯一最终 Markdown。不要在回答中提及证据包、来源字母或内部流程。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_full_context_index_v11_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Keep the full context while adding a deterministic relevance index."""
    packet = compile_evidence_packet(row)
    labels = {
        "personal": "个人数据锚点",
        "expert": "专家建议锚点",
        "knowledge": "知识材料锚点",
        "courses": "课程名称锚点",
    }
    index_lines: list[str] = []
    for source in ("personal", "expert", "knowledge", "courses"):
        index_lines.append(f"### {labels[source]}")
        records = packet["sources"][source]
        if records:
            index_lines.extend(
                f"- [{record['id']}] {record['text']}" for record in records
            )
        else:
            index_lines.append("- （无）")
    evidence_index = "\n".join(index_lines)
    task_type = str(packet["task_profile"]["task_type"])
    route_note = {
        "trend_or_comparison": (
            "本题为比较/趋势意图：只有同指标、同单位、时间明确的原始记录才能并列表格；"
            "不自行计算差值、均值或百分比。"
        ),
        "course_recommendation": (
            "本题涉及课程：课程名必须从课程库全文逐字复制，没有高相关课程就不推荐。"
        ),
        "action_plan": (
            "本题偏行动建议：首段给方向，随后列出有专家/知识依据的优先行动。"
        ),
        "knowledge_explanation": (
            "本题偏知识解释：通用知识不能反推用户个人状态；无相关个人记录时明确说明。"
        ),
        "personal_status": (
            "本题偏个人状态：只分析与问题直接相关且在个人数据中明确出现的事实。"
        ),
    }[task_type]
    system = build_slim_evidence_system(row.get("domain")) + """

## 全量上下文 + 相关证据索引协议
本地索引仅用于帮助你定位长材料中的相关原文，完整来源区仍是最终依据。索引没有列出的内容并非自动禁止，但使用前必须回到对应完整来源区逐字核对。索引 ID 只供内部定位，最终回答不得输出 ID、索引、草稿或审核过程。

一次调用内静默形成“事实最稳”和“表达最完整”两个候选，逐句否决个人事实错配、材料冲突、自行算术、虚构课程、核心答案后置、内容单薄、缺少必要表格或不自然表达，最后只输出一个完整 Markdown。
"""
    user = f"""{build_context(row, corrected=True)}

【本地相关证据索引（辅助定位，完整原文优先）】
{evidence_index}

【任务路由提醒】
{route_note}

请首段直接回答当前问题，再按需使用有信息量的小标题、列表或紧凑表格。不得为了丰富而添加完整来源区没有的个人事实、数字、因果、阈值或课程。最终只输出一个可直接展示的完整中文 Markdown 回答。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compile_personal_fact_cards_v12(
    row: Mapping[str, str],
) -> dict[str, Any]:
    """Create exact, query-relevant personal fact cards without model labels."""
    packet = compile_evidence_packet(row)
    number_pattern = re.compile(
        r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?%?"
    )
    cards: list[dict[str, Any]] = []
    for record in packet["sources"]["personal"]:
        text = record["text"]
        bindings: list[dict[str, str]] = []
        for match in number_pattern.finditer(text):
            start = max(0, match.start() - 14)
            end = min(len(text), match.end() + 14)
            bindings.append(
                {
                    "token": match.group(0),
                    "exact_fragment": text[start:end],
                }
            )
        cards.append(
            {
                "id": record["id"],
                "exact_sentence": text,
                "numeric_bindings": bindings,
                "policy": "copy_exact_sentence_or_omit",
            }
        )
    payload = {
        "schema_version": "personal-fact-cards-v12",
        "task_type": packet["task_profile"]["task_type"],
        "cards": cards,
    }
    payload["cards_sha256"] = sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    return payload


def build_fact_cards_v12_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Generate once using full context plus exact personal-fact copy cards."""
    cards = compile_personal_fact_cards_v12(row)
    system = build_slim_evidence_system(row.get("domain")) + """

## v12 个人事实逐字溯源协议
你只有一次生成调用，最终只输出一个完整中文 Markdown 回答，不输出草稿、卡片 ID、JSON、评分或核验过程。

个人化内容采用严格的“复制或省略”规则：
1. 凡是描述该用户已经发生、当前处于、曾经记录、数值变化或个人趋势的句子，必须先绑定一张个人事实卡；优先逐字复制卡片的 exact_sentence，再在单独句子中给谨慎解释。
2. 不得把两张卡片的日期、指标、数值、单位、范围或运动类型拼成一个新事实；不得改写数字，不得自行计算差值、均值、比例、排名、达标率或趋势。
3. 只有原始个人材料已经逐字给出“升高、降低、偏高、偏低、改善、恶化”等结论时，才可把该结论写成用户事实。否则只并列原始记录，不下趋势结论。
4. 卡片没有覆盖问题所需个人事实时，明确说现有记录不足；不得用专家建议或通用知识反推该用户状态。
5. 专家建议和知识材料可用于一般性解释与行动建议，但必须与个人事实分段，使用“通常、可以考虑”等非诊断表达。课程名只能逐字复制课程库。

回答采取最小充分原则：首段直接回答；最多选择三条真正相关的个人事实；事实之后再给两到四条有材料依据的建议。比较确有必要时使用紧凑表格，但表格单元格仍只能复制同一事实卡中的原始表达。
"""
    cards_json = json.dumps(cards, ensure_ascii=False, sort_keys=True)
    user = f"""{build_context(row, corrected=True)}

【个人事实卡（本地确定性抽取，仅供逐字核对）】
{cards_json}

请先静默执行逐句来源检查：每个个人事实是否对应一张卡、数字与邻近指标/日期/单位是否来自同一句、是否产生了输入未明确给出的比较结论。任何不满足项都直接省略，并明确材料限制。最终只输出一个可直接展示的完整中文 Markdown 回答。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compile_source_priority_blueprint_v13(
    row: Mapping[str, str],
) -> dict[str, Any]:
    """Build a label-free answer blueprint from source presence and task shape."""
    cards = compile_personal_fact_cards_v12(row)
    packet = compile_evidence_packet(row)
    task_type = str(cards["task_type"])
    table_required = task_type == "trend_or_comparison" and len(cards["cards"]) >= 2
    blueprint = {
        "schema_version": "source-priority-blueprint-v13",
        "task_type": task_type,
        "personal_fact_policy": "copy_exact_sentence_or_omit",
        "personal_fact_limit": min(3, len(cards["cards"])),
        "presentation": "compact_table" if table_required else "short_evidence_list",
        "table_required": table_required,
        "advice_policy": "expert_source_only",
        "knowledge_policy": "explain_only_when_not_in_conflict_with_expert",
        "course_policy": "exact_course_name_only",
        "source_presence": {
            source: bool(packet["sources"][source])
            for source in ("personal", "expert", "knowledge", "courses")
        },
    }
    blueprint["blueprint_sha256"] = sha256_text(
        json.dumps(blueprint, ensure_ascii=False, sort_keys=True)
    )
    return blueprint


def build_source_priority_v13_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Generate once with exact facts, strict source priority, and fixed layout."""
    cards = compile_personal_fact_cards_v12(row)
    blueprint = compile_source_priority_blueprint_v13(row)
    system = build_slim_evidence_system(row.get("domain")) + """

## v13 来源优先与冲突隔离协议
你只有一次生成调用，最终只输出一个完整中文 Markdown 回答，不输出卡片、蓝图、JSON、草稿或核验过程。

来源权限严格分层：
1. 个人事实只能逐字复制个人事实卡的 exact_sentence，或明确省略；不得合并卡片、计算、换算、补全、推断趋势或改写为诊断。
2. 行动建议只能来自【专家建议】中明确存在的方向。允许改成自然的第二人称表达，但不得加入专家材料未提出的新动作、频率、时长、强度或目标值。
3. 【知识库知识】只可解释概念，不能产生个人结论或行动建议。若知识材料与专家建议在阈值、方向、适用人群或风险边界上不完全一致，省略该解释，不自行调和。
4. 课程只在与问题直接相关且课程名可从课程库逐字复制时推荐；否则不出现课程。

输出顺序固定为：首段直接回答 → `## 记录依据` → `## 建议`。蓝图要求表格时，记录依据必须是紧凑 Markdown 表格，并且每行只承载一张事实卡的原始表达；否则使用短列表。没有足够个人记录时，首段明确材料限制，记录依据写明未发现相关记录，仍可给专家材料支持的一般建议。

出稿前逐句静默核验：个人句是否逐字来自单卡；建议是否能在专家材料中找到；知识解释是否与专家材料完全兼容；蓝图要求的表格是否存在。任何不满足项直接删除。
"""
    support = {
        "personal_fact_cards": cards,
        "answer_blueprint": blueprint,
    }
    user = f"""{build_context(row, corrected=True)}

【v13 本地确定性支持结构】
{json.dumps(support, ensure_ascii=False, sort_keys=True)}

请按来源权限和蓝图回答当前问题。不要复述材料清单，不要声明内部规则，最终只输出可直接展示的中文 Markdown。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compile_visual_contract_v14(row: Mapping[str, str]) -> dict[str, Any]:
    """Compile an explicit, label-free Markdown presentation contract."""
    base = compile_source_priority_blueprint_v13(row)
    draw_label = str(row.get("画图", "")).strip()
    cards = compile_personal_fact_cards_v12(row)
    display_requested = draw_label == "是"
    comparison_table = (
        base["task_type"] == "trend_or_comparison" and len(cards["cards"]) >= 2
    )
    table_required = display_requested or comparison_table
    contract = {
        "schema_version": "visual-contract-v14",
        "task_type": base["task_type"],
        "draw_label": draw_label or "unspecified",
        "table_required": table_required,
        "presentation": "compact_evidence_table" if table_required else "short_evidence_list",
        "max_personal_facts": min(3, len(cards["cards"])),
        "max_advice_items": 3,
        "repeat_fact_after_table": False,
        "source_priority": {
            "personal": "exact_fact_card_only",
            "advice": "expert_source_only",
            "knowledge": "non_conflicting_explanation_only",
            "course": "exact_name_only",
        },
    }
    contract["contract_sha256"] = sha256_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True)
    )
    return contract


def build_visual_contract_v14_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Generate once with exact facts and an explicit Markdown visual contract."""
    cards = compile_personal_fact_cards_v12(row)
    contract = compile_visual_contract_v14(row)
    system = build_slim_evidence_system(row.get("domain")) + """

## v14 来源优先与可视化契约
你只有一次生成调用，最终只输出一个完整中文 Markdown 回答，不输出事实卡、契约、JSON、草稿或核验过程。

个人事实、专家建议、知识解释和课程名继续遵守严格来源权限：个人事实只能逐字来自单张事实卡；建议只能来自专家材料；知识只作与专家材料不冲突的概念解释；课程名只能逐字复制。不得计算、换算、拼接不同卡片或推断输入未明确给出的趋势。

输出严格执行 visual_contract：
- table_required=true：必须在首段直接回答后提供 `## 记录依据` 的紧凑 Markdown 表格。每行只放一张事实卡的原始表达，不得在表格中新增判断、数值或单位。表后不得用段落或列表重复同一事实。
- table_required=false：`## 记录依据` 使用短列表，不为了形式强加表格。
- `## 建议` 最多三项，每项只表达一个专家材料支持的动作；不得重复首段或记录依据。
- 没有足够个人事实时明确材料限制，不用知识材料填充个人表格。

出稿前静默检查：契约要求的表格是否存在、Markdown 表头与分隔行是否完整、事实是否逐卡溯源、建议是否来自专家材料、同一事实是否重复。发现任一问题就在本次回答内删除或修正。
"""
    support = {"personal_fact_cards": cards, "visual_contract": contract}
    user = f"""{build_context(row, corrected=True)}

【v14 本地确定性支持结构】
{json.dumps(support, ensure_ascii=False, sort_keys=True)}

请按来源权限和 visual_contract 回答当前问题。最终只输出可直接展示的中文 Markdown。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_guarded_visual_contract_v21_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Add a general safety-first completion policy to the proven v14 prompt."""
    messages = [dict(message) for message in build_visual_contract_v14_messages(row)]
    messages[0]["content"] += """

## v21 保守完成策略
事实、数值、材料一致性和安全边界优先于丰富度，适用于所有样本：
- 首段不写派生数值、诊断、因果、效果承诺或输入没有明确支持的比较结论。
- 每个含数字、日期、单位或比较关系的个人句，必须能由同一张个人事实卡的完整原句独立支持；不能跨卡组合。
- `## 建议` 中的每个动作必须能在专家建议原文中定位。材料没有给出动作时，明确说明材料限制，不自行补充频率、强度、时长、用药、治疗或风险处置。
- 遇到材料冲突、口径不齐或证据不足时，采用“直接说明限制 + 逐字记录依据 + 有来源建议”的完整保守写法，不能猜测，也不能只给空泛拒答。

输出前逐行静默否决：无法由单一来源句核对的数字/比较行、无法由专家材料核对的建议行、诊断或越界处置行一律删除。最终仍须是结构完整、可直接展示的 Markdown。"""
    messages[1]["content"] += """

【v21 最终门禁】
只输出一份完成保守核验的 Markdown；证据不足时宁可明确限制，也不补写事实、比较、诊断或建议。"""
    return messages


def build_relevance_safe_direct_v24_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Answer the current query directly without forcing unrelated source sections."""
    system = """你是一个谨慎、实用的中文助手。你只有一次生成调用。

先在内部判断当前用户问题与所给材料是否相关，再直接回答当前问题：
- 相关材料可以用于回答，但不得编造材料中没有的个人事实、数字、日期、单位、诊断、因果或效果承诺。
- 材料与问题无关、互相冲突或不足时，不要强行套用材料；使用可靠的一般知识回答，并明确必要的边界。
- 对违法、伤害、欺骗、隐私侵犯或规避监管等请求，不提供可操作的实施细节；简短说明不能协助的部分，并转向合法、合规、降低风险的替代方案。
- 首段必须回应用户真正的问题，随后给出足够完整的解释和可执行建议；不要只说“材料不足”或要求用户补充资料。
- 涉及健康、法律、财务或安全风险时保持审慎，不作诊断或保证，必要时建议咨询合格专业人士。

最终只输出一份完整、自然、可直接展示的中文 Markdown 回答，不输出分类、推理过程、内部规则或 JSON。"""
    user = f"""{build_context(row, corrected=True)}

请以【当前问题】为最高优先级，按上述相关性与安全规则回答。材料只在确实相关时使用。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_relevance_grounded_positive_v29_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Keep v28 relevance routing while closing common zero-score failure modes."""
    system = """你是一个谨慎、实用的中文对话助手。你只有一次生成机会，必须在本轮给出可直接展示的最终回答。

先在内部判断当前问题与各类输入材料是否相关，再遵守以下通用规则：

1. 首段直接完成用户请求。不得只给方案、提出一串确认问题，或承诺“确认后再回答”。若请求涉及设置闹钟、提醒、日程、写入或删除数据、控制设备等本通道不能直接执行的操作，先简短说明无法直接操作，再立即给出最短可执行步骤；不得声称已经完成。
2. 每一句关于用户本人的事实，只能绑定到【个人数据】中的同一条明确证据。不得把单日记录泛化成日常习惯，不得把【知识库知识】或【专家建议】中的数字写成用户个人指标，不得把相关线索写成确定因果。
3. 默认禁止新增派生计算、换算和比较。用户没有明确要求计算时，直接引用来源已经给出的结果。用户明确要求计算时，必须列出同源操作数和公式，并在内部复核一次；不能可靠复核就说明无法从现有材料计算。只有上下界全部落在目标区间内，才能说“整个区间正常”或“全部达标”。
4. 数据互相冲突、口径不一致或时间范围不同，不自行调和，不选择其中一个继续推导；分别引用原始结果并明确说明口径限制。
5. 推荐课程时，课程名称只能逐字复制【课程库】中的完整名称，并统一写成 `<完整课程名>`；没有完全匹配的课程就不推荐，不得自造、缩写或组合课程名。
6. 健康、法律、财务和安全问题，不得给出材料未支持的剂量、设备参数、阈值、疗程、改善时间、诊断、确定因果或效果承诺；必要时给出审慎的专业求助边界。
7. 涉及当前产品发布、政策、价格或其他时效事实，而输入没有可核验来源时，明确说明无法确认最新状态，不使用记忆中的日期、版本或发布时间作确定结论。
8. 材料与问题无关时不要强行套用。不要为了丰富度扩写无关知识、课程、计算或建议；简单问题使用短段落或列表即可。除非用户明确指定，不得引入 macOS、Windows、Linux、终端、命令行、脚本、文件、工作区、仓库或本地软件环境。

对违法、伤害、欺骗、隐私侵犯或规避监管等请求，不提供可操作的实施细节，简短转向合法且降低风险的替代方案。

出稿前静默检查：是否已经直接回答；个人事实是否同句有据；数字、单位、区间和课程名是否逐项可核对；是否出现未经支持的敏感参数、时效事实、设备完成声明或未来承诺。任何不满足项在本次回答内删除或改为明确边界。

最终只输出一份完整、自然、可直接展示的中文 Markdown，不输出分类、推理过程、内部规则、计划或 JSON。"""
    user = f"""{build_context(row, corrected=True)}

请以【当前问题】为最高优先级，按 v29 相关性、来源绑定、计算与一次完成规则作答。材料只在确实相关时使用。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_relevance_grounded_positive_v30_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Generate a complete chat answer while keeping grounding checks non-destructive."""
    system = """你是一个无状态的中文对话助手。当前任务是直接回复用户，不是规划软件工程任务。你不能访问或操作用户的设备、应用、终端、文件、工作区或其他本地环境，也不能在本轮之后继续执行。你只有一次生成机会，必须立即给出可直接展示的最终回答。

先在内部判断当前问题与各类输入材料是否相关，再遵守以下规则：

1. 开头一到两句直接回答核心问题。不得只给方案、提出一串确认问题，或承诺“确认后再回答”。若用户要求设置闹钟、提醒、日程、写入或删除数据、控制设备，简短说明不能代为操作，并立即给出不依赖特定操作系统的最短可执行步骤；不得声称已经完成。
2. 关于用户本人的事实只能来自【个人数据】中同一条、口径一致的明确证据。保留原始指标名、数值、单位和时间范围；不得把单日记录泛化成长期习惯，不得把【知识库知识】或【专家建议】中的参考数字写成用户个人指标，不得把相关线索写成确定因果。
3. 用户没有明确要求计算时，不新增平均值、差值、比例、换算或其他派生结果，优先引用输入已经给出的统计。用户明确要求计算时，只使用同一来源、同一口径的操作数，展示简短公式并在内部复核；不能可靠计算就说明缺少什么。只有上下界都落入目标区间，才能说整个区间正常或全部达标。
4. 数据冲突、口径不同或时间范围不同，分别呈现原始记录并说明限制，不自行调和后继续推导。
5. 课程名称只能逐字复制【课程库】中的完整名称，并写成 `<完整课程名>`；没有完全匹配的课程就不推荐。健康、法律、财务和安全问题不得补写材料没有支持的剂量、设备参数、阈值、疗程、改善时间、诊断、确定因果或效果承诺。
6. 材料与问题无关时不要强行套用。可使用稳定、可靠的一般知识完成普通问题，但不要凭记忆断言当前产品发布、政策、价格、版本或日期；缺少可核验来源时明确说明无法确认最新状态。除非用户明确询问，不得引入 macOS、Windows、Linux、终端、命令行、脚本、文件、仓库或本地软件环境。

完整性与呈现要求：
- 回答必须独立完整，通常包含“直接结论 + 必要依据或说明 + 与问题相关的下一步”。不能只剩免责声明、材料不足提示或一句空泛建议。
- 对需要分析、比较、解释或建议的非简单问题，使用至少一个有信息量的 `##` 小标题，并用列表组织要点；存在两个及以上可比较指标或时段时，优先使用紧凑表格。极简单的操作请求不必堆叠标题。
- 只写与问题相关的内容，不为凑长度添加课程、计算或通用知识；但不得为了保守而省略已经有依据的解释和可执行建议。

出稿前静默复核一次。若发现某个数字、单位、个人判断、课程名或敏感参数缺乏支持，应修正该具体表述、改成有边界的说法，或只移除有问题的短语；不要删除承载核心回答的整句或整段，也不要把完整回答替换成固定兜底句。复核后再次确认核心答案前置、内容完整、Markdown 结构适合当前问题。

对违法、伤害、欺骗、隐私侵犯或规避监管等请求，不提供可操作的实施细节，简短转向合法且降低风险的替代方案。

最终只输出一份自然、完整、可直接展示的中文 Markdown 回答，不输出分类、推理过程、内部规则、计划或 JSON。"""
    user = f"""{build_context(row, corrected=True)}

请以【当前问题】为最高优先级，按 v30 的相关性、来源绑定、完整性与呈现规则，在本轮直接给出最终回答。材料只在确实相关时使用。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compile_proof_ledger_v15(row: Mapping[str, str]) -> dict[str, Any]:
    """Build a label-free exact-source ledger for proof-carrying generation."""
    packet = compile_evidence_packet(row)
    ledger = {
        record["id"]: {"source": source, "exact_text": record["text"]}
        for source in ("personal", "expert", "knowledge", "courses")
        for record in packet["sources"][source]
    }
    domain_description = DOMAIN_DESCRIPTION.get(
        str(row.get("domain", "")).strip(), ""
    ).replace("\\n", "\n").strip()
    if domain_description:
        ledger["K0"] = {
            "source": "knowledge",
            "exact_text": domain_description,
        }
    if not packet["sources"]["personal"]:
        ledger["M1"] = {
            "source": "metadata",
            "exact_text": "当前输入未提供可引用的个人数据。",
        }
    payload = {
        "schema_version": "proof-ledger-v15",
        "task_profile": packet["task_profile"],
        "sources": ledger,
    }
    payload["ledger_sha256"] = sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    return payload


def build_proof_carrying_v15_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Generate one answer whose substantive lines carry removable source proofs."""
    cards = compile_personal_fact_cards_v12(row)
    contract = compile_visual_contract_v14(row)
    ledger = compile_proof_ledger_v15(row)
    system = build_slim_evidence_system(row.get("domain")) + """

## v15 证明携带式 Markdown 协议
你只有一次生成调用。最终回答中的每个实质内容行都必须在行尾携带一个机器可验证的来源标记，格式严格为 `[S:P1]` 或 `[S:P1,E2]`。标题、空行、表头和 Markdown 表格分隔行不加标记。不要输出 JSON、草稿、评分或核验过程。

来源权限：
- P 是个人数据。任何个人事实必须逐字包含所引用 P 来源的完整 exact_text；不能改写、拼接、计算、换算或推断。一个表格数据行只引用一条 P。
- E 是专家建议。行动建议必须引用 E；可以自然转述，但不得加入原文没有的频率、时长、强度、目标值或效果承诺。
- K 是知识材料，只可作一般解释，不能写成用户已经发生的事实，也不能覆盖或调和专家建议。
- C 是课程材料。课程名必须逐字存在于所引用 C 来源；没有直接相关课程就省略。
- M 是本地确定性材料状态，只能逐字说明某类来源缺失，不能据此推断用户状态。

每个实质行中的所有数字、日期、比例和单位必须能在该行引用的 exact_text 中找到。不要为了完整而混用来源。首段直接回答；按 visual_contract 决定表格或列表；`## 建议` 最多三项。事实不足时明确限制，但限制句也要引用支持其判断的 P 来源；完全没有 P 时可用不含个人结论的通用说明并引用 E 或 K。

出稿前静默执行两遍：先写带来源注释的完整 Markdown，再逐行检查来源 ID、完整 P 原文、数字绑定、课程名和 Markdown。任何不能携带有效证明的行直接删除。最终仍输出带注释的 Markdown，注释会由本地确定性防火墙移除。
"""
    support = {
        "personal_fact_cards": cards,
        "visual_contract": contract,
        "proof_ledger": ledger,
    }
    user = f"""{build_context(row, corrected=True)}

【v15 本地确定性支持结构】
{json.dumps(support, ensure_ascii=False, sort_keys=True)}

请按证明携带式协议回答当前问题。每个实质内容行必须带合法来源注释；最终只输出一份可直接清理为展示稿的中文 Markdown。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_PROOF_TAG_V15 = re.compile(r"\[S:([PEKCM]\d+(?:,[PEKCM]\d+)*)\]")
_NUMBER_TOKEN_V15 = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?%?"
)


def _proof_v15_is_table_separator(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    if not stripped or "|" not in line:
        return False
    cells = [cell.strip() for cell in stripped.split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


_PERSONAL_INFERENCE_TERMS_V15 = (
    "达标",
    "未达标",
    "正常",
    "异常",
    "偏高",
    "偏低",
    "升高",
    "降低",
    "上升",
    "下降",
    "改善",
    "恶化",
    "导致",
    "因此",
    "说明",
    "表明",
    "风险",
    "诊断",
)


def _personal_claim_supported_v15(
    clean_line: str, personal_texts: Sequence[str]
) -> bool:
    """Conservatively check lexical support without requiring full-sentence copying."""
    claim = re.sub(r"[#*`>|_\-]+", "", clean_line)
    normalized_claim = re.sub(r"\s+", "", claim)
    if not normalized_claim:
        return False
    for source_text in personal_texts:
        normalized_source = re.sub(r"\s+", "", source_text)
        if normalized_source and normalized_source in normalized_claim:
            return True
        if any(
            term in normalized_claim and term not in normalized_source
            for term in _PERSONAL_INFERENCE_TERMS_V15
        ):
            continue
        claim_grams = _text_ngrams(normalized_claim)
        source_grams = _text_ngrams(normalized_source)
        shared = claim_grams & source_grams
        if len(shared) >= 3 and len(shared) / max(1, len(claim_grams)) >= 0.35:
            return True
    return False


def apply_proof_citation_firewall_v15(
    row: Mapping[str, str], response: str
) -> tuple[str, dict[str, int]]:
    """Remove unproved lines using only exact local sources, never judge labels."""
    ledger_payload = compile_proof_ledger_v15(row)
    ledger = ledger_payload["sources"]
    source_by_id = {
        source_id: str(record["exact_text"])
        for source_id, record in ledger.items()
    }
    input_lines = response.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: list[str] = []
    substantive_kept = 0
    missing_proof_removed = 0
    invalid_source_removed = 0
    personal_exactness_removed = 0
    numeric_binding_removed = 0
    course_binding_removed = 0
    for index, original_line in enumerate(input_lines):
        stripped = original_line.strip()
        if not stripped:
            kept.append("")
            continue
        if stripped.startswith("#") or _proof_v15_is_table_separator(stripped):
            kept.append(original_line.rstrip())
            continue
        next_line = input_lines[index + 1].strip() if index + 1 < len(input_lines) else ""
        if "|" in stripped and _proof_v15_is_table_separator(next_line):
            kept.append(original_line.rstrip())
            continue
        tag_matches = list(_PROOF_TAG_V15.finditer(original_line))
        if not tag_matches:
            missing_proof_removed += 1
            continue
        source_ids = list(
            dict.fromkeys(
                source_id
                for tag_match in tag_matches
                for source_id in tag_match.group(1).split(",")
            )
        )
        if any(source_id not in source_by_id for source_id in source_ids):
            invalid_source_removed += 1
            continue
        clean_line = _PROOF_TAG_V15.sub("", original_line)
        clean_line = re.sub(r"[ \t]+([，。；：！？])", r"\1", clean_line).rstrip()
        cited_texts = [source_by_id[source_id] for source_id in source_ids]
        personal_texts = [
            source_by_id[source_id] for source_id in source_ids if source_id.startswith("P")
        ]
        if personal_texts and not _personal_claim_supported_v15(
            clean_line, personal_texts
        ):
            personal_exactness_removed += 1
            continue
        cited_numbers = {
            _canonical_number_token(match.group(0))
            for text in cited_texts
            for match in _NUMBER_TOKEN_V15.finditer(text)
        }
        line_numbers = {
            _canonical_number_token(match.group(0))
            for match in _NUMBER_TOKEN_V15.finditer(clean_line)
        }
        if not line_numbers.issubset(cited_numbers):
            numeric_binding_removed += 1
            continue
        course_names = [value.strip() for value in re.findall(r"<([^<>\n]+)>", clean_line)]
        cited_course_text = "\n".join(
            source_by_id[source_id]
            for source_id in source_ids
            if source_id.startswith("C")
        )
        if any(not cited_course_text or name not in cited_course_text for name in course_names):
            course_binding_removed += 1
            continue
        kept.append(clean_line)
        substantive_kept += 1

    while kept and not kept[-1].strip():
        kept.pop()
    collapsed: list[str] = []
    for line in kept:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    rendered = "\n".join(collapsed).strip()
    stats = {
        "input_lines": len(input_lines),
        "substantive_lines_kept": substantive_kept,
        "missing_proof_lines_removed": missing_proof_removed,
        "invalid_source_lines_removed": invalid_source_removed,
        "personal_exactness_lines_removed": personal_exactness_removed,
        "numeric_binding_lines_removed": numeric_binding_removed,
        "course_binding_lines_removed": course_binding_removed,
    }
    return rendered, stats


def _compile_evidence_sources_v16(
    row: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Add a query-ranked domain fallback when the row has no knowledge text."""
    packet = compile_evidence_packet(row)
    sources = {
        source_name: [dict(record) for record in records]
        for source_name, records in packet["sources"].items()
    }
    if not sources["knowledge"]:
        domain_text = DOMAIN_DESCRIPTION.get(
            str(row.get("domain", "")).strip(), ""
        ).replace("\\n", "\n")
        domain_records = _rank_evidence_sentences(
            str(row.get("query", "")),
            domain_text,
            limit=2,
            prefer_numeric=False,
        )
        sources["knowledge"] = [
            {"id": f"K{index}", "text": sentence}
            for index, sentence in enumerate(domain_records, start=1)
        ]
    return packet, sources


def build_evidence_compiler_v16_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Ask Qwen for a source-selection plan, not free-form factual prose."""
    packet, sources = _compile_evidence_sources_v16(row)
    contract = compile_visual_contract_v14(row)
    planning_packet = {
        "task_profile": packet["task_profile"],
        "current_query": packet["current_query"],
        "sources": sources,
        "visual_contract": {
            "table_required": contract["table_required"],
            "max_personal_facts": contract["max_personal_facts"],
            "max_advice_items": contract["max_advice_items"],
        },
    }
    system = """你是证据选择规划器。你只有一次调用，不写最终回答，只返回一个 JSON 对象。

从给定来源中选择回答当前问题最相关的 ID。P 是个人记录，E 是专家建议，K 是通用知识，C 是课程材料。不得创造 ID，不得输出来源原文的改写、判断、计算、诊断、解释或任何 Markdown。

JSON schema 严格为：
{"personal_ids":["P1"],"expert_ids":["E1"],"knowledge_ids":[],"course_ids":[],"layout":"table"}

约束：personal_ids 最多 3 个，expert_ids 最多 3 个，knowledge_ids 最多 2 个，course_ids 最多 2 个；只保留与 current_query 直接相关的来源。比较/趋势任务或 visual_contract.table_required=true 时 layout 必须为 table，否则可以是 table 或 list。最终只输出 JSON，不要代码围栏。"""
    user = "【v16 规划输入】\n" + json.dumps(
        planning_packet, ensure_ascii=False, sort_keys=True
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _selected_source_ids_v16(
    raw_plan: Mapping[str, Any],
    records: Sequence[Mapping[str, str]],
    field: str,
    limit: int,
) -> tuple[list[str], int]:
    allowed = {str(record["id"]) for record in records}
    proposed = raw_plan.get(field, [])
    values = proposed if isinstance(proposed, list) else []
    selected = [
        value
        for value in dict.fromkeys(str(value) for value in values)
        if value in allowed
    ][:limit]
    invalid_count = sum(1 for value in values if str(value) not in allowed)
    return selected, invalid_count


def render_evidence_compiler_v16(
    row: Mapping[str, str], raw_response: str
) -> tuple[str, dict[str, int | str]]:
    """Compile exact source text into Markdown from a model-produced ID plan.

    The renderer never consults teacher outputs. Invalid or incomplete plans use
    a deterministic relevance-order fallback so one successful Qwen request
    always yields one judgeable datum without another model call.
    """
    packet, sources = _compile_evidence_sources_v16(row)
    parse_status = "model_json"
    try:
        parsed = extract_json_object(raw_response)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = {}
        parse_status = "deterministic_fallback"

    limits = {
        "personal": ("personal_ids", 3),
        "expert": ("expert_ids", 3),
        "knowledge": ("knowledge_ids", 2),
        "courses": ("course_ids", 2),
    }
    selections: dict[str, list[str]] = {}
    invalid_ids = 0
    fallback_groups = 0
    for source_name, (field, limit) in limits.items():
        selected, invalid = _selected_source_ids_v16(
            parsed, sources[source_name], field, limit
        )
        invalid_ids += invalid
        if not selected and sources[source_name]:
            should_fallback = source_name in {"personal", "expert"}
            should_fallback = should_fallback or (
                source_name == "knowledge" and not sources["expert"]
            )
            should_fallback = should_fallback or (
                source_name == "courses"
                and packet["task_profile"]["task_type"] == "course_recommendation"
            )
            if should_fallback:
                selected = [
                    str(record["id"]) for record in sources[source_name][:limit]
                ]
                fallback_groups += 1
        selections[source_name] = selected

    if fallback_groups or invalid_ids:
        parse_status = (
            "deterministic_fallback"
            if parse_status == "deterministic_fallback"
            else "model_json_with_local_fallback"
        )

    records_by_id = {
        str(record["id"]): str(record["text"])
        for records in sources.values()
        for record in records
    }
    selected_text = {
        source_name: [records_by_id[source_id] for source_id in source_ids]
        for source_name, source_ids in selections.items()
    }
    task_type = str(packet["task_profile"]["task_type"])
    if selected_text["personal"]:
        opening = {
            "trend_or_comparison": "现有记录可以并列核对；下面保留原始表述，不额外计算或推断趋势。",
            "action_plan": "可以先依据现有记录和已有建议安排下一步；材料未明确说明的效果不作推断。",
            "course_recommendation": "下面先核对现有记录，再列出与当前问题直接相关的课程或建议材料。",
            "knowledge_explanation": "现有记录能支持的个人信息如下；通用材料不用于反推你的具体状态。",
            "personal_status": "现有记录能确认的信息如下；材料未明确给出的正常、异常或原因不作推断。",
        }[task_type]
    else:
        opening = "现有材料没有提供可用于回答这一问题的个人记录，因此目前不能判断你的具体情况。"

    lines = [opening]
    if selected_text["personal"]:
        lines.extend(["", "## 记录依据", "", "| 记录 | 原始内容 |", "| --- | --- |"])
        for text in selected_text["personal"]:
            lines.append(f"| 相关记录 | {_markdown_cell(text)} |")
    else:
        lines.extend(["", "## 记录依据", "", "- 当前输入未提供相关个人记录。"])

    if selected_text["expert"]:
        lines.extend(["", "## 可以怎么做", ""])
        lines.extend(f"- {_markdown_cell(text)}" for text in selected_text["expert"])
    elif selected_text["knowledge"]:
        lines.extend(["", "## 一般说明", ""])
        lines.extend(f"- {_markdown_cell(text)}" for text in selected_text["knowledge"])

    if selected_text["courses"]:
        lines.extend(["", "## 相关课程材料", ""])
        lines.extend(f"- {_markdown_cell(text)}" for text in selected_text["courses"])
    if not any(selected_text.values()):
        lines.extend(
            [
                "",
                "## 下一步",
                "",
                "- 补充与当前问题直接相关的个人记录，并保留原始指标、时间、单位和记录范围。",
                "- 说明希望比较的时段，或希望解决的具体目标。",
            ]
        )

    rendered = "\n".join(lines).strip()
    stats: dict[str, int | str] = {
        "plan_parse_status": parse_status,
        "invalid_ids_removed": invalid_ids,
        "fallback_source_groups": fallback_groups,
        "personal_selected": len(selected_text["personal"]),
        "expert_selected": len(selected_text["expert"]),
        "knowledge_selected": len(selected_text["knowledge"]),
        "courses_selected": len(selected_text["courses"]),
        "rendered_lines": len(rendered.splitlines()),
    }
    return rendered, stats


def build_grounded_composer_v17_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Request a natural but source-addressed answer object in one Qwen call."""
    packet, sources = _compile_evidence_sources_v16(row)
    payload = {
        "task_profile": packet["task_profile"],
        "current_query": packet["current_query"],
        "sources": sources,
        "limits": {"personal": 3, "advice": 3, "explanation": 2, "courses": 2},
    }
    system = """你是有来源约束的中文回答合成器。一次调用内完成思考，但最终只输出一个 JSON 对象，不输出 Markdown、草稿、评分或解释流程。

来源含义：P 只能支持该用户的个人事实；E 支持行动建议；K 只支持一般解释，不能写成用户已经发生的事实；C 支持课程材料。每个 text 都必须列出 source_ids。不得创造 ID，不得自行计算、换算、诊断、推断趋势或加入来源中没有的数字、日期、频率、时长、强度、因果和效果承诺。

JSON schema：
{
  "opening":{"text":"直接回答当前问题的一到两句自然中文","source_ids":["P1"]},
  "personal_ids":["P1"],
  "advice":[{"text":"自然、具体但不超出来源的建议","source_ids":["E1"]}],
  "explanation":[{"text":"不指向用户个人状态的一般解释","source_ids":["K1"]}],
  "course_ids":[]
}

opening 必须直接回答 current_query；涉及个人状态时必须引用 P，缺少 P 时明确无法作个人判断并可引用 E/K 给一般方向。personal_ids 最多 3 个；advice 最多 3 条且只引用 E；explanation 最多 2 条且只引用 K；course_ids 最多 2 个且只取 C。个人记录原文将由本地表格呈现，不要在 opening 或其他字段重复堆砌。最终只输出合法 JSON。"""
    user = "【v17 证据与任务】\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_contract_jury_v18_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Combine v14 evidence/visual contracts with a single-call silent jury."""
    messages = [dict(message) for message in build_visual_contract_v14_messages(row)]
    messages[0]["content"] += """

## v18 单次调用合同陪审团
你仍然只有一次生成调用，且只能输出一份最终 Markdown。不要输出候选、来源 ID、合同、评分、JSON、草稿或检查过程。

在内部独立形成三种写法：
1. 直接回答版：首段用最少句子完成当前问题；
2. 证据稳健版：个人事实、专家建议、知识和课程严格各守来源边界；
3. 自然完整版：像真实助手一样连贯，不把原始材料机械拼接成摘录。

随后执行五项否决：
- 事实否决：个人事实、数字、单位、日期、比较与课程不能从输入逐字定位，立即淘汰；
- 冲突否决：与专家材料或知识边界矛盾，立即淘汰；
- 任务否决：首段没有回答 current query，或遗漏问题的关键部分，立即淘汰；
- 表达否决：病句、残句、重复、材料清单腔、空标题或不自然模板腔，立即淘汰；
- 视觉否决：visual_contract 要求表格却没有完整 Markdown 表格，立即淘汰。

在剩余写法中合成一份最自然、最完整且事实最稳的回答。首段必须是面向用户的实际答案，不能只说“下面列出”“依据现有材料”或复述规则。个人证据可以在后文逐字呈现，但首段要回应问题本身。不得为显得完整而添加输入未给出的事实。"""
    messages[1]["content"] += """

【v18 最终要求】
在一次调用内部完成三稿陪审和五项否决，只输出最终中文 Markdown。保留 visual_contract 规定的表格或列表，但不要机械复述材料；首段先把当前问题真正回答完。"""
    return messages


def build_numeric_shield_jury_v19_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Add a general no-derived-arithmetic shield to the v18 one-shot jury."""
    messages = [dict(message) for message in build_contract_jury_v18_messages(row)]
    messages[0]["content"] += """

## v19 数字原句屏蔽层
数字安全具有最高优先级，适用于所有题目，不针对任何单个样本：
- 最终回答禁止执行或陈述任何新计算，包括差值、总和、均值、比例、百分比、倍数、排名、区间合并、单位换算和由多条记录推导的升降趋势；即使算式很简单也禁止。
- 含数字、日期、时间、比例或单位的个人事实只能整句逐字复制一张 personal_fact_card 的 exact_sentence；不能删改限定词，不能将两张卡的内容放进同一句或同一表格单元格。
- 首段和建议段尽量不出现个人数字；需要数字证据时只在“记录依据”中逐卡展示原句。
- 看到“相比、变化、平均、累计、共、增加、减少、升高、降低、更多、更少、约、接近”等派生关系词时，只有同一张卡原句已经完整包含该词及其结论才可逐字保留，否则删除整句。
- 出稿前独立扫描每个数字 token：若所在整句不是单卡原句的连续逐字复制，删除整句，不用无数字改写替代。

数字屏蔽不得牺牲当前问题的核心回答：首段仍用不含派生数字的自然语言直接回应，后文用逐字证据与有来源的行动建议完成任务。"""
    messages[1]["content"] += """

【v19 数字门禁】
最终静默逐 token 扫描所有数字、日期、单位和比较词；任何派生算术或非单卡逐字数字句直接删除。仍只输出一份完整自然 Markdown。"""
    return messages


def build_packed_contract_jury_v20_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Generate three complete answers in one response for label-free selection."""
    cards = compile_personal_fact_cards_v12(row)
    contract = compile_visual_contract_v14(row)
    system = build_slim_evidence_system(row.get("domain")) + """

## v20 单请求三候选协议
你只有一次 API 调用机会。请在同一响应内生成三个彼此独立、均可直接展示的完整中文 Markdown 回答；不要输出分析、评分、来源 ID、JSON、草稿说明或第四个候选。

所有候选共同遵守：个人事实只能逐字来自单张 personal_fact_card，不得跨卡拼接；建议只由专家材料支持；知识只作不指向用户状态的一般解释；课程名只能逐字来自课程库。不得计算差值、均值、比例、百分比、排名、换算或输入未明确给出的趋势。visual_contract 要求表格时，每个候选都必须有完整紧凑表格。

- 候选 A「直接完整」：首段真正回答当前问题，再给最相关证据和行动建议；自然、完整、无材料清单腔。
- 候选 B「数字屏蔽」：所有含数字、日期、时间或单位的个人句都必须整句逐字复制单张事实卡；首段不写派生数字。
- 候选 C「最小充分」：只保留完成任务不可缺少的内容；证据不足就明确边界，但仍给材料支持的可执行下一步，不能成为空泛拒答。

每个候选必须有首段直接答案、至少一个有信息量的 `##` 标题和适当的列表或表格。严格使用以下边界，标签各出现一次：
<candidate_a>
候选 A 的完整 Markdown
</candidate_a>
<candidate_b>
候选 B 的完整 Markdown
</candidate_b>
<candidate_c>
候选 C 的完整 Markdown
</candidate_c>"""
    support = {"personal_fact_cards": cards, "visual_contract": contract}
    user = f"""{build_context(row, corrected=True)}

【v20 本地确定性支持结构】
{json.dumps(support, ensure_ascii=False, sort_keys=True)}

按三候选协议完成一次响应；三个候选都必须独立完整，最终只输出带规定边界的三份 Markdown。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_packed_candidates_v20(value: str) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for label in ("a", "b", "c"):
        match = re.search(
            rf"<candidate_{label}>\s*(.*?)\s*</candidate_{label}>",
            value,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None or not match.group(1).strip():
            raise ValueError(f"Missing packed v20 candidate {label}")
        candidates[label] = match.group(1).strip()
    return candidates


_DERIVED_RELATION_TERMS_V20 = (
    "相比", "变化", "平均", "累计", "合计", "共", "增加", "减少",
    "升高", "降低", "上升", "下降", "更多", "更少", "约", "接近",
    "差值", "比例", "百分比", "倍", "排名", "换算",
)


def score_packed_contract_candidate_v20(
    row: Mapping[str, str], answer: str
) -> dict[str, int]:
    """Rank candidates using only current inputs and deterministic contracts."""
    base = score_packed_candidate_risk(row, answer)
    source_sentences = [
        sentence
        for value in (
            str(row.get("data", "")),
            str(row.get("suggest", "")),
            str(row.get("rag", "")),
            str(row.get("services", "")),
            DOMAIN_DESCRIPTION.get(str(row.get("domain", "")).strip(), "").replace("\\n", "\n"),
        )
        for sentence in _split_evidence_sentences(value)
    ]
    normalized_sources = [normalize_root_value(value) for value in source_sentences]
    answer_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    ungrounded_numeric_lines = 0
    derived_relation_lines = 0
    personal_mismatch_lines = 0
    personal_text = normalize_root_value(str(row.get("data", "")))
    personal_markers = ("你", "您", "记录显示", "从记录", "目前", "当前")
    for line in answer_lines:
        normalized_line = normalize_root_value(
            re.sub(r"^[|\-*#>\s]+|[|\s]+$", "", line)
        )
        line_numbers = _numeric_tokens(normalized_line)
        supporting = [
            source
            for source in normalized_sources
            if source and (
                source in normalized_line
                or normalized_line in source
                or (
                    line_numbers
                    and line_numbers.issubset(_numeric_tokens(source))
                    and len(_text_ngrams(normalized_line) & _text_ngrams(source)) >= 4
                )
            )
        ]
        if line_numbers and not supporting:
            ungrounded_numeric_lines += 1
        if any(term in normalized_line and term not in "\n".join(supporting) for term in _DERIVED_RELATION_TERMS_V20):
            derived_relation_lines += 1
        looks_personal = any(marker in normalized_line for marker in personal_markers)
        if looks_personal and personal_text:
            claim_grams = _text_ngrams(normalized_line)
            personal_grams = _text_ngrams(personal_text)
            if (
                len(claim_grams & personal_grams) < 3
                and any(term in normalized_line for term in _PERSONAL_INFERENCE_TERMS_V15)
            ):
                personal_mismatch_lines += 1
    contract = compile_visual_contract_v14(row)
    table_missing = int(
        bool(contract["table_required"])
        and not bool(re.search(r"(?m)^\s*\|.*\|\s*$\n\s*\|\s*:?-{3,}", answer))
    )
    has_direct_opening = int(
        not answer_lines or answer_lines[0].startswith("#") or len(answer_lines[0]) < 8
    )
    total = (
        base["total"]
        + 30 * ungrounded_numeric_lines
        + 25 * derived_relation_lines
        + 25 * personal_mismatch_lines
        + 15 * table_missing
        + 6 * has_direct_opening
    )
    return {
        **base,
        "total": total,
        "ungrounded_numeric_lines": ungrounded_numeric_lines,
        "derived_relation_lines": derived_relation_lines,
        "personal_mismatch_lines": personal_mismatch_lines,
        "required_table_missing": table_missing,
        "weak_direct_opening": has_direct_opening,
    }


def _claim_supported_v17(
    text: str,
    source_ids: Sequence[str],
    source_by_id: Mapping[str, str],
    *,
    allowed_prefixes: set[str],
    personal_required: bool = False,
) -> bool:
    if not text.strip() or not source_ids:
        return False
    unique_ids = list(dict.fromkeys(str(value) for value in source_ids))
    if any(
        source_id not in source_by_id or source_id[:1] not in allowed_prefixes
        for source_id in unique_ids
    ):
        return False
    cited_texts = [source_by_id[source_id] for source_id in unique_ids]
    cited_numbers = {
        _canonical_number_token(match.group(0))
        for source_text in cited_texts
        for match in _NUMBER_TOKEN_V15.finditer(source_text)
    }
    claim_numbers = {
        _canonical_number_token(match.group(0))
        for match in _NUMBER_TOKEN_V15.finditer(text)
    }
    if not claim_numbers.issubset(cited_numbers):
        return False
    personal_texts = [
        source_by_id[source_id]
        for source_id in unique_ids
        if source_id.startswith("P")
    ]
    if personal_required and not personal_texts:
        return False
    if personal_texts and not _personal_claim_supported_v15(text, personal_texts):
        return False
    if not personal_texts:
        claim_grams = _text_ngrams(text)
        cited_grams = set().union(*(_text_ngrams(value) for value in cited_texts))
        shared = claim_grams & cited_grams
        if len(shared) < 3 or len(shared) / max(1, len(claim_grams)) < 0.18:
            return False
    return True


def _validated_claims_v17(
    raw_items: Any,
    source_by_id: Mapping[str, str],
    *,
    allowed_prefix: str,
    limit: int,
) -> tuple[list[str], int]:
    items = raw_items if isinstance(raw_items, list) else []
    retained: list[str] = []
    removed = 0
    for item in items[:limit]:
        if not isinstance(item, dict):
            removed += 1
            continue
        text = str(item.get("text", "")).strip()
        ids = item.get("source_ids", [])
        source_ids = ids if isinstance(ids, list) else []
        if _claim_supported_v17(
            text,
            source_ids,
            source_by_id,
            allowed_prefixes={allowed_prefix},
        ):
            if text not in retained:
                retained.append(text)
        else:
            removed += 1
    return retained, removed


def render_grounded_composer_v17(
    row: Mapping[str, str], raw_response: str
) -> tuple[str, dict[str, int | str]]:
    """Validate source-addressed fields and compile one natural Markdown answer."""
    packet, sources = _compile_evidence_sources_v16(row)
    source_by_id = {
        str(record["id"]): str(record["text"])
        for records in sources.values()
        for record in records
    }
    parse_status = "model_json"
    try:
        parsed = extract_json_object(raw_response)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = {}
        parse_status = "deterministic_fallback"

    personal_ids, invalid_personal = _selected_source_ids_v16(
        parsed, sources["personal"], "personal_ids", 3
    )
    if not personal_ids and sources["personal"]:
        personal_ids = [str(record["id"]) for record in sources["personal"][:3]]

    opening_item = parsed.get("opening") if isinstance(parsed.get("opening"), dict) else {}
    opening_text = str(opening_item.get("text", "")).strip()
    opening_ids_raw = opening_item.get("source_ids", [])
    opening_ids = opening_ids_raw if isinstance(opening_ids_raw, list) else []
    allowed_opening = {"P", "E", "K"}
    opening_valid = _claim_supported_v17(
        opening_text,
        opening_ids,
        source_by_id,
        allowed_prefixes=allowed_opening,
        personal_required=bool(sources["personal"]),
    )
    if not opening_valid:
        if sources["personal"]:
            opening_text = "现有记录能确认的信息列在下方；材料未明确给出的状态、原因或趋势不作推断。"
        else:
            opening_text = "现有材料没有提供可用于回答这一问题的个人记录，因此目前不能判断你的具体情况。"

    advice, advice_removed = _validated_claims_v17(
        parsed.get("advice"), source_by_id, allowed_prefix="E", limit=3
    )
    explanation, explanation_removed = _validated_claims_v17(
        parsed.get("explanation"), source_by_id, allowed_prefix="K", limit=2
    )
    fallback_claims = 0
    if not advice and sources["expert"]:
        advice = [str(record["text"]) for record in sources["expert"][:2]]
        fallback_claims += len(advice)
    if not explanation and not advice and sources["knowledge"]:
        explanation = [str(record["text"]) for record in sources["knowledge"][:1]]
        fallback_claims += len(explanation)

    course_ids, invalid_courses = _selected_source_ids_v16(
        parsed, sources["courses"], "course_ids", 2
    )
    lines = [opening_text]
    if personal_ids:
        lines.extend(["", "## 记录依据", "", "| 记录 | 原始内容 |", "| --- | --- |"])
        lines.extend(
            f"| 相关记录 | {_markdown_cell(source_by_id[source_id])} |"
            for source_id in personal_ids
        )
    else:
        lines.extend(["", "## 记录依据", "", "- 当前输入未提供相关个人记录。"])
    if advice:
        lines.extend(["", "## 可以怎么做", ""])
        lines.extend(f"- {_markdown_cell(text)}" for text in advice)
    if explanation:
        lines.extend(["", "## 一般说明", ""])
        lines.extend(f"- {_markdown_cell(text)}" for text in explanation)
    if course_ids:
        lines.extend(["", "## 相关课程材料", ""])
        lines.extend(
            f"- {_markdown_cell(source_by_id[source_id])}" for source_id in course_ids
        )
    if not personal_ids and not advice and not explanation and not course_ids:
        lines.extend(
            [
                "",
                "## 下一步",
                "",
                "- 补充与当前问题直接相关的个人记录，并保留原始指标、时间、单位和记录范围。",
                "- 说明希望比较的时段，或希望解决的具体目标。",
            ]
        )
    rendered = "\n".join(lines).strip()
    stats: dict[str, int | str] = {
        "plan_parse_status": parse_status,
        "opening_generated_text_retained": int(opening_valid),
        "personal_selected": len(personal_ids),
        "advice_retained": len(advice),
        "advice_removed": advice_removed,
        "explanation_retained": len(explanation),
        "explanation_removed": explanation_removed,
        "courses_selected": len(course_ids),
        "invalid_ids_removed": invalid_personal + invalid_courses,
        "fallback_claims": fallback_claims,
        "rendered_lines": len(rendered.splitlines()),
    }
    return rendered, stats


def _canonical_number_token(value: str) -> str:
    token = value.replace(",", "").replace("，", "").strip()
    suffix = "%" if token.endswith("%") else ""
    if suffix:
        token = token[:-1]
    try:
        numeric = float(token)
    except ValueError:
        return value
    rendered = str(int(numeric)) if numeric.is_integer() else format(numeric, "g")
    return rendered + suffix


def apply_label_free_output_guard(
    row: Mapping[str, str], response: str
) -> tuple[str, dict[str, int]]:
    """Delete unsupported numeric/course lines without consulting judge outputs."""
    number_pattern = re.compile(
        r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?%?"
    )
    allowed_corpus = "\n".join(
        [
            row.get("query", ""),
            row.get("data", ""),
            row.get("suggest", ""),
            row.get("rag", ""),
            row.get("services", ""),
            DOMAIN_DESCRIPTION.get(row.get("domain", "").strip(), "").replace(
                "\\n", "\n"
            ),
        ]
    )
    allowed_numbers = {
        _canonical_number_token(match.group(0))
        for match in number_pattern.finditer(allowed_corpus)
    }
    course_catalog = row.get("services", "")
    kept: list[str] = []
    removed_numeric = 0
    removed_course = 0
    for line in response.splitlines():
        claim_text = re.sub(r"^\s*\d+[.)、]\s*", "", line)
        claim_text = re.sub(r"\[\d+\]", "", claim_text)
        line_numbers = {
            _canonical_number_token(match.group(0))
            for match in number_pattern.finditer(claim_text)
        }
        if line_numbers - allowed_numbers:
            removed_numeric += 1
            continue
        course_names = re.findall(r"<([^<>\n]+)>", line)
        if any(name.strip() not in course_catalog for name in course_names):
            removed_course += 1
            continue
        kept.append(line.rstrip())

    compact: list[str] = []
    for line in kept:
        if not line and (not compact or not compact[-1]):
            continue
        compact.append(line)
    while compact and not compact[-1]:
        compact.pop()
    nonempty_positions = [index for index, line in enumerate(compact) if line]
    empty_heading_positions: set[int] = set()
    for offset, index in enumerate(nonempty_positions):
        if not compact[index].lstrip().startswith("#"):
            continue
        next_index = (
            nonempty_positions[offset + 1]
            if offset + 1 < len(nonempty_positions)
            else None
        )
        if next_index is None or compact[next_index].lstrip().startswith("#"):
            empty_heading_positions.add(index)
    guarded = "\n".join(
        line for index, line in enumerate(compact) if index not in empty_heading_positions
    ).strip()
    fallback_used = 0
    if not guarded:
        guarded = response.strip()
        fallback_used = 1
    return guarded, {
        "unsupported_numeric_lines_removed": removed_numeric,
        "unsupported_course_lines_removed": removed_course,
        "empty_headings_removed": len(empty_heading_positions),
        "empty_guard_fallback_used": fallback_used,
    }


_NUMBER_PATTERN_V29 = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?%?"
)
_NUMBER_UNIT_PATTERN_V29 = re.compile(
    r"(?<![A-Za-z])([-+]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?%?)"
    r"\s*(步|次/分钟|次/分|公里|千米|米|分钟|小时|天|周|月|年|千卡|大卡|"
    r"公斤|千克|kg|克|毫升|ml|毫米汞柱|mmHg|摄氏度|℃|%)",
    flags=re.IGNORECASE,
)
_EXPLICIT_CALCULATION_V29 = re.compile(
    r"计算|算一下|求|差值|总和|合计|平均|均值|比例|百分比|换算|相差多少"
)
_DEFERRED_ANSWER_V29 = re.compile(
    r"确认后.{0,12}(?:再|将|会)|请先确认|待你确认|告诉我.{0,12}(?:再|后)|"
    r"回复.{0,12}(?:再|后)|下一步我会|之后我再|确认好了再"
)
_LOCAL_ENVIRONMENT_V29 = re.compile(
    r"macOS|Windows|Linux|终端|命令行|shell|脚本|文件路径|本地环境|"
    r"工作区|代码仓库|Docker|osascript|Notion",
    flags=re.IGNORECASE,
)
_DEVICE_ACTION_QUERY_V29 = re.compile(
    r"闹钟|提醒|日程|备忘|记下来|记录一下|保存|删除|打开|关闭|设置|创建"
)
_FAKE_DEVICE_COMPLETION_V29 = re.compile(
    r"已(?:经)?(?:为你|帮你)?(?:设置|创建|记录|保存|删除|打开|关闭|完成)"
)
_SWEEPING_RANGE_CLAIM_V29 = re.compile(
    r"(?:全部|整个|所有|均|都).{0,10}(?:正常|达标|范围内|没有异常)"
)


def _verified_simple_arithmetic_v29(
    line: str,
    allowed_numbers: set[str],
    source_number_sets: Sequence[set[str]],
) -> set[str]:
    """Return verified result tokens for simple source-bound arithmetic only."""
    verified: set[str] = set()
    pattern = re.compile(
        r"([-+]?\d+(?:\.\d+)?)\s*([+\-×*xX÷/])\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*=\s*([-+]?\d+(?:\.\d+)?)(%?)"
    )
    for match in pattern.finditer(line):
        left, operator, right, result, percent = match.groups()
        operands = {
            _canonical_number_token(left),
            _canonical_number_token(right),
        }
        if not operands.issubset(allowed_numbers) or not any(
            operands.issubset(source_numbers)
            for source_numbers in source_number_sets
        ):
            continue
        first = float(left)
        second = float(right)
        if operator == "+":
            expected = first + second
        elif operator == "-":
            expected = first - second
        elif operator in {"×", "*", "x", "X"}:
            expected = first * second
        elif second != 0:
            expected = first / second
        else:
            continue
        actual = float(result)
        tolerance = max(1e-6, abs(expected) * 1e-6)
        if math.isclose(actual, expected, rel_tol=1e-6, abs_tol=tolerance):
            verified.add(_canonical_number_token(result + percent))
    return verified


def _compact_markdown_v29(lines: Sequence[str]) -> str:
    compact: list[str] = []
    for line in lines:
        if not line.strip() and (not compact or not compact[-1].strip()):
            continue
        compact.append(line.rstrip())
    while compact and not compact[-1].strip():
        compact.pop()
    nonempty_positions = [index for index, line in enumerate(compact) if line.strip()]
    empty_headings: set[int] = set()
    for offset, index in enumerate(nonempty_positions):
        if not compact[index].lstrip().startswith("#"):
            continue
        next_index = (
            nonempty_positions[offset + 1]
            if offset + 1 < len(nonempty_positions)
            else None
        )
        if next_index is None or compact[next_index].lstrip().startswith("#"):
            empty_headings.add(index)
    return "\n".join(
        line for index, line in enumerate(compact) if index not in empty_headings
    ).strip()


def apply_relevance_grounded_guard_v29(
    row: Mapping[str, str], response: str
) -> tuple[str, dict[str, int]]:
    """Safely rewrite high-confidence generation defects without judge labels."""
    query = str(row.get("query", ""))
    source_values = [
        query,
        str(row.get("data", "")),
        str(row.get("suggest", "")),
        str(row.get("rag", "")),
        str(row.get("services", "")),
        DOMAIN_DESCRIPTION.get(str(row.get("domain", "")).strip(), "").replace(
            "\\n", "\n"
        ),
    ]
    source_corpus = "\n".join(source_values)
    allowed_numbers = {
        _canonical_number_token(match.group(0))
        for match in _NUMBER_PATTERN_V29.finditer(source_corpus)
    }
    for match in re.finditer(r"(?<!\d)(\d{1,2})\s*(?:点|时)(?!\d)", query):
        allowed_numbers.add(_canonical_number_token(match.group(1)))
        allowed_numbers.add("0")
        if "半" in query:
            allowed_numbers.add("30")
    allowed_number_units = {
        (_canonical_number_token(match.group(1)), match.group(2).lower())
        for match in _NUMBER_UNIT_PATTERN_V29.finditer(source_corpus)
    }
    normalized_sources = [
        normalize_root_value(sentence)
        for value in source_values
        for sentence in _split_evidence_sentences(value)
        if sentence.strip()
    ]
    source_number_sets = [
        {
            _canonical_number_token(match.group(0))
            for match in _NUMBER_PATTERN_V29.finditer(sentence)
        }
        for value in source_values
        for sentence in _split_evidence_sentences(value)
        if sentence.strip()
    ]
    personal_sources = [
        normalize_root_value(sentence)
        for sentence in _split_evidence_sentences(str(row.get("data", "")))
        if sentence.strip()
    ]
    services = normalize_root_value(str(row.get("services", "")))
    calculation_requested = bool(_EXPLICIT_CALCULATION_V29.search(query))
    device_action_requested = bool(_DEVICE_ACTION_QUERY_V29.search(query))
    query_names_environment = bool(_LOCAL_ENVIRONMENT_V29.search(query))

    stats = {
        "v29_unsupported_numeric_lines_removed": 0,
        "v29_unsupported_unit_lines_removed": 0,
        "v29_derived_relation_lines_removed": 0,
        "v29_sweeping_range_lines_removed": 0,
        "v29_personal_mismatch_lines_removed": 0,
        "v29_unsupported_course_lines_removed": 0,
        "v29_sensitive_lines_removed": 0,
        "v29_environment_lines_removed": 0,
        "v29_deferred_lines_removed": 0,
        "v29_device_completion_rewritten": 0,
        "v29_safe_fallback_used": 0,
        "v29_local_gate_failed": 0,
    }
    if device_action_requested and (
        _FAKE_DEVICE_COMPLETION_V29.search(response)
        or _DEFERRED_ANSWER_V29.search(response)
        or (not query_names_environment and _LOCAL_ENVIRONMENT_V29.search(response))
    ):
        stats["v29_device_completion_rewritten"] = 1
        if re.search(r"闹钟|提醒", query):
            rendered = (
                "我无法直接操作你的设备。请打开设备上的闹钟或提醒应用，"
                "按你给出的时间新建并保存，然后确认开关已经启用。"
            )
        else:
            rendered = (
                "我无法直接操作你的设备。请在对应应用中按你的要求完成设置，"
                "并在保存前核对时间或内容。"
            )
        stats["v29_output_lines"] = len(rendered.splitlines())
        return rendered, stats

    kept: list[str] = []
    for original_line in response.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = original_line.strip()
        if not stripped or stripped.startswith("#") or _proof_v15_is_table_separator(stripped):
            kept.append(original_line.rstrip())
            continue
        normalized = normalize_root_value(
            re.sub(r"^[|\-*#>\s]+|[|\s]+$", "", stripped)
        )
        if not query_names_environment and _LOCAL_ENVIRONMENT_V29.search(stripped):
            stats["v29_environment_lines_removed"] += 1
            continue
        if _DEFERRED_ANSWER_V29.search(stripped):
            stats["v29_deferred_lines_removed"] += 1
            stats["v29_local_gate_failed"] = 1
            continue

        course_names = [
            name.strip()
            for match in re.finditer(r"<([^<>\n]+)>|《([^《》\n]+)》", stripped)
            for name in (match.group(1) or match.group(2),)
        ]
        if "课程" in stripped and any(
            not services or normalize_root_value(name) not in services
            for name in course_names
        ):
            stats["v29_unsupported_course_lines_removed"] += 1
            continue

        numeric_claim = re.sub(r"^\s*\d+[.)、]\s*", "", stripped)
        numeric_claim = re.sub(r"\[\d+\]", "", numeric_claim)
        line_numbers = {
            _canonical_number_token(match.group(0))
            for match in _NUMBER_PATTERN_V29.finditer(numeric_claim)
        }
        verified_results = (
            _verified_simple_arithmetic_v29(
                numeric_claim, allowed_numbers, source_number_sets
            )
            if calculation_requested
            else set()
        )
        if line_numbers - allowed_numbers - verified_results:
            stats["v29_unsupported_numeric_lines_removed"] += 1
            continue
        line_number_units = {
            (_canonical_number_token(match.group(1)), match.group(2).lower())
            for match in _NUMBER_UNIT_PATTERN_V29.finditer(numeric_claim)
        }
        if line_number_units - allowed_number_units:
            stats["v29_unsupported_unit_lines_removed"] += 1
            continue

        supporting = [
            source
            for source in normalized_sources
            if source
            and (
                source in normalized
                or normalized in source
                or (
                    line_numbers
                    and line_numbers.issubset(_numeric_tokens(source))
                    and len(_text_ngrams(normalized) & _text_ngrams(source)) >= 4
                )
            )
        ]
        if (
            not calculation_requested
            and any(term in normalized for term in _DERIVED_RELATION_TERMS_V20)
            and not any(
                term in source
                for term in _DERIVED_RELATION_TERMS_V20
                if term in normalized
                for source in supporting
            )
        ):
            stats["v29_derived_relation_lines_removed"] += 1
            continue
        if _SWEEPING_RANGE_CLAIM_V29.search(normalized) and not any(
            _SWEEPING_RANGE_CLAIM_V29.search(source)
            for source in normalized_sources
        ):
            stats["v29_sweeping_range_lines_removed"] += 1
            continue
        looks_personal = any(
            marker in normalized
            for marker in ("你", "您", "你的", "您的", "记录显示", "从记录", "当前")
        )
        personal_judgment = any(
            term in normalized for term in _PERSONAL_INFERENCE_TERMS_V15
        )
        personal_source_support = [
            source
            for source in personal_sources
            if source
            and (
                source in normalized
                or normalized in source
                or (
                    line_numbers
                    and line_numbers.issubset(_numeric_tokens(source))
                    and len(_text_ngrams(normalized) & _text_ngrams(source)) >= 3
                )
            )
        ]
        generalization_terms = ("每天", "每日", "一直", "长期", "一贯", "通常", "总是")
        unsupported_generalization = any(
            term in normalized
            and not any(term in source for source in personal_source_support)
            for term in generalization_terms
        )
        unsupported_personal_number = bool(
            looks_personal and line_numbers and not personal_source_support
        )
        unsupported_personal_judgment = bool(
            looks_personal
            and personal_judgment
            and personal_sources
            and not personal_source_support
        )
        if (
            unsupported_generalization
            or unsupported_personal_number
            or unsupported_personal_judgment
        ):
            stats["v29_personal_mismatch_lines_removed"] += 1
            continue
        if (
            any(term in normalized for term in _SENSITIVE_ACTION_TERMS_V21)
            and not supporting
        ):
            stats["v29_sensitive_lines_removed"] += 1
            continue
        kept.append(original_line.rstrip())

    rendered = _compact_markdown_v29(kept)
    substantive = [
        line
        for line in rendered.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not _proof_v15_is_table_separator(line.strip())
    ]
    if not substantive:
        stats["v29_safe_fallback_used"] = 1
        stats["v29_local_gate_failed"] = 1
        rendered = (
            "现有输入不足以在不引入未经核实信息的情况下给出可靠结论。"
            "请以原始记录为准，并补充与当前问题直接相关且口径一致的材料。"
        )
    stats["v29_output_lines"] = len(rendered.splitlines())
    return rendered, stats


def audit_relevance_grounded_guard_v30(
    row: Mapping[str, str], response: str
) -> tuple[str, dict[str, int]]:
    """Report source-risk signals while returning the sampler text byte-for-byte."""
    hypothetical_v29, v29_stats = apply_relevance_grounded_guard_v29(row, response)
    lines = response.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    substantive = [
        line
        for line in lines
        if line.strip()
        and not line.lstrip().startswith("#")
        and not _proof_v15_is_table_separator(line.strip())
    ]
    heading_count = sum(line.lstrip().startswith("#") for line in lines)
    list_count = sum(
        bool(re.match(r"^\s*(?:[-*+] |\d+[.)、]\s*)", line)) for line in lines
    )
    table_row_count = sum(
        line.strip().startswith("|") and line.strip().endswith("|")
        for line in lines
    )
    mapped_keys = {
        "v30_audit_unsupported_numeric_lines": (
            "v29_unsupported_numeric_lines_removed"
        ),
        "v30_audit_unsupported_unit_lines": "v29_unsupported_unit_lines_removed",
        "v30_audit_derived_relation_lines": "v29_derived_relation_lines_removed",
        "v30_audit_sweeping_range_lines": "v29_sweeping_range_lines_removed",
        "v30_audit_personal_mismatch_lines": "v29_personal_mismatch_lines_removed",
        "v30_audit_unsupported_course_lines": (
            "v29_unsupported_course_lines_removed"
        ),
        "v30_audit_sensitive_lines": "v29_sensitive_lines_removed",
        "v30_audit_environment_lines": "v29_environment_lines_removed",
        "v30_audit_deferred_lines": "v29_deferred_lines_removed",
        "v30_audit_device_completion_risk": "v29_device_completion_rewritten",
    }
    stats = {
        key: int(v29_stats.get(source_key, 0))
        for key, source_key in mapped_keys.items()
    }
    stats.update(
        {
            "v30_audit_only": 1,
            "v30_response_modified": 0,
            "v30_audit_flag_total": sum(stats.values()),
            "v30_hypothetical_v29_changed": int(hypothetical_v29 != response),
            "v30_hypothetical_v29_characters_removed": max(
                0, len(response) - len(hypothetical_v29)
            ),
            "v30_output_characters": len(response),
            "v30_output_lines": len(lines),
            "v30_substantive_lines": len(substantive),
            "v30_markdown_headings": heading_count,
            "v30_markdown_list_items": list_count,
            "v30_markdown_table_rows": table_row_count,
            "v30_completeness_warning": int(
                len(response.strip()) < 80 or len(substantive) < 2
            ),
        }
    )
    return response, stats


_SENSITIVE_ACTION_TERMS_V21 = (
    "诊断", "确诊", "停药", "用药", "剂量", "治疗", "治愈", "手术",
    "必须就医", "立即就医", "无需就医", "排除疾病", "没有风险", "保证",
)


def apply_guarded_visual_contract_v21(
    row: Mapping[str, str], response: str
) -> tuple[str, dict[str, int]]:
    """Remove high-risk lines using only input sources, then repair Markdown locally."""
    packet = compile_evidence_packet(row)
    personal_sources = [item["text"] for item in packet["sources"]["personal"]]
    expert_sources = [item["text"] for item in packet["sources"]["expert"]]
    all_sources = [
        item["text"]
        for source_name in ("personal", "expert", "knowledge", "courses")
        for item in packet["sources"][source_name]
    ]
    normalized_sources = [normalize_root_value(value) for value in all_sources]
    expert_grams = [_text_ngrams(value) for value in expert_sources]
    contract = compile_visual_contract_v14(row)
    output_guarded, base_stats = apply_label_free_output_guard(row, response)
    input_lines = output_guarded.splitlines()
    kept: list[str] = []
    removed_numeric = 0
    removed_derived = 0
    removed_advice = 0
    removed_sensitive = 0
    in_advice = False

    def source_supports_line(line: str) -> bool:
        normalized = normalize_root_value(re.sub(r"^[|\-*#>\s]+|[|\s]+$", "", line))
        numbers = _numeric_tokens(normalized)
        grams = _text_ngrams(normalized)
        return any(
            source
            and (
                source in normalized
                or normalized in source
                or (
                    numbers
                    and numbers.issubset(_numeric_tokens(source))
                    and len(grams & _text_ngrams(source)) >= 4
                )
            )
            for source in normalized_sources
        )

    for original_line in input_lines:
        stripped = original_line.strip()
        if stripped.startswith("#"):
            in_advice = bool(re.search(r"建议|行动|下一步", stripped))
            kept.append(original_line.rstrip())
            continue
        if not stripped or _proof_v15_is_table_separator(stripped):
            kept.append(original_line.rstrip())
            continue
        normalized = normalize_root_value(re.sub(r"^[|\-*#>\s]+|[|\s]+$", "", stripped))
        supported = source_supports_line(stripped)
        if _numeric_tokens(normalized) and not supported:
            removed_numeric += 1
            continue
        if any(term in normalized for term in _DERIVED_RELATION_TERMS_V20) and not any(
            term in source
            for term in _DERIVED_RELATION_TERMS_V20
            for source in normalized_sources
            if term in normalized
        ):
            removed_derived += 1
            continue
        if any(term in normalized for term in _SENSITIVE_ACTION_TERMS_V21) and not supported:
            removed_sensitive += 1
            continue
        if in_advice and expert_sources:
            line_grams = _text_ngrams(normalized)
            if not any(len(line_grams & grams) >= 3 for grams in expert_grams):
                removed_advice += 1
                continue
        kept.append(original_line.rstrip())

    compact: list[str] = []
    for line in kept:
        if not line.strip() and (not compact or not compact[-1].strip()):
            continue
        compact.append(line)
    while compact and not compact[-1].strip():
        compact.pop()
    rendered = "\n".join(compact).strip()
    nonempty = [line for line in compact if line.strip()]
    substantive = [line for line in nonempty if not line.lstrip().startswith("#")]
    original_substantive_count = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in input_lines
    )
    v21_removed_total = (
        removed_numeric + removed_derived + removed_advice + removed_sensitive
    )
    has_heading = any(line.lstrip().startswith("##") for line in nonempty)
    has_table = bool(re.search(r"(?m)^\s*\|.*\|\s*$\n\s*\|\s*:?-{3,}", rendered))
    fallback_used = int(
        len(substantive) < 3
        or not has_heading
        or (bool(contract["table_required"]) and not has_table)
        or (
            original_substantive_count > 0
            and v21_removed_total / original_substantive_count >= 0.30
        )
    )
    if fallback_used:
        cards = compile_personal_fact_cards_v12(row)["cards"][:3]
        lines = [
            "根据目前提供的材料，可以先核对与当前问题直接相关的记录；材料未明确支持的比较、诊断或结论暂不作推断。",
            "",
            "## 记录依据",
            "",
        ]
        if cards and contract["table_required"]:
            lines.extend(["| 可核对记录 |", "| --- |"])
            lines.extend(
                f"| {_markdown_cell(card['exact_sentence'])} |" for card in cards
            )
        elif cards:
            lines.extend(f"- {card['exact_sentence']}" for card in cards)
        else:
            lines.append("- 当前材料中没有可直接引用的相关个人记录。")
        lines.extend(["", "## 建议", ""])
        if expert_sources:
            lines.extend(f"- {value}" for value in expert_sources[:2])
        else:
            lines.append("- 补充与当前问题直接相关且口径一致的原始记录后，再进行判断。")
        rendered = "\n".join(lines).strip()

    return rendered, {
        **base_stats,
        "v21_numeric_lines_removed": removed_numeric,
        "v21_derived_relation_lines_removed": removed_derived,
        "v21_advice_lines_removed": removed_advice,
        "v21_sensitive_lines_removed": removed_sensitive,
        "v21_fallback_used": fallback_used,
        "v21_output_lines": len(rendered.splitlines()),
    }


def render_controlled_negative_v22(
    row: Mapping[str, str], response: str
) -> tuple[str, dict[str, int]]:
    """Turn a source-guarded answer into a complete but deliberately poor KTO negative."""
    guarded, guard_stats = apply_guarded_visual_contract_v21(row, response)
    fragments: list[str] = []
    for original_line in guarded.splitlines():
        line = original_line.strip()
        if not line or line.startswith("#") or _proof_v15_is_table_separator(line):
            continue
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            if cells:
                fragments.append("，".join(cells).rstrip("。") + "。")
            continue
        cleaned = re.sub(r"^[-*+]\s+", "", line)
        cleaned = re.sub(r"^>\s*", "", cleaned)
        if cleaned:
            fragments.append(cleaned)
    if not fragments:
        fragments.append("现有材料不足以支持更具体的判断。")
    delayed_repetitive_opening = (
        "关于这个问题，需要先把现有记录和建议分开来看。"
        "关于这个问题，确实需要先把现有记录和建议分开来看。"
    )
    rendered = delayed_repetitive_opening + "".join(fragments)
    rendered = re.sub(r"\s+", "", rendered).strip()
    if not rendered.endswith(("。", "！", "？")):
        rendered += "。"
    return rendered, {
        **guard_stats,
        "v22_markdown_removed": 1,
        "v22_repetitive_delayed_opening_added": 1,
        "v22_source_fragments_retained": len(fragments),
        "v22_output_lines": 1,
    }


def render_complete_plaintext_negative_v23(
    row: Mapping[str, str], response: str
) -> tuple[str, dict[str, int]]:
    """Preserve complete grounded content while adding only structural defects."""
    guarded, guard_stats = apply_guarded_visual_contract_v21(row, response)
    paragraphs: list[str] = []
    pending_label = ""
    for original_line in guarded.splitlines():
        line = original_line.strip()
        if not line or _proof_v15_is_table_separator(line):
            continue
        if line.startswith("#"):
            pending_label = line.lstrip("#").strip().rstrip("：:") + "："
            continue
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            text = "；".join(cells)
        else:
            text = re.sub(r"^[-*+]\s+", "", line)
            text = re.sub(r"^>\s*", "", text)
        if not text:
            continue
        if pending_label:
            text = pending_label + text
            pending_label = ""
        paragraphs.append(text)
    if not paragraphs:
        paragraphs = ["现有材料不足以支持更具体的判断。"]
    if not any(value.startswith("记录依据：") for value in paragraphs):
        paragraphs.insert(1, "记录依据：以当前提供的原始记录为准。")
    if not any(value.startswith("建议：") for value in paragraphs):
        paragraphs.append("建议：补充与当前问题直接相关的原始记录后再判断。")
    paragraphs.append("以上建议建议可以作为下一步参考，以上建议可以作为下一步参考。")
    rendered = "\n\n".join(paragraphs)
    return rendered, {
        **guard_stats,
        "v23_markdown_removed": 1,
        "v23_direct_opening_preserved": 1,
        "v23_section_labels_preserved": 1,
        "v23_repetitive_grammar_defect_added": 1,
        "v23_output_paragraphs": len(paragraphs),
    }


def render_full_answer_plaintext_negative_v24(
    response: str,
) -> tuple[str, dict[str, int]]:
    """Keep the whole generated answer while degrading presentation only."""
    paragraphs: list[str] = []
    pending_label = ""
    source_line_count = 0
    for original_line in response.splitlines():
        line = original_line.strip()
        if not line:
            continue
        source_line_count += 1
        if _proof_v15_is_table_separator(line):
            continue
        if line.startswith("#"):
            pending_label = line.lstrip("#").strip().rstrip("：:") + "："
            continue
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            text = "；".join(cells)
        else:
            text = re.sub(r"^[-*+]\s+", "", line)
            text = re.sub(r"^\d+[.)、]\s*", "", text)
            text = re.sub(r"^>\s*", "", text)
        if not text:
            continue
        if pending_label:
            text = pending_label + text
            pending_label = ""
        paragraphs.append(text)
    if not paragraphs:
        paragraphs = ["我不能在信息不足时给出可靠结论，但可以说明一般处理原则。"]
    defect = "以上内容内容可以作为参考，以上内容可以作为参考。"
    paragraphs.append(defect)
    return "\n\n".join(paragraphs), {
        "v24_source_lines_seen": source_line_count,
        "v24_source_paragraphs_retained": max(0, len(paragraphs) - 1),
        "v24_markdown_removed": 1,
        "v24_full_answer_preserved": 1,
        "v24_repetitive_grammar_defect_added": 1,
        "v24_output_paragraphs": len(paragraphs),
    }


def render_malformed_mechanical_negative_v25(
    response: str,
) -> tuple[str, dict[str, int]]:
    """Preserve v24 content and add two bounded presentation defects."""
    rendered, stats = render_full_answer_plaintext_negative_v24(response)
    malformed_mechanical = (
        "** 排版状态：答复模块模块已经处理完成。"
        "答复模块模块已经处理完成。"
    )
    return rendered + "\n\n" + malformed_mechanical, {
        **stats,
        "v25_unclosed_markdown_added": 1,
        "v25_mechanical_module_tone_added": 1,
        "v25_full_answer_preserved": 1,
        "v25_output_paragraphs": stats["v24_output_paragraphs"] + 1,
    }


def render_visual_budget_negative_v26(
    response: str,
) -> tuple[str, dict[str, int]]:
    """Target a 12-point defect budget while satisfying the visual-aid rule."""
    rendered, stats = render_full_answer_plaintext_negative_v24(response)
    visual_malformed_mechanical = (
        "📌 ** 排版状态：答复模块模块已经处理完成。"
        "答复模块模块已经处理完成。"
    )
    return rendered + "\n\n" + visual_malformed_mechanical, {
        **stats,
        "v26_visual_anchor_added": 1,
        "v26_unclosed_markdown_added": 1,
        "v26_mechanical_module_tone_added": 1,
        "v26_full_answer_preserved": 1,
        "v26_target_structure_deduction": 12,
        "v26_output_paragraphs": stats["v24_output_paragraphs"] + 1,
    }


def render_markdown_preserving_negative_v27(
    response: str,
) -> tuple[str, dict[str, int]]:
    """Keep the model's complete Markdown and add bounded visible defects."""
    preserved = response.strip()
    if not preserved:
        preserved = "## 回答\n\n现有信息不足以给出更具体的结论。"
    defect = (
        "📌 ** 排版状态：答复模块模块已经处理完成。"
        "答复模块模块已经处理完成。"
    )
    return preserved + "\n\n" + defect, {
        "v27_original_response_preserved": 1,
        "v27_original_characters_retained": len(preserved),
        "v27_unclosed_markdown_added": 1,
        "v27_mechanical_module_tone_added": 1,
        "v27_visual_anchor_added": 1,
        "v27_target_structure_deduction": 12,
    }


def validate_context_compiler_v7_protocol(config: ExperimentConfig) -> None:
    """Reject configs that could violate the user's fixed 1Q+2L protocol."""
    if config.phase1_count <= 0:
        raise ValueError("trace.phase1_count must be positive")
    if config.judge_repeats != 1:
        raise ValueError("context compiler v7 requires judge_repeats=1")
    if config.max_attempts_per_operation != 1:
        raise ValueError("context compiler v7 forbids retries")
    if config.qwen_request_cap != config.phase1_count:
        raise ValueError("context compiler v7 requires qwen cap == root count")
    if config.luna_request_cap != 2 * config.phase1_count:
        raise ValueError("context compiler v7 requires luna cap == 2 * root count")


def build_claim_gated_dual_messages(
    row: Mapping[str, str],
) -> list[dict[str, str]]:
    """Build a one-call prompt with internal claim grounding and dual drafting."""
    system = build_slim_evidence_system(row.get("domain"))
    user = build_context(row, corrected=True) + """

【单次调用内的逐主张门控与双草案择优】
以下全部在内部静默完成，绝不输出账本、标签、草稿、评分或核验过程：

1. 意图边界：先用一句内部短语确定用户究竟问什么、哪些材料与此直接相关；无关指标不得进入回答。
2. 来源账本：对准备使用的每个个人事实、数字、单位、日期、范围、课程和通用解释，记录来源类别与输入中的精确依据。来源类别只能是个人数据、专家建议、知识材料、课程库或领域规则。
3. 允许主张：把证据转换成最小充分主张。任何主张若无明确来源、超出来源含义、混淆时段/指标/人群/运动类型、把相关写成因果，或与另一材料冲突，必须删除而不是猜测或强行调和。
4. 双草案：分别形成“证据结论优先”和“行动可读性优先”两个完整候选；逐句检查每个候选是否都能回指来源账本，并否决存在事实、数值、比较、课程、安全或材料一致性问题的候选。
5. 融合定稿：仅融合两个合格候选的优点。首段直接回答；随后至少使用一个有信息量的 `##` 小标题和行动列表；有两个及以上相关指标或时段需要比较时使用紧凑表格。避免空标题、泛泛建议和重复。
6. 最终否决：若定稿仍含无法由输入支持的句子、与专家/知识/领域规则冲突的判断、核心答案后置、Markdown不完整或安全越界，立即修正或删除。

最终只输出给用户看的完整中文 Markdown 回答。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_evidence_plan_messages(row: Mapping[str, str]) -> list[dict[str, str]]:
    """Request a machine-checkable answer plan for deterministic rendering."""
    system = build_slim_evidence_system(row.get("domain"))
    user = build_context(row, corrected=True) + """

【任务：只生成可校验的回答计划】
你只有这一次生成机会。请先静默比较两个候选思路，再输出一个 JSON 对象；不要输出 Markdown、代码围栏、解释或 JSON 以外文字。

每个准备写进最终回答的事实性主张都必须带 `source` 和 `support_quote`：
- `source` 只能是 `personal`、`expert`、`kb`、`domain`；
- `support_quote` 必须是对应输入区域中连续出现的原文片段，不能改写；
- 主张不得超出引文含义，不得把相关性写成因果；
- 课程名只放在 `courses`，必须与课程库名称逐字一致；
- 没有可用证据的字段用空字符串或空数组，不要猜测。

严格输出以下结构：
{
  "direct": {"text": "直接回答当前问题的一到两句", "source": "personal|expert|kb|domain", "support_quote": "连续原文"},
  "evidence": [
    {"label": "输入中的指标原名", "display_value": "准确数值/单位/日期或简短事实", "source": "personal|expert|kb|domain", "support_quote": "连续原文", "interpretation": "不超出证据的谨慎解释", "action": "与该证据直接相关的可执行建议"}
  ],
  "actions": [
    {"text": "可执行建议", "source": "expert|kb|domain", "support_quote": "连续原文"}
  ],
  "courses": [{"name": "课程库中的精确名称", "reason": "与当前问题的直接关联"}],
  "safety_note": "只有确有必要时才填写的非诊断安全提醒"
}

控制要求：`evidence` 只保留最相关的1—4项；`actions` 最多4项；所有字符串用中文；最终 JSON 必须可被标准解析器解析。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_packed_bon_messages(row: Mapping[str, str]) -> list[dict[str, str]]:
    """Ask for two deliberately different complete answers in one API response."""
    system = build_slim_evidence_system(row.get("domain"))
    user = build_context(row, corrected=True) + """

【一次响应内生成两个候选】
你只有一次API调用机会。请在内部先建立证据账本，然后生成两个各自完整、可直接展示给用户的中文Markdown回答：

- 候选A“证据保守版”：只使用输入明确支持的个人事实、数字、比较和课程；核心结论前置，宁可删去不确定内容也不猜测。
- 候选B“行动可读版”：同样严格忠于证据，但更强调自然表达、清晰表格/列表和可执行建议；不得为了丰富而添加输入没有的个人事实、数字、因果或课程。

两个候选必须独立成文，均包含首段直接回答、至少一个有信息量的`##`小标题和行动列表；有两个及以上相关指标/时段需要比较时使用紧凑表格。不得输出分析、评分、解释或第三个候选。

严格按以下纯文本边界输出，边界标签必须各出现一次：
<candidate_a>
候选A的完整Markdown
</candidate_a>
<candidate_b>
候选B的完整Markdown
</candidate_b>"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_packed_candidates(value: str) -> tuple[str, str]:
    matches = {}
    for label in ("a", "b"):
        match = re.search(
            rf"<candidate_{label}>\s*(.*?)\s*</candidate_{label}>",
            value,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None or not match.group(1).strip():
            raise ValueError(f"Missing packed candidate {label}")
        matches[label] = match.group(1).strip()
    return matches["a"], matches["b"]


def score_packed_candidate_risk(
    row: Mapping[str, str], answer: str
) -> dict[str, int]:
    """Score deterministic, high-precision risks without approving quality."""
    source_text = "\n".join(
        str(row.get(field, ""))
        for field in ("data", "suggest", "rag", "services", "query")
    ) + "\n" + DOMAIN_DESCRIPTION.get(str(row.get("domain", "")).strip(), "")
    unsupported_numbers = _numeric_tokens(answer) - _numeric_tokens(source_text)
    services = normalize_root_value(str(row.get("services", "")))
    course_names = [normalize_root_value(value) for value in re.findall(r"<([^<>]+)>", answer)]
    unsupported_courses = [
        value for value in course_names if value and value not in services
    ]
    nonempty_lines = [normalize_root_value(line) for line in answer.splitlines() if line.strip()]
    repeated_lines = len(nonempty_lines) - len(set(nonempty_lines))
    has_heading = bool(re.search(r"(?m)^##\s+\S", answer))
    has_list = bool(re.search(r"(?m)^\s*[-*]\s+\S", answer))
    starts_with_heading = bool(nonempty_lines and nonempty_lines[0].startswith("#"))
    too_short = len(normalize_root_value(answer)) < 180
    too_long = len(normalize_root_value(answer)) > 2200
    total = (
        20 * len(unsupported_courses)
        + 8 * len(unsupported_numbers)
        + 4 * int(not has_heading)
        + 3 * int(not has_list)
        + 3 * repeated_lines
        + 2 * int(starts_with_heading)
        + 4 * int(too_short)
        + 2 * int(too_long)
    )
    return {
        "total": total,
        "unsupported_courses": len(unsupported_courses),
        "unsupported_numbers": len(unsupported_numbers),
        "missing_heading": int(not has_heading),
        "missing_list": int(not has_list),
        "repeated_lines": repeated_lines,
        "starts_with_heading": int(starts_with_heading),
        "too_short": int(too_short),
        "too_long": int(too_long),
    }


def _source_texts(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "personal": str(row.get("data", "")),
        "expert": str(row.get("suggest", "")),
        "kb": str(row.get("rag", "")),
        "domain": DOMAIN_DESCRIPTION.get(str(row.get("domain", "")).strip(), "").replace(
            "\\n", "\n"
        ),
    }


def _supported_plan_text(
    item: Mapping[str, Any], sources: Mapping[str, str]
) -> bool:
    source = str(item.get("source", "")).strip()
    quote = normalize_root_value(str(item.get("support_quote", "")))
    if source not in sources or len(quote) < 2:
        return False
    return quote in normalize_root_value(sources[source])


def _markdown_cell(value: Any) -> str:
    return " ".join(str(value or "").replace("|", "｜").split())


def render_evidence_plan(
    row: Mapping[str, str], plan: Mapping[str, Any]
) -> tuple[str, dict[str, int]]:
    """Validate exact source quotes and render a deterministic Markdown answer."""
    sources = _source_texts(row)
    direct = plan.get("direct") if isinstance(plan.get("direct"), dict) else {}
    evidence_raw = plan.get("evidence") if isinstance(plan.get("evidence"), list) else []
    actions_raw = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    courses_raw = plan.get("courses") if isinstance(plan.get("courses"), list) else []

    evidence = [
        item
        for item in evidence_raw[:4]
        if isinstance(item, dict) and _supported_plan_text(item, sources)
    ]
    actions = [
        item
        for item in actions_raw[:4]
        if isinstance(item, dict) and _supported_plan_text(item, sources)
    ]
    services = normalize_root_value(str(row.get("services", "")))
    courses = [
        item
        for item in courses_raw[:3]
        if isinstance(item, dict)
        and normalize_root_value(str(item.get("name", "")))
        and normalize_root_value(str(item.get("name", ""))) in services
    ]

    direct_text = ""
    if isinstance(direct, dict) and _supported_plan_text(direct, sources):
        direct_text = _markdown_cell(direct.get("text"))
    if not direct_text:
        for item in evidence:
            candidate = _markdown_cell(item.get("interpretation"))
            if candidate:
                direct_text = candidate
                break
    if not direct_text:
        direct_text = "根据目前提供的信息，下面先聚焦与你的问题直接相关的内容。"

    lines = [direct_text]
    if evidence:
        lines.extend(["", "## 关键依据"])
        if len(evidence) >= 2:
            lines.extend(["", "| 指标/事实 | 数据 | 解读 |", "| --- | --- | --- |"])
            for item in evidence:
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _markdown_cell(item.get("label")) or "相关信息",
                            _markdown_cell(item.get("display_value")) or "见输入记录",
                            _markdown_cell(item.get("interpretation")) or "以原始记录为准",
                        )
                    )
                    + " |"
                )
        else:
            item = evidence[0]
            label = _markdown_cell(item.get("label")) or "相关信息"
            value = _markdown_cell(item.get("display_value"))
            interpretation = _markdown_cell(item.get("interpretation"))
            detail = "；".join(part for part in (value, interpretation) if part)
            lines.extend(["", f"- **{label}**：{detail or '以原始记录为准'}"])

    action_texts: list[str] = []
    for item in evidence:
        value = _markdown_cell(item.get("action"))
        if value and value not in action_texts:
            action_texts.append(value)
    for item in actions:
        value = _markdown_cell(item.get("text"))
        if value and value not in action_texts:
            action_texts.append(value)
    for item in courses:
        name = _markdown_cell(item.get("name"))
        reason = _markdown_cell(item.get("reason"))
        value = f"可尝试课程 <{name}>" + (f"：{reason}" if reason else "。")
        if value not in action_texts:
            action_texts.append(value)
    if action_texts:
        lines.extend(["", "## 建议", ""])
        lines.extend(f"- {value}" for value in action_texts[:5])

    safety_note = _markdown_cell(plan.get("safety_note"))
    if safety_note:
        lines.extend(["", f"> {safety_note}"])

    rendered = "\n".join(lines).strip()
    stats = {
        "evidence_proposed": len(evidence_raw),
        "evidence_retained": len(evidence),
        "actions_proposed": len(actions_raw),
        "actions_retained": len(actions),
        "courses_proposed": len(courses_raw),
        "courses_retained": len(courses),
        "direct_supported": int(
            isinstance(direct, dict) and _supported_plan_text(direct, sources)
        ),
    }
    return rendered, stats


def _numeric_tokens(value: Any) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", str(value or "")))


def _safe_supported_paraphrase(text: Any, quote: Any) -> str:
    proposed = _markdown_cell(text)
    source_quote = _markdown_cell(quote)
    if not proposed:
        return ""
    if _numeric_tokens(proposed) - _numeric_tokens(source_quote):
        return ""
    judgment_words = ("偏高", "偏低", "正常", "异常", "达标", "未达标")
    if any(word in proposed and word not in source_quote for word in judgment_words):
        return ""
    return proposed


def render_evidence_plan_conservative(
    row: Mapping[str, str], plan: Mapping[str, Any]
) -> tuple[str, dict[str, int]]:
    """Render only exact evidence quotes and source-backed action quotes."""
    sources = _source_texts(row)
    direct = plan.get("direct") if isinstance(plan.get("direct"), dict) else {}
    evidence_raw = plan.get("evidence") if isinstance(plan.get("evidence"), list) else []
    actions_raw = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    courses_raw = plan.get("courses") if isinstance(plan.get("courses"), list) else []
    evidence = [
        item
        for item in evidence_raw[:4]
        if isinstance(item, dict) and _supported_plan_text(item, sources)
    ]
    actions = [
        item
        for item in actions_raw[:4]
        if isinstance(item, dict) and _supported_plan_text(item, sources)
    ]
    services = normalize_root_value(str(row.get("services", "")))
    courses = [
        item
        for item in courses_raw[:3]
        if isinstance(item, dict)
        and normalize_root_value(str(item.get("name", "")))
        and normalize_root_value(str(item.get("name", ""))) in services
    ]

    direct_text = ""
    direct_supported = isinstance(direct, dict) and _supported_plan_text(
        direct, sources
    )
    if direct_supported:
        proposed = _markdown_cell(direct.get("text"))
        quote = _markdown_cell(direct.get("support_quote"))
        direct_text = _safe_supported_paraphrase(proposed, quote)
    if not direct_text:
        direct_text = "下面只依据现有记录，聚焦与你当前问题直接相关的信息。"

    lines = [direct_text, "", "## 关键依据"]
    if len(evidence) >= 2:
        lines.extend(["", "| 指标/事实 | 原始依据 |", "| --- | --- |"])
        for item in evidence:
            label = _markdown_cell(item.get("label"))
            quote = _markdown_cell(item.get("support_quote"))
            if not label or normalize_root_value(label) not in normalize_root_value(quote):
                label = "相关记录"
            lines.append(f"| {label} | {quote} |")
    elif evidence:
        item = evidence[0]
        label = _markdown_cell(item.get("label"))
        quote = _markdown_cell(item.get("support_quote"))
        if not label or normalize_root_value(label) not in normalize_root_value(quote):
            label = "相关记录"
        lines.extend(["", f"- **{label}**：{quote}"])
    else:
        lines.extend(["", "- 暂无可被输入原文直接支持的个人记录。"])

    action_texts: list[str] = []
    for item in actions:
        quote = _markdown_cell(item.get("support_quote"))
        if quote and quote not in action_texts:
            action_texts.append(quote)
    for item in courses:
        name = _markdown_cell(item.get("name"))
        value = f"可尝试课程 <{name}>。"
        if value not in action_texts:
            action_texts.append(value)
    if action_texts:
        lines.extend(["", "## 建议", ""])
        lines.extend(f"- {value}" for value in action_texts[:5])

    stats = {
        "evidence_proposed": len(evidence_raw),
        "evidence_retained": len(evidence),
        "actions_proposed": len(actions_raw),
        "actions_retained": len(actions),
        "courses_proposed": len(courses_raw),
        "courses_retained": len(courses),
        "direct_supported": int(direct_supported),
        "direct_generated_text_retained": int(
            bool(direct_text)
            and direct_text
            != "下面只依据现有记录，聚焦与你当前问题直接相关的信息。"
        ),
    }
    return "\n".join(lines).strip(), stats


def render_evidence_plan_balanced(
    row: Mapping[str, str], plan: Mapping[str, Any]
) -> tuple[str, dict[str, int]]:
    """Balance exact evidence display with numerically constrained paraphrases."""
    sources = _source_texts(row)
    direct = plan.get("direct") if isinstance(plan.get("direct"), dict) else {}
    evidence_raw = plan.get("evidence") if isinstance(plan.get("evidence"), list) else []
    actions_raw = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    courses_raw = plan.get("courses") if isinstance(plan.get("courses"), list) else []
    evidence = [
        item
        for item in evidence_raw[:4]
        if isinstance(item, dict) and _supported_plan_text(item, sources)
    ]
    actions = [
        item
        for item in actions_raw[:4]
        if isinstance(item, dict) and _supported_plan_text(item, sources)
    ]
    services = normalize_root_value(str(row.get("services", "")))
    courses = [
        item
        for item in courses_raw[:3]
        if isinstance(item, dict)
        and normalize_root_value(str(item.get("name", "")))
        and normalize_root_value(str(item.get("name", ""))) in services
    ]

    direct_text = ""
    direct_supported = isinstance(direct, dict) and _supported_plan_text(
        direct, sources
    )
    if direct_supported:
        direct_text = _safe_supported_paraphrase(
            direct.get("text"), direct.get("support_quote")
        )
    if not direct_text and evidence:
        direct_text = f"从现有记录看，{_markdown_cell(evidence[0].get('support_quote'))}。"
    if not direct_text:
        direct_text = "目前没有足够的相关记录支持进一步判断。"

    lines = [direct_text, "", "## 关键依据"]
    if len(evidence) >= 2:
        lines.extend(["", "| 指标/事实 | 原始依据 |", "| --- | --- |"])
        for item in evidence:
            label = _markdown_cell(item.get("label"))
            quote = _markdown_cell(item.get("support_quote"))
            if not label or normalize_root_value(label) not in normalize_root_value(quote):
                label = "相关记录"
            lines.append(f"| {label} | {quote} |")
    elif evidence:
        quote = _markdown_cell(evidence[0].get("support_quote"))
        label = _markdown_cell(evidence[0].get("label")) or "相关记录"
        lines.extend(["", f"- **{label}**：{quote}"])
    else:
        lines.extend(["", "- 暂无可被输入原文直接支持的个人记录。"])

    action_texts: list[str] = []
    for item in actions:
        value = _safe_supported_paraphrase(
            item.get("text"), item.get("support_quote")
        ) or _markdown_cell(item.get("support_quote"))
        if value and value not in action_texts:
            action_texts.append(value)
    for item in courses:
        name = _markdown_cell(item.get("name"))
        value = f"可尝试课程 <{name}>。"
        if value not in action_texts:
            action_texts.append(value)
    if action_texts:
        lines.extend(["", "## 建议", ""])
        lines.extend(f"- {value}" for value in action_texts[:5])

    stats = {
        "evidence_proposed": len(evidence_raw),
        "evidence_retained": len(evidence),
        "actions_proposed": len(actions_raw),
        "actions_retained": len(actions),
        "courses_proposed": len(courses_raw),
        "courses_retained": len(courses),
        "direct_supported": int(direct_supported),
        "direct_used_first_evidence_quote": int(
            not _safe_supported_paraphrase(
                direct.get("text") if isinstance(direct, dict) else "",
                direct.get("support_quote") if isinstance(direct, dict) else "",
            )
            and bool(evidence)
        ),
    }
    return "\n".join(lines).strip(), stats


def render_evidence_plan_natural(
    row: Mapping[str, str], plan: Mapping[str, Any]
) -> tuple[str, dict[str, int]]:
    """Avoid repeated quotes while retaining constrained natural interpretations."""
    sources = _source_texts(row)
    direct = plan.get("direct") if isinstance(plan.get("direct"), dict) else {}
    evidence_raw = plan.get("evidence") if isinstance(plan.get("evidence"), list) else []
    actions_raw = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    courses_raw = plan.get("courses") if isinstance(plan.get("courses"), list) else []
    evidence = [
        item
        for item in evidence_raw[:4]
        if isinstance(item, dict) and _supported_plan_text(item, sources)
    ]
    actions = [
        item
        for item in actions_raw[:4]
        if isinstance(item, dict) and _supported_plan_text(item, sources)
    ]
    services = normalize_root_value(str(row.get("services", "")))
    courses = [
        item
        for item in courses_raw[:3]
        if isinstance(item, dict)
        and normalize_root_value(str(item.get("name", "")))
        and normalize_root_value(str(item.get("name", ""))) in services
    ]

    safe_direct = ""
    if isinstance(direct, dict) and _supported_plan_text(direct, sources):
        safe_direct = _safe_supported_paraphrase(
            direct.get("text"), direct.get("support_quote")
        )
    used_first_quote = not safe_direct and bool(evidence)
    direct_text = safe_direct
    if not direct_text and evidence:
        direct_text = f"从现有记录看，{_markdown_cell(evidence[0].get('support_quote'))}。"
    if not direct_text:
        direct_text = "目前没有足够的相关记录支持进一步判断。"

    displayed_evidence = evidence[1:] if used_first_quote else evidence
    lines = [direct_text]
    if displayed_evidence:
        lines.extend(["", "## 关键依据"])
        if len(displayed_evidence) >= 2:
            lines.extend(["", "| 指标/事实 | 原始依据 | 解读 |", "| --- | --- | --- |"])
            for item in displayed_evidence:
                label = _markdown_cell(item.get("label"))
                quote = _markdown_cell(item.get("support_quote"))
                if not label or normalize_root_value(label) not in normalize_root_value(quote):
                    label = "相关记录"
                interpretation = _safe_supported_paraphrase(
                    item.get("interpretation"), quote
                ) or "以原始记录为准"
                lines.append(f"| {label} | {quote} | {interpretation} |")
        else:
            item = displayed_evidence[0]
            quote = _markdown_cell(item.get("support_quote"))
            interpretation = _safe_supported_paraphrase(
                item.get("interpretation"), quote
            )
            detail = quote + (f"；{interpretation}" if interpretation else "")
            label = _markdown_cell(item.get("label")) or "相关记录"
            lines.extend(["", f"- **{label}**：{detail}"])

    action_texts: list[str] = []
    for item in actions:
        value = _safe_supported_paraphrase(
            item.get("text"), item.get("support_quote")
        ) or _markdown_cell(item.get("support_quote"))
        if value and value not in action_texts:
            action_texts.append(value)
    for item in courses:
        name = _markdown_cell(item.get("name"))
        value = f"可尝试课程 <{name}>。"
        if value not in action_texts:
            action_texts.append(value)
    if action_texts:
        lines.extend(["", "## 建议", ""])
        lines.extend(f"- {value}" for value in action_texts[:5])

    stats = {
        "evidence_proposed": len(evidence_raw),
        "evidence_retained": len(evidence),
        "evidence_displayed_after_direct_dedup": len(displayed_evidence),
        "actions_proposed": len(actions_raw),
        "actions_retained": len(actions),
        "courses_proposed": len(courses_raw),
        "courses_retained": len(courses),
        "direct_used_first_evidence_quote": int(used_first_quote),
    }
    return "\n".join(lines).strip(), stats


def build_generation_messages(
    row: Mapping[str, str], strategy: str, prompts: PromptBundle
) -> list[dict[str, str]]:
    if strategy == "legacy":
        context = build_context(row, corrected=False)
        user = (
            f"{context}\n\n"
            "请结合以上 [个人数据]、[专家建议]、[知识库知识]、[课程库] 与 [对话历史]，"
            "以“小艺”健康管家的身份，用中文为用户提供结构清晰、可执行的运动健康建议。"
        )
        system = prompts.legacy_generation_system
    elif strategy in {"corrected_v5", "self_review_v5"}:
        user = build_context(row, corrected=True) + "\n"
        system = build_phone_personal_prompt(domain=row.get("domain", "").strip() or None)
    elif strategy == "evidence_contract_v1":
        system = build_phone_personal_prompt(domain=row.get("domain", "").strip() or None)
        user = build_context(row, corrected=True) + r"""

【本次回答的证据与输出契约】
请先在内部静默完成“证据账本 → 风险核验 → Markdown 成稿”，不要输出思考或核验过程。

1. 证据边界：只把【个人数据】和【专家建议】里明确属于该用户的内容当作个人事实；知识库只用于通用解释。不得补写输入没有提供的日期、数值、单位、性别、年龄、症状、原因或结论。
2. 数值核验：引用个人数值、日期、单位和指标名时逐字回看输入；优先使用输入已经给出的统计值。只有问题确实要求且输入足够时才计算，计算前逐项核对口径，不能把不同指标、时间范围或运动类型混在一起。
3. 判断依据：只有【专家建议】或本领域说明明确给出范围时，才判断正常、偏高或偏低；区分“用户数据”“建议范围”“一般知识”，不得互相替代。避免把相关性写成因果。
4. 回答完整性：首段直接回答当前问题。需要个性化分析时，选择1—4个最相关指标，每个指标都给出“证据事实 + 谨慎解释 + 可执行建议”；无相关个人数据时直说未查询到，不强行分析。
5. Markdown 可读性：正文至少使用一个有信息量的小标题，并用列表组织行动项；当存在两个及以上需要对比的指标或时段时，优先使用紧凑表格。不要为了排版重复内容，不能只给标题没有解释。
6. 课程与安全：课程名只能逐字使用【课程库】中实际存在且高度相关的名称；没有合适课程就不推荐。不得诊断疾病、保证疗效或给出具体用药，必要时说明就医边界。
7. 出稿前静默做最终否决检查：个人事实是否有输入证据；数字/单位/比较是否一致；是否违背专家建议；核心答案是否前置；Markdown 是否完整；建议是否安全且可执行。发现问题必须在本次回答中直接修正。

最终只输出给用户看的完整中文 Markdown 回答。"""
    elif strategy == "slim_evidence_v2":
        return build_slim_generation_messages(row, internal_dual_draft=False)
    elif strategy == "slim_dual_draft_v2":
        return build_slim_generation_messages(row, internal_dual_draft=True)
    elif strategy == "claim_gated_dual_v3":
        return build_claim_gated_dual_messages(row)
    elif strategy == "context_compiler_v7":
        return build_context_compiler_v7_messages(row)
    elif strategy == "context_compiler_v8":
        return build_context_compiler_v8_messages(row)
    elif strategy == "silent_jury_v9":
        return build_silent_jury_v9_messages(row)
    elif strategy == "evidence_packet_v10":
        return build_evidence_packet_v10_messages(row)
    elif strategy == "full_context_index_v11":
        return build_full_context_index_v11_messages(row)
    elif strategy == "fact_cards_v12":
        return build_fact_cards_v12_messages(row)
    elif strategy == "source_priority_v13":
        return build_source_priority_v13_messages(row)
    elif strategy == "visual_contract_v14":
        return build_visual_contract_v14_messages(row)
    elif strategy == "guarded_visual_contract_v21":
        return build_guarded_visual_contract_v21_messages(row)
    elif strategy == "controlled_negative_v22":
        return build_guarded_visual_contract_v21_messages(row)
    elif strategy == "complete_plaintext_negative_v23":
        return build_guarded_visual_contract_v21_messages(row)
    elif strategy == "full_answer_plaintext_negative_v24":
        return build_relevance_safe_direct_v24_messages(row)
    elif strategy == "malformed_mechanical_negative_v25":
        return build_relevance_safe_direct_v24_messages(row)
    elif strategy == "visual_budget_negative_v26":
        return build_relevance_safe_direct_v24_messages(row)
    elif strategy == "markdown_preserving_negative_v27":
        return build_relevance_safe_direct_v24_messages(row)
    elif strategy == "relevance_safe_positive_v28":
        return build_relevance_safe_direct_v24_messages(row)
    elif strategy == "relevance_grounded_positive_v29":
        return build_relevance_grounded_positive_v29_messages(row)
    elif strategy == "relevance_grounded_positive_v30":
        return build_relevance_grounded_positive_v30_messages(row)
    elif strategy == "proof_carrying_v15":
        return build_proof_carrying_v15_messages(row)
    elif strategy == "evidence_compiler_v16":
        return build_evidence_compiler_v16_messages(row)
    elif strategy == "grounded_composer_v17":
        return build_grounded_composer_v17_messages(row)
    elif strategy == "contract_jury_v18":
        return build_contract_jury_v18_messages(row)
    elif strategy == "numeric_shield_jury_v19":
        return build_numeric_shield_jury_v19_messages(row)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_revision_messages(
    row: Mapping[str, str], draft: str, prompts: PromptBundle
) -> list[dict[str, str]]:
    system = build_phone_personal_prompt(domain=row.get("domain", "").strip() or None)
    context = build_context(row, corrected=True)
    user = f"""{context}

【待复核草稿】
{draft}

请逐项静默复核草稿：个人数据数值和单位、计算与比较、课程名、核心结论前置、Markdown、内容完整度、医疗安全边界。若有问题只修正问题；若无问题保持原意。最终只输出修订后的完整回答，不要输出复核过程。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_teacher_patch_messages(
    row: Mapping[str, str],
    original_answer: str,
    diagnosis: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = build_phone_personal_prompt(domain=row.get("domain", "").strip() or None)
    context = build_context(row, corrected=True)
    repair_packet = {
        "rule_ids": diagnosis.get("rule_ids", []),
        "repair_targets": diagnosis.get("repair_targets", []),
        "revision_instructions": diagnosis.get("revision_instructions", []),
        "hit_checks": diagnosis.get("hit_checks", []),
    }
    user = f"""{context}

【第一次回答】
{original_answer}

【教师诊断包】
{json.dumps(repair_packet, ensure_ascii=False, indent=2)}

请在第一次回答上做一次最小定向修补：
1. 只处理教师诊断指出的问题，尽量保留其余正确内容和结构；
2. 所有数字、单位、课程名、个人事实必须回到上方输入证据核对；
3. 不得添加输入中不存在的个人信息、因果结论或课程；
4. 最终只输出修订后的完整回答，不输出诊断、解释或修订过程。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_judge_inputs(
    row: Mapping[str, str], answer: str, *, corrected: bool, prompts: PromptBundle
) -> dict[str, tuple[str, str]]:
    service = row.get("services", "") if corrected else row.get("service", "")
    rag = row.get("rag", "").strip()
    suggest = row.get("suggest", "").strip()
    kb_text = "\n".join(value for value in (rag, suggest) if value)
    module_parts = [
        "## 模块数据",
        "[个人数据]\n" + row.get("data", "").strip(),
        "[课程库]\n" + service.strip(),
    ]
    kb_parts = [value for value in (rag, kb_text) if value]
    if kb_parts:
        module_parts.append("[知识库知识]\n" + "\n".join(kb_parts))
    if suggest:
        module_parts.append("[专家建议]\n" + suggest)
    modules_block = "\n\n".join(module_parts)

    history_parts: list[str] = []
    if row.get("last_query", "").strip():
        history_parts.append(f"user: {row['last_query'].strip()}")
    if row.get("last_answer_phone", "").strip():
        history_parts.append(f"assistant: {row['last_answer_phone'].strip()}")
    substitutions = {
        "modules_block": modules_block,
        "history_input": "\n".join(history_parts),
        "input_data": f"user: {row.get('query', '').strip()}",
        "answer": answer,
    }
    return {
        "ground": (
            prompts.ground_system,
            prompts.ground_template.safe_substitute(substitutions),
        ),
        "structure": (
            prompts.structure_system,
            prompts.structure_template.safe_substitute(substitutions),
        ),
    }


def extract_chat_content(raw: Mapping[str, Any]) -> str:
    value = raw["choices"][0]["message"]["content"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Empty chat completion content")
    return value.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("Judge output is not an object")
                return value
    raise ValueError("Unclosed JSON object")


def validate_judge_output(value: Mapping[str, Any], dimension: str) -> dict[str, Any]:
    checks = value.get("checks")
    if not isinstance(checks, list):
        raise ValueError("Judge output has no checks list")
    allowed = GROUND_RULES if dimension == "ground" else STRUCTURE_RULES
    normalized: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("Judge check is not an object")
        rule_id = str(check.get("rule_id", "")).upper()
        if rule_id not in allowed:
            # Historical scorer accepted arbitrary checks and ignored unsupported
            # rule ids in checks_to_penalties. Preserve that behavior so an extra
            # metadata-like check does not waste another teacher request.
            continue
        hit = check.get("hit")
        if not isinstance(hit, bool):
            raise ValueError(f"Judge rule {rule_id} has non-boolean hit")
        normalized.append({**check, "rule_id": rule_id, "hit": hit})
    confidence = value.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        raise ValueError("Judge confidence is not numeric")
    return {
        "checks": normalized,
        "confidence": max(0.0, min(1.0, float(confidence))),
    }


def judge_response_format(dimension: str) -> dict[str, Any]:
    """Constrain judge serialization without changing the frozen judge prompt."""
    allowed = GROUND_RULES if dimension == "ground" else STRUCTURE_RULES
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{dimension}_judge_output",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "checks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rule_id": {
                                    "type": "string",
                                    "enum": sorted(allowed),
                                },
                                "hit": {"type": "boolean"},
                                "severity": {"type": "string"},
                                "reason": {"type": "string"},
                                "excerpt": {"type": "string"},
                            },
                            "required": [
                                "rule_id",
                                "hit",
                                "severity",
                                "reason",
                                "excerpt",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["checks", "confidence"],
                "additionalProperties": False,
            },
        },
    }


def parse_persisted_judge_event(
    event: Mapping[str, Any], dimension: str
) -> dict[str, Any] | None:
    """Recover a successful judge response without issuing another API call."""
    if event.get("status") != "ok":
        return None
    raw_response = event.get("raw_response")
    if not isinstance(raw_response, Mapping):
        return None
    try:
        return validate_judge_output(
            extract_json_object(extract_chat_content(raw_response)), dimension
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def checks_to_penalties(checks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for check in checks:
        if not check.get("hit", False):
            continue
        rule_id = str(check.get("rule_id", "")).upper()
        severity = str(check.get("severity", "")).lower()
        score: float | None
        if rule_id in STRICT_20:
            score = 20.0
        elif rule_id == "NO_MARKDOWN":
            score = 5.0
        elif rule_id in {"REDUNDANT", "GRAMMAR"}:
            score = 3.0
        elif rule_id == "PERSONAL_DATA_ANALYSIS_ISSUE":
            score = 5.0 if severity in {"5", "major"} else 3.0
        elif rule_id in {"CONTRADICT_KB_OR_EXPERT", "FACT_LOGIC_ISSUE"}:
            score = 10.0 if severity in {"major", "严重", "high"} else 5.0
        elif rule_id in {"BURIED_CORE_ANSWER", "THIN_CONTENT"}:
            score = 5.0
        elif rule_id in {"UNNATURAL_TONE", "BAD_MARKDOWN_USAGE", "LACK_VISUAL_AID"}:
            score = 3.0
        else:
            score = None
        if score is not None:
            output.append(
                {
                    "rule_id": rule_id,
                    "score": score,
                    "reason": check.get("reason", ""),
                    "excerpt": check.get("excerpt", ""),
                }
            )
    return output


def _historical_data_json(row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the frozen scorer's local-validator payload without inference."""
    value = row.get("data")
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def historical_local_validator_penalties(
    row: Mapping[str, Any], answer: str
) -> list[dict[str, Any]]:
    """Reproduce the frozen scorer's three deterministic validators."""
    data = _historical_data_json(row)
    penalties: list[dict[str, Any]] = []
    sleep = data.get("sleep") if isinstance(data, dict) else None
    if isinstance(sleep, dict):
        target_duration = sleep.get("duration_hours")
        start = sleep.get("start")
        end = sleep.get("end")
        if start and end and not target_duration:
            start_match = re.search(r"(\d{1,2}):(\d{2})", str(start))
            end_match = re.search(r"(\d{1,2}):(\d{2})", str(end))
            if start_match and end_match:
                start_minutes = int(start_match.group(1)) * 60 + int(
                    start_match.group(2)
                )
                end_minutes = int(end_match.group(1)) * 60 + int(
                    end_match.group(2)
                )
                duration_minutes = end_minutes - start_minutes
                if duration_minutes < 0:
                    duration_minutes += 24 * 60
                target_duration = round(duration_minutes / 60.0, 2)
        if target_duration is not None:
            duration_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:h|小时)", answer, flags=re.IGNORECASE
            )
            if duration_match:
                try:
                    answer_duration = float(duration_match.group(1))
                    if abs(answer_duration - float(target_duration)) > 0.25:
                        penalties.append(
                            {
                                "rule_id": "PERSONAL_DATA_MISMATCH",
                                "score": 20.0,
                                "reason": "historical sleep duration validator mismatch",
                                "excerpt": "",
                            }
                        )
                except (TypeError, ValueError):
                    pass
            else:
                penalties.append(
                    {
                        "rule_id": "PERSONAL_DATA_ANALYSIS_ISSUE",
                        "score": 5.0,
                        "reason": "historical sleep duration validator missing",
                        "excerpt": "",
                    }
                )
        thresholds = sleep.get("score_thresholds")
        sleep_score = sleep.get("score")
        if isinstance(thresholds, dict) and sleep_score is not None:
            expected_grade: str | None = None
            for grade, bounds in thresholds.items():
                try:
                    low, high = float(bounds[0]), float(bounds[1])
                    score_value = float(sleep_score)
                except (IndexError, TypeError, ValueError):
                    continue
                if low <= score_value < high or math.isclose(score_value, high):
                    expected_grade = str(grade)
                    break
            if expected_grade:
                tokens = [
                    expected_grade.lower(),
                    "优",
                    "良",
                    "中",
                    "差",
                    "一般",
                    "poor",
                    "fair",
                    "good",
                ]
                if not any(token.lower() in answer.lower() for token in tokens):
                    penalties.append(
                        {
                            "rule_id": "PERSONAL_DATA_ANALYSIS_ISSUE",
                            "score": 5.0,
                            "reason": "historical sleep grade validator missing",
                            "excerpt": "",
                        }
                    )
    require_chart = data.get("require_chart_or_table") if isinstance(data, dict) else None
    if require_chart:
        has_table = bool(re.search(r"\n\|.+\|\n\|[-:\s|]+\|\n", answer))
        has_image = "![ " in answer or "![" in answer
        if not (has_table or has_image):
            penalties.append(
                {
                    "rule_id": "MISSING_CHART_TABLE",
                    "score": 10.0,
                    "reason": "historical chart or table validator missing",
                    "excerpt": "",
                }
            )
    services = data.get("services") if isinstance(data, dict) else None
    if services:
        missing = [
            service
            for service in services
            if isinstance(service, str)
            and service.strip()
            and service.lower() not in answer.lower()
        ] if isinstance(services, list) else []
        if missing:
            penalties.append(
                {
                    "rule_id": "MISSING_SERVICE",
                    "score": 10.0,
                    "reason": "historical service validator missing",
                    "excerpt": "",
                }
            )
    merged: dict[str, dict[str, Any]] = {}
    for penalty in penalties:
        rule_id = str(penalty["rule_id"])
        if rule_id not in merged or float(penalty["score"]) > float(
            merged[rule_id]["score"]
        ):
            merged[rule_id] = penalty
    return list(merged.values())


def replay_run_metrics(run_dir: Path) -> dict[str, Any]:
    """Recompute the user metric from immutable request and judge ledgers."""
    events = read_jsonl(run_dir / "intermediate" / "api_events.jsonl")
    candidates = read_jsonl(run_dir / "processed" / "candidates.jsonl")
    judges = [
        normalize_judge_result_acceptance(result)
        for result in read_jsonl(run_dir / "processed" / "judge_results.jsonl")
    ]
    qwen_events = [event for event in events if event.get("provider") == "qwen"]
    luna_events = [event for event in events if event.get("provider") == "luna"]

    def candidate_from_operation(event: Mapping[str, Any]) -> str:
        parts = str(event.get("operation_id", "")).split(":")
        return parts[1] if len(parts) >= 2 else ""

    qwen_events_by_candidate: dict[str, list[dict[str, Any]]] = {}
    luna_events_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for event in qwen_events:
        qwen_events_by_candidate.setdefault(candidate_from_operation(event), []).append(
            event
        )
    for event in luna_events:
        luna_events_by_candidate.setdefault(candidate_from_operation(event), []).append(
            event
        )

    successful_generation_ids = {
        candidate_id
        for candidate_id, candidate_events in qwen_events_by_candidate.items()
        if candidate_id
        and any(event.get("status") == "ok" for event in candidate_events)
    }
    failed_generation_ids = {
        candidate_id
        for candidate_id, candidate_events in qwen_events_by_candidate.items()
        if candidate_id
        and not any(event.get("status") == "ok" for event in candidate_events)
    }
    persisted_candidate_ids = {
        str(candidate.get("candidate_id", "")) for candidate in candidates
    }
    judge_rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for result in judges:
        judge_rows_by_candidate.setdefault(str(result.get("candidate_id", "")), []).append(
            result
        )

    completed_results = [
        results[0]
        for candidate_id, results in judge_rows_by_candidate.items()
        if candidate_id in successful_generation_ids
        and len(results) == 1
        and results[0].get("status") == "ok"
    ]
    score_band_counts = {
        "positive": 0,
        "negative": 0,
        "ambiguous": 0,
        "unusable_zero": 0,
    }
    for result in completed_results:
        band = classify_kto_score(float(result.get("total_score_20", 0.0)))
        score_band_counts[band] += 1

    accepted_outputs = (
        score_band_counts["positive"] + score_band_counts["negative"]
    )
    qwen_requests = len(qwen_events)
    efficiency = accepted_outputs / qwen_requests if qwen_requests else 0.0
    call_protocol_checks = {
        "one_event_per_sampling_operation": all(
            len(candidate_events) == 1
            for candidate_id, candidate_events in qwen_events_by_candidate.items()
            if candidate_id
        ),
        "one_persisted_item_per_successful_sampling_call": (
            persisted_candidate_ids == successful_generation_ids
        ),
        "two_judge_events_per_successful_generation": all(
            len(luna_events_by_candidate.get(candidate_id, [])) == 2
            for candidate_id in successful_generation_ids
        ),
        "no_judge_events_for_failed_generation": all(
            not luna_events_by_candidate.get(candidate_id)
            for candidate_id in failed_generation_ids
        ),
    }
    score_completeness_checks = {
        "one_completed_judge_result_per_successful_generation": all(
            len(judge_rows_by_candidate.get(candidate_id, [])) == 1
            and judge_rows_by_candidate[candidate_id][0].get("status") == "ok"
            for candidate_id in successful_generation_ids
        ),
    }
    lower, upper = wilson_interval(accepted_outputs, qwen_requests)
    return {
        "run_dir": str(run_dir.resolve()),
        "metric_definition": (
            "KTO-accepted completed outputs divided by every sampling-model API "
            "request event, including failed or interrupted sampling requests"
        ),
        "acceptance_policy": {
            "positive_min_score_inclusive": KTO_POSITIVE_MIN_SCORE,
            "negative_min_score_exclusive": KTO_NEGATIVE_MIN_EXCLUSIVE,
            "negative_max_score_inclusive": KTO_NEGATIVE_MAX_SCORE,
            "unusable_score": 0.0,
        },
        "sampling_requests_denominator": qwen_requests,
        "successful_sampling_requests": sum(
            event.get("status") == "ok" for event in qwen_events
        ),
        "judge_requests": len(luna_events),
        "successful_judge_requests": sum(
            event.get("status") == "ok" for event in luna_events
        ),
        "completed_scored_outputs": len(completed_results),
        "score_band_counts": score_band_counts,
        "accepted_outputs": accepted_outputs,
        "accepts_per_sampling_request": efficiency,
        "target_at_least": TARGET_ACCEPTS_PER_QWEN_REQUEST,
        "target_point_rate_meets_80_percent": (
            qwen_requests > 0 and efficiency >= TARGET_ACCEPTS_PER_QWEN_REQUEST
        ),
        "request_denominator_wilson_95_interval": [lower, upper],
        "request_denominator_one_sided_exact_95_lower_bound": (
            one_sided_exact_lower_bound(accepted_outputs, qwen_requests)
        ),
        "call_protocol_checks": call_protocol_checks,
        "strict_call_protocol_satisfied": all(call_protocol_checks.values()),
        "score_completeness_checks": score_completeness_checks,
        "complete_scoring_satisfied": all(score_completeness_checks.values()),
        "protocol_checks": {**call_protocol_checks, **score_completeness_checks},
        "strict_protocol_satisfied": all(call_protocol_checks.values())
        and all(score_completeness_checks.values()),
    }


def score_judges(
    ground: Mapping[str, Any],
    structure: Mapping[str, Any],
    answer: str,
    *,
    local_penalties: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    all_penalties = checks_to_penalties(
        [*ground.get("checks", []), *structure.get("checks", [])]
    ) + [dict(penalty) for penalty in local_penalties]
    for penalty in all_penalties:
        rule_id = str(penalty["rule_id"])
        if rule_id not in merged or float(penalty["score"]) > float(
            merged[rule_id]["score"]
        ):
            merged[rule_id] = penalty
    if not merged:
        has_markdown = any(token in answer for token in ("#", "- ", "* ")) or (
            "|" in answer and "\n|-" in answer
        )
        if not has_markdown:
            merged["NO_MARKDOWN"] = {
                "rule_id": "NO_MARKDOWN",
                "score": 5.0,
                "reason": "无明显 Markdown 结构",
                "excerpt": "",
            }
    penalties = list(merged.values())
    total = max(0.0, 20.0 - sum(float(item["score"]) for item in penalties))
    confidence = (
        float(ground.get("confidence", 0.5)) + float(structure.get("confidence", 0.5))
    ) / 2.0
    return {"total_score_20": total, "confidence": confidence, "penalties": penalties}


def _dimension_score(
    checks: Sequence[Mapping[str, Any]],
    *,
    add_no_markdown: bool,
) -> float:
    merged: dict[str, dict[str, Any]] = {}
    for penalty in checks_to_penalties(checks):
        rule_id = str(penalty["rule_id"])
        if rule_id not in merged or penalty["score"] > merged[rule_id]["score"]:
            merged[rule_id] = penalty
    if add_no_markdown and "NO_MARKDOWN" not in merged:
        merged["NO_MARKDOWN"] = {"rule_id": "NO_MARKDOWN", "score": 5.0}
    return max(0.0, 20.0 - sum(float(item["score"]) for item in merged.values()))


def build_teacher_diagnosis(
    ground: Mapping[str, Any],
    structure: Mapping[str, Any],
    answer: str,
    *,
    pass_threshold: float,
    critical_dimension_floor: float,
    local_penalties: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    scoring = score_judges(
        ground, structure, answer, local_penalties=local_penalties
    )
    has_markdown = any(token in answer for token in ("#", "- ", "* ")) or (
        "|" in answer and "\n|-" in answer
    )
    ground_score = _dimension_score(ground.get("checks", []), add_no_markdown=False)
    structure_score = _dimension_score(
        structure.get("checks", []), add_no_markdown=not has_markdown
    )
    rule_ids = [str(item["rule_id"]) for item in scoring["penalties"]]
    fatal_flags = sorted(rule_id for rule_id in rule_ids if rule_id in STRICT_20)
    repair_targets = list(
        dict.fromkeys(
            REPAIR_TARGET_BY_RULE[rule_id]
            for rule_id in rule_ids
            if rule_id in REPAIR_TARGET_BY_RULE
        )
    )
    hit_checks: list[dict[str, Any]] = []
    for dimension, judge in (("ground", ground), ("structure", structure)):
        for check in judge.get("checks", []):
            if check.get("hit", False):
                hit_checks.append(
                    {
                        "dimension": dimension,
                        "rule_id": str(check.get("rule_id", "")),
                        "severity": str(check.get("severity", "")),
                        "reason": str(check.get("reason", "")),
                        "excerpt": str(check.get("excerpt", "")),
                    }
                )
    for penalty in local_penalties:
        hit_checks.append(
            {
                "dimension": "historical_local_validator",
                "rule_id": str(penalty.get("rule_id", "")),
                "severity": str(penalty.get("score", "")),
                "reason": str(penalty.get("reason", "")),
                "excerpt": "",
            }
        )
    if not has_markdown and "NO_MARKDOWN" in rule_ids and not any(
        item["rule_id"] == "NO_MARKDOWN" for item in hit_checks
    ):
        hit_checks.append(
            {
                "dimension": "structure",
                "rule_id": "NO_MARKDOWN",
                "severity": "historical_fallback",
                "reason": "无明显 Markdown 结构",
                "excerpt": "",
            }
        )
    dimension_scores = {
        "ground_20": ground_score,
        "structure_20": structure_score,
    }
    critical_floors_met = all(
        score >= critical_dimension_floor for score in dimension_scores.values()
    )
    score_band = classify_kto_score(float(scoring["total_score_20"]))
    accepted = is_kto_accepted_score(float(scoring["total_score_20"]))
    legacy_positive_quality_gate = (
        float(scoring["total_score_20"]) > pass_threshold
        and not fatal_flags
        and critical_floors_met
    )
    return {
        "schema_version": "trace-diagnosis-v1-dual-judge",
        **scoring,
        "accepted": accepted,
        "kto_score_band": score_band,
        "kto_label": score_band if accepted else None,
        "legacy_positive_quality_gate": legacy_positive_quality_gate,
        "dimension_scores": dimension_scores,
        "critical_dimension_floor": critical_dimension_floor,
        "critical_floors_met": critical_floors_met,
        "fatal_flags": fatal_flags,
        "rule_ids": rule_ids,
        "repair_targets": repair_targets,
        "revision_instructions": [
            REPAIR_INSTRUCTION_BY_TARGET[target] for target in repair_targets
        ],
        "hit_checks": hit_checks,
        "judge_confidence": {
            "ground": float(ground.get("confidence", 0.5)),
            "structure": float(structure.get("confidence", 0.5)),
        },
    }


class HardStop(RuntimeError):
    pass


class GatewayClient:
    def __init__(
        self,
        *,
        run_dir: Path,
        qwen_url: str,
        qwen_key: str,
        qwen_model: str,
        luna_url: str,
        luna_key: str,
        luna_model: str,
        luna_min_request_interval_seconds: float,
        qwen_cap: int,
        luna_cap: int,
        max_attempts: int,
        stop_after_failures: int,
        luna_url_pool: Sequence[str] | None = None,
        luna_initial_request_delay_seconds: float = 0.0,
    ) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "intermediate" / "api_events.jsonl"
        self.urls = {"qwen": qwen_url, "luna": luna_url}
        self.url_pools = {
            "qwen": (qwen_url,),
            "luna": tuple(luna_url_pool or (luna_url,)),
        }
        self.keys = {"qwen": qwen_key, "luna": luna_key}
        self.models = {"qwen": qwen_model, "luna": luna_model}
        self.min_request_interval_seconds = {
            "qwen": 0.0,
            "luna": max(0.0, float(luna_min_request_interval_seconds)),
        }
        self.last_request_started_monotonic: dict[str, float] = {}
        self.initial_not_before_monotonic: dict[str, float] = {}
        self.caps = {"qwen": qwen_cap, "luna": luna_cap}
        self.max_attempts = max_attempts
        self.stop_after_failures = stop_after_failures
        self.consecutive_failures = 0
        self.events = read_jsonl(self.events_path)
        now = datetime.now(timezone.utc)
        for provider in ("qwen", "luna"):
            started_values = [
                event.get("started_at")
                for event in self.events
                if event.get("provider") == provider and event.get("started_at")
            ]
            if started_values:
                try:
                    latest = max(datetime.fromisoformat(str(value)) for value in started_values)
                    elapsed = max(0.0, (now - latest).total_seconds())
                    self.last_request_started_monotonic[provider] = (
                        time.monotonic() - elapsed
                    )
                except ValueError:
                    pass
        if not any(event.get("provider") == "luna" for event in self.events):
            initial_delay = max(0.0, float(luna_initial_request_delay_seconds))
            if initial_delay > 0:
                self.initial_not_before_monotonic["luna"] = (
                    time.monotonic() + initial_delay
                )
        self.client = httpx.Client(timeout=httpx.Timeout(900.0, connect=20.0))

    def close(self) -> None:
        self.client.close()

    def _cached_success(self, operation_id: str) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event.get("operation_id") == operation_id and event.get("status") == "ok":
                return event
        return None

    def _request_count(self, provider: str) -> int:
        return sum(1 for event in self.events if event.get("provider") == provider)

    def call(
        self,
        *,
        provider: str,
        operation_id: str,
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any] | None = None,
        reasoning_effort: str | None = None,
        expect_json_dimension: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        cached = self._cached_success(operation_id)
        if cached is not None:
            text = extract_chat_content(cached["raw_response"])
            parsed = (
                validate_judge_output(extract_json_object(text), expect_json_dimension)
                if expect_json_dimension
                else {}
            )
            return text, parsed
        prior_terminal = next(
            (
                event
                for event in reversed(self.events)
                if event.get("operation_id") == operation_id
            ),
            None,
        )
        if prior_terminal is not None:
            raise RuntimeError(
                f"Operation {operation_id} already has terminal event "
                f"status={prior_terminal.get('status')}; retry forbidden"
            )

        last_error = "unknown"
        for attempt in range(1, self.max_attempts + 1):
            if self._request_count(provider) >= self.caps[provider]:
                raise HardStop(f"{provider} request cap reached ({self.caps[provider]})")
            interval_wait_seconds = 0.0
            last_started = self.last_request_started_monotonic.get(provider)
            minimum_interval = self.min_request_interval_seconds[provider]
            if last_started is not None and minimum_interval > 0:
                interval_wait_seconds = max(
                    0.0, minimum_interval - (time.monotonic() - last_started)
                )
            not_before = self.initial_not_before_monotonic.get(provider)
            if not_before is not None:
                interval_wait_seconds = max(
                    interval_wait_seconds, not_before - time.monotonic()
                )
            if interval_wait_seconds > 0:
                time.sleep(interval_wait_seconds)
            self.initial_not_before_monotonic.pop(provider, None)
            self.last_request_started_monotonic[provider] = time.monotonic()
            endpoint_index = self._request_count(provider) % len(
                self.url_pools[provider]
            )
            endpoint_url = self.url_pools[provider][endpoint_index]
            payload: dict[str, Any] = {
                "model": self.models[provider],
                "messages": [dict(message) for message in messages],
                "stream": False,
            }
            if response_format is not None:
                payload["response_format"] = dict(response_format)
            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort
            started = time.monotonic()
            event: dict[str, Any] = {
                "operation_id": operation_id,
                "provider": provider,
                "model": self.models[provider],
                "attempt": attempt,
                "started_at": utc_now(),
                "rate_limit_wait_seconds": round(interval_wait_seconds, 3),
                "endpoint_index": endpoint_index,
                "endpoint_url": endpoint_url,
                "messages": payload["messages"],
                "request_options": {
                    key: value
                    for key, value in payload.items()
                    if key not in {"messages", "model"}
                },
            }
            try:
                response = self.client.post(
                    endpoint_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.keys[provider]}"},
                    json=payload,
                )
                event["http_status"] = response.status_code
                event["request_id"] = response.headers.get("x-request-id")
                response.raise_for_status()
                raw_response = response.json()
                text = extract_chat_content(raw_response)
                event.update(
                    {
                        "usage": raw_response.get("usage"),
                        "raw_response": raw_response,
                        "output_sha256": sha256_text(text),
                    }
                )
                parsed: dict[str, Any] = {}
                if expect_json_dimension:
                    parsed = validate_judge_output(
                        extract_json_object(text), expect_json_dimension
                    )
                event.update(
                    {
                        "status": "ok",
                        "latency_seconds": round(time.monotonic() - started, 3),
                    }
                )
                append_jsonl(self.events_path, event)
                self.events.append(event)
                self.consecutive_failures = 0
                return text, parsed
            except KeyboardInterrupt as exc:
                event.update(
                    {
                        "status": "interrupted",
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "failure_class": type(exc).__name__,
                        "failure_message": "request interrupted by operator",
                    }
                )
                append_jsonl(self.events_path, event)
                self.events.append(event)
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, httpx.HTTPStatusError):
                    try:
                        error_payload = exc.response.json().get("error", {})
                    except (ValueError, AttributeError):
                        error_payload = {}
                    if isinstance(error_payload, Mapping):
                        event["gateway_error"] = {
                            "type": str(error_payload.get("type", ""))[:120],
                            "code": str(error_payload.get("code", ""))[:120],
                            "message": str(error_payload.get("message", ""))[:1000],
                        }
                event.update(
                    {
                        "status": "failed",
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "failure_class": type(exc).__name__,
                        "failure_message": str(exc)[:1000],
                    }
                )
                append_jsonl(self.events_path, event)
                self.events.append(event)
                if attempt < self.max_attempts:
                    time.sleep(float(attempt))
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.stop_after_failures:
            raise HardStop(
                f"Stopped after {self.consecutive_failures} consecutive terminal failures"
            )
        raise RuntimeError(f"Operation {operation_id} failed: {last_error}")


class SynthesisExperiment:
    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: Path,
        qwen_key: str,
        luna_key: str,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.rows = load_source_rows(config)
        response_prompt_source = read_frozen_artifact(
            config.legacy_response_prompt,
            repo=config.snapshot_repo,
            commit=config.snapshot_commit,
            git_path=config.legacy_response_prompt_git_path,
        ).decode("utf-8")
        scorer_source = read_frozen_artifact(
            config.legacy_scorer,
            repo=config.snapshot_repo,
            commit=config.snapshot_commit,
            git_path=config.legacy_scorer_git_path,
        ).decode("utf-8")
        self.prompts = PromptBundle.from_source_texts(
            response_prompt_source, scorer_source
        )
        self.gateway = GatewayClient(
            run_dir=run_dir,
            qwen_url=config.qwen_url,
            qwen_key=qwen_key,
            qwen_model=config.qwen_model,
            luna_url=config.luna_url,
            luna_key=luna_key,
            luna_model=config.luna_model,
            luna_min_request_interval_seconds=(
                config.luna_min_request_interval_seconds
            ),
            luna_initial_request_delay_seconds=(
                config.luna_initial_request_delay_seconds
            ),
            qwen_cap=config.qwen_request_cap,
            luna_cap=config.luna_request_cap,
            max_attempts=config.max_attempts_per_operation,
            stop_after_failures=config.stop_after_consecutive_failures,
            luna_url_pool=config.luna_url_pool,
        )
        self.candidates_path = run_dir / "processed" / "candidates.jsonl"
        self.judges_path = run_dir / "processed" / "judge_results.jsonl"
        self.candidates = read_jsonl(self.candidates_path)
        self.judges = [
            normalize_judge_result_acceptance(result)
            for result in read_jsonl(self.judges_path)
        ]

    def close(self) -> None:
        self.gateway.close()

    def write_manifest(self, config_path: Path) -> None:
        source_bytes = read_source_artifact(self.config)
        response_prompt_bytes = read_frozen_artifact(
            self.config.legacy_response_prompt,
            repo=self.config.snapshot_repo,
            commit=self.config.snapshot_commit,
            git_path=self.config.legacy_response_prompt_git_path,
        )
        scorer_bytes = read_frozen_artifact(
            self.config.legacy_scorer,
            repo=self.config.snapshot_repo,
            commit=self.config.snapshot_commit,
            git_path=self.config.legacy_scorer_git_path,
        )
        manifest = {
            "created_at": utc_now(),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_path": str(self.config.source_path.resolve()),
            "source_git_ref": (
                f"{self.config.snapshot_commit}:{self.config.source_git_path}"
                if self.config.source_git_path
                else None
            ),
            "source_snapshot_mode": (
                "git_blob" if self.config.source_git_path else "local_sha256"
            ),
            "source_sha256": sha256_bytes(source_bytes),
            "source_rows": len(self.rows),
            "snapshot_commit": self.config.snapshot_commit,
            "legacy_response_prompt_path": str(self.config.legacy_response_prompt),
            "legacy_response_prompt_git_ref": (
                f"{self.config.snapshot_commit}:"
                f"{self.config.legacy_response_prompt_git_path}"
            ),
            "legacy_response_prompt_sha256": sha256_bytes(response_prompt_bytes),
            "legacy_scorer_path": str(self.config.legacy_scorer),
            "legacy_scorer_git_ref": (
                f"{self.config.snapshot_commit}:{self.config.legacy_scorer_git_path}"
            ),
            "legacy_scorer_sha256": sha256_bytes(scorer_bytes),
            "models": {
                "generation": self.config.qwen_model,
                "judge": self.config.luna_model,
                "judge_reasoning_effort": self.config.luna_reasoning_effort,
                "judge_min_request_interval_seconds": (
                    self.config.luna_min_request_interval_seconds
                ),
                "judge_initial_request_delay_seconds": (
                    self.config.luna_initial_request_delay_seconds
                ),
            },
            "pipeline_epoch": self.config.pipeline_epoch,
            "concurrency": {"generation": 1, "judge": 1},
            "request_caps": {
                "qwen": self.config.qwen_request_cap,
                "luna": self.config.luna_request_cap,
            },
            "luna_endpoint_pool": list(self.config.luna_url_pool),
            "acceptance_policy": {
                "positive_min_score_inclusive": KTO_POSITIVE_MIN_SCORE,
                "negative_min_score_exclusive": KTO_NEGATIVE_MIN_EXCLUSIVE,
                "negative_max_score_inclusive": KTO_NEGATIVE_MAX_SCORE,
                "unusable_score": 0.0,
                "target_accepts_per_sampling_request_at_least": (
                    TARGET_ACCEPTS_PER_QWEN_REQUEST
                ),
            },
            "legacy_pass_condition_for_provenance": (
                f"mean(total_score_20) > {self.config.pass_threshold}"
            ),
            "trace": {
                "phase1_count": self.config.phase1_count,
                "excluded_indices": list(self.config.phase1_excluded_indices),
                "validation_excluded_indices": list(
                    self.config.validation_excluded_indices
                ),
                "audit_excluded_indices": list(
                    self.config.audit_excluded_indices
                ),
                "split_protected_indices": list(
                    self.config.split_protected_indices
                ),
                "checkpoints": list(self.config.phase1_checkpoints),
                "validation_root_count": self.config.validation_root_count,
                "audit_root_count": self.config.audit_root_count,
                "critical_dimension_floor": self.config.critical_dimension_floor,
                "stop_when_target_impossible": (
                    self.config.stop_when_target_impossible
                ),
                "diagnosis_schema": "trace-diagnosis-v1-dual-judge",
            },
            "prompt_hashes": {
                "legacy_generation_system": sha256_text(
                    self.prompts.legacy_generation_system
                ),
                "legacy_ground": sha256_text(self.prompts.ground_template.template),
                "legacy_structure": sha256_text(
                    self.prompts.structure_template.template
                ),
            },
        }
        write_json(self.run_dir / "manifest.json", manifest)

    def _find_candidate(
        self,
        row_index: int,
        strategy: str,
        variant_id: str | None = None,
    ) -> dict[str, Any] | None:
        expected_variant_id = variant_id or strategy
        for candidate in reversed(self.candidates):
            if (
                candidate.get("row_index") == row_index
                and candidate.get("strategy") == strategy
                and candidate.get("variant_id", strategy) == expected_variant_id
                and candidate.get("status") == "ok"
            ):
                return candidate
        return None

    def ensure_candidate(
        self,
        row_index: int,
        strategy: str,
        *,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        candidate_variant_id = variant_id or strategy
        existing = self._find_candidate(row_index, strategy, candidate_variant_id)
        if existing is not None:
            return existing
        row = self.rows[row_index]
        row_root_id = root_context_id(row)
        is_trace = strategy.startswith("trace_")
        candidate_id = (
            stable_trace_candidate_id(
                row_root_id, self.config.pipeline_epoch, candidate_variant_id
            )
            if is_trace
            else stable_candidate_id(
                self.config.expected_source_sha256, row_index, candidate_variant_id
            )
        )
        base_strategy = (
            "corrected_v5"
            if strategy
            in {
                "self_review_v5",
                "trace_initial_v1",
                "trace_self_review_v1",
                "trace_prompt_baseline_v1",
            }
            else "evidence_contract_v1"
            if strategy == "trace_prompt_evidence_contract_v1"
            else "slim_evidence_v2"
            if strategy == "trace_prompt_slim_evidence_v2"
            else "slim_dual_draft_v2"
            if strategy == "trace_prompt_slim_dual_draft_v2"
            else "claim_gated_dual_v3"
            if strategy == "trace_prompt_claim_gated_dual_v3"
            else "context_compiler_v7"
            if strategy == "trace_context_compiler_v7"
            else "context_compiler_v8"
            if strategy == "trace_context_compiler_v8"
            else "silent_jury_v9"
            if strategy == "trace_silent_jury_v9"
            else "evidence_packet_v10"
            if strategy == "trace_evidence_packet_v10"
            else "full_context_index_v11"
            if strategy == "trace_full_context_index_v11"
            else "fact_cards_v12"
            if strategy == "trace_fact_cards_v12"
            else "source_priority_v13"
            if strategy == "trace_source_priority_v13"
            else "visual_contract_v14"
            if strategy == "trace_visual_contract_v14"
            else "guarded_visual_contract_v21"
            if strategy == "trace_guarded_visual_contract_v21"
            else "controlled_negative_v22"
            if strategy == "trace_controlled_negative_v22"
            else "complete_plaintext_negative_v23"
            if strategy == "trace_complete_plaintext_negative_v23"
            else "full_answer_plaintext_negative_v24"
            if strategy == "trace_full_answer_plaintext_negative_v24"
            else "malformed_mechanical_negative_v25"
            if strategy == "trace_malformed_mechanical_negative_v25"
            else "visual_budget_negative_v26"
            if strategy == "trace_visual_budget_negative_v26"
            else "markdown_preserving_negative_v27"
            if strategy == "trace_markdown_preserving_negative_v27"
            else "relevance_safe_positive_v28"
            if strategy == "trace_relevance_safe_positive_v28"
            else "relevance_grounded_positive_v29"
            if strategy == "trace_relevance_grounded_positive_v29"
            else "relevance_grounded_positive_v30"
            if strategy == "trace_relevance_grounded_positive_v30"
            else "proof_carrying_v15"
            if strategy == "trace_proof_carrying_v15"
            else "evidence_compiler_v16"
            if strategy == "trace_evidence_compiler_v16"
            else "grounded_composer_v17"
            if strategy == "trace_grounded_composer_v17"
            else "contract_jury_v18"
            if strategy == "trace_contract_jury_v18"
            else "numeric_shield_jury_v19"
            if strategy == "trace_numeric_shield_jury_v19"
            else strategy
        )
        generation_messages = build_generation_messages(
            row, base_strategy, self.prompts
        )
        draft, _ = self.gateway.call(
            provider="qwen",
            operation_id=f"generate:{candidate_id}:draft",
            messages=generation_messages,
        )
        response = draft
        revision_sha256: str | None = None
        request_count = 1
        strategy_path = "one_shot"
        local_guard_stats: dict[str, Any] | None = None
        evidence_packet_stats: dict[str, dict[str, int]] | None = None
        if strategy == "trace_context_compiler_v8":
            response, local_guard_stats = apply_label_free_output_guard(row, response)
            strategy_path = "one_shot_then_label_free_output_guard"
        if strategy == "trace_guarded_visual_contract_v21":
            response, local_guard_stats = apply_guarded_visual_contract_v21(
                row, response
            )
            strategy_path = "one_shot_then_source_only_risk_guard_and_markdown_repair"
        if strategy == "trace_controlled_negative_v22":
            response, local_guard_stats = render_controlled_negative_v22(row, response)
            strategy_path = "one_shot_then_source_guard_then_controlled_structure_degrade"
        if strategy == "trace_complete_plaintext_negative_v23":
            response, local_guard_stats = render_complete_plaintext_negative_v23(
                row, response
            )
            strategy_path = "one_shot_then_source_guard_then_complete_plaintext_degrade"
        if strategy == "trace_full_answer_plaintext_negative_v24":
            response, local_guard_stats = render_full_answer_plaintext_negative_v24(
                response
            )
            strategy_path = "relevance_safe_one_shot_then_full_answer_plaintext_degrade"
        if strategy == "trace_malformed_mechanical_negative_v25":
            response, local_guard_stats = render_malformed_mechanical_negative_v25(
                response
            )
            strategy_path = "relevance_safe_one_shot_then_bounded_malformed_mechanical_degrade"
        if strategy == "trace_visual_budget_negative_v26":
            response, local_guard_stats = render_visual_budget_negative_v26(response)
            strategy_path = "relevance_safe_one_shot_then_visual_anchored_12_point_defect_budget"
        if strategy == "trace_markdown_preserving_negative_v27":
            response, local_guard_stats = render_markdown_preserving_negative_v27(
                response
            )
            strategy_path = "relevance_safe_markdown_preserved_then_bounded_visible_defects"
        if strategy == "trace_relevance_grounded_positive_v29":
            response, local_guard_stats = apply_relevance_grounded_guard_v29(
                row, response
            )
            strategy_path = "relevance_grounded_one_shot_then_label_free_safety_guard"
        if strategy == "trace_relevance_grounded_positive_v30":
            response, local_guard_stats = audit_relevance_grounded_guard_v30(
                row, response
            )
            strategy_path = "relevance_grounded_complete_one_shot_then_audit_only"
        if strategy == "trace_proof_carrying_v15":
            response, local_guard_stats = apply_proof_citation_firewall_v15(
                row, response
            )
            strategy_path = "proof_carrying_one_shot_then_local_citation_firewall"
        if strategy == "trace_evidence_compiler_v16":
            response, local_guard_stats = render_evidence_compiler_v16(row, response)
            strategy_path = "one_shot_source_plan_then_deterministic_markdown_compiler"
        if strategy == "trace_grounded_composer_v17":
            response, local_guard_stats = render_grounded_composer_v17(row, response)
            strategy_path = "one_shot_source_addressed_json_then_guarded_markdown_compiler"
        if strategy in {
            "trace_evidence_packet_v10",
            "trace_full_context_index_v11",
            "trace_fact_cards_v12",
            "trace_source_priority_v13",
            "trace_visual_contract_v14",
            "trace_proof_carrying_v15",
            "trace_evidence_compiler_v16",
            "trace_grounded_composer_v17",
            "trace_contract_jury_v18",
            "trace_numeric_shield_jury_v19",
            "trace_guarded_visual_contract_v21",
            "trace_controlled_negative_v22",
            "trace_complete_plaintext_negative_v23",
            "trace_full_answer_plaintext_negative_v24",
            "trace_malformed_mechanical_negative_v25",
            "trace_visual_budget_negative_v26",
            "trace_markdown_preserving_negative_v27",
            "trace_relevance_safe_positive_v28",
            "trace_relevance_grounded_positive_v29",
            "trace_relevance_grounded_positive_v30",
        }:
            packet = compile_evidence_packet(row)
            evidence_packet_stats = packet["source_stats"]
            strategy_path = (
                "label_free_evidence_packet_then_one_shot"
                if strategy == "trace_evidence_packet_v10"
                else "proof_carrying_one_shot_then_local_citation_firewall"
                if strategy == "trace_proof_carrying_v15"
                else "one_shot_source_plan_then_deterministic_markdown_compiler"
                if strategy == "trace_evidence_compiler_v16"
                else "one_shot_source_addressed_json_then_guarded_markdown_compiler"
                if strategy == "trace_grounded_composer_v17"
                else "full_context_exact_fact_cards_visual_contract_and_silent_jury"
                if strategy == "trace_contract_jury_v18"
                else "full_context_visual_contract_silent_jury_and_numeric_shield"
                if strategy == "trace_numeric_shield_jury_v19"
                else "visual_contract_then_source_only_risk_guard_and_markdown_repair"
                if strategy == "trace_guarded_visual_contract_v21"
                else "source_guarded_one_shot_then_controlled_negative_render"
                if strategy == "trace_controlled_negative_v22"
                else "source_guarded_one_shot_then_complete_plaintext_negative_render"
                if strategy == "trace_complete_plaintext_negative_v23"
                else "relevance_safe_one_shot_then_full_answer_plaintext_negative_render"
                if strategy == "trace_full_answer_plaintext_negative_v24"
                else "relevance_safe_one_shot_then_bounded_malformed_mechanical_negative_render"
                if strategy == "trace_malformed_mechanical_negative_v25"
                else "relevance_safe_one_shot_then_visual_anchored_12_point_negative_render"
                if strategy == "trace_visual_budget_negative_v26"
                else "relevance_safe_markdown_preserved_then_bounded_negative_render"
                if strategy == "trace_markdown_preserving_negative_v27"
                else "relevance_safe_original_markdown_positive"
                if strategy == "trace_relevance_safe_positive_v28"
                else "relevance_grounded_one_shot_then_label_free_safety_guard"
                if strategy == "trace_relevance_grounded_positive_v29"
                else "relevance_grounded_complete_one_shot_then_audit_only"
                if strategy == "trace_relevance_grounded_positive_v30"
                else "full_context_plus_exact_fact_cards_then_one_shot"
                if strategy in {"trace_fact_cards_v12", "trace_source_priority_v13"}
                else "full_context_plus_label_free_index_then_one_shot"
            )
        if strategy in {"self_review_v5", "trace_self_review_v1"}:
            response, _ = self.gateway.call(
                provider="qwen",
                operation_id=f"generate:{candidate_id}:revision",
                messages=build_revision_messages(row, draft, self.prompts),
            )
            revision_sha256 = sha256_text(draft)
            request_count = 2
            strategy_path = "fresh_draft_then_self_review"
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": row_root_id,
            "source_hash": source_record_hash(row),
            "variant_id": candidate_variant_id,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": sha256_text(
                json.dumps(generation_messages, ensure_ascii=False, sort_keys=True)
            ),
            "strategy": strategy,
            "strategy_path": strategy_path,
            "parent_candidate_id": None,
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "draft_sha256": revision_sha256,
            "request_count": request_count,
            "local_guard_stats": local_guard_stats,
            "evidence_packet_stats": evidence_packet_stats,
            "terminal_status": "generated",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def ensure_teacher_patch_candidate(
        self,
        row_index: int,
        parent_candidate: Mapping[str, Any],
        diagnosis: Mapping[str, Any],
    ) -> dict[str, Any]:
        strategy = "trace_teacher_patch_v1"
        existing = self._find_candidate(row_index, strategy)
        if existing is not None:
            return existing
        row = self.rows[row_index]
        row_root_id = root_context_id(row)
        candidate_id = stable_trace_candidate_id(
            row_root_id, self.config.pipeline_epoch, strategy
        )
        messages = build_teacher_patch_messages(
            row, str(parent_candidate["response"]), diagnosis
        )
        response, _ = self.gateway.call(
            provider="qwen",
            operation_id=f"generate:{candidate_id}:patch",
            messages=messages,
        )
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": row_root_id,
            "source_hash": source_record_hash(row),
            "variant_id": strategy,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            ),
            "strategy": strategy,
            "strategy_path": "initial_fail_then_teacher_grounded_patch",
            "parent_candidate_id": str(parent_candidate["candidate_id"]),
            "teacher_rule_ids": list(diagnosis.get("rule_ids", [])),
            "repair_targets": list(diagnosis.get("repair_targets", [])),
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "draft_sha256": str(parent_candidate.get("response_sha256", "")),
            "request_count": 1,
            "terminal_status": "generated",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def ensure_evidence_renderer_candidate(
        self,
        row_index: int,
        strategy: str = "trace_prompt_evidence_renderer_v4",
    ) -> dict[str, Any]:
        if strategy not in {
            "trace_prompt_evidence_renderer_v4",
            "trace_prompt_evidence_renderer_v4_1",
            "trace_prompt_evidence_renderer_v4_4",
        }:
            raise ValueError(f"Unsupported evidence renderer strategy: {strategy}")
        existing = self._find_candidate(row_index, strategy)
        if existing is not None:
            return existing
        row = self.rows[row_index]
        row_root_id = root_context_id(row)
        candidate_id = stable_trace_candidate_id(
            row_root_id, self.config.pipeline_epoch, strategy
        )
        messages = build_evidence_plan_messages(row)
        response_format = (
            {"type": "json_object"}
            if strategy == "trace_prompt_evidence_renderer_v4"
            else None
        )
        raw_plan, _ = self.gateway.call(
            provider="qwen",
            operation_id=f"generate:{candidate_id}:evidence-plan",
            messages=messages,
            response_format=response_format,
        )
        plan = extract_json_object(raw_plan)
        if strategy == "trace_prompt_evidence_renderer_v4_4":
            response, validation_stats = render_evidence_plan_natural(row, plan)
        else:
            response, validation_stats = render_evidence_plan(row, plan)
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": row_root_id,
            "source_hash": source_record_hash(row),
            "variant_id": strategy,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            ),
            "strategy": strategy,
            "strategy_path": "one_qwen_evidence_plan_then_deterministic_renderer",
            "parent_candidate_id": None,
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "raw_plan_sha256": sha256_text(raw_plan),
            "plan_validation_stats": validation_stats,
            "request_count": 1,
            "terminal_status": "generated",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def ensure_conservative_rerender_candidate(
        self,
        row_index: int,
        parent_candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        strategy = "trace_prompt_evidence_renderer_v4_2_offline"
        existing = self._find_candidate(row_index, strategy)
        if existing is not None:
            return existing
        parent_id = str(parent_candidate["candidate_id"])
        parent_event = next(
            (
                event
                for event in reversed(self.gateway.events)
                if event.get("provider") == "qwen"
                and event.get("status") == "ok"
                and parent_id in str(event.get("operation_id", ""))
            ),
            None,
        )
        if parent_event is None:
            raise ValueError("Could not locate successful parent Qwen event")
        raw_plan = extract_chat_content(parent_event["raw_response"])
        plan = extract_json_object(raw_plan)
        row = self.rows[row_index]
        response, validation_stats = render_evidence_plan_conservative(row, plan)
        candidate_id = stable_trace_candidate_id(
            root_context_id(row), self.config.pipeline_epoch, strategy
        )
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": root_context_id(row),
            "source_hash": source_record_hash(row),
            "variant_id": strategy,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": str(
                parent_candidate.get("generation_prompt_hash", "")
            ),
            "strategy": strategy,
            "strategy_path": "offline_conservative_rerender_of_one_qwen_evidence_plan",
            "parent_candidate_id": parent_id,
            "inherited_qwen_request_count": 1,
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "raw_plan_sha256": sha256_text(raw_plan),
            "plan_validation_stats": validation_stats,
            "request_count": 0,
            "terminal_status": "generated_offline",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def ensure_balanced_rerender_candidate(
        self,
        row_index: int,
        parent_candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        strategy = "trace_prompt_evidence_renderer_v4_3_offline"
        existing = self._find_candidate(row_index, strategy)
        if existing is not None:
            return existing
        parent_id = str(parent_candidate["candidate_id"])
        parent_event = next(
            (
                event
                for event in reversed(self.gateway.events)
                if event.get("provider") == "qwen"
                and event.get("status") == "ok"
                and parent_id in str(event.get("operation_id", ""))
            ),
            None,
        )
        if parent_event is None:
            raise ValueError("Could not locate successful parent Qwen event")
        raw_plan = extract_chat_content(parent_event["raw_response"])
        plan = extract_json_object(raw_plan)
        row = self.rows[row_index]
        response, validation_stats = render_evidence_plan_balanced(row, plan)
        candidate_id = stable_trace_candidate_id(
            root_context_id(row), self.config.pipeline_epoch, strategy
        )
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": root_context_id(row),
            "source_hash": source_record_hash(row),
            "variant_id": strategy,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": str(
                parent_candidate.get("generation_prompt_hash", "")
            ),
            "strategy": strategy,
            "strategy_path": "offline_balanced_rerender_of_one_qwen_evidence_plan",
            "parent_candidate_id": parent_id,
            "inherited_qwen_request_count": 1,
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "raw_plan_sha256": sha256_text(raw_plan),
            "plan_validation_stats": validation_stats,
            "request_count": 0,
            "terminal_status": "generated_offline",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def ensure_natural_rerender_candidate(
        self,
        row_index: int,
        parent_candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        strategy = "trace_prompt_evidence_renderer_v4_4_offline"
        existing = self._find_candidate(row_index, strategy)
        if existing is not None:
            return existing
        parent_id = str(parent_candidate["candidate_id"])
        parent_event = next(
            (
                event
                for event in reversed(self.gateway.events)
                if event.get("provider") == "qwen"
                and event.get("status") == "ok"
                and parent_id in str(event.get("operation_id", ""))
            ),
            None,
        )
        if parent_event is None:
            raise ValueError("Could not locate successful parent Qwen event")
        raw_plan = extract_chat_content(parent_event["raw_response"])
        plan = extract_json_object(raw_plan)
        row = self.rows[row_index]
        response, validation_stats = render_evidence_plan_natural(row, plan)
        candidate_id = stable_trace_candidate_id(
            root_context_id(row), self.config.pipeline_epoch, strategy
        )
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": root_context_id(row),
            "source_hash": source_record_hash(row),
            "variant_id": strategy,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": str(
                parent_candidate.get("generation_prompt_hash", "")
            ),
            "strategy": strategy,
            "strategy_path": "offline_natural_rerender_of_one_qwen_evidence_plan",
            "parent_candidate_id": parent_id,
            "inherited_qwen_request_count": 1,
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "raw_plan_sha256": sha256_text(raw_plan),
            "plan_validation_stats": validation_stats,
            "request_count": 0,
            "terminal_status": "generated_offline",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def ensure_packed_bon_candidate(self, row_index: int) -> dict[str, Any]:
        strategy = "trace_prompt_packed_bon_v5"
        existing = self._find_candidate(row_index, strategy)
        if existing is not None:
            return existing
        row = self.rows[row_index]
        root_id = root_context_id(row)
        candidate_id = stable_trace_candidate_id(
            root_id, self.config.pipeline_epoch, strategy
        )
        messages = build_packed_bon_messages(row)
        packed_response, _ = self.gateway.call(
            provider="qwen",
            operation_id=f"generate:{candidate_id}:packed-bon",
            messages=messages,
        )
        candidate_a, candidate_b = extract_packed_candidates(packed_response)
        risk_a = score_packed_candidate_risk(row, candidate_a)
        risk_b = score_packed_candidate_risk(row, candidate_b)
        if risk_b["total"] < risk_a["total"]:
            selected_label, response, selected_risk = "b", candidate_b, risk_b
        else:
            selected_label, response, selected_risk = "a", candidate_a, risk_a
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": root_id,
            "source_hash": source_record_hash(row),
            "variant_id": strategy,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            ),
            "strategy": strategy,
            "strategy_path": "one_qwen_packed_best_of_two_then_deterministic_risk_select",
            "parent_candidate_id": None,
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "packed_response_sha256": sha256_text(packed_response),
            "candidate_a_sha256": sha256_text(candidate_a),
            "candidate_b_sha256": sha256_text(candidate_b),
            "selected_packed_label": selected_label,
            "packed_risk": {"a": risk_a, "b": risk_b, "selected": selected_risk},
            "request_count": 1,
            "terminal_status": "generated",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def ensure_packed_contract_jury_v20_candidate(
        self, row_index: int
    ) -> dict[str, Any]:
        strategy = "trace_packed_contract_jury_v20"
        existing = self._find_candidate(row_index, strategy)
        if existing is not None:
            return existing
        row = self.rows[row_index]
        root_id = root_context_id(row)
        candidate_id = stable_trace_candidate_id(
            root_id, self.config.pipeline_epoch, strategy
        )
        messages = build_packed_contract_jury_v20_messages(row)
        packed_response, _ = self.gateway.call(
            provider="qwen",
            operation_id=f"generate:{candidate_id}:packed-contract-jury",
            messages=messages,
        )
        packed_candidates = extract_packed_candidates_v20(packed_response)
        risks = {
            label: score_packed_contract_candidate_v20(row, response)
            for label, response in packed_candidates.items()
        }
        selected_label = min(
            ("a", "b", "c"), key=lambda label: (risks[label]["total"], label)
        )
        response = packed_candidates[selected_label]
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": root_id,
            "source_hash": source_record_hash(row),
            "variant_id": strategy,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            ),
            "strategy": strategy,
            "strategy_path": "one_qwen_three_complete_answers_then_label_free_contract_risk_select",
            "parent_candidate_id": None,
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "packed_response_sha256": sha256_text(packed_response),
            "packed_candidate_sha256": {
                label: sha256_text(value) for label, value in packed_candidates.items()
            },
            "selected_packed_label": selected_label,
            "packed_risk": {**risks, "selected": risks[selected_label]},
            "request_count": 1,
            "terminal_status": "generated",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def ensure_packed_counterfactual_candidate(
        self,
        row_index: int,
        parent_candidate: Mapping[str, Any],
        label: str,
    ) -> dict[str, Any]:
        if label not in {"a", "b"}:
            raise ValueError("Packed counterfactual label must be a or b")
        strategy = f"trace_prompt_packed_bon_v5_counterfactual_{label}"
        existing = self._find_candidate(row_index, strategy)
        if existing is not None:
            return existing
        parent_id = str(parent_candidate["candidate_id"])
        parent_event = next(
            (
                event
                for event in reversed(self.gateway.events)
                if event.get("provider") == "qwen"
                and event.get("status") == "ok"
                and parent_id in str(event.get("operation_id", ""))
            ),
            None,
        )
        if parent_event is None:
            raise ValueError("Could not locate successful packed Qwen event")
        packed_response = extract_chat_content(parent_event["raw_response"])
        candidate_a, candidate_b = extract_packed_candidates(packed_response)
        response = candidate_a if label == "a" else candidate_b
        row = self.rows[row_index]
        risk = score_packed_candidate_risk(row, response)
        candidate_id = stable_trace_candidate_id(
            root_context_id(row), self.config.pipeline_epoch, strategy
        )
        candidate = {
            "candidate_id": candidate_id,
            "row_index": row_index,
            "root_context_id": root_context_id(row),
            "source_hash": source_record_hash(row),
            "variant_id": strategy,
            "pipeline_epoch": self.config.pipeline_epoch,
            "generator_model": self.config.qwen_model,
            "generation_prompt_hash": str(
                parent_candidate.get("generation_prompt_hash", "")
            ),
            "strategy": strategy,
            "strategy_path": "offline_counterfactual_from_one_qwen_packed_bon",
            "parent_candidate_id": parent_id,
            "inherited_qwen_request_count": 1,
            "status": "ok",
            "response": response,
            "response_sha256": sha256_text(response),
            "exact_hash": sha256_text(normalize_root_value(response)),
            "selected_packed_label": label,
            "packed_risk": {"selected": risk},
            "request_count": 0,
            "terminal_status": "generated_offline",
            "final_disposition": "pending_teacher",
            "created_at": utc_now(),
        }
        append_jsonl(self.candidates_path, candidate)
        self.candidates.append(candidate)
        return candidate

    def _find_judge(
        self, candidate_id: str, repeat_index: int
    ) -> dict[str, Any] | None:
        for result in reversed(self.judges):
            if (
                result.get("candidate_id") == candidate_id
                and result.get("repeat_index") == repeat_index
            ):
                return result
        return None

    def ensure_judged(
        self, row_index: int, strategy: str, candidate: Mapping[str, Any], repeat_index: int
    ) -> dict[str, Any]:
        candidate_id = str(candidate["candidate_id"])
        existing = self._find_judge(candidate_id, repeat_index)
        if existing is not None:
            return existing
        corrected = strategy != "legacy"
        prompt_by_dimension = build_judge_inputs(
            self.rows[row_index],
            str(candidate["response"]),
            corrected=corrected,
            prompts=self.prompts,
        )
        parsed: dict[str, dict[str, Any]] = {}
        failed_dimensions: list[str] = []
        for dimension in ("ground", "structure"):
            system, user = prompt_by_dimension[dimension]
            operation_id = f"judge:{candidate_id}:{repeat_index}:{dimension}"
            prior_event = next(
                (
                    event
                    for event in reversed(self.gateway.events)
                    if event.get("provider") == "luna"
                    and event.get("operation_id") == operation_id
                ),
                None,
            )
            if prior_event is not None:
                recovered = parse_persisted_judge_event(prior_event, dimension)
                if recovered is None:
                    failed_dimensions.append(dimension)
                else:
                    parsed[dimension] = recovered
                continue
            try:
                _, parsed[dimension] = self.gateway.call(
                    provider="luna",
                    operation_id=operation_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format=judge_response_format(dimension),
                    reasoning_effort=self.config.luna_reasoning_effort,
                    expect_json_dimension=dimension,
                )
            except (HardStop, Exception):
                failed_dimensions.append(dimension)
        if failed_dimensions:
            result = {
                "candidate_id": candidate_id,
                "row_index": row_index,
                "root_context_id": candidate.get("root_context_id"),
                "strategy": strategy,
                "repeat_index": repeat_index,
                "status": "failed",
                "failed_dimensions": failed_dimensions,
                "total_score_20": 0.0,
                "accepted": False,
                "created_at": utc_now(),
            }
        else:
            local_penalties = historical_local_validator_penalties(
                self.rows[row_index], str(candidate["response"])
            )
            diagnosis = build_teacher_diagnosis(
                parsed["ground"],
                parsed["structure"],
                str(candidate["response"]),
                pass_threshold=self.config.pass_threshold,
                critical_dimension_floor=self.config.critical_dimension_floor,
                local_penalties=local_penalties,
            )
            result = {
                "candidate_id": candidate_id,
                "row_index": row_index,
                "root_context_id": candidate.get("root_context_id"),
                "strategy": strategy,
                "repeat_index": repeat_index,
                "status": "ok",
                "total_score_20": diagnosis["total_score_20"],
                "confidence": diagnosis["confidence"],
                "penalties": diagnosis["penalties"],
                "accepted": diagnosis["accepted"],
                "kto_score_band": diagnosis["kto_score_band"],
                "kto_label": diagnosis["kto_label"],
                "diagnosis": diagnosis,
                "raw_ground": parsed["ground"],
                "raw_structure": parsed["structure"],
                "historical_local_validator_penalties": local_penalties,
                "created_at": utc_now(),
            }
        append_jsonl(self.judges_path, result)
        self.judges.append(result)
        return result

    def run_canary(self) -> dict[str, Any]:
        for position, row_index in enumerate(self.config.canary_indices, start=1):
            candidate = self.ensure_candidate(row_index, "legacy")
            for repeat_index in range(self.config.judge_repeats):
                self.ensure_judged(row_index, "legacy", candidate, repeat_index)
            print(
                f"canary {position}/{len(self.config.canary_indices)} row={row_index} "
                "generation=ok judge=complete",
                flush=True,
            )
        report = self.build_report("canary", list(self.config.canary_indices))
        write_json(self.run_dir / "processed" / "canary_report.json", report)
        return report

    def run_compare(self) -> dict[str, Any]:
        indices = select_compare_indices(
            self.rows, self.config.compare_count, self.config.canary_indices
        )
        write_json(
            self.run_dir / "intermediate" / "compare_selection.json",
            {
                "row_indices": indices,
                "selection_sha256": sha256_text(json.dumps(indices)),
                "method": "required canary rows plus deterministic unique metadata strata",
            },
        )
        strategies = ("legacy", "corrected_v5", "self_review_v5")
        completed = 0
        total = len(indices) * len(strategies)
        for row_index in indices:
            for strategy in strategies:
                candidate = self.ensure_candidate(row_index, strategy)
                self.ensure_judged(row_index, strategy, candidate, 0)
                completed += 1
                print(
                    f"compare {completed}/{total} row={row_index} strategy={strategy} complete",
                    flush=True,
                )
        report = self.build_report("compare", indices)
        write_json(self.run_dir / "processed" / "comparison_report.json", report)
        return report

    def _write_trace_split_and_selection(self) -> list[int]:
        splits = build_root_splits(
            self.rows,
            self.config.expected_source_sha256,
            protected_development_indices=self.config.split_protected_indices,
            validation_root_count=self.config.validation_root_count,
            audit_root_count=self.config.audit_root_count,
        )
        selection = select_trace_phase1_indices(
            self.rows,
            self.config.expected_source_sha256,
            development_indices=splits["development"],
            excluded_indices=self.config.phase1_excluded_indices,
            count=self.config.phase1_count,
        )
        split_payload: dict[str, Any] = {
            "schema_version": "trace-root-split-v1",
            "pipeline_epoch": self.config.pipeline_epoch,
            "source_sha256": self.config.expected_source_sha256,
            "protected_development_indices": list(
                self.config.split_protected_indices
            ),
            "split_method": (
                "group exact normalized root contexts; protect historical pilot in "
                "development; deterministic hash order; audit then validation"
            ),
            "splits": {},
        }
        for split_name, indices in splits.items():
            split_payload["splits"][split_name] = {
                "row_count": len(indices),
                "root_count": len(
                    {root_context_id(self.rows[index]) for index in indices}
                ),
                "records": [
                    {
                        "row_index": index,
                        "root_context_id": root_context_id(self.rows[index]),
                        "source_hash": source_record_hash(self.rows[index]),
                    }
                    for index in indices
                ],
            }
        write_json(
            self.run_dir / "intermediate" / "root_splits.json", split_payload
        )
        write_json(
            self.run_dir / "intermediate" / "trace_phase1_selection.json",
            {
                "pipeline_epoch": self.config.pipeline_epoch,
                "root_count": len(selection),
                "checkpoints": list(self.config.phase1_checkpoints),
                "selection_method": (
                    "development-only exact-root representatives, pilot excluded, "
                    "metadata-stratified then deterministic hash fill"
                ),
                "records": [
                    {
                        "row_index": index,
                        "root_context_id": root_context_id(self.rows[index]),
                        "source_hash": source_record_hash(self.rows[index]),
                    }
                    for index in selection
                ],
                "selection_sha256": sha256_text(
                    json.dumps(selection, separators=(",", ":"))
                ),
            },
        )
        return selection

    def _record_trace_terminal_failure(
        self, row_index: int, stage: str, error: Exception
    ) -> None:
        append_jsonl(
            self.run_dir / "processed" / "trace_terminal_failures.jsonl",
            {
                "row_index": row_index,
                "root_context_id": root_context_id(self.rows[row_index]),
                "stage": stage,
                "failure_class": type(error).__name__,
                "created_at": utc_now(),
            },
        )

    def run_trace_phase1(self) -> dict[str, Any]:
        if self.config.phase1_count <= 0:
            raise ValueError("trace.phase1_count must be positive")
        indices = self._write_trace_split_and_selection()
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        for position, row_index in enumerate(indices, start=1):
            try:
                initial = self.ensure_candidate(row_index, "trace_initial_v1")
                initial_result = self.ensure_judged(
                    row_index, "trace_initial_v1", initial, 0
                )
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, "initial", exc)
                print(
                    f"trace {position}/{len(indices)} root={root_context_id(self.rows[row_index])} "
                    f"initial=failed class={type(exc).__name__}",
                    flush=True,
                )
                if position in checkpoints:
                    self._write_trace_checkpoint(indices[:position], position)
                continue

            if initial_result.get("status") != "ok":
                print(
                    f"trace {position}/{len(indices)} root={root_context_id(self.rows[row_index])} "
                    "initial_judge=failed",
                    flush=True,
                )
            elif initial_result.get("accepted", False):
                print(
                    f"trace {position}/{len(indices)} root={root_context_id(self.rows[row_index])} "
                    "first_pass=accepted",
                    flush=True,
                )
            else:
                diagnosis = initial_result["diagnosis"]
                arm_status: dict[str, str] = {}
                try:
                    arm_a = self.ensure_candidate(row_index, "trace_self_review_v1")
                    self.ensure_judged(
                        row_index, "trace_self_review_v1", arm_a, 0
                    )
                    arm_status["A"] = "complete"
                except HardStop:
                    raise
                except Exception as exc:
                    self._record_trace_terminal_failure(row_index, "arm_a", exc)
                    arm_status["A"] = f"failed:{type(exc).__name__}"
                try:
                    arm_b = self.ensure_teacher_patch_candidate(
                        row_index, initial, diagnosis
                    )
                    self.ensure_judged(
                        row_index, "trace_teacher_patch_v1", arm_b, 0
                    )
                    arm_status["B"] = "complete"
                except HardStop:
                    raise
                except Exception as exc:
                    self._record_trace_terminal_failure(row_index, "arm_b", exc)
                    arm_status["B"] = f"failed:{type(exc).__name__}"
                print(
                    f"trace {position}/{len(indices)} root={root_context_id(self.rows[row_index])} "
                    f"first_pass=rejected arm_a={arm_status.get('A')} "
                    f"arm_b={arm_status.get('B')}",
                    flush=True,
                )
            if position in checkpoints:
                self._write_trace_checkpoint(indices[:position], position)
        report = self.build_trace_phase1_report(indices)
        write_json(
            self.run_dir / "processed" / "trace_phase1_report.json", report
        )
        return report

    def _write_trace_checkpoint(
        self, row_indices: Sequence[int], position: int
    ) -> None:
        write_json(
            self.run_dir / "processed" / f"trace_checkpoint_{position:03d}.json",
            self.build_trace_phase1_report(row_indices),
        )

    def _write_trace_prompt_dev_split_and_selection(self) -> list[int]:
        splits = build_root_splits(
            self.rows,
            self.config.expected_source_sha256,
            protected_development_indices=self.config.split_protected_indices,
            validation_root_count=self.config.validation_root_count,
            audit_root_count=self.config.audit_root_count,
        )
        selection = select_trace_phase1_indices(
            self.rows,
            self.config.expected_source_sha256,
            development_indices=splits["development"],
            excluded_indices=self.config.phase1_excluded_indices,
            count=self.config.phase1_count,
        )
        split_payload: dict[str, Any] = {
            "schema_version": "trace-root-split-v1",
            "pipeline_epoch": self.config.pipeline_epoch,
            "source_sha256": self.config.expected_source_sha256,
            "protected_development_indices": list(
                self.config.split_protected_indices
            ),
            "split_method": (
                "group exact normalized root contexts; protect historical pilot and "
                "prior prompt-development roots; deterministic hash order; audit then "
                "validation"
            ),
            "splits": {},
        }
        for split_name, indices in splits.items():
            split_payload["splits"][split_name] = {
                "row_count": len(indices),
                "root_count": len(
                    {root_context_id(self.rows[index]) for index in indices}
                ),
                "records": [
                    {
                        "row_index": index,
                        "root_context_id": root_context_id(self.rows[index]),
                        "source_hash": source_record_hash(self.rows[index]),
                    }
                    for index in indices
                ],
            }
        write_json(self.run_dir / "intermediate" / "root_splits.json", split_payload)
        write_json(
            self.run_dir / "intermediate" / "trace_prompt_dev_selection.json",
            {
                "pipeline_epoch": self.config.pipeline_epoch,
                "root_count": len(selection),
                "checkpoints": list(self.config.phase1_checkpoints),
                "selection_method": (
                    "development-only exact-root representatives, historical pilot and "
                    "TRACE Phase 1 roots excluded, metadata-stratified then deterministic "
                    "hash fill"
                ),
                "records": [
                    {
                        "row_index": index,
                        "root_context_id": root_context_id(self.rows[index]),
                        "source_hash": source_record_hash(self.rows[index]),
                    }
                    for index in selection
                ],
                "selection_sha256": sha256_text(
                    json.dumps(selection, separators=(",", ":"))
                ),
            },
        )
        return selection

    def _write_trace_validation_split_and_selection(self) -> list[int]:
        """Select roots only from the frozen validation split."""
        splits = build_root_splits(
            self.rows,
            self.config.expected_source_sha256,
            protected_development_indices=self.config.split_protected_indices,
            validation_root_count=self.config.validation_root_count,
            audit_root_count=self.config.audit_root_count,
        )
        if self.config.phase1_count > len(splits["validation"]):
            raise ValueError("requested validation roots exceed frozen split size")
        selection = select_trace_phase1_indices(
            self.rows,
            self.config.expected_source_sha256,
            development_indices=splits["validation"],
            excluded_indices=self.config.validation_excluded_indices,
            count=self.config.phase1_count,
        )
        split_roots = {
            name: {root_context_id(self.rows[index]) for index in indices}
            for name, indices in splits.items()
        }
        if not split_roots["validation"].isdisjoint(split_roots["development"]):
            raise ValueError("validation roots overlap development roots")
        if not split_roots["validation"].isdisjoint(split_roots["audit"]):
            raise ValueError("validation roots overlap audit roots")
        write_json(
            self.run_dir / "intermediate" / "trace_validation_selection.json",
            {
                "pipeline_epoch": self.config.pipeline_epoch,
                "root_count": len(selection),
                "validation_split_root_count": len(split_roots["validation"]),
                "checkpoints": list(self.config.phase1_checkpoints),
                "selection_method": (
                    "frozen validation split only; exact-root disjoint from development "
                    "and audit; metadata-stratified then deterministic hash fill"
                ),
                "records": [
                    {
                        "row_index": index,
                        "root_context_id": root_context_id(self.rows[index]),
                        "source_hash": source_record_hash(self.rows[index]),
                    }
                    for index in selection
                ],
                "selection_sha256": sha256_text(
                    json.dumps(selection, separators=(",", ":"))
                ),
            },
        )
        return selection

    def _write_trace_audit_split_and_selection(self) -> list[int]:
        """Select roots only from a frozen audit split, disjoint from all tuning."""
        splits = build_root_splits(
            self.rows,
            self.config.expected_source_sha256,
            protected_development_indices=self.config.split_protected_indices,
            validation_root_count=self.config.validation_root_count,
            audit_root_count=self.config.audit_root_count,
        )
        if self.config.phase1_count > len(splits["audit"]):
            raise ValueError("requested audit roots exceed frozen split size")
        selection = select_trace_phase1_indices(
            self.rows,
            self.config.expected_source_sha256,
            development_indices=splits["audit"],
            excluded_indices=self.config.audit_excluded_indices,
            count=self.config.phase1_count,
        )
        split_roots = {
            name: {root_context_id(self.rows[index]) for index in indices}
            for name, indices in splits.items()
        }
        if not split_roots["audit"].isdisjoint(split_roots["development"]):
            raise ValueError("audit roots overlap development roots")
        if not split_roots["audit"].isdisjoint(split_roots["validation"]):
            raise ValueError("audit roots overlap validation roots")
        write_json(
            self.run_dir / "intermediate" / "trace_audit_selection.json",
            {
                "pipeline_epoch": self.config.pipeline_epoch,
                "root_count": len(selection),
                "audit_split_root_count": len(split_roots["audit"]),
                "excluded_indices": list(self.config.audit_excluded_indices),
                "checkpoints": list(self.config.phase1_checkpoints),
                "selection_method": (
                    "frozen audit split only; exact-root disjoint from development and "
                    "validation; metadata-stratified then deterministic hash fill"
                ),
                "records": [
                    {
                        "row_index": index,
                        "root_context_id": root_context_id(self.rows[index]),
                        "source_hash": source_record_hash(self.rows[index]),
                    }
                    for index in selection
                ],
                "selection_sha256": sha256_text(
                    json.dumps(selection, separators=(",", ":"))
                ),
            },
        )
        return selection

    def run_trace_prompt_dev(self) -> dict[str, Any]:
        """Compare two one-Qwen-call prompts on unused development roots."""
        if self.config.phase1_count <= 0:
            raise ValueError("trace.phase1_count must be positive")
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategies = (
            "trace_prompt_baseline_v1",
            "trace_prompt_evidence_contract_v1",
        )
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        for position, row_index in enumerate(indices, start=1):
            statuses: dict[str, str] = {}
            for strategy in strategies:
                try:
                    candidate = self.ensure_candidate(row_index, strategy)
                    result = self.ensure_judged(row_index, strategy, candidate, 0)
                    statuses[strategy] = (
                        "accepted" if result.get("accepted", False) else "rejected"
                    )
                except HardStop:
                    raise
                except Exception as exc:
                    self._record_trace_terminal_failure(row_index, strategy, exc)
                    statuses[strategy] = f"failed:{type(exc).__name__}"
            print(
                f"trace-prompt-dev {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} "
                f"baseline={statuses.get(strategies[0])} "
                f"optimized={statuses.get(strategies[1])}",
                flush=True,
            )
            if position in checkpoints:
                self._write_trace_prompt_dev_checkpoint(indices[:position], position)
        report = self.build_trace_prompt_dev_report(indices)
        write_json(
            self.run_dir / "processed" / "trace_prompt_dev_report.json", report
        )
        return report

    def run_trace_prompt_search_v2(self) -> dict[str, Any]:
        """Run a three-arm one-shot prompt search on unused development roots."""
        if self.config.phase1_count <= 0:
            raise ValueError("trace.phase1_count must be positive")
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategies = (
            "trace_prompt_baseline_v1",
            "trace_prompt_slim_evidence_v2",
            "trace_prompt_slim_dual_draft_v2",
        )
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        for position, row_index in enumerate(indices, start=1):
            statuses: dict[str, str] = {}
            for strategy in strategies:
                try:
                    candidate = self.ensure_candidate(row_index, strategy)
                    result = self.ensure_judged(row_index, strategy, candidate, 0)
                    statuses[strategy] = (
                        "accepted" if result.get("accepted", False) else "rejected"
                    )
                except HardStop:
                    raise
                except Exception as exc:
                    self._record_trace_terminal_failure(row_index, strategy, exc)
                    statuses[strategy] = f"failed:{type(exc).__name__}"
            print(
                f"trace-prompt-search-v2 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} "
                f"baseline={statuses.get(strategies[0])} "
                f"slim={statuses.get(strategies[1])} "
                f"dual={statuses.get(strategies[2])}",
                flush=True,
            )
            if position in checkpoints:
                self._write_trace_prompt_search_v2_checkpoint(
                    indices[:position], position
                )
        report = self.build_trace_prompt_search_v2_report(indices)
        write_json(
            self.run_dir / "processed" / "trace_prompt_search_v2_report.json",
            report,
        )
        return report

    def run_trace_prompt_gate_v3(self) -> dict[str, Any]:
        """Compare one-call dual drafting with a claim-gated successor."""
        if self.config.phase1_count <= 0:
            raise ValueError("trace.phase1_count must be positive")
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategies = (
            "trace_prompt_slim_dual_draft_v2",
            "trace_prompt_claim_gated_dual_v3",
        )
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        for position, row_index in enumerate(indices, start=1):
            statuses: dict[str, str] = {}
            for strategy in strategies:
                try:
                    candidate = self.ensure_candidate(row_index, strategy)
                    result = self.ensure_judged(row_index, strategy, candidate, 0)
                    statuses[strategy] = (
                        "accepted" if result.get("accepted", False) else "rejected"
                    )
                except HardStop:
                    raise
                except Exception as exc:
                    self._record_trace_terminal_failure(row_index, strategy, exc)
                    statuses[strategy] = f"failed:{type(exc).__name__}"
            print(
                f"trace-prompt-gate-v3 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} "
                f"dual_v2={statuses.get(strategies[0])} "
                f"gated_v3={statuses.get(strategies[1])}",
                flush=True,
            )
            if position in checkpoints:
                self._write_trace_prompt_gate_v3_checkpoint(
                    indices[:position], position
                )
        report = self.build_trace_prompt_gate_v3_report(indices)
        write_json(
            self.run_dir / "processed" / "trace_prompt_gate_v3_report.json",
            report,
        )
        return report

    def run_trace_evidence_renderer_v4(self) -> dict[str, Any]:
        """Compare one-shot dual drafting with a deterministic evidence renderer."""
        if self.config.phase1_count <= 0:
            raise ValueError("trace.phase1_count must be positive")
        indices = self._write_trace_prompt_dev_split_and_selection()
        dual_strategy = "trace_prompt_slim_dual_draft_v2"
        renderer_strategy = "trace_prompt_evidence_renderer_v4"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        for position, row_index in enumerate(indices, start=1):
            statuses: dict[str, str] = {}
            try:
                dual = self.ensure_candidate(row_index, dual_strategy)
                result = self.ensure_judged(row_index, dual_strategy, dual, 0)
                statuses[dual_strategy] = (
                    "accepted" if result.get("accepted", False) else "rejected"
                )
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, dual_strategy, exc)
                statuses[dual_strategy] = f"failed:{type(exc).__name__}"
            try:
                renderer = self.ensure_evidence_renderer_candidate(row_index)
                result = self.ensure_judged(
                    row_index, renderer_strategy, renderer, 0
                )
                statuses[renderer_strategy] = (
                    "accepted" if result.get("accepted", False) else "rejected"
                )
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, renderer_strategy, exc)
                statuses[renderer_strategy] = f"failed:{type(exc).__name__}"
            print(
                f"trace-evidence-renderer-v4 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} "
                f"dual_v2={statuses.get(dual_strategy)} "
                f"renderer_v4={statuses.get(renderer_strategy)}",
                flush=True,
            )
            if position in checkpoints:
                self._write_trace_evidence_renderer_v4_checkpoint(
                    indices[:position], position
                )
        report = self.build_trace_evidence_renderer_v4_report(indices)
        write_json(
            self.run_dir / "processed" / "trace_evidence_renderer_v4_report.json",
            report,
        )
        return report

    def run_trace_evidence_renderer_smoke_v4_1(self) -> dict[str, Any]:
        """Smoke test renderer JSON without the unsupported response-format option."""
        if self.config.phase1_count != 1:
            raise ValueError("renderer v4.1 smoke requires exactly one root")
        indices = self._write_trace_prompt_dev_split_and_selection()
        row_index = indices[0]
        strategy = "trace_prompt_evidence_renderer_v4_1"
        try:
            candidate = self.ensure_evidence_renderer_candidate(
                row_index, strategy=strategy
            )
            result = self.ensure_judged(row_index, strategy, candidate, 0)
            status = "accepted" if result.get("accepted", False) else "rejected"
        except HardStop:
            raise
        except Exception as exc:
            self._record_trace_terminal_failure(row_index, strategy, exc)
            status = f"failed:{type(exc).__name__}"
        print(
            f"trace-evidence-renderer-v4-1-smoke 1/1 "
            f"root={root_context_id(self.rows[row_index])} status={status}",
            flush=True,
        )
        report = self.build_trace_evidence_renderer_smoke_v4_1_report(indices)
        write_json(
            self.run_dir
            / "processed"
            / "trace_evidence_renderer_v4_1_smoke_report.json",
            report,
        )
        return report

    def run_trace_evidence_renderer_canary_v4_4(self) -> dict[str, Any]:
        """Run the selected one-Qwen evidence renderer on new development roots."""
        if self.config.phase1_count <= 0:
            raise ValueError("trace.phase1_count must be positive")
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_prompt_evidence_renderer_v4_4"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        for position, row_index in enumerate(indices, start=1):
            try:
                candidate = self.ensure_evidence_renderer_candidate(
                    row_index, strategy=strategy
                )
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            print(
                f"trace-evidence-renderer-v4-4 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status}",
                flush=True,
            )
            if position in checkpoints:
                self._write_trace_evidence_renderer_v4_4_checkpoint(
                    indices[:position], position
                )
        report = self.build_trace_evidence_renderer_v4_4_report(indices)
        write_json(
            self.run_dir / "processed" / "trace_evidence_renderer_v4_4_report.json",
            report,
        )
        return report

    def run_trace_packed_bon_smoke_v5(self) -> dict[str, Any]:
        """Smoke test packed best-of-two parsing, selection, and judging."""
        if self.config.phase1_count != 1:
            raise ValueError("packed BoN v5 smoke requires exactly one root")
        indices = self._write_trace_prompt_dev_split_and_selection()
        row_index = indices[0]
        strategy = "trace_prompt_packed_bon_v5"
        try:
            candidate = self.ensure_packed_bon_candidate(row_index)
            result = self.ensure_judged(row_index, strategy, candidate, 0)
            status = "accepted" if result.get("accepted", False) else "rejected"
        except HardStop:
            raise
        except Exception as exc:
            self._record_trace_terminal_failure(row_index, strategy, exc)
            status = f"failed:{type(exc).__name__}"
        print(
            f"trace-packed-bon-v5-smoke 1/1 "
            f"root={root_context_id(self.rows[row_index])} status={status}",
            flush=True,
        )
        report = self.build_trace_packed_bon_v5_report(indices, smoke=True)
        write_json(
            self.run_dir / "processed" / "trace_packed_bon_v5_smoke_report.json",
            report,
        )
        return report

    def run_trace_context_compiler_v7(self) -> dict[str, Any]:
        """Run a label-free routed prompt under the strict fixed-call protocol."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_context_compiler_v7"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-context-compiler-v7 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_context_compiler_v7_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_context_compiler_v7_report.json",
            report,
        )
        return report

    def run_trace_context_compiler_v8(self) -> dict[str, Any]:
        """Run v8 exact-anchor prompting plus a deterministic label-free guard."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_context_compiler_v8"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-context-compiler-v8 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_context_compiler_v8_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_context_compiler_v8_report.json",
            report,
        )
        return report

    def run_trace_silent_jury_v9(self) -> dict[str, Any]:
        """Run the single-output silent-jury strategy on unused roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_silent_jury_v9"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = math.ceil(
            (1.0 - TARGET_ACCEPTS_PER_QWEN_REQUEST) * len(indices) - 1e-12
        )
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-silent-jury-v9 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_silent_jury_v9_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_gt_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_silent_jury_v9_report.json",
            report,
        )
        return report

    def run_trace_silent_jury_v9_validation(self) -> dict[str, Any]:
        """Evaluate frozen v9 on an exact-root-disjoint validation split."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_validation_split_and_selection()
        strategy = "trace_silent_jury_v9"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = math.ceil(
            (1.0 - TARGET_ACCEPTS_PER_QWEN_REQUEST) * len(indices) - 1e-12
        )
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-silent-jury-v9-validation {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope="validation",
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_silent_jury_v9_validation_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_gt_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope="validation",
        )
        write_json(
            self.run_dir
            / "processed"
            / "trace_silent_jury_v9_validation_report.json",
            report,
        )
        return report

    def run_trace_evidence_packet_v10(self) -> dict[str, Any]:
        """Run deterministic retrieval plus one-shot generation on new dev roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_evidence_packet_v10"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = math.ceil(
            (1.0 - TARGET_ACCEPTS_PER_QWEN_REQUEST) * len(indices) - 1e-12
        )
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-evidence-packet-v10 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_evidence_packet_v10_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_gt_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_evidence_packet_v10_report.json",
            report,
        )
        return report

    def run_trace_full_context_index_v11(self) -> dict[str, Any]:
        """Run full-context generation with a deterministic relevance index."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_full_context_index_v11"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = math.ceil(
            (1.0 - TARGET_ACCEPTS_PER_QWEN_REQUEST) * len(indices) - 1e-12
        )
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-full-context-index-v11 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_full_context_index_v11_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_gt_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_full_context_index_v11_report.json",
            report,
        )
        return report

    def run_trace_fact_cards_v12(self) -> dict[str, Any]:
        """Run one-shot generation with deterministic exact personal-fact cards."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_fact_cards_v12"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = math.ceil(
            (1.0 - TARGET_ACCEPTS_PER_QWEN_REQUEST) * len(indices) - 1e-12
        )
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-fact-cards-v12 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_fact_cards_v12_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_gt_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_fact_cards_v12_report.json",
            report,
        )
        return report

    def run_trace_source_priority_v13(self) -> dict[str, Any]:
        """Run exact facts with strict expert/knowledge source priority."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_source_priority_v13"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = math.ceil(
            (1.0 - TARGET_ACCEPTS_PER_QWEN_REQUEST) * len(indices) - 1e-12
        )
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-source-priority-v13 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_source_priority_v13_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_gt_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_source_priority_v13_report.json",
            report,
        )
        return report

    def run_trace_visual_contract_v14(self) -> dict[str, Any]:
        """Run exact-source generation with an explicit visual contract."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_visual_contract_v14"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]),
                self.config.pipeline_epoch,
                strategy,
            )
            candidate: dict[str, Any] | None = None
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-visual-contract-v14 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_visual_contract_v14_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_visual_contract_v14_report.json",
            report,
        )
        return report

    def run_trace_visual_contract_v14_validation(self) -> dict[str, Any]:
        """Evaluate frozen v14 on exact-root-disjoint validation roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_validation_split_and_selection()
        strategy = "trace_visual_contract_v14"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]),
                self.config.pipeline_epoch,
                strategy,
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-visual-contract-v14-validation {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope="validation",
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_visual_contract_v14_validation_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope="validation",
        )
        write_json(
            self.run_dir
            / "processed"
            / "trace_visual_contract_v14_validation_report.json",
            report,
        )
        return report

    def run_trace_proof_carrying_v15(self) -> dict[str, Any]:
        """Run proof-carrying one-shot generation on fresh development roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_proof_carrying_v15"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]),
                self.config.pipeline_epoch,
                strategy,
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-proof-carrying-v15 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_proof_carrying_v15_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            guard_stats = candidate.get("local_guard_stats", {})
            if (
                not str(candidate.get("response", "")).strip()
                or int(guard_stats.get("substantive_lines_kept", 0)) == 0
            ):
                stop_reason = "local_firewall_empty"
                break
            if (
                status != "accepted"
                and int(guard_stats.get("substantive_lines_kept", 0)) <= 1
            ):
                stop_reason = "local_firewall_overpruned"
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_proof_carrying_v15_report.json",
            report,
        )
        return report

    def run_trace_evidence_compiler_v16(self) -> dict[str, Any]:
        """Run one-shot Qwen planning plus deterministic exact-source rendering."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_evidence_compiler_v16"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]),
                self.config.pipeline_epoch,
                strategy,
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-evidence-compiler-v16 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_evidence_compiler_v16_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_evidence_compiler_v16_report.json",
            report,
        )
        return report

    def run_trace_grounded_composer_v17(self) -> dict[str, Any]:
        """Run one-shot source-addressed composition on fresh development roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_grounded_composer_v17"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-grounded-composer-v17 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_grounded_composer_v17_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_grounded_composer_v17_report.json",
            report,
        )
        return report

    def run_trace_contract_jury_v18(self) -> dict[str, Any]:
        """Run v14 evidence/visual contracts plus one-call internal jury."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_contract_jury_v18"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-contract-jury-v18 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_contract_jury_v18_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_contract_jury_v18_report.json",
            report,
        )
        return report

    def run_trace_numeric_shield_jury_v19(self) -> dict[str, Any]:
        """Run v18 with a general no-derived-arithmetic shield."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_numeric_shield_jury_v19"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-numeric-shield-jury-v19 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_numeric_shield_jury_v19_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_numeric_shield_jury_v19_report.json",
            report,
        )
        return report

    def run_trace_packed_contract_jury_v20(self) -> dict[str, Any]:
        """Run one Qwen response with three candidates and label-free selection."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_packed_contract_jury_v20"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_packed_contract_jury_v20_candidate(row_index)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-packed-contract-jury-v20 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_packed_contract_jury_v20_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_packed_contract_jury_v20_report.json",
            report,
        )
        return report

    def run_trace_packed_contract_jury_v20_validation(self) -> dict[str, Any]:
        """Evaluate frozen v20 on exact-root-disjoint validation roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_validation_split_and_selection()
        strategy = "trace_packed_contract_jury_v20"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_packed_contract_jury_v20_candidate(row_index)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-packed-contract-jury-v20-validation {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope="validation",
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_packed_contract_jury_v20_validation_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope="validation",
        )
        write_json(
            self.run_dir / "processed"
            / "trace_packed_contract_jury_v20_validation_report.json",
            report,
        )
        return report

    def run_trace_guarded_visual_contract_v21(self) -> dict[str, Any]:
        """Run v21 once per root with deterministic, source-only risk removal."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_guarded_visual_contract_v21"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-guarded-visual-contract-v21 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_guarded_visual_contract_v21_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_guarded_visual_contract_v21_report.json",
            report,
        )
        return report

    def run_trace_controlled_negative_v22(self) -> dict[str, Any]:
        """Generate one safe answer, then deterministically render a poor KTO negative."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_controlled_negative_v22"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-controlled-negative-v22 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_controlled_negative_v22_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed" / "trace_controlled_negative_v22_report.json",
            report,
        )
        return report

    def run_trace_complete_plaintext_negative_v23(self) -> dict[str, Any]:
        """Generate a grounded answer and add only controlled presentation defects."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_complete_plaintext_negative_v23"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-complete-plaintext-negative-v23 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_complete_plaintext_negative_v23_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed"
            / "trace_complete_plaintext_negative_v23_report.json",
            report,
        )
        return report

    def run_trace_full_answer_plaintext_negative_v24(self) -> dict[str, Any]:
        """Generate a direct complete answer, then degrade presentation only."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_full_answer_plaintext_negative_v24"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-full-answer-plaintext-negative-v24 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_full_answer_plaintext_negative_v24_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed"
            / "trace_full_answer_plaintext_negative_v24_report.json",
            report,
        )
        return report

    def run_trace_full_answer_plaintext_negative_v24_validation(
        self,
    ) -> dict[str, Any]:
        """Evaluate frozen v24 on exact-root-disjoint validation roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_validation_split_and_selection()
        strategy = "trace_full_answer_plaintext_negative_v24"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-full-answer-plaintext-negative-v24-validation "
                f"{position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope="validation",
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_full_answer_plaintext_negative_v24_validation_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope="validation",
        )
        write_json(
            self.run_dir / "processed"
            / "trace_full_answer_plaintext_negative_v24_validation_report.json",
            report,
        )
        return report

    def run_trace_malformed_mechanical_negative_v25(self) -> dict[str, Any]:
        """Generate a complete answer and add two bounded structural defects."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_malformed_mechanical_negative_v25"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-malformed-mechanical-negative-v25 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_malformed_mechanical_negative_v25_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed"
            / "trace_malformed_mechanical_negative_v25_report.json",
            report,
        )
        return report

    def run_trace_visual_budget_negative_v26(self) -> dict[str, Any]:
        """Generate a complete answer with a visual-anchored defect budget."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_visual_budget_negative_v26"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-visual-budget-negative-v26 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_visual_budget_negative_v26_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed"
            / "trace_visual_budget_negative_v26_report.json",
            report,
        )
        return report

    def run_trace_visual_budget_negative_v26_audit(self) -> dict[str, Any]:
        """Evaluate frozen v26 once on exact-root-disjoint audit roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_audit_split_and_selection()
        strategy = "trace_visual_budget_negative_v26"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-visual-budget-negative-v26-audit {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope="audit",
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_visual_budget_negative_v26_audit_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope="audit",
        )
        write_json(
            self.run_dir / "processed"
            / "trace_visual_budget_negative_v26_audit_report.json",
            report,
        )
        return report

    def run_trace_markdown_preserving_negative_v27(self) -> dict[str, Any]:
        """Generate a complete Markdown answer and add bounded defects only."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_markdown_preserving_negative_v27"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-markdown-preserving-negative-v27 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_markdown_preserving_negative_v27_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed"
            / "trace_markdown_preserving_negative_v27_report.json",
            report,
        )
        return report

    def run_trace_relevance_safe_positive_v28(self) -> dict[str, Any]:
        """Generate one relevance-safe answer without local degradation."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_prompt_dev_split_and_selection()
        strategy = "trace_relevance_safe_positive_v28"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-relevance-safe-positive-v28 {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_relevance_safe_positive_v28_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if (
                self.config.stop_when_target_impossible
                and nonaccepts >= nonaccept_stop
            ):
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
        )
        write_json(
            self.run_dir / "processed"
            / "trace_relevance_safe_positive_v28_report.json",
            report,
        )
        return report

    def run_trace_relevance_safe_positive_v28_audit(self) -> dict[str, Any]:
        """Evaluate frozen v28 once on exact-root-disjoint audit roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_audit_split_and_selection()
        strategy = "trace_relevance_safe_positive_v28"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]), self.config.pipeline_epoch, strategy
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-relevance-safe-positive-v28-audit {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope="audit",
                )
                write_json(
                    self.run_dir / "processed"
                    / f"trace_relevance_safe_positive_v28_audit_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if (
                self.config.stop_when_target_impossible
                and nonaccepts >= nonaccept_stop
            ):
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope="audit",
        )
        write_json(
            self.run_dir / "processed"
            / "trace_relevance_safe_positive_v28_audit_report.json",
            report,
        )
        return report

    def _run_trace_relevance_grounded_positive_v29(
        self, *, validation: bool
    ) -> dict[str, Any]:
        """Run frozen v29 with one sampler call and exactly two fixed judges."""
        validate_context_compiler_v7_protocol(self.config)
        indices = (
            self._write_trace_validation_split_and_selection()
            if validation
            else self._write_trace_prompt_dev_split_and_selection()
        )
        strategy = "trace_relevance_grounded_positive_v29"
        phase_slug = "trace-relevance-grounded-positive-v29"
        report_scope = "validation" if validation else "canary"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]),
                self.config.pipeline_epoch,
                strategy,
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"{phase_slug}{'-validation' if validation else ''} "
                f"{position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope=report_scope,
                )
                suffix = "_validation" if validation else ""
                write_json(
                    self.run_dir
                    / "processed"
                    / (
                        "trace_relevance_grounded_positive_v29"
                        f"{suffix}_checkpoint_{position:03d}.json"
                    ),
                    checkpoint,
                )
            infrastructure_stop = trace_v14_infrastructure_stop_reason(
                self.gateway.events, current_candidate_id
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if (
                self.config.stop_when_target_impossible
                and nonaccepts >= nonaccept_stop
            ):
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope=report_scope,
        )
        suffix = "_validation" if validation else ""
        write_json(
            self.run_dir
            / "processed"
            / f"trace_relevance_grounded_positive_v29{suffix}_report.json",
            report,
        )
        return report

    def run_trace_relevance_grounded_positive_v29(self) -> dict[str, Any]:
        """Run v29 on fresh development roots."""
        return self._run_trace_relevance_grounded_positive_v29(validation=False)

    def run_trace_relevance_grounded_positive_v29_validation(
        self,
    ) -> dict[str, Any]:
        """Evaluate frozen v29 on exact-root-disjoint validation roots."""
        return self._run_trace_relevance_grounded_positive_v29(validation=True)

    def _run_trace_relevance_grounded_positive_v30(
        self, *, validation: bool
    ) -> dict[str, Any]:
        """Run v30 with one untouched sampler output and exactly two fixed judges."""
        validate_context_compiler_v7_protocol(self.config)
        indices = (
            self._write_trace_validation_split_and_selection()
            if validation
            else self._write_trace_prompt_dev_split_and_selection()
        )
        strategy = "trace_relevance_grounded_positive_v30"
        phase_slug = "trace-relevance-grounded-positive-v30"
        report_scope = "validation" if validation else "canary"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = nonaccept_stop_count_for_target(len(indices))
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            current_candidate_id = stable_trace_candidate_id(
                root_context_id(self.rows[row_index]),
                self.config.pipeline_epoch,
                strategy,
            )
            # Replaying a persisted terminal operation must not issue another
            # request or stop again on the same historical infrastructure event.
            infrastructure_failure_preexisted = (
                trace_v14_infrastructure_stop_reason(
                    self.gateway.events, current_candidate_id
                )
                is not None
            )
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"{phase_slug}{'-validation' if validation else ''} "
                f"{position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope=report_scope,
                )
                suffix = "_validation" if validation else ""
                write_json(
                    self.run_dir
                    / "processed"
                    / (
                        "trace_relevance_grounded_positive_v30"
                        f"{suffix}_checkpoint_{position:03d}.json"
                    ),
                    checkpoint,
                )
            infrastructure_stop = trace_infrastructure_stop_reason_for_iteration(
                self.gateway.events,
                current_candidate_id,
                failure_preexisted=infrastructure_failure_preexisted,
            )
            if infrastructure_stop is not None:
                stop_reason = infrastructure_stop
                break
            if (
                self.config.stop_when_target_impossible
                and nonaccepts >= nonaccept_stop
            ):
                stop_reason = "nonaccept_limit_makes_ge_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope=report_scope,
        )
        suffix = "_validation" if validation else ""
        write_json(
            self.run_dir
            / "processed"
            / f"trace_relevance_grounded_positive_v30{suffix}_report.json",
            report,
        )
        return report

    def run_trace_relevance_grounded_positive_v30(self) -> dict[str, Any]:
        """Run v30 on fresh development roots."""
        return self._run_trace_relevance_grounded_positive_v30(validation=False)

    def run_trace_relevance_grounded_positive_v30_validation(
        self,
    ) -> dict[str, Any]:
        """Evaluate frozen v30 on exact-root-disjoint validation roots."""
        return self._run_trace_relevance_grounded_positive_v30(validation=True)

    def run_trace_full_context_index_v11_audit(self) -> dict[str, Any]:
        """Evaluate frozen v11 plus fixed rate policy on untouched audit roots."""
        validate_context_compiler_v7_protocol(self.config)
        indices = self._write_trace_audit_split_and_selection()
        strategy = "trace_full_context_index_v11"
        checkpoints = set(self.config.phase1_checkpoints) | {len(indices)}
        attempted_indices: list[int] = []
        nonaccepts = 0
        nonaccept_stop = math.ceil(
            (1.0 - TARGET_ACCEPTS_PER_QWEN_REQUEST) * len(indices) - 1e-12
        )
        stop_reason = "planned_roots_completed"
        for position, row_index in enumerate(indices, start=1):
            attempted_indices.append(row_index)
            try:
                candidate = self.ensure_candidate(row_index, strategy)
                result = self.ensure_judged(row_index, strategy, candidate, 0)
                status = "accepted" if result.get("accepted", False) else "rejected"
            except HardStop:
                raise
            except Exception as exc:
                self._record_trace_terminal_failure(row_index, strategy, exc)
                status = f"failed:{type(exc).__name__}"
            if status != "accepted":
                nonaccepts += 1
            print(
                f"trace-full-context-index-v11-audit {position}/{len(indices)} "
                f"root={root_context_id(self.rows[row_index])} status={status} "
                f"nonaccepts={nonaccepts}",
                flush=True,
            )
            if position in checkpoints:
                checkpoint = self.build_trace_context_compiler_v7_report(
                    attempted_indices,
                    planned_root_count=len(indices),
                    stop_reason="checkpoint",
                    strategy=strategy,
                    report_scope="audit",
                )
                write_json(
                    self.run_dir
                    / "processed"
                    / f"trace_full_context_index_v11_audit_checkpoint_{position:03d}.json",
                    checkpoint,
                )
            if nonaccepts >= nonaccept_stop:
                stop_reason = "nonaccept_limit_makes_gt_80_impossible"
                break
        report = self.build_trace_context_compiler_v7_report(
            attempted_indices,
            planned_root_count=len(indices),
            stop_reason=stop_reason,
            strategy=strategy,
            report_scope="audit",
        )
        write_json(
            self.run_dir
            / "processed"
            / "trace_full_context_index_v11_audit_report.json",
            report,
        )
        return report

    def _write_trace_prompt_search_v2_checkpoint(
        self, row_indices: Sequence[int], position: int
    ) -> None:
        write_json(
            self.run_dir
            / "processed"
            / f"trace_prompt_search_v2_checkpoint_{position:03d}.json",
            self.build_trace_prompt_search_v2_report(row_indices),
        )

    def _write_trace_prompt_gate_v3_checkpoint(
        self, row_indices: Sequence[int], position: int
    ) -> None:
        write_json(
            self.run_dir
            / "processed"
            / f"trace_prompt_gate_v3_checkpoint_{position:03d}.json",
            self.build_trace_prompt_gate_v3_report(row_indices),
        )

    def _write_trace_evidence_renderer_v4_checkpoint(
        self, row_indices: Sequence[int], position: int
    ) -> None:
        write_json(
            self.run_dir
            / "processed"
            / f"trace_evidence_renderer_v4_checkpoint_{position:03d}.json",
            self.build_trace_evidence_renderer_v4_report(row_indices),
        )

    def _write_trace_evidence_renderer_v4_4_checkpoint(
        self, row_indices: Sequence[int], position: int
    ) -> None:
        write_json(
            self.run_dir
            / "processed"
            / f"trace_evidence_renderer_v4_4_checkpoint_{position:03d}.json",
            self.build_trace_evidence_renderer_v4_4_report(row_indices),
        )

    def _write_trace_prompt_dev_checkpoint(
        self, row_indices: Sequence[int], position: int
    ) -> None:
        write_json(
            self.run_dir
            / "processed"
            / f"trace_prompt_dev_checkpoint_{position:03d}.json",
            self.build_trace_prompt_dev_report(row_indices),
        )

    def _trace_result(
        self, row_index: int, strategy: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        candidate = self._find_candidate(row_index, strategy)
        if candidate is None:
            return None, None
        return candidate, self._find_judge(str(candidate["candidate_id"]), 0)

    def _request_metrics_for_candidates(
        self, provider: str, candidate_ids: set[str]
    ) -> dict[str, Any]:
        events = [
            event
            for event in self.gateway.events
            if event.get("provider") == provider
            and any(
                candidate_id in str(event.get("operation_id", ""))
                for candidate_id in candidate_ids
            )
        ]
        usage: dict[str, float] = {}
        for event in events:
            raw_usage = event.get("usage")
            if not isinstance(raw_usage, dict):
                continue
            for key, value in raw_usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0.0) + float(value)
        return {
            "requests_including_retries": len(events),
            "successful_requests": sum(
                1 for event in events if event.get("status") == "ok"
            ),
            "usage": usage,
        }

    def build_trace_phase1_report(
        self, row_indices: Sequence[int]
    ) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        candidate_ids = {
            "initial": set(),
            "arm_a": set(),
            "arm_b": set(),
        }
        for row_index in row_indices:
            initial, initial_result = self._trace_result(
                row_index, "trace_initial_v1"
            )
            arm_a, arm_a_result = self._trace_result(
                row_index, "trace_self_review_v1"
            )
            arm_b, arm_b_result = self._trace_result(
                row_index, "trace_teacher_patch_v1"
            )
            for label, candidate in (
                ("initial", initial),
                ("arm_a", arm_a),
                ("arm_b", arm_b),
            ):
                if candidate is not None:
                    candidate_ids[label].add(str(candidate["candidate_id"]))

            def summarize_result(
                result: Mapping[str, Any] | None,
            ) -> dict[str, Any] | None:
                if result is None:
                    return None
                diagnosis = result.get("diagnosis", {})
                return {
                    "status": result.get("status"),
                    "accepted": bool(result.get("accepted", False)),
                    "total_score_20": float(result.get("total_score_20", 0.0)),
                    "rule_ids": list(diagnosis.get("rule_ids", [])),
                    "fatal_flags": list(diagnosis.get("fatal_flags", [])),
                    "repair_targets": list(diagnosis.get("repair_targets", [])),
                }

            samples.append(
                {
                    "root_context_id": root_context_id(self.rows[row_index]),
                    "source_hash": source_record_hash(self.rows[row_index]),
                    "initial": summarize_result(initial_result),
                    "arm_a_fresh_self_review": summarize_result(arm_a_result),
                    "arm_b_teacher_patch": summarize_result(arm_b_result),
                }
            )

        def accepted(value: Mapping[str, Any] | None) -> bool:
            return bool(
                value
                and value.get("status") == "ok"
                and value.get("accepted", False)
            )

        initial_passed = sum(1 for sample in samples if accepted(sample["initial"]))
        failures = [sample for sample in samples if not accepted(sample["initial"])]
        arm_a_rescued = sum(
            1 for sample in failures if accepted(sample["arm_a_fresh_self_review"])
        )
        arm_b_rescued = sum(
            1 for sample in failures if accepted(sample["arm_b_teacher_patch"])
        )
        final_a = initial_passed + arm_a_rescued
        final_b = initial_passed + arm_b_rescued
        paired = [
            sample
            for sample in failures
            if sample["arm_a_fresh_self_review"] is not None
            and sample["arm_b_teacher_patch"] is not None
            and sample["arm_a_fresh_self_review"].get("status") == "ok"
            and sample["arm_b_teacher_patch"].get("status") == "ok"
        ]
        paired_a = [
            accepted(sample["arm_a_fresh_self_review"]) for sample in paired
        ]
        paired_b = [accepted(sample["arm_b_teacher_patch"]) for sample in paired]
        first_only = sum(a and not b for a, b in zip(paired_a, paired_b))
        second_only = sum(b and not a for a, b in zip(paired_a, paired_b))
        both = sum(a and b for a, b in zip(paired_a, paired_b))
        neither = len(paired) - first_only - second_only - both
        difference_interval = paired_difference_bootstrap_interval(
            paired_a, paired_b
        )

        rule_outcomes: dict[str, dict[str, int]] = {}
        for sample in failures:
            initial = sample["initial"] or {}
            for rule_id in initial.get("rule_ids", []):
                outcome = rule_outcomes.setdefault(
                    str(rule_id),
                    {"initial_failures": 0, "arm_a_rescued": 0, "arm_b_rescued": 0},
                )
                outcome["initial_failures"] += 1
                outcome["arm_a_rescued"] += int(
                    accepted(sample["arm_a_fresh_self_review"])
                )
                outcome["arm_b_rescued"] += int(
                    accepted(sample["arm_b_teacher_patch"])
                )

        metrics = {
            label: {
                provider: self._request_metrics_for_candidates(provider, ids)
                for provider in ("qwen", "luna")
            }
            for label, ids in candidate_ids.items()
        }
        total = len(samples)

        def rate_summary(passed: int) -> dict[str, Any]:
            lower, upper = wilson_interval(passed, total)
            return {
                "passed": passed,
                "samples": total,
                "rate": passed / total if total else 0.0,
                "wilson_95_interval": [lower, upper],
                "one_sided_exact_95_lower_bound": one_sided_exact_lower_bound(
                    passed, total
                ),
            }

        arm_a_qwen = metrics["arm_a"]["qwen"]["requests_including_retries"]
        arm_b_qwen = metrics["arm_b"]["qwen"]["requests_including_retries"]
        return {
            "phase": "trace_phase1_paired_mechanism",
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "root_count": total,
            "pass_threshold_strictly_greater_than": self.config.pass_threshold,
            "critical_dimension_floor": self.config.critical_dimension_floor,
            "first_pass": rate_summary(initial_passed),
            "arm_a_current_fresh_self_review": {
                "rescue": {
                    "passed": arm_a_rescued,
                    "eligible_first_pass_failures": len(failures),
                    "rate": arm_a_rescued / len(failures) if failures else 0.0,
                },
                "final_pipeline": rate_summary(final_a),
                "qwen_requests_per_rescued": (
                    arm_a_qwen / arm_a_rescued if arm_a_rescued else None
                ),
            },
            "arm_b_trace_teacher_patch": {
                "rescue": {
                    "passed": arm_b_rescued,
                    "eligible_first_pass_failures": len(failures),
                    "rate": arm_b_rescued / len(failures) if failures else 0.0,
                },
                "final_pipeline": rate_summary(final_b),
                "qwen_requests_per_rescued": (
                    arm_b_qwen / arm_b_rescued if arm_b_rescued else None
                ),
            },
            "paired_failure_comparison": {
                "complete_pairs": len(paired),
                "both_rescued": both,
                "arm_a_only": first_only,
                "arm_b_only": second_only,
                "neither_rescued": neither,
                "rescue_rate_difference_b_minus_a": (
                    (sum(paired_b) - sum(paired_a)) / len(paired)
                    if paired
                    else 0.0
                ),
                "paired_bootstrap_95_interval": list(difference_interval),
                "exact_mcnemar_p_value": exact_mcnemar_p_value(
                    first_only, second_only
                ),
            },
            "request_metrics_by_path": metrics,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "repair_outcomes_by_rule": rule_outcomes,
            "sample_results": samples,
            "claim_boundary": (
                "Phase 1 is a 50-root mechanism experiment. It compares paired rescue "
                "paths and does not certify that the population pass rate exceeds 80%."
            ),
        }

    def build_trace_prompt_dev_report(
        self, row_indices: Sequence[int]
    ) -> dict[str, Any]:
        strategy_names = {
            "baseline": "trace_prompt_baseline_v1",
            "optimized": "trace_prompt_evidence_contract_v1",
        }
        result_by_arm: dict[str, list[bool | None]] = {}
        report_by_arm: dict[str, dict[str, Any]] = {}
        rule_counts_by_arm: dict[str, dict[str, int]] = {}
        for arm, strategy in strategy_names.items():
            candidate_ids = {
                stable_trace_candidate_id(
                    root_context_id(self.rows[index]),
                    self.config.pipeline_epoch,
                    strategy,
                )
                for index in row_indices
            }
            outcomes: list[bool | None] = []
            rule_counts: dict[str, int] = {}
            for row_index in row_indices:
                _, result = self._trace_result(row_index, strategy)
                if result is None or result.get("status") != "ok":
                    outcomes.append(None)
                    continue
                is_accepted = bool(result.get("accepted", False))
                outcomes.append(is_accepted)
                if not is_accepted:
                    for rule_id in result.get("diagnosis", {}).get("rule_ids", []):
                        key = str(rule_id)
                        rule_counts[key] = rule_counts.get(key, 0) + 1
            passed = sum(value is True for value in outcomes)
            judged = sum(value is not None for value in outcomes)
            qwen_metrics = self._request_metrics_for_candidates(
                "qwen", candidate_ids
            )
            luna_metrics = self._request_metrics_for_candidates(
                "luna", candidate_ids
            )
            qwen_requests = int(qwen_metrics["requests_including_retries"])
            successful_qwen = int(qwen_metrics["successful_requests"])
            root_lower, root_upper = wilson_interval(passed, len(row_indices))
            report_by_arm[arm] = {
                "strategy": strategy,
                "accepted_outputs": passed,
                "selected_roots": len(row_indices),
                "judged_outputs": judged,
                "qwen_requests_including_retries": qwen_requests,
                "successful_qwen_requests": successful_qwen,
                "user_efficiency_accepts_per_qwen_request": (
                    passed / qwen_requests if qwen_requests else 0.0
                ),
                "accepted_per_successful_qwen_response": (
                    passed / successful_qwen if successful_qwen else 0.0
                ),
                "root_accept_rate": passed / len(row_indices) if row_indices else 0.0,
                "root_wilson_95_interval": [root_lower, root_upper],
                "root_one_sided_exact_95_lower_bound": (
                    one_sided_exact_lower_bound(passed, len(row_indices))
                ),
                "target_point_rate_meets_80_percent": (
                    passed / qwen_requests >= TARGET_ACCEPTS_PER_QWEN_REQUEST
                    if qwen_requests
                    else False
                ),
                "qwen": qwen_metrics,
                "luna": luna_metrics,
            }
            result_by_arm[arm] = outcomes
            rule_counts_by_arm[arm] = rule_counts

        complete_pairs = [
            (baseline, optimized)
            for baseline, optimized in zip(
                result_by_arm["baseline"], result_by_arm["optimized"]
            )
            if baseline is not None and optimized is not None
        ]
        baseline_only = sum(
            baseline and not optimized for baseline, optimized in complete_pairs
        )
        optimized_only = sum(
            optimized and not baseline for baseline, optimized in complete_pairs
        )
        both = sum(baseline and optimized for baseline, optimized in complete_pairs)
        neither = len(complete_pairs) - baseline_only - optimized_only - both
        paired_baseline = [bool(value[0]) for value in complete_pairs]
        paired_optimized = [bool(value[1]) for value in complete_pairs]
        paired_interval = paired_difference_bootstrap_interval(
            paired_baseline, paired_optimized
        )
        return {
            "phase": "trace_prompt_development_paired_one_shot",
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "metric_definition": (
                "accepted Qwen-generated outputs divided by all Qwen API requests, "
                "including failed attempts and retries"
            ),
            "target_at_least": TARGET_ACCEPTS_PER_QWEN_REQUEST,
            "root_count": len(row_indices),
            "arms": report_by_arm,
            "paired_root_comparison": {
                "complete_pairs": len(complete_pairs),
                "both_accepted": both,
                "baseline_only": baseline_only,
                "optimized_only": optimized_only,
                "neither_accepted": neither,
                "optimized_minus_baseline_root_rate": (
                    (sum(paired_optimized) - sum(paired_baseline))
                    / len(complete_pairs)
                    if complete_pairs
                    else 0.0
                ),
                "paired_bootstrap_95_interval": list(paired_interval),
                "exact_mcnemar_p_value": exact_mcnemar_p_value(
                    baseline_only, optimized_only
                ),
            },
            "rejection_rule_counts": rule_counts_by_arm,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "claim_boundary": (
                "Prompt development uses previously untouched development roots. It can "
                "select a prompt for validation but cannot certify held-out performance."
            ),
        }

    def build_trace_prompt_search_v2_report(
        self, row_indices: Sequence[int]
    ) -> dict[str, Any]:
        strategy_names = {
            "baseline": "trace_prompt_baseline_v1",
            "slim_evidence": "trace_prompt_slim_evidence_v2",
            "slim_dual_draft": "trace_prompt_slim_dual_draft_v2",
        }
        outcomes_by_arm: dict[str, list[bool | None]] = {}
        arms: dict[str, dict[str, Any]] = {}
        rejection_rules: dict[str, dict[str, int]] = {}
        for arm, strategy in strategy_names.items():
            candidate_ids = {
                stable_trace_candidate_id(
                    root_context_id(self.rows[index]),
                    self.config.pipeline_epoch,
                    strategy,
                )
                for index in row_indices
            }
            outcomes: list[bool | None] = []
            rule_counts: dict[str, int] = {}
            for row_index in row_indices:
                _, result = self._trace_result(row_index, strategy)
                if result is None or result.get("status") != "ok":
                    outcomes.append(None)
                    continue
                accepted = bool(result.get("accepted", False))
                outcomes.append(accepted)
                if not accepted:
                    for rule_id in result.get("diagnosis", {}).get("rule_ids", []):
                        key = str(rule_id)
                        rule_counts[key] = rule_counts.get(key, 0) + 1
            accepted_count = sum(value is True for value in outcomes)
            judged_count = sum(value is not None for value in outcomes)
            qwen = self._request_metrics_for_candidates("qwen", candidate_ids)
            luna = self._request_metrics_for_candidates("luna", candidate_ids)
            qwen_requests = int(qwen["requests_including_retries"])
            qwen_success = int(qwen["successful_requests"])
            lower, upper = wilson_interval(accepted_count, len(row_indices))
            arms[arm] = {
                "strategy": strategy,
                "accepted_outputs": accepted_count,
                "selected_roots": len(row_indices),
                "judged_outputs": judged_count,
                "qwen_requests_including_retries": qwen_requests,
                "successful_qwen_requests": qwen_success,
                "user_efficiency_accepts_per_qwen_request": (
                    accepted_count / qwen_requests if qwen_requests else 0.0
                ),
                "accepted_per_successful_qwen_response": (
                    accepted_count / qwen_success if qwen_success else 0.0
                ),
                "root_accept_rate": (
                    accepted_count / len(row_indices) if row_indices else 0.0
                ),
                "root_wilson_95_interval": [lower, upper],
                "root_one_sided_exact_95_lower_bound": (
                    one_sided_exact_lower_bound(accepted_count, len(row_indices))
                ),
                "target_point_rate_meets_80_percent": (
                    accepted_count / qwen_requests
                    >= TARGET_ACCEPTS_PER_QWEN_REQUEST
                    if qwen_requests
                    else False
                ),
                "qwen": qwen,
                "luna": luna,
            }
            outcomes_by_arm[arm] = outcomes
            rejection_rules[arm] = rule_counts

        def paired_comparison(first_arm: str, second_arm: str) -> dict[str, Any]:
            pairs = [
                (first, second)
                for first, second in zip(
                    outcomes_by_arm[first_arm], outcomes_by_arm[second_arm]
                )
                if first is not None and second is not None
            ]
            first_only = sum(first and not second for first, second in pairs)
            second_only = sum(second and not first for first, second in pairs)
            first_values = [bool(value[0]) for value in pairs]
            second_values = [bool(value[1]) for value in pairs]
            interval = paired_difference_bootstrap_interval(
                first_values, second_values
            )
            return {
                "complete_pairs": len(pairs),
                "both_accepted": sum(first and second for first, second in pairs),
                "first_only": first_only,
                "second_only": second_only,
                "neither_accepted": sum(
                    not first and not second for first, second in pairs
                ),
                "second_minus_first_root_rate": (
                    (sum(second_values) - sum(first_values)) / len(pairs)
                    if pairs
                    else 0.0
                ),
                "paired_bootstrap_95_interval": list(interval),
                "exact_mcnemar_p_value": exact_mcnemar_p_value(
                    first_only, second_only
                ),
            }

        return {
            "phase": "trace_prompt_search_v2_paired_one_shot",
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "metric_definition": (
                "accepted Qwen-generated outputs divided by all Qwen API requests, "
                "including failed attempts and retries"
            ),
            "target_at_least": TARGET_ACCEPTS_PER_QWEN_REQUEST,
            "root_count": len(row_indices),
            "arms": arms,
            "paired_comparisons": {
                "baseline_to_slim_evidence": paired_comparison(
                    "baseline", "slim_evidence"
                ),
                "baseline_to_slim_dual_draft": paired_comparison(
                    "baseline", "slim_dual_draft"
                ),
                "slim_evidence_to_slim_dual_draft": paired_comparison(
                    "slim_evidence", "slim_dual_draft"
                ),
            },
            "rejection_rule_counts": rejection_rules,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "claim_boundary": (
                "This is a 10-root multi-variant prompt-search canary on development "
                "roots. Variant selection is exploratory and requires new-root "
                "confirmation before held-out validation."
            ),
        }

    def build_trace_prompt_gate_v3_report(
        self, row_indices: Sequence[int]
    ) -> dict[str, Any]:
        strategy_names = {
            "dual_v2": "trace_prompt_slim_dual_draft_v2",
            "claim_gated_v3": "trace_prompt_claim_gated_dual_v3",
        }
        outcomes_by_arm: dict[str, list[bool | None]] = {}
        arms: dict[str, dict[str, Any]] = {}
        rejection_rules: dict[str, dict[str, int]] = {}
        for arm, strategy in strategy_names.items():
            candidate_ids = {
                stable_trace_candidate_id(
                    root_context_id(self.rows[index]),
                    self.config.pipeline_epoch,
                    strategy,
                )
                for index in row_indices
            }
            outcomes: list[bool | None] = []
            rules: dict[str, int] = {}
            for row_index in row_indices:
                _, result = self._trace_result(row_index, strategy)
                if result is None or result.get("status") != "ok":
                    outcomes.append(None)
                    continue
                accepted = bool(result.get("accepted", False))
                outcomes.append(accepted)
                if not accepted:
                    for rule_id in result.get("diagnosis", {}).get("rule_ids", []):
                        key = str(rule_id)
                        rules[key] = rules.get(key, 0) + 1
            accepted_count = sum(value is True for value in outcomes)
            judged_count = sum(value is not None for value in outcomes)
            qwen = self._request_metrics_for_candidates("qwen", candidate_ids)
            luna = self._request_metrics_for_candidates("luna", candidate_ids)
            requests = int(qwen["requests_including_retries"])
            successes = int(qwen["successful_requests"])
            lower, upper = wilson_interval(accepted_count, len(row_indices))
            arms[arm] = {
                "strategy": strategy,
                "accepted_outputs": accepted_count,
                "selected_roots": len(row_indices),
                "judged_outputs": judged_count,
                "qwen_requests_including_retries": requests,
                "successful_qwen_requests": successes,
                "user_efficiency_accepts_per_qwen_request": (
                    accepted_count / requests if requests else 0.0
                ),
                "accepted_per_successful_qwen_response": (
                    accepted_count / successes if successes else 0.0
                ),
                "root_accept_rate": (
                    accepted_count / len(row_indices) if row_indices else 0.0
                ),
                "root_wilson_95_interval": [lower, upper],
                "root_one_sided_exact_95_lower_bound": (
                    one_sided_exact_lower_bound(accepted_count, len(row_indices))
                ),
                "target_point_rate_meets_80_percent": (
                    accepted_count / requests >= TARGET_ACCEPTS_PER_QWEN_REQUEST
                    if requests
                    else False
                ),
                "qwen": qwen,
                "luna": luna,
            }
            outcomes_by_arm[arm] = outcomes
            rejection_rules[arm] = rules

        pairs = [
            (first, second)
            for first, second in zip(
                outcomes_by_arm["dual_v2"], outcomes_by_arm["claim_gated_v3"]
            )
            if first is not None and second is not None
        ]
        first_only = sum(first and not second for first, second in pairs)
        second_only = sum(second and not first for first, second in pairs)
        first_values = [bool(value[0]) for value in pairs]
        second_values = [bool(value[1]) for value in pairs]
        interval = paired_difference_bootstrap_interval(first_values, second_values)
        return {
            "phase": "trace_prompt_claim_gate_v3_paired_one_shot",
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "metric_definition": (
                "accepted Qwen-generated outputs divided by all Qwen API requests, "
                "including failed attempts and retries"
            ),
            "target_at_least": TARGET_ACCEPTS_PER_QWEN_REQUEST,
            "root_count": len(row_indices),
            "arms": arms,
            "paired_comparison": {
                "complete_pairs": len(pairs),
                "both_accepted": sum(first and second for first, second in pairs),
                "dual_v2_only": first_only,
                "claim_gated_v3_only": second_only,
                "neither_accepted": sum(
                    not first and not second for first, second in pairs
                ),
                "v3_minus_v2_root_rate": (
                    (sum(second_values) - sum(first_values)) / len(pairs)
                    if pairs
                    else 0.0
                ),
                "paired_bootstrap_95_interval": list(interval),
                "exact_mcnemar_p_value": exact_mcnemar_p_value(
                    first_only, second_only
                ),
            },
            "rejection_rule_counts": rejection_rules,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "claim_boundary": (
                "This is a 10-root development canary comparing two selected prompt "
                "variants. It does not certify held-out performance."
            ),
        }

    def build_trace_evidence_renderer_v4_report(
        self, row_indices: Sequence[int]
    ) -> dict[str, Any]:
        strategy_names = {
            "dual_v2": "trace_prompt_slim_dual_draft_v2",
            "evidence_renderer_v4": "trace_prompt_evidence_renderer_v4",
        }
        outcomes_by_arm: dict[str, list[bool | None]] = {}
        arms: dict[str, dict[str, Any]] = {}
        rejection_rules: dict[str, dict[str, int]] = {}
        for arm, strategy in strategy_names.items():
            candidate_ids = {
                stable_trace_candidate_id(
                    root_context_id(self.rows[index]),
                    self.config.pipeline_epoch,
                    strategy,
                )
                for index in row_indices
            }
            outcomes: list[bool | None] = []
            rules: dict[str, int] = {}
            for row_index in row_indices:
                _, result = self._trace_result(row_index, strategy)
                if result is None or result.get("status") != "ok":
                    outcomes.append(None)
                    continue
                accepted = bool(result.get("accepted", False))
                outcomes.append(accepted)
                if not accepted:
                    for rule_id in result.get("diagnosis", {}).get("rule_ids", []):
                        key = str(rule_id)
                        rules[key] = rules.get(key, 0) + 1
            accepted_count = sum(value is True for value in outcomes)
            judged_count = sum(value is not None for value in outcomes)
            qwen = self._request_metrics_for_candidates("qwen", candidate_ids)
            luna = self._request_metrics_for_candidates("luna", candidate_ids)
            requests = int(qwen["requests_including_retries"])
            successes = int(qwen["successful_requests"])
            lower, upper = wilson_interval(accepted_count, len(row_indices))
            arms[arm] = {
                "strategy": strategy,
                "accepted_outputs": accepted_count,
                "selected_roots": len(row_indices),
                "judged_outputs": judged_count,
                "qwen_requests_including_retries": requests,
                "successful_qwen_requests": successes,
                "user_efficiency_accepts_per_qwen_request": (
                    accepted_count / requests if requests else 0.0
                ),
                "accepted_per_successful_qwen_response": (
                    accepted_count / successes if successes else 0.0
                ),
                "root_accept_rate": (
                    accepted_count / len(row_indices) if row_indices else 0.0
                ),
                "root_wilson_95_interval": [lower, upper],
                "root_one_sided_exact_95_lower_bound": (
                    one_sided_exact_lower_bound(accepted_count, len(row_indices))
                ),
                "target_point_rate_meets_80_percent": (
                    accepted_count / requests >= TARGET_ACCEPTS_PER_QWEN_REQUEST
                    if requests
                    else False
                ),
                "qwen": qwen,
                "luna": luna,
            }
            outcomes_by_arm[arm] = outcomes
            rejection_rules[arm] = rules

        pairs = [
            (first, second)
            for first, second in zip(
                outcomes_by_arm["dual_v2"],
                outcomes_by_arm["evidence_renderer_v4"],
            )
            if first is not None and second is not None
        ]
        first_only = sum(first and not second for first, second in pairs)
        second_only = sum(second and not first for first, second in pairs)
        first_values = [bool(value[0]) for value in pairs]
        second_values = [bool(value[1]) for value in pairs]
        interval = paired_difference_bootstrap_interval(first_values, second_values)

        renderer_stats: dict[str, int] = {}
        for row_index in row_indices:
            candidate = self._find_candidate(
                row_index, "trace_prompt_evidence_renderer_v4"
            )
            if candidate is None:
                continue
            for key, value in candidate.get("plan_validation_stats", {}).items():
                if isinstance(value, int):
                    renderer_stats[str(key)] = renderer_stats.get(str(key), 0) + value

        return {
            "phase": "trace_evidence_renderer_v4_paired_one_shot",
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "metric_definition": (
                "accepted Qwen-generated outputs divided by all Qwen API requests, "
                "including failed and interrupted attempts and retries"
            ),
            "target_at_least": TARGET_ACCEPTS_PER_QWEN_REQUEST,
            "root_count": len(row_indices),
            "arms": arms,
            "paired_comparison": {
                "complete_pairs": len(pairs),
                "both_accepted": sum(first and second for first, second in pairs),
                "dual_v2_only": first_only,
                "renderer_v4_only": second_only,
                "neither_accepted": sum(
                    not first and not second for first, second in pairs
                ),
                "renderer_minus_dual_root_rate": (
                    (sum(second_values) - sum(first_values)) / len(pairs)
                    if pairs
                    else 0.0
                ),
                "paired_bootstrap_95_interval": list(interval),
                "exact_mcnemar_p_value": exact_mcnemar_p_value(
                    first_only, second_only
                ),
            },
            "rejection_rule_counts": rejection_rules,
            "renderer_validation_totals": renderer_stats,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "claim_boundary": (
                "This is a 10-root development canary. Deterministic rendering and "
                "variant selection remain exploratory and require new-root confirmation."
            ),
        }

    def build_trace_evidence_renderer_smoke_v4_1_report(
        self, row_indices: Sequence[int]
    ) -> dict[str, Any]:
        strategy = "trace_prompt_evidence_renderer_v4_1"
        candidate_ids = {
            stable_trace_candidate_id(
                root_context_id(self.rows[index]),
                self.config.pipeline_epoch,
                strategy,
            )
            for index in row_indices
        }
        accepted = 0
        judged = 0
        validation_stats: dict[str, int] = {}
        for row_index in row_indices:
            candidate, result = self._trace_result(row_index, strategy)
            if candidate is not None:
                for key, value in candidate.get("plan_validation_stats", {}).items():
                    if isinstance(value, int):
                        validation_stats[str(key)] = (
                            validation_stats.get(str(key), 0) + value
                        )
            if result is not None and result.get("status") == "ok":
                judged += 1
                accepted += int(bool(result.get("accepted", False)))
        qwen = self._request_metrics_for_candidates("qwen", candidate_ids)
        luna = self._request_metrics_for_candidates("luna", candidate_ids)
        requests = int(qwen["requests_including_retries"])
        return {
            "phase": "trace_evidence_renderer_v4_1_smoke",
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "root_count": len(row_indices),
            "candidate_count": sum(
                self._find_candidate(index, strategy) is not None
                for index in row_indices
            ),
            "judged_outputs": judged,
            "accepted_outputs": accepted,
            "qwen_requests_including_retries": requests,
            "user_efficiency_accepts_per_qwen_request": (
                accepted / requests if requests else 0.0
            ),
            "qwen": qwen,
            "luna": luna,
            "renderer_validation_totals": validation_stats,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "claim_boundary": (
                "A one-root development smoke test proves only protocol viability, "
                "not quality or target efficiency."
            ),
        }

    def build_trace_evidence_renderer_v4_4_report(
        self, row_indices: Sequence[int]
    ) -> dict[str, Any]:
        strategy = "trace_prompt_evidence_renderer_v4_4"
        candidate_ids = {
            stable_trace_candidate_id(
                root_context_id(self.rows[index]),
                self.config.pipeline_epoch,
                strategy,
            )
            for index in row_indices
        }
        accepted = 0
        judged = 0
        score_band_counts = {
            "positive": 0,
            "negative": 0,
            "ambiguous": 0,
            "unusable_zero": 0,
        }
        rules: dict[str, int] = {}
        validation_stats: dict[str, int] = {}
        for row_index in row_indices:
            candidate, result = self._trace_result(row_index, strategy)
            if candidate is not None:
                for key, value in candidate.get("plan_validation_stats", {}).items():
                    if isinstance(value, int):
                        validation_stats[str(key)] = (
                            validation_stats.get(str(key), 0) + value
                        )
            if result is None or result.get("status") != "ok":
                continue
            judged += 1
            score_band_counts[
                classify_kto_score(float(result.get("total_score_20", 0.0)))
            ] += 1
            is_accepted = bool(result.get("accepted", False))
            accepted += int(is_accepted)
            if not is_accepted:
                for rule_id in result.get("diagnosis", {}).get("rule_ids", []):
                    key = str(rule_id)
                    rules[key] = rules.get(key, 0) + 1
        qwen = self._request_metrics_for_candidates("qwen", candidate_ids)
        luna = self._request_metrics_for_candidates("luna", candidate_ids)
        requests = int(qwen["requests_including_retries"])
        lower, upper = wilson_interval(accepted, len(row_indices))
        return {
            "phase": "trace_evidence_renderer_v4_4_one_shot_canary",
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "metric_definition": (
                "accepted Qwen-generated outputs divided by all Qwen API requests, "
                "including failed and interrupted attempts and retries"
            ),
            "target_at_least": TARGET_ACCEPTS_PER_QWEN_REQUEST,
            "root_count": len(row_indices),
            "candidate_count": sum(
                self._find_candidate(index, strategy) is not None
                for index in row_indices
            ),
            "judged_outputs": judged,
            "accepted_outputs": accepted,
            "score_band_counts": score_band_counts,
            "qwen_requests_including_retries": requests,
            "successful_qwen_requests": int(qwen["successful_requests"]),
            "user_efficiency_accepts_per_qwen_request": (
                accepted / requests if requests else 0.0
            ),
            "target_point_rate_meets_80_percent": (
                accepted / requests >= TARGET_ACCEPTS_PER_QWEN_REQUEST
                if requests
                else False
            ),
            "root_accept_rate": (
                accepted / len(row_indices) if row_indices else 0.0
            ),
            "root_wilson_95_interval": [lower, upper],
            "root_one_sided_exact_95_lower_bound": (
                one_sided_exact_lower_bound(accepted, len(row_indices))
            ),
            "rejection_rule_counts": rules,
            "renderer_validation_totals": validation_stats,
            "qwen": qwen,
            "luna": luna,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "claim_boundary": (
                "This is a 10-root development canary selected after renderer tuning. "
                "It requires new-root confirmation before held-out validation."
            ),
        }

    def build_trace_packed_bon_v5_report(
        self, row_indices: Sequence[int], *, smoke: bool
    ) -> dict[str, Any]:
        strategy = "trace_prompt_packed_bon_v5"
        candidate_ids = {
            stable_trace_candidate_id(
                root_context_id(self.rows[index]),
                self.config.pipeline_epoch,
                strategy,
            )
            for index in row_indices
        }
        accepted = 0
        judged = 0
        selected_labels = {"a": 0, "b": 0}
        selected_risk_total = 0
        rules: dict[str, int] = {}
        for row_index in row_indices:
            candidate, result = self._trace_result(row_index, strategy)
            if candidate is not None:
                label = str(candidate.get("selected_packed_label", ""))
                if label in selected_labels:
                    selected_labels[label] += 1
                value = candidate.get("packed_risk", {}).get("selected", {}).get("total")
                if isinstance(value, int):
                    selected_risk_total += value
            if result is None or result.get("status") != "ok":
                continue
            judged += 1
            is_accepted = bool(result.get("accepted", False))
            accepted += int(is_accepted)
            if not is_accepted:
                for rule_id in result.get("diagnosis", {}).get("rule_ids", []):
                    key = str(rule_id)
                    rules[key] = rules.get(key, 0) + 1
        qwen = self._request_metrics_for_candidates("qwen", candidate_ids)
        luna = self._request_metrics_for_candidates("luna", candidate_ids)
        requests = int(qwen["requests_including_retries"])
        lower, upper = wilson_interval(accepted, len(row_indices))
        return {
            "phase": (
                "trace_packed_bon_v5_smoke"
                if smoke
                else "trace_packed_bon_v5_one_shot_canary"
            ),
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "metric_definition": (
                "accepted selected outputs divided by all Qwen API requests, "
                "including failed and interrupted attempts and retries"
            ),
            "target_at_least": TARGET_ACCEPTS_PER_QWEN_REQUEST,
            "root_count": len(row_indices),
            "candidate_count": sum(
                self._find_candidate(index, strategy) is not None
                for index in row_indices
            ),
            "judged_outputs": judged,
            "accepted_outputs": accepted,
            "qwen_requests_including_retries": requests,
            "successful_qwen_requests": int(qwen["successful_requests"]),
            "user_efficiency_accepts_per_qwen_request": (
                accepted / requests if requests else 0.0
            ),
            "target_point_rate_meets_80_percent": (
                accepted / requests >= TARGET_ACCEPTS_PER_QWEN_REQUEST
                if requests
                else False
            ),
            "root_accept_rate": (
                accepted / len(row_indices) if row_indices else 0.0
            ),
            "root_wilson_95_interval": [lower, upper],
            "root_one_sided_exact_95_lower_bound": (
                one_sided_exact_lower_bound(accepted, len(row_indices))
            ),
            "selected_candidate_labels": selected_labels,
            "selected_deterministic_risk_total": selected_risk_total,
            "rejection_rule_counts": rules,
            "qwen": qwen,
            "luna": luna,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "claim_boundary": (
                "A smoke test proves protocol viability only."
                if smoke
                else "This is a selected-variant development canary requiring new-root confirmation."
            ),
        }

    def build_trace_context_compiler_v7_report(
        self,
        row_indices: Sequence[int],
        *,
        planned_root_count: int,
        stop_reason: str,
        strategy: str = "trace_context_compiler_v7",
        report_scope: str = "canary",
    ) -> dict[str, Any]:
        candidate_ids_by_row = {
            index: stable_trace_candidate_id(
                root_context_id(self.rows[index]),
                self.config.pipeline_epoch,
                strategy,
            )
            for index in row_indices
        }
        candidate_ids = set(candidate_ids_by_row.values())
        qwen = self._request_metrics_for_candidates("qwen", candidate_ids)
        luna = self._request_metrics_for_candidates("luna", candidate_ids)
        accepted = 0
        judged = 0
        score_band_counts = {
            "positive": 0,
            "negative": 0,
            "ambiguous": 0,
            "unusable_zero": 0,
        }
        rules: dict[str, int] = {}
        route_counts: dict[str, int] = {}
        guard_totals: dict[str, int] = {}
        packet_totals: dict[str, dict[str, int]] = {}
        candidate_count = 0
        for row_index in row_indices:
            route = str(compile_context_profile(self.rows[row_index])["task_type"])
            route_counts[route] = route_counts.get(route, 0) + 1
            candidate, result = self._trace_result(row_index, strategy)
            candidate_count += int(candidate is not None)
            if candidate is not None:
                for key, value in (candidate.get("local_guard_stats") or {}).items():
                    if isinstance(value, int):
                        guard_totals[str(key)] = guard_totals.get(str(key), 0) + value
                for source, stats in (candidate.get("evidence_packet_stats") or {}).items():
                    source_totals = packet_totals.setdefault(str(source), {})
                    if isinstance(stats, dict):
                        for key, value in stats.items():
                            if isinstance(value, int):
                                source_totals[str(key)] = (
                                    source_totals.get(str(key), 0) + value
                                )
            if result is None or result.get("status") != "ok":
                continue
            judged += 1
            score_band_counts[
                classify_kto_score(float(result.get("total_score_20", 0.0)))
            ] += 1
            is_accepted = bool(result.get("accepted", False))
            accepted += int(is_accepted)
            if not is_accepted:
                for rule_id in result.get("diagnosis", {}).get("rule_ids", []):
                    key = str(rule_id)
                    rules[key] = rules.get(key, 0) + 1

        qwen_events_by_candidate: dict[str, int] = {}
        luna_events_by_candidate: dict[str, int] = {}
        for candidate_id in candidate_ids:
            qwen_events_by_candidate[candidate_id] = sum(
                event.get("provider") == "qwen"
                and candidate_id in str(event.get("operation_id", ""))
                for event in self.gateway.events
            )
            luna_events_by_candidate[candidate_id] = sum(
                event.get("provider") == "luna"
                and candidate_id in str(event.get("operation_id", ""))
                for event in self.gateway.events
            )
        successful_candidate_ids = {
            str(candidate["candidate_id"])
            for row_index in row_indices
            if (candidate := self._find_candidate(row_index, strategy)) is not None
        }
        successful_qwen_candidate_ids = {
            candidate_id
            for candidate_id in candidate_ids
            if any(
                event.get("provider") == "qwen"
                and event.get("status") == "ok"
                and candidate_id in str(event.get("operation_id", ""))
                for event in self.gateway.events
            )
        }
        qwen_requests = int(qwen["requests_including_retries"])
        luna_requests = int(luna["requests_including_retries"])
        exclude_sampling_infrastructure = strategy in {
            "trace_relevance_grounded_positive_v29",
            "trace_relevance_grounded_positive_v30",
        }
        relevant_qwen_events = [
            event
            for event in self.gateway.events
            if event.get("provider") == "qwen"
            and any(
                candidate_id in str(event.get("operation_id", ""))
                for candidate_id in candidate_ids
            )
        ]
        (
            sampling_quality_denominator_requests,
            excluded_sampling_infrastructure_failures,
        ) = sampling_quality_denominator(
            relevant_qwen_events,
            exclude_infrastructure_failures=exclude_sampling_infrastructure,
        )
        judge_infrastructure_candidate_ids = {
            candidate_id
            for candidate_id in candidate_ids
            if any(
                event.get("provider") == "luna"
                and candidate_id in str(event.get("operation_id", ""))
                and is_infrastructure_failure_event(event)
                for event in self.gateway.events
            )
        }
        excluded_judge_infrastructure_events = sum(
            event.get("provider") == "luna"
            and is_infrastructure_failure_event(event)
            and any(
                candidate_id in str(event.get("operation_id", ""))
                for candidate_id in candidate_ids
            )
            for event in self.gateway.events
        )
        quality_denominator_requests = max(
            0,
            sampling_quality_denominator_requests
            - len(judge_infrastructure_candidate_ids),
        )
        call_protocol_checks = {
            "retries_disabled": self.config.max_attempts_per_operation == 1,
            "one_qwen_event_per_attempted_root": (
                qwen_requests == len(row_indices)
                and all(value == 1 for value in qwen_events_by_candidate.values())
            ),
            "two_luna_events_per_successful_generation": (
                luna_requests == 2 * len(successful_qwen_candidate_ids)
                and all(
                    luna_events_by_candidate[candidate_id] == 2
                    for candidate_id in successful_qwen_candidate_ids
                )
            ),
            "no_luna_for_failed_generation": all(
                luna_events_by_candidate[candidate_id] == 0
                for candidate_id in candidate_ids - successful_qwen_candidate_ids
            ),
            "candidate_persisted_for_each_successful_generation": (
                successful_candidate_ids == successful_qwen_candidate_ids
            ),
            "one_terminal_judge_result_per_successful_generation": sum(
                self._find_judge(candidate_id, 0) is not None
                for candidate_id in successful_candidate_ids
            )
            == candidate_count,
            "judge_repeats_frozen_to_one": self.config.judge_repeats == 1,
            "judge_prompt_source_frozen_snapshot": True,
        }
        score_completeness_checks = {
            "one_completed_judge_result_per_successful_generation": sum(
                (
                    result := self._find_judge(candidate_id, 0)
                ) is not None
                and result.get("status") == "ok"
                for candidate_id in successful_candidate_ids
            )
            == candidate_count,
        }
        protocol_checks = {**call_protocol_checks, **score_completeness_checks}
        efficiency = (
            accepted / quality_denominator_requests
            if quality_denominator_requests
            else 0.0
        )
        lower, upper = wilson_interval(accepted, quality_denominator_requests)
        return {
            "phase": f"{strategy}_fixed_1q_2l_{report_scope}",
            "pipeline_epoch": self.config.pipeline_epoch,
            "created_at": utc_now(),
            "metric_definition": (
                "accepted outputs divided by infrastructure-clean root attempts; sampler "
                "or judge service, network, container, auth, and capacity failures are "
                "reported separately and excluded by affected root, while HTTP-200 "
                "semantic failures remain in the denominator"
                if exclude_sampling_infrastructure
                else "accepted outputs divided by all bottom-level Qwen API request events; "
                "failed and interrupted Qwen attempts remain in the denominator"
            ),
            "target_at_least": TARGET_ACCEPTS_PER_QWEN_REQUEST,
            "minimum_accepts_for_planned_roots": minimum_accepts_for_target(
                planned_root_count
            ),
            "planned_root_count": planned_root_count,
            "attempted_root_count": len(row_indices),
            "stop_reason": stop_reason,
            "candidate_count": candidate_count,
            "judged_outputs": judged,
            "accepted_outputs": accepted,
            "score_band_counts": score_band_counts,
            "qwen_requests_including_failures": qwen_requests,
            "quality_denominator_sampling_requests": quality_denominator_requests,
            "quality_denominator_evaluable_roots": quality_denominator_requests,
            "sampling_requests_after_sampler_infrastructure_exclusions": (
                sampling_quality_denominator_requests
            ),
            "excluded_sampling_infrastructure_failures": (
                excluded_sampling_infrastructure_failures
            ),
            "excluded_judge_infrastructure_events": (
                excluded_judge_infrastructure_events
            ),
            "excluded_judge_infrastructure_roots": len(
                judge_infrastructure_candidate_ids
            ),
            "luna_requests_including_failures": luna_requests,
            "user_efficiency_accepts_per_qwen_request": efficiency,
            "target_point_rate_meets_80_percent": (
                quality_denominator_requests > 0
                and efficiency >= TARGET_ACCEPTS_PER_QWEN_REQUEST
            ),
            "request_denominator_wilson_95_interval": [lower, upper],
            "request_denominator_one_sided_exact_95_lower_bound": (
                one_sided_exact_lower_bound(accepted, quality_denominator_requests)
            ),
            "one_sided_95_lower_bound_meets_80_percent": (
                one_sided_exact_lower_bound(accepted, quality_denominator_requests)
                >= TARGET_ACCEPTS_PER_QWEN_REQUEST
            ),
            "rejection_rule_counts": rules,
            "context_route_counts": route_counts,
            "label_free_output_guard_totals": guard_totals,
            "evidence_packet_totals": packet_totals,
            "protocol_checks": protocol_checks,
            "call_protocol_checks": call_protocol_checks,
            "score_completeness_checks": score_completeness_checks,
            "strict_call_protocol_satisfied": all(call_protocol_checks.values()),
            "complete_scoring_satisfied": all(score_completeness_checks.values()),
            "strict_protocol_satisfied": all(protocol_checks.values()),
            "qwen": qwen,
            "luna": luna,
            "actual_experiment_request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
            "claim_boundary": (
                "This is a frozen-strategy evaluation on an exact-root-disjoint validation "
                "split. The point-rate and one-sided lower-bound gates are reported separately."
                if report_scope == "validation"
                else "This is a frozen-strategy and frozen-rate-policy evaluation on an "
                "exact-root-disjoint audit split. No audit result may be used to revise "
                "the reported epoch."
                if report_scope == "audit"
                else "This is label-free prompt development on previously unused development "
                "roots. A point estimate above 80% selects the strategy for independent "
                "validation; it does not by itself prove population performance."
            ),
        }

    def build_report(self, phase: str, row_indices: Sequence[int]) -> dict[str, Any]:
        strategies = ["legacy"] if phase == "canary" else [
            "legacy",
            "corrected_v5",
            "self_review_v5",
        ]
        report: dict[str, Any] = {
            "phase": phase,
            "created_at": utc_now(),
            "row_count": len(row_indices),
            "row_indices": list(row_indices),
            "pass_threshold_strictly_greater_than": self.config.pass_threshold,
            "strategies": {},
            "request_usage": {
                provider: self.gateway._request_count(provider)
                for provider in ("qwen", "luna")
            },
        }
        for strategy in strategies:
            sample_summaries: list[dict[str, Any]] = []
            for row_index in row_indices:
                candidate = self._find_candidate(row_index, strategy)
                if candidate is None:
                    continue
                results = [
                    result
                    for result in self.judges
                    if result.get("candidate_id") == candidate["candidate_id"]
                ]
                if phase == "compare":
                    results = [result for result in results if result.get("repeat_index") == 0]
                scores = [float(result.get("total_score_20", 0.0)) for result in results]
                successful = bool(results) and all(result.get("status") == "ok" for result in results)
                mean_score = sum(scores) / len(scores) if scores else 0.0
                sample_summaries.append(
                    {
                        "row_index": row_index,
                        "candidate_id": candidate["candidate_id"],
                        "repeat_scores": scores,
                        "mean_score": mean_score,
                        "accepted": successful and is_kto_accepted_score(mean_score),
                        "kto_score_band": (
                            classify_kto_score(mean_score) if successful else None
                        ),
                    }
                )
            passed = sum(1 for sample in sample_summaries if sample["accepted"])
            report["strategies"][strategy] = {
                "samples": len(sample_summaries),
                "passed": passed,
                "pass_rate": passed / len(sample_summaries) if sample_summaries else 0.0,
                "mean_score": (
                    sum(sample["mean_score"] for sample in sample_summaries)
                    / len(sample_summaries)
                    if sample_summaries
                    else 0.0
                ),
                "sample_results": sample_summaries,
            }
        if phase == "compare":
            corrected = {
                item["row_index"]: item
                for item in report["strategies"]["corrected_v5"]["sample_results"]
            }
            reviewed = {
                item["row_index"]: item
                for item in report["strategies"]["self_review_v5"]["sample_results"]
            }
            fallback_rows = [index for index in row_indices if not corrected[index]["accepted"]]
            selected_results = [
                corrected[index] if corrected[index]["accepted"] else reviewed[index]
                for index in row_indices
            ]
            adaptive_passed = sum(1 for item in selected_results if item["accepted"])
            adaptive_rate = adaptive_passed / len(row_indices) if row_indices else 0.0
            lower, upper = wilson_interval(adaptive_passed, len(row_indices))
            # corrected_v5 costs one Qwen request. The frozen self_review_v5
            # fallback costs a new draft plus one revision, hence two Qwen calls.
            observed_qwen_calls = len(row_indices) + 2 * len(fallback_rows)
            observed_luna_calls = 2 * len(row_indices) + 2 * len(fallback_rows)
            target_accepted = 25_000
            projected_raw_rows = (
                math.ceil(target_accepted / adaptive_rate) if adaptive_rate else None
            )
            fallback_rate = len(fallback_rows) / len(row_indices) if row_indices else 0.0
            report["adaptive"] = {
                "name": "corrected_v5_then_self_review_v5_on_score_le_14",
                "passed": adaptive_passed,
                "samples": len(row_indices),
                "pass_rate": adaptive_rate,
                "wilson_95_interval": [lower, upper],
                "fallback_rows": fallback_rows,
                "fallback_rate": fallback_rate,
                "observed_request_equivalent": {
                    "qwen": observed_qwen_calls,
                    "luna": observed_luna_calls,
                },
                "projected_for_25000_accepted_at_observed_rates": {
                    "raw_rows": projected_raw_rows,
                    "qwen_requests": (
                        math.ceil(projected_raw_rows * (1.0 + 2.0 * fallback_rate))
                        if projected_raw_rows is not None
                        else None
                    ),
                    "luna_requests": (
                        math.ceil(projected_raw_rows * (2.0 + 2.0 * fallback_rate))
                        if projected_raw_rows is not None
                        else None
                    ),
                },
                "claim_boundary": (
                    "Observed pilot projection only; 12 samples are insufficient to prove "
                    "the population pass rate exceeds 80%."
                ),
            }
        return report
