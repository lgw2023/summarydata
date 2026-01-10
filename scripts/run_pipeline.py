from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Dict, List, Sequence, Any

from collections import defaultdict
import logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.data_loader.excel_loader import Sample, SampleLoader, export_samples
from src.data_loader.context_builder import (
    build_context_samples,
    export_context_samples_jsonl,
)
from src.generators.base import build_generators, Candidate, generate_candidates
from src.utils.io import write_jsonl, ensure_dir, read_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env

from prompts.system_prompt_v5_yixuan import build_phone_personal_prompt
from prompts.system_prompt_v5_yixuan import build_phone_general_prompt
from prompts.system_prompt_v5_yixuan import build_watch_personal_prompt
from prompts.system_prompt_v5_yixuan import build_watch_general_prompt
from prompts.system_prompt_v5_yixuan import DOMAIN_DESCRIPTION as _DOMAIN_DESCRIPTION_MAP


# generate 阶段增量写盘时，每批并行生成的样本数（通过环境变量可覆盖）
try:
    DEFAULT_GENERATE_BATCH_SIZE = max(
        1, int(os.getenv("GENERATE_BATCH_SIZE", "16"))
    )
except ValueError:
    DEFAULT_GENERATE_BATCH_SIZE = 16


# ===== Gemini 2.5 特殊后处理：裁剪掉「思考过程」 =====
_GEMINI_MODEL_NAME_FROM_ENV = os.getenv("LLM_MODEL_GEMINI25_NAME")
# 与默认配置文件中保持兼容
_GEMINI_MODEL_DEFAULT_NAME = "google/gemini-2.5-flash"
_GEMINI_MODEL_NAMES = {
    name for name in (_GEMINI_MODEL_NAME_FROM_ENV, _GEMINI_MODEL_DEFAULT_NAME) if name
}

_SAMPLE_ID_VARIANT_SEP = "::"


def _extract_reference_answer_keys(generator_cfgs: Sequence[dict[str, Any]]) -> list[str]:
    """
    从 YAML 配置（pipeline.generators）中提取 reference 生成器的 answer_key 列名列表。

    设计目标：
    - 参考答案的“列名”必须以 YAML 为准，避免在代码中写死 a_answer/b_answer/answer_phone 等；
    - 允许存在多个 reference 生成器，对应多个参考答案列；
    - 去重并保持出现顺序。
    """
    keys: list[str] = []
    seen: set[str] = set()
    for cfg in generator_cfgs:
        # 兼容旧配置 name: reference，以及按设备拆分后的：
        # - reference_phone
        # - reference_watch
        if cfg.get("name") not in {"reference", "reference_phone", "reference_watch"}:
            continue
        k = str(cfg.get("answer_key") or "").strip()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        keys.append(k)
    return keys


def _normalize_table_cell_to_str(v: Any) -> str | None:
    """
    将 CSV/Excel 读取出来的单元格值归一为 str | None。
    - pandas 的 NaN/NaT -> None
    - 数字/布尔值 -> str
    """
    if v is None:
        return None
    # NaN(float) 判断：v != v 仅对 NaN 成立
    if isinstance(v, float) and v != v:
        return None
    try:
        import pandas as pd  # type: ignore

        if bool(pd.isna(v)):
            return None
    except Exception:
        pass
    s = str(v)
    return s if s != "" else None


def _normalize_sample_id_value(v: Any, fallback: str) -> str:
    """
    统一处理 sample_id：
    - Excel 中常见的 1 / 1.0 -> "1"
    - 其他情况按字符串处理；空值则回退 fallback
    """
    if v is None:
        return fallback
    if isinstance(v, float):
        if v != v:  # NaN
            return fallback
        if v.is_integer():
            return str(int(v))
    if isinstance(v, int):
        return str(v)
    s = str(v).strip()
    return s or fallback


def _load_raw_fieldnames(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "读取 Excel 需要依赖 pandas + openpyxl。请先安装：pip install pandas openpyxl"
            ) from exc
        df = pd.read_excel(path, nrows=0)
        return [str(c) for c in df.columns.tolist()]

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def _iter_raw_rows(path: Path) -> tuple[list[str], Iterable[dict[str, Any]]]:
    """
    统一读取 CSV/Excel 原始行（用于 patch 回溯等场景）。
    返回 (fieldnames, rows_iterable)。
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "读取 Excel 需要依赖 pandas + openpyxl。请先安装：pip install pandas openpyxl"
            ) from exc
        df = pd.read_excel(path, dtype=object)
        fieldnames = [str(c) for c in df.columns.tolist()]
        return fieldnames, df.to_dict(orient="records")

    def _csv_iter() -> Iterable[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, _csv_iter()


def _extract_base_sample_id(sample_id: Any) -> str:
    """
    将形如 "<base>::phone" / "<base>::watch" 的变体 sample_id 还原成 base id。
    若不符合该格式，则原样返回。
    """
    sid = str(sample_id or "").strip()
    if not sid:
        return ""
    if _SAMPLE_ID_VARIANT_SEP in sid:
        return sid.split(_SAMPLE_ID_VARIANT_SEP, 1)[0]
    return sid


def _extract_variant_device_from_sample_id(sample_id: Any) -> str:
    """
    从形如 "<base>::phone" / "<base>::watch" 的 sample_id 中提取 device。
    若不符合该格式，则返回空字符串。
    """
    sid = str(sample_id or "").strip()
    if not sid or _SAMPLE_ID_VARIANT_SEP not in sid:
        return ""
    device = sid.split(_SAMPLE_ID_VARIANT_SEP, 1)[1].strip().lower()
    return device


def _has_personal_data(sample: Sample) -> bool:
    return bool((sample.data or "").strip())


def _normalize_domain(domain: Any) -> str | None:
    d = str(domain).strip() if domain is not None else ""
    return d or None


def _remove_trailing_commas_in_json(text: str) -> str:
    """
    允许 prompts/domain.jsonl 里出现诸如：
    - {"a": 1,}
    - [1,2,]
    这类非严格 JSON 的尾随逗号，解析前做一次清理。
    """
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _extract_top_level_json_objects(text: str) -> list[str]:
    """
    从文本中提取所有顶层 JSON 对象（以 { } 包裹），按出现顺序返回。
    用于解析 prompts/domain.jsonl（里面包含多个 JSON 对象 + 注释）。
    """
    objs: list[str] = []
    buf: list[str] = []
    level = 0
    in_str = False
    esc = False

    for ch in text:
        if level == 0:
            if ch == "{":
                level = 1
                buf = ["{"]
                in_str = False
                esc = False
            continue

        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            level += 1
            continue
        if ch == "}":
            level -= 1
            if level == 0:
                objs.append("".join(buf))
                buf = []

    return objs


@lru_cache(maxsize=1)
def _load_domain_specs() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    读取 prompts/domain.jsonl，得到：
    - domain_groups: {"健康":[...], "运动":[...], "其他":[...]}
    - subdomain_map: {"跑步":[...], "骑行":[...], ...}

    若读取失败，则返回空 dict，逻辑会回退到兜底策略。
    """
    domain_path = PROJECT_ROOT / "prompts" / "domain.jsonl"
    try:
        raw = domain_path.read_text(encoding="utf-8")
    except Exception:
        return {}, {}

    # 去掉注释行（以 # 开头）
    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        if line.lstrip().startswith("#"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    objs = _extract_top_level_json_objects(cleaned)
    if len(objs) < 2:
        return {}, {}

    try:
        domain_groups = json.loads(_remove_trailing_commas_in_json(objs[0]))
        subdomain_map = json.loads(_remove_trailing_commas_in_json(objs[1]))
    except Exception:
        return {}, {}

    if not isinstance(domain_groups, dict) or not isinstance(subdomain_map, dict):
        return {}, {}

    # 规范化为 dict[str, list[str]]
    dg: dict[str, list[str]] = {}
    for k, v in domain_groups.items():
        if isinstance(k, str) and isinstance(v, list):
            dg[k] = [str(x) for x in v if x is not None]

    sd: dict[str, list[str]] = {}
    for k, v in subdomain_map.items():
        if isinstance(k, str) and isinstance(v, list):
            sd[k] = [str(x) for x in v if x is not None]

    return dg, sd


def _resolve_domain_for_prompt(domain: Any) -> tuple[str | None, str | None]:
    """
    将输入 CSV 的 domain 映射到 system_prompt_v5_yixuan.DOMAIN_DESCRIPTION 的 key。

    返回 (domain_key, domain_description)：
    - 若 domain_key 能命中预置字典，则 domain_description 为对应说明；
    - 若未命中，则仍返回一段最小可用的说明，避免“领域说明”留空。
    """
    raw = _normalize_domain(domain)
    if not raw:
        return None, None

    # 命中预置字典
    if raw in _DOMAIN_DESCRIPTION_MAP:
        return raw, _DOMAIN_DESCRIPTION_MAP.get(raw) or ""

    domain_groups, subdomain_map = _load_domain_specs()
    health_domains = set(domain_groups.get("健康") or [])
    sport_domains = set(domain_groups.get("运动") or [])
    other_domains = set(domain_groups.get("其他") or [])

    # 1) 规范 domain（健康类）通常与 DOMAIN_DESCRIPTION key 一致
    if raw in health_domains and raw in _DOMAIN_DESCRIPTION_MAP:
        return raw, _DOMAIN_DESCRIPTION_MAP.get(raw) or ""

    # 2) 运动大类：跑步/骑行/子类（户外跑步等）统一映射到“运动”
    is_sport = raw in sport_domains
    if not is_sport:
        # 原始 domain 是某个运动的 subdomain（如“户外跑步”）
        for subs in subdomain_map.values():
            if raw in subs:
                is_sport = True
                break
    if is_sport and "运动" in _DOMAIN_DESCRIPTION_MAP:
        return "运动", _DOMAIN_DESCRIPTION_MAP.get("运动") or ""

    # 3) 其他
    if raw in other_domains and "其他" in _DOMAIN_DESCRIPTION_MAP:
        return "其他", _DOMAIN_DESCRIPTION_MAP.get("其他") or ""

    # 兜底：直接把原 domain 写进领域说明
    return None, ""


def _expand_samples_phone_watch(samples: Sequence[Sample]) -> List[Sample]:
    """
    将每条原始样本拆成两条独立样本：
    - <base>::phone
    - <base>::watch

    同时按规则写入每条样本的 system_prompt：
    - data 非空 -> personal prompt（并按 domain 注入领域说明）
    - data 为空 -> general prompt
    """
    expanded: List[Sample] = []
    for s in samples:
        base_id = str(s.sample_id)
        has_personal = _has_personal_data(s)
        raw_domain = _normalize_domain(getattr(s, "domain", None))
        domain_key, domain_desc = _resolve_domain_for_prompt(raw_domain)

        for device in ("phone", "watch"):
            system_prompt_type = f"{device}_{'personal' if has_personal else 'general'}"
            # 仅 personal prompt 才涉及 domain；general 置为空串
            system_prompt_domain = (domain_key or raw_domain or "") if has_personal else ""
            if has_personal:
                system_prompt = (
                    build_phone_personal_prompt(domain=domain_key, domain_description=domain_desc)
                    if device == "phone"
                    else build_watch_personal_prompt(domain=domain_key, domain_description=domain_desc)
                )
            else:
                system_prompt = (
                    build_phone_general_prompt()
                    if device == "phone"
                    else build_watch_general_prompt()
                )

            expanded.append(
                replace(
                    s,
                    sample_id=f"{base_id}{_SAMPLE_ID_VARIANT_SEP}{device}",
                    base_sample_id=base_id,
                    device=device,
                    system_prompt=system_prompt,
                    system_prompt_type=system_prompt_type,
                    system_prompt_domain=system_prompt_domain,
                )
            )
    return expanded


def _postprocess_think_tags(candidates: Iterable[Candidate]) -> List[Candidate]:
    """
    - 若响应中包含 "</think>"，仅保留该标记之后的内容（不含 "</think>" 本身）。
    """
    THINK_END = "</think>"
    processed: List[Candidate] = []
    for cand in candidates:
        if cand.model_name in _GEMINI_MODEL_NAMES and THINK_END in cand.response:
            idx = cand.response.rfind(THINK_END)
            # 仅保留 </think> 之后的文本，并去掉前后空白
            new_resp = cand.response[idx + len(THINK_END) :].lstrip()
            cand.response = new_resp
        processed.append(cand)
    return processed


def run_generation(samples, generator_configs) -> list[Candidate]:
    """
    在同一模型内按样本批量生成，不同模型之间并发执行。
    同时对特定模型（如 Gemini 2.5）做统一后处理。
    """
    generators = build_generators(generator_configs)
    raw_candidates = list(generate_candidates(samples, generators))
    return _postprocess_think_tags(raw_candidates)

def _load_completed_generated_rows(config: PipelineConfig) -> Dict[str, Dict]:
    """
    从已有的 generated_responses.jsonl 中筛选出「已完成所有当前配置生成器」的样本行。

    设计原则：
    - 仅当某个 sample_id 对所有当前 PipelineConfig.generators 中的 (model_type, model_name)
      至少各有一条候选时，才认为该样本“已完成”，后续跑 generate 阶段时可以跳过重新生成；
    - 若生成器配置发生变化（新增 / 删除模型，或修改 model_name），则旧结果不再视为完整，
      会对对应样本重新生成，从而保证配置变更后结果不会“错误复用”；
    - 对于此前中途失败、候选不全的样本，不会出现在返回结果里，后续会整条样本重新生成。
    """
    logger = logging.getLogger(__name__)
    output_path = config.output_files.generated_responses
    rows = read_jsonl(output_path)
    if not rows:
        return {}

    # 针对可能存在的「同一 sample_id 多条记录」场景，保留**最后一条**作为最新结果。
    latest_rows_by_id: Dict[str, Dict] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if not sample_id:
            continue
        latest_rows_by_id[sample_id] = row

    # 根据当前配置构建一次 generator，仅用于拿到 (model_type, model_name) 组合。
    generators = build_generators(config.generators)
    if not generators:
        return {}

    # “完成度”仅用于决定是否跳过大模型生成；
    # reference* 候选属于“表格派生数据”，应允许随原始表格变化而刷新，因此不纳入完成度判断。
    #
    # 注意：由于样本会被拆成 <id>::phone / <id>::watch 两条变体，
    # 完成度也必须按 device 过滤后判定，否则在同时包含 phone+watch 生成器的配置下，
    # 单条变体样本会永远无法覆盖“全局期望集合”，从而无法被识别为 completed。
    def _is_reference_model_type(model_type: Any) -> bool:
        mt = str(model_type or "").strip().lower()
        return bool(mt) and mt.startswith("reference")

    non_ref_generators = [g for g in generators if not _is_reference_model_type(getattr(g, "model_type", None))]
    if not non_ref_generators:
        return {}

    def _expected_pairs_for_device(device: str) -> set[tuple[str, str]]:
        d = (device or "").strip().lower()
        pairs: set[tuple[str, str]] = set()
        for g in non_ref_generators:
            td = getattr(g, "target_device", None)
            if isinstance(td, str) and td.strip():
                # sample 的 device 为空（旧数据）时不过滤，尽量保持兼容
                if d and d != td.strip().lower():
                    continue
            pairs.add((g.model_type, g.model_name))
        return pairs

    completed: Dict[str, Dict] = {}
    for sample_id, row in latest_rows_by_id.items():
        candidates = row.get("candidates") or []
        seen_pairs = {
            (str(c.get("model_type")), str(c.get("model_name"))) for c in candidates
        }
        device = _extract_variant_device_from_sample_id(sample_id)
        expected_pairs = _expected_pairs_for_device(device)
        if not expected_pairs:
            # 没有“适用于该样本 device 的非 reference 生成器”时，不将其视为 completed
            # （保持与此前：无 expected_pairs 则不复用的策略一致）
            continue
        # 仅当所有当前配置（按 device 过滤后）的 (model_type, model_name) 都已出现时，视为该样本“完整”
        if expected_pairs.issubset(seen_pairs):
            completed[sample_id] = row
    if completed:
        def _try_int(x):
            try:
                return int(x)
            except Exception:
                return x
        sample_ids_preview = sorted(completed.keys()) # , key=_try_int)
        logger.info(
            "Loaded %d completed samples from existing generated_responses file %s. "
            "Sample IDs:\n%s",
            len(completed),
            output_path,
            ", ".join(sample_ids_preview),
        )
    return completed


def _load_latest_generated_rows(path: str | Path) -> Dict[str, Dict]:
    """
    从 generated_responses.jsonl 读取并返回「每个 sample_id 的最新一条记录」。
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    rows = read_jsonl(path_obj)
    if not rows:
        return {}
    latest: Dict[str, Dict] = {}
    for row in rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        latest[sid] = row
    return latest


def _refresh_reference_and_compact_generated_responses(
    *,
    config: PipelineConfig,
    samples: Sequence[Sample],
    context_lookup: Dict[str, str],
) -> int:
    """
    在不重跑大模型的前提下：
    - 依据 YAML 中 reference 生成器的 answer_key，从当前 raw_data 重新抽取参考答案（已在 Sample 中）；
    - 重新生成 reference 候选，并与“最新的非 reference 候选”合并；
    - 同时将 generated_responses.jsonl 压缩为“每个 sample_id 仅保留最新一条”（避免人工误读旧行）。
    """
    logger = logging.getLogger(__name__)

    output_path = Path(config.output_files.generated_responses)
    latest_rows_by_id = _load_latest_generated_rows(output_path)

    try:
        generators = build_generators(config.generators)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to build generators for reference refresh: %r", exc)
        return 0

    ref_gens = [
        g
        for g in generators
        if str(getattr(g, "model_type", "") or "").strip().lower().startswith("reference")
    ]
    if not ref_gens and not latest_rows_by_id:
        return 0

    ensure_dir(output_path.parent)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    updated = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            sid = str(sample.sample_id)
            existing_row = latest_rows_by_id.get(sid) or {}
            existing_candidates = existing_row.get("candidates") or []

            def _is_reference_candidate(c: dict[str, Any]) -> bool:
                mt = str(c.get("model_type") or "").strip().lower()
                return bool(mt) and mt.startswith("reference")

            # 保留旧的非 reference* 候选（大模型输出），reference* 用最新值覆盖
            kept_non_ref: list[dict[str, Any]] = [
                c for c in existing_candidates if not _is_reference_candidate(c)
            ]
            old_ref_by_name: dict[str, dict[str, Any]] = {
                str(c.get("model_name") or ""): c
                for c in existing_candidates
                if _is_reference_candidate(c) and str(c.get("model_name") or "")
            }

            # 仅对当前 device 生成对应 reference，避免 phone 样本混入 watch reference（反之亦然）
            sample_device = (getattr(sample, "device", None) or "").strip().lower()
            applicable_ref_gens = []
            for g in ref_gens:
                td = getattr(g, "target_device", None)
                if isinstance(td, str) and td.strip():
                    # 若 sample.device 为空（旧数据），则不启用过滤；否则严格匹配
                    if sample_device and sample_device == td.strip().lower():
                        applicable_ref_gens.append(g)
                else:
                    applicable_ref_gens.append(g)

            new_ref: list[Candidate] = [g.generate(sample) for g in applicable_ref_gens]

            changed = False
            if applicable_ref_gens:
                for c in new_ref:
                    old = old_ref_by_name.get(str(c.model_name))
                    old_resp = str((old or {}).get("response") or "")
                    if old is None or old_resp != str(c.response or ""):
                        changed = True
                        break

            if changed:
                updated += 1

            row = {
                "sample_id": sid,
                "system_prompt": str(getattr(sample, "system_prompt", "") or ""),
                "system_prompt_type": str(getattr(sample, "system_prompt_type", "") or ""),
                "system_prompt_domain": str(getattr(sample, "system_prompt_domain", "") or ""),
                "context": context_lookup.get(sid, ""),
                "question": sample.query,
                "candidates": kept_non_ref + [c.to_dict() for c in new_ref],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tmp_path.replace(output_path)
    if ref_gens:
        logger.info(
            "Refreshed reference candidates (from YAML answer_key) for %d samples; compacted output to %s",
            updated,
            output_path,
        )
    else:
        logger.info("Compacted generated_responses.jsonl to latest rows: %s", output_path)
    return updated


def run_pipeline(
    config_path: str | Path,
    raw_data_path: str | Path,
    stage: str = "all",
    max_rows: int | None = 3,
) -> None:
    """
    一键运行 pipeline，支持通过 --stage 只执行当前阶段。

    stage 取值：
    - generate：从原始数据构建样本 + 上下文，并生成候选回复，结果写入 JSONL；
    - all（默认）：等价于 generate。
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Loading pipeline config from %s (stage=%s, raw_data=%s)",
        config_path,
        stage,
        raw_data_path,
    )
    config = PipelineConfig.from_yaml(config_path, raw_data_path=raw_data_path)

    # ==== reference 生成器动态降级（严格按 YAML 的 answer_key 判断列是否存在） ====
    try:
        fieldnames = set(_load_raw_fieldnames(config.raw_data))
    except Exception as exc:  # pragma: no cover - 防御性日志
        fieldnames = set()
        logging.getLogger(__name__).warning(
            "Failed to read raw_data header for reference generator check: %r", exc
        )

    if fieldnames:
        original_generators = list(config.generators)
        filtered_generators: list[dict[str, Any]] = []
        skipped: list[str] = []

        for gen_cfg in original_generators:
            if gen_cfg.get("name") in {"reference", "reference_phone", "reference_watch"}:
                answer_key = str(gen_cfg.get("answer_key") or "").strip()
                model_name = str(gen_cfg.get("model_name") or "").strip()
                # 若 YAML 明确指定 answer_key，则以其作为“参考答案列名”的唯一来源
                if answer_key and answer_key not in fieldnames:
                    skipped.append(model_name or answer_key)
                    continue

            filtered_generators.append(gen_cfg)

        if skipped:
            logger.info(
                "输入文件 %s 中未找到参考答案列，以下 reference 生成器将被跳过：%s",
                config.raw_data,
                ", ".join(sorted(skipped)),
            )
        config.generators = filtered_generators

    ensure_dir(config.processed_dir)
    ensure_dir(config.intermediate_dir)

    # ==== Stage: generate ====
    if stage in ("all", "generate"):
        # 参考答案列名完全由 YAML 决定（reference generator.answer_key）
        reference_cols = _extract_reference_answer_keys(config.generators)
        raw_samples = SampleLoader(config.raw_data, reference_columns=reference_cols).load(max_rows=max_rows)
        logger.info("Loaded %d raw samples from %s", len(raw_samples), config.raw_data)

        # 按“手机/手表 × 是否含个人数据”规则拆分样本，并注入 system_prompt
        samples = _expand_samples_phone_watch(raw_samples)
        logger.info(
            "Expanded raw samples into %d device variants (phone/watch for each row).",
            len(samples),
        )
        # 1) 导出原始样本快照，便于后续对齐 & 调试
        write_jsonl(config.output_files.samples, export_samples(samples))

        # 2) 构建并导出上下文样本（TASK.md 2.2）
        context_samples = build_context_samples(samples)
        export_context_samples_jsonl(config.intermediate_dir, context_samples)
        context_lookup: Dict[str, str] = {cs.sample_id: cs.context for cs in context_samples}

        # 3) 断点续跑 + 实时写盘：
        #    - 先从已有 generated_responses.jsonl 中识别出“已完成”的样本；
        #    - 对未完成样本按批次调用原有的并行生成逻辑（run_generation）；
        #    - 每个样本生成完一整批候选后立刻 append 写入 JSONL；
        #    - 若本次运行中途被中断，下次重跑会自动跳过已完成样本，仅补齐缺失部分。
        completed_rows_by_id = _load_completed_generated_rows(config)
        completed_sample_ids = set(completed_rows_by_id.keys())
        if completed_sample_ids:
            logger.info(
                "Detected %d completed samples from previous run in %s, "
                "will skip regeneration for them.",
                len(completed_sample_ids),
                config.output_files.generated_responses,
            )

        # 仅对“未完成”的样本重新生成；完整样本后续 judge/rank 直接复用旧结果。
        samples_to_generate: List[Sample] = [
            s for s in samples if s.sample_id not in completed_sample_ids
        ]

        if not samples_to_generate:
            logger.info(
                "All %d samples already have completed generated responses for current config. "
                "Nothing to regenerate.",
                len(samples),
            )
        else:
            output_path = config.output_files.generated_responses
            ensure_dir(output_path.parent)

            total_samples = len(samples)
            reused_count = len(completed_sample_ids)
            newly_generated_count = 0

            batch_size = DEFAULT_GENERATE_BATCH_SIZE
            logger.info(
                "Starting incremental generation for %d samples (batch_size=%d, reused_from_previous_runs=%d)",
                len(samples_to_generate),
                batch_size,
                reused_count,
            )

            # 预先根据当前配置推导出“应生成的模型集合”，便于逐样本完成度判断。
            # 由于样本被拆成 <id>::phone / <id>::watch 两条变体，期望集合也必须按 device 过滤。
            generators_for_check: list[Any] = []
            try:
                generators_for_check = list(build_generators(config.generators))
            except Exception as exc:  # pragma: no cover - 防御性日志
                logger.warning("未能解析生成器配置以做完成度校验：%r", exc)

            def _expected_model_pairs_for_sample(sample: Sample) -> set[tuple[str, str]]:
                if not generators_for_check:
                    return set()
                d = (getattr(sample, "device", None) or "").strip().lower()
                pairs: set[tuple[str, str]] = set()
                for gen in generators_for_check:
                    td = getattr(gen, "target_device", None)
                    if isinstance(td, str) and td.strip():
                        if d and d != td.strip().lower():
                            continue
                    pairs.add((gen.model_type, gen.model_name))
                return pairs

            # 采用 append 方式实时写盘：每处理完一批样本，就将这些样本的完整候选写入文件。
            with Path(output_path).open("a", encoding="utf-8") as f:
                for start in range(0, len(samples_to_generate), batch_size):
                    batch = samples_to_generate[start : start + batch_size]
                    try:
                        batch_candidates = run_generation(batch, config.generators)
                    except Exception as exc:  # pragma: no cover - 防御性日志
                        logger.exception(
                            "run_generation failed on batch starting at index %d: %r",
                            start,
                            exc,
                        )
                        continue

                    grouped_candidates: Dict[str, List[Candidate]] = defaultdict(list)
                    for cand in batch_candidates:
                        grouped_candidates[cand.sample_id].append(cand)

                    for sample in batch:
                        sid = sample.sample_id
                        per_sample_candidates = grouped_candidates.get(sid, [])
                        if not per_sample_candidates:
                            logger.warning(
                                "No candidates generated for sample %s in current batch, skipping write for this sample.",
                                sid,
                            )
                            continue

                        row = {
                            "sample_id": sid,
                            "system_prompt": str(getattr(sample, "system_prompt", "") or ""),
                            "system_prompt_type": str(getattr(sample, "system_prompt_type", "") or ""),
                            "system_prompt_domain": str(getattr(sample, "system_prompt_domain", "") or ""),
                            "context": context_lookup.get(sid, ""),
                            "question": sample.query,
                            "candidates": [c.to_dict() for c in per_sample_candidates],
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        newly_generated_count += 1

                        # 当前样本的所有模型回复已就绪，打印一次标记便于前端观察进度
                        expected_model_pairs = _expected_model_pairs_for_sample(sample)
                        if expected_model_pairs:
                            generated_pairs = {
                                (c.model_type, c.model_name) for c in per_sample_candidates
                            }
                            matched_pairs = generated_pairs & expected_model_pairs
                            if expected_model_pairs.issubset(generated_pairs):
                                logger.info(
                                    "Sample %s 已完成所有模型生成（%d/%d），模型列表：%s",
                                    sid,
                                    len(matched_pairs),
                                    len(expected_model_pairs),
                                    ", ".join(sorted({c.model_name for c in per_sample_candidates})),
                                )

            logger.info(
                "Generation stage finished: total_samples=%d, newly_generated=%d, reused_from_previous_runs=%d",
                total_samples,
                newly_generated_count,
                reused_count,
            )

        # generate 阶段结束后：按 YAML 配置的 reference.answer_key 刷新 reference，并压缩输出文件
        # 避免“append 多次运行后同一 sample_id 多条旧行导致人工误读”。
        _refresh_reference_and_compact_generated_responses(
            config=config,
            samples=samples,
            context_lookup=context_lookup,
        )

        # 如果只跑 generate，则直接返回；否则继续后续阶段。
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end data generation pipeline")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file (模型与打分配置等，不再内置原始数据路径)",
    )
    parser.add_argument(
        "--raw-data",
        required=True,
        help="本次实验的输入数据文件路径（例如 CSV/Excel），用于决定读取样本以及 data/<输入文件名>/ 下的输出目录",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "generate"],
        default="all",
        help="Which stage to run (all/generate; default: all).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum number of rows to load from the raw data (default: 0) all.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 加载 .env 中的模型与代理配置（若存在）
    load_env()
    init_logger()
    args = parse_args()
    run_pipeline(
        args.config,
        raw_data_path=args.raw_data,
        stage=args.stage,
        max_rows=args.max_rows,
    )
