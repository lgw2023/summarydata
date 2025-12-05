from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.data_loader.excel_loader import SampleLoader
from src.generators.base import build_generators, BaseGenerator, Candidate, FALLBACK_RESPONSE_DEFAULT
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env


def _build_candidate_key(model_type: str, model_name: str) -> Tuple[str, str]:
    """
    生成用于去重 / 匹配的候选键。
    - 不依赖 candidate_id，避免哈希实现细节耦合；
    - 只使用 (model_type, model_name) 识别同一模型的候选。
    """
    return (str(model_type), str(model_name))


def _needs_regeneration(response_text: str) -> bool:
    """
    判定某个候选回复是否视为“缺失，需要重算”：
    - 为空或全是空白字符；
    - 等于默认占位文案（表示上游 LLM 调用失败后写入的占位回复）。

    注意：若用户在 gen_config 中自定义了 fallback_response，则默认视为“用户显式接受占位”，
    此处不再强制重算，以免与用户意图冲突。
    """
    text = (response_text or "").strip()
    if not text:
        return True
    if text == FALLBACK_RESPONSE_DEFAULT:
        return True
    return False


def run_repair_generated(config_path: str | Path) -> None:
    """
    针对已生成的 generated_responses.jsonl 做一次“补洞”：

    - 逐样本检查是否已经包含所有配置中的生成器（experimental / open_source / closed_source / reference 等）；
    - 对于缺失的候选，或 response 为空 / 默认占位文案的候选，按需重新调用对应模型生成；
    - 将新生成的候选补写回原有 generated_responses.jsonl 中（原文件会被整体覆盖写入）。
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading pipeline config from %s", config_path)
    config = PipelineConfig.from_yaml(config_path)

    # 1) 加载原始样本与现有 generated_responses 结果
    samples = SampleLoader(config.raw_data).load()
    sample_lookup: Dict[str, object] = {s.sample_id: s for s in samples}

    generated_path = config.output_files.generated_responses
    existing_rows = read_jsonl(generated_path)
    rows_by_sample_id: Dict[str, Dict] = {}
    if existing_rows:
        for row in existing_rows:
            sid = str(row.get("sample_id"))
            rows_by_sample_id[sid] = row

    # 2) 构建生成器实例，用于按需补算
    generators: List[BaseGenerator] = build_generators(config.generators)
    if not generators:
        logger.warning("No generators configured; nothing to repair.")
        return

    logger.info(
        "Loaded %d samples, %d existing generated rows, %d generators",
        len(samples),
        len(rows_by_sample_id),
        len(generators),
    )

    total_new_candidates = 0
    total_updated_candidates = 0

    repaired_rows: List[Dict] = []
    for sample in samples:
        sid = str(sample.sample_id)
        row = rows_by_sample_id.get(sid)

        # 若该样本完全没有行，则创建一个基础行结构
        if not row:
            row = {
                "sample_id": sid,
                # 这里不依赖中间产物，直接按需重建上下文与问题字段
                "context": None,  # 占位，后面按需填充
                "question": sample.query,
                "candidates": [],
            }

        cand_list: List[Dict] = list(row.get("candidates") or [])
        existing_by_key: Dict[Tuple[str, str], Dict] = {}
        for c in cand_list:
            key = _build_candidate_key(c.get("model_type", ""), c.get("model_name", ""))
            existing_by_key[key] = c

        # 逐 generator 检查 / 补算
        for gen in generators:
            key = _build_candidate_key(getattr(gen, "model_type", ""), getattr(gen, "model_name", ""))
            existing = existing_by_key.get(key)

            need_regen = False
            if existing is None:
                need_regen = True
            else:
                resp_text = str(existing.get("response", "") or "")
                if _needs_regeneration(resp_text):
                    need_regen = True

            if not need_regen:
                continue

            try:
                new_cand: Candidate = gen.generate(sample)  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - 防御性兜底
                logger.exception(
                    "Failed to regenerate candidate for sample_id=%s, model=%s/%s: %r",
                    sid,
                    getattr(gen, "model_type", "unknown"),
                    getattr(gen, "model_name", "unknown"),
                    exc,
                )
                # 若此前已有候选，保留原值；若完全缺失，则暂时继续缺失
                continue

            if existing is None:
                cand_list.append(new_cand.to_dict())
                total_new_candidates += 1
            else:
                # 就地更新已有候选的内容，保持其它字段不变（如 candidate_id 一致性）
                existing.update(new_cand.to_dict())
                total_updated_candidates += 1

        # 若行中尚未填充 context，则按需构建
        if not row.get("context"):
            # 延迟导入，以避免与 build_generators 中的导入产生潜在循环
            from src.data_loader.context_builder import build_context_text  # noqa: WPS433

            row["context"] = build_context_text(sample)

        row["candidates"] = cand_list
        repaired_rows.append(row)

    # 3) 覆盖写回 generated_responses.jsonl
    write_jsonl(generated_path, repaired_rows)
    logger.info(
        "Repair finished. New candidates added: %d, updated existing candidates: %d. "
        "Total samples written: %d. Output: %s",
        total_new_candidates,
        total_updated_candidates,
        len(repaired_rows),
        generated_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair missing or placeholder responses in generated_responses.jsonl by "
            "re-generating candidates only where needed."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    init_logger()
    args = parse_args()
    run_repair_generated(args.config)


