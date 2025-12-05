from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Any, Dict, List, Optional

from src.data_loader.excel_loader import Sample
from src.utils.io import write_jsonl, ensure_dir


@dataclass
class ContextSample:
    """
    与 TASK.md 2.2 小节约定基本对齐的中间样本结构。

    说明：
    - context 的具体拼接模板目前针对当前 data.csv 约定进行实现，
      后续可通过配置化模板 / 多语言版本进一步扩展。
    """

    sample_id: str
    context: str
    question: str
    meta: Dict[str, Any]
    reference_answer: Optional[str] = None


def build_context_text(sample: Sample) -> str:
    """
    根据当前项目的字段约定构建上下文字符串。

    结构大致参考 response_prompt.py 中的说明：
    - [个人数据]      -> data
    - [专家建议]      -> suggest
    - [知识库知识]    -> rag
    - [课程库]        -> service
    - [对话历史]      -> last_query + last_answer_phone（若有）
    - [用户提问]      -> 当前 query
    """
    blocks: List[str] = []

    # 个人数据
    blocks.append("[个人数据]")
    blocks.append((sample.data or "").strip() or "（无）")
    blocks.append("")  # 空行分隔

    # 专家建议
    blocks.append("[专家建议]")
    blocks.append((sample.suggest or "").strip() or "（无）")
    blocks.append("")

    # 知识库知识
    blocks.append("[知识库知识]")
    blocks.append((sample.rag or "").strip() or "（无）")
    blocks.append("")

    # 课程库
    blocks.append("[课程库]")
    if sample.service:
        blocks.append(sample.service.strip())
    else:
        blocks.append("（无）")
    blocks.append("")

    # 对话历史（当前 Demo 支持上一轮 query + 助手回复）
    blocks.append("[对话历史]")
    has_history = bool((getattr(sample, "last_query", None) or "").strip() or (sample.last_answer_phone or "").strip())
    if has_history:
        if getattr(sample, "last_query", None):
            blocks.append(f"user: {sample.last_query.strip()}")
        if sample.last_answer_phone:
            blocks.append(f"assistant: {sample.last_answer_phone.strip()}")
    else:
        blocks.append("（无）")
    blocks.append("")

    # 当前用户提问
    blocks.append("[用户提问]")
    blocks.append(sample.query.strip())

    return "\n".join(blocks)


def build_context_samples(samples: Iterable[Sample]) -> List[ContextSample]:
    """
    将原始 Sample 列表转换为带有上下文的 ContextSample 列表。

    这里的 source_row_index 简单使用 enumerate 的顺序，
    若未来从 Excel/CSV 读取时保留行号，可在此处替换。
    """
    context_samples: List[ContextSample] = []
    for idx, sample in enumerate(samples):
        reference_answer: Optional[str] = None
        # 当前 CSV 结构中 reference_answers 是一个 dict[str, str] | None
        if sample.reference_answers:
            # 这里只是为了中间产物 / 可视化方便，将 a_answer / b_answer
            # 简单拼成一个字符串；真正用于评估与生成的仍是
            # sample.reference_answers 字典中按 key 拆分的多条参考答案。
            reference_answer = " ||| ".join(
                f"{key}:{value}"
                for key, value in sample.reference_answers.items()
                if value
            )

        context_samples.append(
            ContextSample(
                sample_id=sample.sample_id,
                context=build_context_text(sample),
                question=sample.query,
                meta={
                    "source_row_index": idx,
                    "extra": {
                        "category_level1": sample.category_level1,
                        "category_level2": sample.category_level2,
                        "category_level3": sample.category_level3,
                        "domain": sample.domain,
                    },
                },
                reference_answer=reference_answer,
            )
        )
    return context_samples


def export_context_samples_jsonl(
    output_dir: str | Path, context_samples: Iterable[ContextSample]
) -> Path:
    """
    将 ContextSample 序列化为 JSONL，写入 data/intermediate/context_samples.jsonl。
    """
    output_dir_path = ensure_dir(output_dir)
    output_path = output_dir_path / "context_samples.jsonl"
    rows = [asdict(sample) for sample in context_samples]
    write_jsonl(output_path, rows)
    return output_path


