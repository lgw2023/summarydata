from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Any, Dict, Tuple


def _is_missing(v: Any) -> bool:
    # pandas 会用 NaN/NaT 表示缺失；这里做一个尽量轻量的兼容判断
    try:
        import pandas as pd  # type: ignore

        return v is None or (isinstance(v, float) and v != v) or bool(pd.isna(v))
    except Exception:
        # 没有 pandas 时退化判断：None 或 NaN(float)
        return v is None or (isinstance(v, float) and v != v)


def _to_text(v: Any) -> str | None:
    if _is_missing(v):
        return None
    # Excel 里可能是数字/布尔值，统一转为字符串；保留换行等内容
    s = str(v)
    return s if s != "" else None


def _load_tabular_rows(
    path: Path, max_rows: int | None = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    统一读取 CSV / Excel，返回 (fieldnames, rows)。
    - fieldnames: 原始列名列表
    - rows: 每行 dict（key 为原始列名，value 为原始单元格值）
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "读取 Excel 需要依赖 pandas + openpyxl。请先安装：pip install pandas openpyxl"
            ) from exc

        nrows = None
        if max_rows is not None and max_rows != 0:
            nrows = int(max_rows)

        # 默认读取第一个 sheet；dtype=object 保留原始类型，便于后续统一转字符串
        df = pd.read_excel(path, dtype=object, nrows=nrows)
        fieldnames = [str(c) for c in df.columns.tolist()]
        rows: List[Dict[str, Any]] = df.to_dict(orient="records")
        return fieldnames, rows

    # 默认按 CSV 处理（兼容 .csv / 其他纯文本表格）
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows and max_rows != 0:
                break
            rows.append(row)
        return fieldnames, rows


@dataclass
class Sample:
    sample_id: str
    query: str
    # 变体信息（用于同一条原始数据拆成 phone/watch 两条独立样本）
    base_sample_id: str | None = None
    device: str | None = None  # "phone" | "watch"
    system_prompt: str | None = None
    # system_prompt 元信息（用于写盘到 context_samples / generated_responses）
    # system_prompt_type: "phone_personal" | "phone_general" | "watch_personal" | "watch_general"
    system_prompt_type: str | None = None
    # 仅在 *_personal 时写入 domain key（否则为空串/None）
    system_prompt_domain: str | None = None
    # 上一轮对话（可选）
    last_query: str | None = None
    category_level1: str | None = None
    category_level2: str | None = None
    category_level3: str | None = None
    domain: str | None = None
    data: str | None = None
    suggest: str | None = None
    rag: str | None = None
    service: str | None = None
    last_answer_phone: str | None = None
    # 手表端上一轮助手回复（可选；与 last_query 组成 watch 的历史对话）
    last_answer_watch: str | None = None
    reference_answers: dict[str, str] | None = None


REQUIRED_COLUMNS = {
    "query": "query",
}

OPTIONAL_COLUMNS = {
    "一级分类": "category_level1",
    "二级分类": "category_level2",
    "三级分类": "category_level3",
    "domain": "domain",
    "last_query": "last_query",
    "data": "data",
    "suggest": "suggest",
    "rag": "rag",
    "service": "service",
    "last_answer_phone": "last_answer_phone",
    "last_answer_watch": "last_answer_watch",
    "a_answer": "a_answer",
    "b_answer": "b_answer",
    "winner": "winner",
}


class SampleLoader:
    def __init__(self, path: str | Path, reference_columns: List[str] | None = None):
        self.path = Path(path)
        # 参考答案列由上层（通常来自 YAML 配置中的 reference generator.answer_key）显式指定。
        # 设计目标：避免在此处写死任何列名或做“猜列名”的逻辑。
        self.reference_columns = [c for c in (reference_columns or []) if str(c).strip()]

    def load(self, max_rows: int | None = None) -> List[Sample]:
        fieldnames, rows = _load_tabular_rows(self.path, max_rows=max_rows)

        missing = [col for col in REQUIRED_COLUMNS if col not in set(fieldnames)]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

        samples: List[Sample] = []
        for idx, row in enumerate(rows):
            # 将 dict 的 value 统一归一成 str|None，避免 Excel 中出现 float/bool 导致下游不稳定
            norm_row: Dict[str, str | None] = {k: _to_text(v) for k, v in (row or {}).items()}

            mapping = {
                internal: norm_row.get(external) for external, internal in OPTIONAL_COLUMNS.items()
            }

            # 参考答案列：完全由上层显式指定（通常来自 YAML 中 reference 生成器的 answer_key）。
            # 这里只做“按列取值 + 过滤空值”，不做任何列名推断。
            reference: Dict[str, str] = {}
            for key in self.reference_columns:
                val = norm_row.get(key)
                if val:
                    reference[key] = val

            sample_id_val = norm_row.get("sample_id")
            sample_id = (sample_id_val.strip() if sample_id_val else str(idx + 1))

            query_val = norm_row.get("query")
            if not query_val or not query_val.strip():
                # 跳过空 query 行（对 Excel/CSV 都一致）
                continue

            samples.append(
                Sample(
                    sample_id=sample_id,
                    query=query_val,
                    last_query=mapping.get("last_query"),
                    category_level1=mapping.get("category_level1"),
                    category_level2=mapping.get("category_level2"),
                    category_level3=mapping.get("category_level3"),
                    domain=mapping.get("domain"),
                    data=mapping.get("data"),
                    suggest=mapping.get("suggest"),
                    rag=mapping.get("rag"),
                    service=mapping.get("service"),
                    last_answer_phone=mapping.get("last_answer_phone"),
                    last_answer_watch=mapping.get("last_answer_watch"),
                    reference_answers=reference or None,
                )
            )

        return samples


def export_samples(samples: Iterable[Sample]) -> List[dict]:
    return [sample.__dict__ for sample in samples]
