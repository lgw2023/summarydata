from __future__ import annotations

import hashlib


def make_candidate_id(sample_id: str, model_name: str, index: int | None = None) -> str:
    """
    为候选生成一个**稳定可复现**的候选 ID。

    规则：
    - 基本形式：{sample_id}::{model_name}
    - 如果同一模型为同一样本生成多条回复，则附加 `#{index}` 后缀。
    - 再追加一个短 hash 以降低碰撞概率。
    """
    base = f"{sample_id}::{model_name}"
    if index is not None:
        base = f"{base}#{index}"
    # 生成一个很短的 hash 作为防碰撞后缀
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{base}::{h}"


