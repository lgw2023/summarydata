from __future__ import annotations

from cProfile import label
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import markdown
import pandas as pd
import streamlit as st


def _inject_compact_radio_css() -> None:
    """
    调整 Streamlit 默认 Radio 的行高与间距，使「人工排序」区域更紧凑，
    让每行 5–0 选项的高度更接近左侧表格行高。
    """
    st.markdown(
        """
<style>
/* 收紧整个 Radio 控件之间的上下间距（多行模型之间更紧凑） */
div[data-baseweb="radio"] {
    margin-top: 0.05rem;
    margin-bottom: 0.05rem;
}

/* 收紧单选按钮每一行之间的垂直间距（同一行 5–0 选项之间） */
div[data-baseweb="radio"] > div {
    margin-bottom: 0.05rem;
}

/* 收紧单选按钮内部的上下 padding，并减小字体与行高 */
div[data-baseweb="radio"] label {
    padding-top: 0.02rem;
    padding-bottom: 0.02rem;
    font-size: 0.75rem;
    line-height: 1.1;
}

/* 放大 st.dataframe 内部单元格与表头的字体（新 DataFrame 网格实现） */
div[data-testid="stDataFrame"] div[role="columnheader"] {
    font-size: 24px !important;
}
div[data-testid="stDataFrame"] div[role="gridcell"] {
    font-size: 24px !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def find_data_files() -> List[Path]:
    """递归查找 data/ 目录下的所有 JSON/JSONL 文件。"""
    if not DATA_DIR.exists():
        return []

    jsonl_files = list(DATA_DIR.rglob("*.jsonl"))
    json_files = list(DATA_DIR.rglob("*.json"))

    all_files = jsonl_files + json_files
    # 按相对路径排序，方便浏览
    return sorted(all_files, key=lambda p: str(p.relative_to(DATA_DIR)))


def load_records(path: Path) -> List[Any]:
    """从 JSON 或 JSONL 文件中加载记录列表。"""
    if not path.exists():
        return []

    if path.suffix == ".jsonl":
        records: List[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception as exc:  # noqa: BLE001
                    # 解析失败时保留原始信息，避免整文件报错
                    records.append(
                        {
                            "_parse_error": str(exc),
                            "_line_no": line_no,
                            "_raw": line,
                        }
                    )
        return records

    # 普通 JSON 文件：可能是对象或列表
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # 单个字典/标量时，包一层方便统一处理
    return [data]


def build_record_title(idx: int, record: Any) -> str:
    """为每条记录构造在 expander 上展示的标题。"""
    base = f"#{idx}"

    if isinstance(record, dict):
        # 尝试用常见字段做标题
        candidate_keys: Iterable[str] = (
            "sample_id",
            "id",
            "question",
            "query",
            "candidate_id",
            "model_name",
        )
        for key in candidate_keys:
            if key in record and record[key]:
                text = str(record[key])
                if len(text) > 60:
                    text = text[:60] + "..."
                return f"{base} | {key}: {text}"

    return base


def _render_split_reference_answer(text: str) -> bool:
    """
    识别类似
    a_answer:... ||| b_answer:...
    或
    answer_phone:... ||| think_phone:...
    的结构，并拆成多段渲染。

    返回值表示是否已处理。
    """
    if "|||" not in text:
        return False

    try:
        # 目前约定上下文中参考答案使用 "xxx:... ||| yyy:..." 的形式拼接，
        # 这里统一只按第一个 "|||" 切成两段，便于兼容多种 key 命名。
        a_part, b_part = text.split("|||", 1)
    except ValueError:
        return False

    a_part = a_part.strip()
    b_part = b_part.strip()

    def _split_label(body: str) -> tuple[str | None, str]:
        """尝试从 'label:内容' 结构中拆出 label 与内容；失败时返回 (None, 原文)。"""
        if ":" in body:
            label, content = body.split(":", 1)
            label = label.strip()
            content = content.lstrip()
            return (label or None, content)
        return (None, body)

    a_label, a_content = _split_label(a_part)
    b_label, b_content = _split_label(b_part)

    # 若能解析出 label，则用 label 作为标题，否则退化为 answer_1 / answer_2
    a_title = a_label or "answer_1"
    b_title = b_label or "answer_2"

    st.markdown(f"**{a_title}**")
    st.markdown(a_content)
    st.markdown("---")
    st.markdown(f"**{b_title}**")
    st.markdown(b_content)
    return True


def _render_reference_answers_dict(data: dict[str, Any]) -> bool:
    """
    识别形如 {"a_answer": ..., "b_answer": ...}
    或 {"answer_phone": ..., "think_phone": ...} 的结构，
    将各参考答案字段分别用 Markdown 渲染。

    返回值表示是否已处理。
    """
    # 兼容旧字段（a_answer/b_answer）与新字段（answer_phone/think_phone）
    ref_keys = ["a_answer", "b_answer", "answer_phone", "think_phone"]
    present_keys = [k for k in ref_keys if k in data and isinstance(data[k], str)]

    if not present_keys:
        return False

    for key in present_keys:
        st.markdown(f"**{key}**")
        st.markdown(data[key])
        st.markdown("---")

    # 其它键如果存在，可选地再以 json 形式展示
    other_keys = {k: v for k, v in data.items() if k not in set(ref_keys)}
    if other_keys:
        st.markdown("**其他字段**")
        st.json(other_keys)

    return True


def render_value(key: str | None, value: Any) -> None:
    """根据键名和类型选择合适的展示方式。"""
    if isinstance(value, str):
        # 特殊处理 a_answer / b_answer 合并在一个字段里的情况
        if key in {"reference_answer", "ref_answer", "answers"} and _render_split_reference_answer(value):
            return

        # 对于较长或多行文本，用 Markdown 渲染
        if len(value) > 120 or "\n" in value:
            st.markdown(value)
        else:
            st.write(value)
    elif isinstance(value, (int, float, bool)) or value is None:
        st.write(value)
    elif isinstance(value, dict):
        # 对 samples.jsonl 中的 reference_answers 专门处理
        if key == "reference_answers" and _render_reference_answers_dict(value):
            return

        # 其它嵌套结构直接用 json 展示，避免 UI 过于复杂
        st.json(value)
    elif isinstance(value, list):
        # 列表统一 json 展示（候选列表等在更外层单独处理）
        st.json(value)
    else:
        st.write(repr(value))


def render_candidates(candidates: list[dict[str, Any]]) -> None:
    """专门渲染 generated_responses.jsonl 中的 candidates 列表。"""
    for idx, cand in enumerate(candidates, start=1):
        title_parts: list[str] = []
        if "candidate_id" in cand:
            title_parts.append(str(cand["candidate_id"]))
        if "model_name" in cand:
            title_parts.append(str(cand["model_name"]))
        if "model_type" in cand:
            title_parts.append(str(cand["model_type"]))

        title = " | ".join(title_parts) if title_parts else f"candidate #{idx}"

        with st.expander(title, expanded=False):
            # 先展示元信息
            meta_keys = ["candidate_id", "model_type", "model_name"]
            meta_items = {k: cand.get(k) for k in meta_keys if k in cand}
            if meta_items:
                st.markdown("**元信息（metadata）**")
                st.json(meta_items)
                st.markdown("---")

            # 再展示 response
            if "response" in cand:
                st.markdown("**response**")
                render_value("response", cand["response"])
                st.markdown("---")

            # 其余字段单独展示
            for k, v in cand.items():
                if k in meta_keys or k == "response":
                    continue
                st.markdown(f"**{k}**")
                render_value(k, v)
                st.markdown("---")


@st.cache_data(show_spinner=False)
def load_generated_response_index() -> Dict[str, str]:
    """
    从 data/processed/generated_responses.jsonl 构建
    candidate_id -> response 的索引，方便在 judge_results 里回显模型回复。
    """
    idx: Dict[str, str] = {}
    path = DATA_DIR / "processed" / "generated_responses.jsonl"
    if not path.exists():
        return idx

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue

            candidates = rec.get("candidates")
            if not isinstance(candidates, list):
                continue

            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                cid = cand.get("candidate_id")
                resp = cand.get("response")
                if isinstance(cid, str) and isinstance(resp, str):
                    idx[cid] = resp

    return idx


def _maybe_parse_json_from_markdown_block(text: str) -> Any | None:
    """
    尝试从带 ```json / ``` 包裹的内容中解析 JSON。
    解析失败时返回 None。
    """
    stripped = text.strip()

    # 去掉 ```lang 前后包裹
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # 去掉第一行 ``` 或 ```json
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # 去掉最后一行 ```
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except Exception:  # noqa: BLE001
        return None


def _render_response_box(resp: str) -> None:
    """
    将模型回复渲染在类似“竖版手机屏幕比例（16:9）”的方框内。

    使用 python-markdown 将 Markdown 文本转为 HTML，然后包在一个固定比例的 div 里。
    为了保证文本在框内滚动而不是撑开外层布局，增加 overflow-y: auto。
    """
    # 将 Markdown 转为 HTML。
    # 开启 tables / fenced_code 等扩展，以便在方框内正确渲染表格等 GitHub 风格 Markdown。
    html_body = markdown.markdown(
        resp,
        extensions=[
            "extra",        # 包含 tables、fenced_code 等常用扩展
            "sane_lists",
        ],
    )

    st.markdown(
        f"""
<div style="
    border: 1px solid #dddddd;
    border-radius: 16px;
    padding: 16px 14px;
    margin: 8px 0 18px 0;
    width: 100%;
    max-width: 100%;
    aspect-ratio: 9 / 16;  /* 竖版 16:9（宽:高=9:16） */
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    background-color: #ffffff;
    overflow-y: auto;
    box-sizing: border-box;
">
  <div style="font-size: 14px; line-height: 1.5; white-space: normal;">
    {html_body}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _extract_non_full_checks_from_score(score_obj: dict[str, Any]) -> list[dict[str, Any]]:
    """
    从单个维度的 score 对象中抽取“非满分”的检查项。

    预期格式（raw_judge_output 解析后）：
    {
      "checks": [
        {
          "rule_id": "...",
          "score": 4,
          "reason": "...",
          "excerpt": "..."
        },
        ...
      ]
    }
    """
    raw = score_obj.get("raw_judge_output")
    if not isinstance(raw, str) or not raw.strip():
        return []

    parsed = _maybe_parse_json_from_markdown_block(raw)
    if not isinstance(parsed, dict):
        return []

    checks = parsed.get("checks")
    if not isinstance(checks, list):
        return []

    non_full: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        s = check.get("score")
        if not isinstance(s, (int, float)):
            continue
        # 约定每条检查的满分为 5 分；只展示 < 5 分的条目
        if s >= 5:
            continue
        non_full.append(
            {
                "rule_id": check.get("rule_id"),
                "score": s,
                "reason": check.get("reason"),
                "excerpt": check.get("excerpt"),
            }
        )

    return non_full


def _extract_hit_checks_from_score(score_obj: dict[str, Any]) -> list[dict[str, Any]]:
    """
    从单个维度的 score 对象中抽取 hit == true 的检查项。

    预期格式（raw_judge_output 解析后）：
    {
      "checks": [
        {
          "rule_id": "...",
          "hit": true/false,
          "severity": "...",
          "reason": "...",
          "excerpt": "..."
        },
        ...
      ]
    }
    """
    raw = score_obj.get("raw_judge_output")
    if not isinstance(raw, str) or not raw.strip():
        return []

    parsed = _maybe_parse_json_from_markdown_block(raw)
    if not isinstance(parsed, dict):
        return []

    checks = parsed.get("checks")
    if not isinstance(checks, list):
        return []

    hits: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if not check.get("hit", False):
            continue
        hits.append(
            {
                "rule_id": check.get("rule_id"),
                "severity": check.get("severity"),
                "reason": check.get("reason"),
                "excerpt": check.get("excerpt"),
            }
        )

    return hits


def _default_manual_score_from_total_20(total_score_20: Any) -> int:
    """
    根据 total_score_20（0–20 分）推导人工打分默认值（0–5 分）。

    约定映射：
    - 20 分 -> 5 分
    - 19–16 分 -> 4 分
    - 15–14 分 -> 3 分
    - 13–12 分 -> 2 分
    - 1–11 分 -> 1 分
    - 0 分或异常值 -> 0 分
    """
    try:
        v = float(total_score_20)
    except (TypeError, ValueError):
        return 0

    # 明确 0 分 单独为 0
    if v <= 0:
        return 0
    if v >= 20:
        # 理论上最大为 20，这里 >=20 都视作 20 分处理
        return 5
    if 16 <= v < 20:
        return 4
    if 14 <= v < 16:
        return 3
    if 12 <= v < 14:
        return 2
    if 0 < v < 12:
        return 1
    return 0


def _render_non_full_checks_for_item(item: dict[str, Any]) -> None:
    """
    在候选回复方框下方展示“非满分”的检查项摘要。
    仅展示 ground / structure 两个维度里，得分 < 满分 的检查。
    """
    scores = item.get("scores")
    if not isinstance(scores, dict):
        return

    sections: list[tuple[str, list[dict[str, Any]]]] = []
    for score_key in ("ground", "structure"):
        score_obj = scores.get(score_key)
        if not isinstance(score_obj, dict):
            continue
        checks = _extract_non_full_checks_from_score(score_obj)
        if checks:
            sections.append((score_key, checks))

    if not sections:
        return

    st.markdown("**非满分得分条目（仅展示扣分项）**")
    for score_key, checks in sections:
        st.markdown(f"- **{score_key}**")
        for chk in checks:
            rule_id = chk.get("rule_id") or "unknown_rule"
            score = chk.get("score")
            reason = chk.get("reason") or ""
            excerpt = chk.get("excerpt") or ""
            # 为了紧凑展示，这里使用缩进的 Markdown 文本
            st.markdown(
                f"  - `{rule_id}`：得分 **{score} / 5**"
                + (f"；原因：{reason}" if reason else "")
            )
            if excerpt:
                st.markdown(f"    - 示例：{excerpt}")


def _render_hit_checks_for_item(item: dict[str, Any]) -> None:
    """
    在候选回复方框下方展示 KTO 评估中命中的规则项（hit == true）。
    """
    scores = item.get("scores")
    if not isinstance(scores, dict):
        return

    sections: list[tuple[str, list[dict[str, Any]]]] = []
    for score_key, score_obj in scores.items():
        if not isinstance(score_obj, dict):
            continue
        checks = _extract_hit_checks_from_score(score_obj)
        if checks:
            sections.append((score_key, checks))

    if not sections:
        return

    st.markdown("**命中规则（hit == true）**")
    for score_key, checks in sections:
        st.markdown(f"- **{score_key}**")
        for chk in checks:
            rule_id = chk.get("rule_id") or "unknown_rule"
            severity = chk.get("severity") or ""
            reason = chk.get("reason") or ""
            excerpt = chk.get("excerpt") or ""

            line = f"  - `{rule_id}`"
            if severity:
                line += f"（severity: {severity}）"
            if reason:
                line += f"；原因：{reason}"
            st.markdown(line)
            if excerpt:
                st.markdown(f"    - 示例：{excerpt}")


def render_judge_results(
    results: list[dict[str, Any]],
    is_kto_results: bool = False,
    sample_id: str | None = None,
    source_rel: str | None = None,
) -> None:
    """
    专门渲染 judge_results.jsonl 中的 results 列表：
    - 顶部展示各候选的汇总得分表
    - 下方逐个候选展开查看详细打分与原始 judge 输出
    """
    # 统一按 aggregate_score 降序排序，作为“默认排序”
    def _sort_key_item(item: dict[str, Any]) -> tuple[bool, float]:
        score = item.get("aggregate_score")
        if isinstance(score, (int, float)):
            return (False, float(score))
        return (True, 0.0)

    sorted_results = sorted(results, key=_sort_key_item, reverse=True)

    # 先构建一个总览表
    summary_rows: list[dict[str, Any]] = []
    for item in sorted_results:
        row = {
            "candidate_id": item.get("candidate_id"),
            "model_name": item.get("model_name"),
            "model_type": item.get("model_type"),
            "ground_score": item.get("ground_score"),
            "structure_score": item.get("structure_score"),
            "total_score_20_list": item.get("total_score_20_list"),
            "total_score_20": item.get("total_score_20"),
            "aggregate_score_list": item.get("aggregate_score_list"),
            "aggregate_score": item.get("aggregate_score"),
        }
        summary_rows.append(row)

    if summary_rows:
        # 使用两列布局：左侧为评分表格，右侧为一行一个模型的手动 5–0 打分
        col_table, col_manual = st.columns([4, 1])

        with col_table:
            st.markdown("**候选整体评分一览（按 aggregate_score 降序）**")

            df = pd.DataFrame(summary_rows)

            def _highlight_total(row: pd.Series) -> list[str]:
                """total_score_20 >= 阈值（默认 12）时，将整行底色标为淡绿色。"""
                try:
                    val = float(row.get("total_score_20"))
                except (TypeError, ValueError):
                    return [""] * len(row)

                if val >= 12:
                    return ["background-color: #e6f9e6"] * len(row)
                return [""] * len(row)

            # 通过 pandas Styler 在表格内部直接设置字体大小和行间距，
            # 同时隐藏行索引，并改用 st.table 避免内部滚动条、展示全部数据。
            styled_df = df.style.apply(_highlight_total, axis=1)
            try:
                # 新版 pandas 推荐的隐藏索引方式
                styled_df = styled_df.hide(axis="index")
            except Exception:  # noqa: BLE001
                # 兼容老版本 pandas（没有 hide 方法时退化为保留索引）
                pass

            styled_df = styled_df.set_table_styles(
                [
                    {
                        "selector": "th, td",
                        "props": [
                            ("font-size", "36px"),      # 字体再放大一些
                            ("line-height", "2.0"),     # 行高增大，整体高度约放大 1.5 倍
                            ("padding-top", "1.6rem"),
                            ("padding-bottom", "1.6rem"),
                        ],
                    }
                ]
            )

            # 使用 st.table 静态渲染，取消内部滚动条，页面整体滚动展示所有行
            st.table(styled_df)

        with col_manual:
            if sample_id is not None:
                st.markdown("**人工排序（5–0）**")
                st.markdown("**决定是否用于组成训练样本**")
                st.markdown("**5～1分代表保留，0分代表删除**")
                # st.caption("与左侧表格顺序一致；每行一个模型，仅展示 model_name。默认值为 0。")
                for item in sorted_results:
                    cid = item.get("candidate_id")
                    if cid is None:
                        continue
                    cid_str = str(cid)
                    # 使用 model_name 作为当前行打分控件的可见标签
                    label_text = str(item.get("model_name") or cid_str)
                    key = f"manual_rank__{source_rel or ''}__{sample_id}__{cid_str}"
                    options = [5, 4, 3, 2, 1, 0]
                    total_20 = item.get("total_score_20")

                    # 通过 session_state 统一控制默认值：
                    # - 若已有历史值（包括从 *_rank_manually.jsonl 恢复的），则直接使用；
                    # - 若不存在，则基于 total_score_20 计算一个智能默认值，
                    #   避免所有候选都从 0 分开始。
                    if key not in st.session_state:
                        st.session_state[key] = _default_manual_score_from_total_20(total_20)
                    else:
                        # 防御性地将已有值裁剪到 0–5 区间内
                        try:
                            v = int(st.session_state[key])
                        except Exception:  # noqa: BLE001
                            v = 0
                        if v < 0 or v > 5:
                            v = 0
                        st.session_state[key] = v

                    st.radio(
                        label_text,
                        options=options,
                        key=key,
                        horizontal=True,
                    )

        st.markdown("---")

    # 表格之后：按“手机屏幕比例”的方框展示每个候选的回复文本（Markdown）
    # 并在页面中每行展示 4 个回复卡片（使用 Streamlit 的列布局）
    resp_index = load_generated_response_index()
    if resp_index:
        # 先过滤出有有效回复文本的候选
        entries: list[tuple[dict[str, Any], str]] = []
        for item in sorted_results:
            cid = item.get("candidate_id")
            if not isinstance(cid, str):
                continue
            resp = resp_index.get(cid)
            if not isinstance(resp, str) or not resp.strip():
                continue
            entries.append((item, resp))

        if entries:
            st.markdown("**候选回复文本展示（按 aggregate_score 降序）**")

            cols_per_row = 4
            for row_start in range(0, len(entries), cols_per_row):
                row_entries = entries[row_start : row_start + cols_per_row]
                cols = st.columns(len(row_entries))

                for col, (item, resp) in zip(cols, row_entries):
                    with col:
                        cid = item.get("candidate_id")
                        title_parts: list[str] = []
                        if "model_name" in item:
                            title_parts.append(str(item["model_name"]))
                        if "model_type" in item:
                            title_parts.append(str(item["model_type"]))
                        score_txt = f"aggregate_score={item.get('aggregate_score')}"
                        title = "  ".join(title_parts) if title_parts else str(cid)

                        st.markdown(f"**{title}**（{score_txt}）")
                        # 在竖版 16:9 比例的方框中渲染回复文本
                        _render_response_box(resp)
                        # 在方框下方展示该候选的“非满分得分条目”或 KTO 命中规则摘要
                        if is_kto_results:
                            _render_hit_checks_for_item(item)
                        else:
                            _render_non_full_checks_for_item(item)

            st.markdown("---")

    # 逐个候选展开展示详细信息
    for idx, item in enumerate(sorted_results, start=1):
        title_parts: list[str] = [f"#{idx}"]
        if "candidate_id" in item:
            title_parts.append(str(item["candidate_id"]))
        if "model_name" in item:
            title_parts.append(str(item["model_name"]))
        if "model_type" in item:
            title_parts.append(str(item["model_type"]))

        title = " | ".join(title_parts)

        with st.expander(title, expanded=False):
            base_meta_keys = [
                "candidate_id",
                "model_name",
                "model_type",
                "judge",
                "judge_meta",
                "notes",
            ]
            meta = {k: item.get(k) for k in base_meta_keys if k in item}
            if meta:
                st.markdown("**基础信息**")
                st.json(meta)
                st.markdown("---")

            # 汇总分数
            st.markdown("**汇总得分**")
            st.write(
                {
                    "ground_score": item.get("ground_score"),
                    "structure_score": item.get("structure_score"),
                    "aggregate_score": item.get("aggregate_score"),
                }
            )
            st.markdown("---")

            # 详细 judge 打分
            scores = item.get("scores")
            if isinstance(scores, dict):
                for score_key in ("ground", "structure"):
                    score_obj = scores.get(score_key)
                    if not isinstance(score_obj, dict):
                        continue

                    st.markdown(f"**{score_key} 评分**")
                    st.write(
                        {
                            "score": score_obj.get("score"),
                            "max_score": score_obj.get("max_score"),
                        }
                    )

                    prompt = score_obj.get("raw_judge_prompt")
                    if isinstance(prompt, str) and prompt.strip():
                        with st.expander(f"{score_key} raw_judge_prompt", expanded=False):
                            # prompt 一般是原始的系统/用户指令，直接以纯文本展示
                            st.text(prompt)

                    raw = score_obj.get("raw_judge_output")
                    if isinstance(raw, str) and raw.strip():
                        with st.expander(f"{score_key} raw_judge_output", expanded=False):
                            parsed = _maybe_parse_json_from_markdown_block(raw)
                            if parsed is not None:
                                st.json(parsed)
                            else:
                                # 退化为纯文本展示
                                st.text(raw)
                    st.markdown("---")

            # 其余未展示的字段，统一以 json 展示，方便调试
            shown_keys = set(base_meta_keys) | {
                "ground_score",
                "structure_score",
                "aggregate_score",
                "scores",
                "candidate",
            }
            other = {k: v for k, v in item.items() if k not in shown_keys}
            if other:
                st.markdown("**其它字段（raw）**")
                st.json(other)


def render_record(
    record: Any,
    is_kto_results: bool = False,
    source_rel: str | None = None,
) -> None:
    """渲染单条记录的所有键值内容。"""
    if isinstance(record, dict):
        for key, value in record.items():
            # 对 generated_responses.jsonl 中的 candidates 做专门展开
            if key == "candidates" and isinstance(value, list):
                st.markdown("**candidates**")
                render_candidates(value)
                st.markdown("---")
                continue

            # 对 judge_results.jsonl / judge_results_kto.jsonl 中的 results 做专门渲染
            if key == "results" and isinstance(value, list):
                st.markdown("**results（judge 评估结果）**")
                render_judge_results(
                    value,
                    is_kto_results=is_kto_results,
                    sample_id=str(record.get("sample_id"))
                    if isinstance(record, dict) and "sample_id" in record
                    else None,
                    source_rel=source_rel,
                )
                st.markdown("---")
                continue

            st.markdown(f"**{key}**")
            render_value(key, value)
            st.markdown("---")
    elif isinstance(record, list):
        for i, item in enumerate(record):
            st.markdown(f"**[{i}]**")
            render_value(None, item)
            st.markdown("---")
    else:
        render_value(None, record)


def _save_manual_ranking_for_sample(
    record: dict[str, Any],
    selected_rel: str,
    overwrite: bool = True,
) -> dict[str, Any] | None:
    """
    将当前样本在前端选择的手动打分结果写入对应 judge_results 文件专属的 *_rank_manually.jsonl。

    每一行结构大致为：
    {
      "sample_id": "...",
      "source_file": "processed/judge_results_kto.jsonl",
      "context": {... 原始 record 中除 results 之外的字段 ...},
      "manual_ranks": [
        {
          "candidate_id": "...",
          "model_name": "...",
          "model_type": "...",
          "aggregate_score": ...,
          "ground_score": ...,
          "structure_score": ...,
          "manual_score": 0-5
        },
        ...
      ]
    }
    """
    if not isinstance(record, dict):
        return None

    sample_id = record.get("sample_id")
    if sample_id is None:
        return None
    sample_id_str = str(sample_id)

    results = record.get("results")
    if not isinstance(results, list):
        return None

    manual_ranks: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        cid = item.get("candidate_id")
        if cid is None:
            continue
        cid_str = str(cid)
        key = f"manual_rank__{selected_rel or ''}__{sample_id_str}__{cid_str}"
        score = st.session_state.get(key, 0)
        # 仅允许数值 0-5，其它情况强行归 0
        try:
            score_int = int(score)
        except Exception:
            score_int = 0
        if score_int < 0 or score_int > 5:
            score_int = 0

        manual_ranks.append(
            {
                "candidate_id": cid,
                "model_name": item.get("model_name"),
                "model_type": item.get("model_type"),
                "aggregate_score": item.get("aggregate_score"),
                "ground_score": item.get("ground_score"),
                "structure_score": item.get("structure_score"),
                "manual_score": score_int,
            }
        )

    if not manual_ranks:
        return None

    context_fields = {k: v for k, v in record.items() if k != "results"}

    payload: dict[str, Any] = {
        "sample_id": sample_id_str,
        "source_file": selected_rel,
        "context": context_fields,
        "manual_ranks": manual_ranks,
    }

    output_path = _build_manual_rank_output_path(selected_rel)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 若开启 overwrite，则先读出原文件内容，删除同一 sample_id + source_file 的旧记录，再回写
    if overwrite and output_path.exists():
        remaining_lines: list[str] = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except Exception:  # noqa: BLE001
                    # 解析失败时保留原始行，避免破坏文件
                    remaining_lines.append(line)
                    continue
                if (
                    isinstance(rec, dict)
                    and str(rec.get("sample_id")) == sample_id_str
                    and str(rec.get("source_file")) == selected_rel
                ):
                    # 跳过同一 sample_id + source_file 的旧记录，实现覆盖
                    continue
                remaining_lines.append(line)

        with output_path.open("w", encoding="utf-8") as f:
            for l in remaining_lines:
                f.write(l)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return payload


def _build_manual_rank_output_path(selected_rel: str) -> Path:
    """
    根据当前选中的 judge_results 相对路径，构造对应的
    *_rank_manually.jsonl 完整存储路径。
    """
    rel_path = Path(selected_rel)
    if rel_path.is_absolute():
        output_dir = rel_path.parent
    else:
        output_dir = DATA_DIR / rel_path.parent

    stem = rel_path.stem
    suffix = rel_path.suffix or ".jsonl"
    output_name = f"{stem}_rank_manually{suffix}"
    return output_dir / output_name


def _load_manual_ranks_for_sample(
    selected_rel: str,
    sample_id: str,
) -> dict[str, int]:
    """
    从对应的 *_rank_manually.jsonl 文件中读取指定 sample_id 的手动打分结果。

    返回:
        {candidate_id(str): manual_score(int)}
    """
    output_path = _build_manual_rank_output_path(selected_rel)

    if not output_path.exists():
        return {}

    result: dict[str, int] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(rec, dict):
                continue
            if str(rec.get("sample_id")) != sample_id:
                continue
            if str(rec.get("source_file")) != selected_rel:
                continue
            manual_ranks = rec.get("manual_ranks")
            if not isinstance(manual_ranks, list):
                continue
            for item in manual_ranks:
                if not isinstance(item, dict):
                    continue
                cid = item.get("candidate_id")
                score = item.get("manual_score")
                if cid is None:
                    continue
                try:
                    score_int = int(score)
                except Exception:
                    continue
                result[str(cid)] = score_int
            # 理论上同一 sample_id + source_file 只会有一条记录
    return result

def main() -> None:
    st.set_page_config(page_title="data 目录可视化浏览", layout="wide")
    # 收紧全局 Radio 样式，使右侧人工排序按钮区域高度更贴近表格行高
    _inject_compact_radio_css()
    st.title("data 目录可视化浏览")
    st.caption("自动加载 data/ 下的 JSON / JSONL 文件；每条记录可展开查看，长文本使用 Markdown 渲染。")

    files = find_data_files()
    if not files:
        st.warning("未在 data/ 目录下找到任何 .json 或 .jsonl 文件。")
        return

    rel_paths = [str(p.relative_to(DATA_DIR)) for p in files]

    st.sidebar.header("文件选择")
    # 默认优先选中 data/processed/judge_results_kto.jsonl 或
    # data/processed/judge_results.jsonl（若存在）
    default_index = 0
    preferred_defaults = [
        "processed/judge_results_kto.jsonl",
        "processed/judge_results_kto_v1.jsonl",
        "processed/judge_results.jsonl",
    ]
    for rel in preferred_defaults:
        if rel in rel_paths or "processed/judge_results_kto" in rel_paths:
            default_index = rel_paths.index(rel)
            break

    selected_rel = st.sidebar.selectbox("选择要查看的文件", rel_paths, index=default_index)
    selected_path = files[rel_paths.index(selected_rel)]

    st.sidebar.caption(f"当前文件：{selected_rel}")

    records = load_records(selected_path)
    total = len(records)

    st.subheader(f"文件：{selected_rel}")
    st.write(f"共 **{total}** 条记录。")

    if total == 0:
        st.info("文件为空或未能解析出记录。")
        return

    # 对 judge_results.jsonl / judge_results_kto*.jsonl 做特殊处理：
    # 按“样本 = 页码”分页展示，且不使用整体折叠
    if selected_rel in {
        "processed/judge_results.jsonl",
        "processed/judge_results_kto.jsonl",
        "processed/judge_results_kto_v1.jsonl",
    } or "processed/judge_results_kto" in selected_rel:
        # 使用 session_state 记录当前页码和上一页码，以便在切换页码时自动保存上一条样本的人工打分
        page_state_key = f"current_page__{selected_rel}"
        prev_state_key = f"{page_state_key}__prev"

        # 当前页默认值：来自当前页 state（没有则为 1），先做边界修正
        default_page_raw = st.session_state.get(page_state_key, 1)
        try:
            default_page = int(default_page_raw)
        except Exception:  # noqa: BLE001
            default_page = 1
        if default_page < 1:
            default_page = 1
        if default_page > total:
            default_page = total

        current_page = st.sidebar.number_input(
            "选择样本页码",
            min_value=1,
            max_value=total,
            value=default_page,
            step=1,
            key=page_state_key,
        )

        # 上一页页码：单独存一份，默认等于当前页（首次进入不触发自动保存）
        prev_page_raw = st.session_state.get(prev_state_key, current_page)
        try:
            prev_page = int(prev_page_raw)
        except Exception:  # noqa: BLE001
            prev_page = current_page
        if prev_page < 1:
            prev_page = 1
        if prev_page > total:
            prev_page = total

        # 若检测到页码发生变化，则对上一条样本自动保存人工打分
        if current_page != prev_page:
            prev_idx = int(prev_page)
            if 1 <= prev_idx <= total:
                prev_record = records[prev_idx - 1]
                if isinstance(prev_record, dict) and "results" in prev_record:
                    _ = _save_manual_ranking_for_sample(prev_record, selected_rel, overwrite=True)

        # 更新上一页页码为当前页，供下一次刷新时使用
        st.session_state[prev_state_key] = int(current_page)

        st.divider()

        idx = int(current_page)
        record = records[idx - 1]
        st.markdown(f"**样本 #{idx}**")

        # 在渲染当前样本之前，若存在历史手动打分，则加载并写入 session_state，
        # 这样右侧 Radio 的默认值会显示为历史结果，而不是每次都从 0 开始。
        if isinstance(record, dict) and "results" in record:
            sample_id = record.get("sample_id")
            if sample_id is not None:
                sample_id_str = str(sample_id)
                existing_manual = _load_manual_ranks_for_sample(selected_rel, sample_id_str)
                if existing_manual:
                    for item in record.get("results", []):
                        if not isinstance(item, dict):
                            continue
                        cid = item.get("candidate_id")
                        if cid is None:
                            continue
                        cid_str = str(cid)
                        if cid_str not in existing_manual:
                            continue
                        key = f"manual_rank__{selected_rel or ''}__{sample_id_str}__{cid_str}"
                        # 避免覆盖当前会话中尚未保存的最新选择，只在 key 不存在时初始化
                        if key not in st.session_state:
                            try:
                                st.session_state[key] = int(existing_manual[cid_str])
                            except Exception:
                                continue

        # 当为 KTO 结果文件时，启用 hit==true 的规则展示
        is_kto_results = (
            selected_rel == "processed/judge_results_kto.jsonl"
            or "processed/judge_results_kto" in selected_rel
        )

        # 在侧边栏增加保存按钮：将当前样本的人工打分写入当前 judge_results 文件对应的 *_rank_manually.jsonl
        if isinstance(record, dict) and "results" in record:
            if st.sidebar.button(
                "保存当前样本的手动打分",
                key=f"save_manual_rank_{selected_rel}_{idx}",
            ):
                saved = _save_manual_ranking_for_sample(record, selected_rel, overwrite=True)
                if saved is not None:
                    output_path = _build_manual_rank_output_path(selected_rel)
                    try:
                        display_path = str(output_path.relative_to(PROJECT_ROOT))
                    except ValueError:
                        display_path = str(output_path)
                    st.sidebar.success(f"已保存当前样本的手动打分到文件：{display_path}")
                else:
                    st.sidebar.warning("未找到可保存的手动打分或结果结构异常，未写入文件。")

        render_record(record, is_kto_results=is_kto_results, source_rel=selected_rel)
        return

    # 其它文件仍使用“最多展示前 N 条记录 + 折叠”的方式
    max_default = min(100, total)
    max_show = st.sidebar.number_input(
        "最多展示前 N 条记录",
        min_value=1,
        max_value=total,
        value=max_default,
        step=1,
    )

    st.divider()

    for idx, record in enumerate(records[: int(max_show)], start=1):
        title = build_record_title(idx, record)
        with st.expander(title, expanded=False):
            render_record(record)


if __name__ == "__main__":
    main()


