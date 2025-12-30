from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple


# ==== 项目根目录 / sys.path ====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.env import load_env  # noqa: E402

# 复用现有 KTO 打分逻辑中的工具函数
from src.scoring import (  # noqa: E402
    kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats as kto_mod,
)


try:  # noqa: E402
    from openai import OpenAI
except Exception:  # pragma: no cover - 环境缺失时的防御
    OpenAI = None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_think_tags(text: str) -> str:
    """
    对新增待评估的模型回复做轻量清洗，避免把思考过程混入 answer：

    - 若同时出现 <think> 与 </think>：删除所有成对块（含块内内容）
    - 若只出现 </think>：删除该 tag 及其之前的所有内容
    - 删除后若产生冗余空白/换行，做最小化清理
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return s

    original = s

    has_open = "<think>" in s
    has_close = "</think>" in s

    if has_open and has_close:
        # 删除所有成对的 <think>...</think> 块（非贪婪，跨行）
        s = re.sub(r"<think>[\s\S]*?</think>", "", s)
    elif (not has_open) and has_close:
        # 只出现 </think>：删除该 tag 及其之前的内容（取最后一个更稳妥）
        idx = s.rfind("</think>")
        s = s[idx + len("</think>") :]

    # 仅当发生过处理时，做“冗余空白/换行”清理，尽量不扰动原文本格式
    if s != original:
        # 清理开头/结尾多余空白（常见于删除 think 块后留下的换行）
        s = s.strip()
        # 把连续 3+ 个换行压缩为 2 个，避免空洞过大
        s = re.sub(r"\n{3,}", "\n\n", s)
        # 删除空行里的纯空白
        s = re.sub(r"[ \t]+\n", "\n", s)

    return s


def _build_processed_dir_from_raw(raw_data: str | Path) -> Path:
    """
    根据 --raw-data 推导 processed 目录：
    data/<raw_data_stem>/processed
    """
    raw_path = Path(raw_data)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    return PROJECT_ROOT / "data" / raw_path.stem / "processed"


def _load_base_generated_and_judge(
    processed_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    加载基础的 generated_responses.jsonl 与 judge_results_kto.jsonl。

    返回：
    - base_samples: {sample_id -> sample_json}
    - base_results: {sample_id -> [result_entries]}
    """
    base_gen_path = processed_dir / "generated_responses.jsonl"
    base_judge_path = processed_dir / "judge_results_kto.jsonl"

    if not base_gen_path.exists():
        raise FileNotFoundError(f"找不到基础 generated_responses.jsonl: {base_gen_path}")
    if not base_judge_path.exists():
        raise FileNotFoundError(f"找不到基础 judge_results_kto.jsonl: {base_judge_path}")

    base_gen_rows = _read_jsonl(base_gen_path)
    base_judge_rows = _read_jsonl(base_judge_path)

    base_samples: Dict[str, Dict[str, Any]] = {}
    for row in base_gen_rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        # 若存在多条，以最后一条为准（与其它组件保持一致）
        base_samples[sid] = row

    base_results: Dict[str, List[Dict[str, Any]]] = {}
    for row in base_judge_rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        results = row.get("results") or []
        if not isinstance(results, list):
            continue
        base_results[sid] = results

    return base_samples, base_results


def _load_existing_generated_and_judge_incremental(
    processed_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    增量更新模式下加载“已有结果”：

    - 若存在 generated_responses_merged.jsonl，则优先用它作为基线 generated；
      否则回退到 generated_responses.jsonl。
    - 若存在 judge_results_kto_merger.jsonl，则优先用它作为基线 judge；
      否则回退到 judge_results_kto.jsonl。

    返回：
    - base_samples: {sample_id -> sample_json}
    - base_results: {sample_id -> [result_entries]}
    - meta: 记录使用了哪些基线文件路径，便于日志输出
    """
    gen_merged_path = processed_dir / "generated_responses_merged.jsonl"
    judge_merged_path = processed_dir / "judge_results_kto_merger.jsonl"

    base_gen_path = processed_dir / "generated_responses.jsonl"
    base_judge_path = processed_dir / "judge_results_kto.jsonl"

    gen_path = gen_merged_path if gen_merged_path.exists() else base_gen_path
    judge_path = judge_merged_path if judge_merged_path.exists() else base_judge_path

    if not gen_path.exists():
        raise FileNotFoundError(f"找不到 generated_responses 文件: {gen_path}")
    if not judge_path.exists():
        raise FileNotFoundError(f"找不到 judge_results 文件: {judge_path}")

    gen_rows = _read_jsonl(gen_path)
    judge_rows = _read_jsonl(judge_path)

    base_samples: Dict[str, Dict[str, Any]] = {}
    for row in gen_rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        base_samples[sid] = row

    base_results: Dict[str, List[Dict[str, Any]]] = {}
    for row in judge_rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        results = row.get("results") or []
        if not isinstance(results, list):
            continue
        base_results[sid] = results

    meta = {
        "generated_path": str(gen_path),
        "judge_path": str(judge_path),
        "used_generated_merged": bool(gen_merged_path.exists()),
        "used_judge_merged": bool(judge_merged_path.exists()),
    }
    return base_samples, base_results, meta


def _discover_extra_generated_files(
    processed_dir: Path,
    explicit_files: List[str] | None = None,
) -> List[Path]:
    """
    发现需要增量合并的 generated_responses_*.jsonl 文件。

    优先使用命令行显式指定的文件列表；若未指定，则自动从
    processed_dir 中查找所有形如 generated_responses_*.jsonl 的文件，
    并排除：
    - generated_responses.jsonl
    - generated_responses_merged.jsonl
    """
    if explicit_files:
        out: List[Path] = []
        for p in explicit_files:
            path_obj = Path(p)
            if not path_obj.is_absolute():
                path_obj = processed_dir / path_obj
            if path_obj.exists():
                out.append(path_obj)
        return out

    candidates: List[Path] = []
    for p in processed_dir.glob("generated_responses_*.jsonl"):
        name = p.name
        if name in {"generated_responses.jsonl", "generated_responses_merged.jsonl"}:
            continue
        candidates.append(p)
    return sorted(candidates)


def _build_row_for_kto(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    """
    为 judge_one_answer 构造 row / user_input / history_input。

    逻辑与 kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py
    中 process_jsonl_sample 保持一致。
    """
    sample_id = sample.get("sample_id", "")
    context = sample.get("context", "")
    question = sample.get("question", "")

    parsed_context = kto_mod.parse_context_string(str(context or ""))

    row = {
        "sample_id": sample_id,
        "data": parsed_context.get("data", ""),
        "suggest": parsed_context.get("suggest", ""),
        "rag": parsed_context.get("rag", ""),
        "service": parsed_context.get("service", ""),
        "last_answer_phone": parsed_context.get("last_answer_phone", ""),
    }

    user_input = f"user: {question}"
    history_input = ""
    if parsed_context.get("last_answer_phone"):
        history_input = f"assistant: {parsed_context['last_answer_phone']}"

    return row, user_input, history_input


def _build_single_run_candidate_entry(
    judged: Dict[str, Any],
    row: Dict[str, Any],
    user_input: str,
    history_input: str,
    candidate_id: str,
    model_type: str,
    model_name: str,
    answer: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    复用原脚本中 _process_one_candidate 的核心逻辑，先构造“单轮打分结果 e”，
    再在本函数外做「单轮版聚合」以便形状与原先聚合后的 judge_results_kto.jsonl
    尽量保持一致。
    """
    # ==== 先按原脚本逻辑得到单轮的基础字段 ====
    # 从统一 penalties 中按维度拆分 0–5 分
    penalties_all = judged.get("penalties", []) or []

    def _dim_score_5(rule_set: set[str]) -> float:
        total_deduction = 0.0
        for p in penalties_all:
            rid = (p.get("rule_id") or "").upper()
            if rid in rule_set:
                try:
                    total_deduction += float(p.get("score", 0.0))
                except Exception:
                    continue
        local_total_20 = kto_mod.FIXED_MAX_SCORE - total_deduction
        if not args.allow_negative:
            local_total_20 = max(0.0, local_total_20)
        return local_total_20 * 5.0 / kto_mod.FIXED_MAX_SCORE

    ground_score_5 = _dim_score_5(kto_mod.GROUND_DIM_RULE_IDS)
    structure_score_5 = _dim_score_5(kto_mod.STRUCT_DIM_RULE_IDS)

    total_score_20 = float(judged.get("total_score", 0.0))
    aggregate_score_5 = total_score_20 / 20.0 * 5.0
    confidence = float(judged.get("confidence", 0.5))
    label = int(judged.get("_label", 0))
    weight = float(judged.get("_weight", 1.0))

    data_json = judged.get("_data_json", {}) or {}
    kb_text = judged.get("_kb_text", "") or ""
    jg = judged.get("_ground_judge", {}) or {}
    js = judged.get("_struct_judge", {}) or {}

    modules_block = kto_mod.build_modules_block(row, data_json, kb_text)
    ground_prompt = kto_mod.GROUND_PROMPT_TPL.safe_substitute(
        input_data=user_input,
        history_input=history_input,
        answer=answer,
        modules_block=modules_block,
    )
    struct_prompt = kto_mod.STRUCT_PROMPT_TPL.safe_substitute(
        input_data=user_input,
        history_input=history_input,
        answer=answer,
        modules_block=modules_block,
    )

    single_entry = {
        "candidate_id": candidate_id,
        "candidate": candidate_id,
        "model_type": model_type,
        "model_name": model_name,
        "scores": {
            "ground": {
                "score": ground_score_5,
                "max_score": 5.0,
                "raw_judge_output": json.dumps(jg, ensure_ascii=False),
                "raw_judge_prompt": ground_prompt,
            },
            "structure": {
                "score": structure_score_5,
                "max_score": 5.0,
                "raw_judge_output": json.dumps(js, ensure_ascii=False),
                "raw_judge_prompt": struct_prompt,
            },
        },
        "ground_score": ground_score_5,
        "structure_score": structure_score_5,
        "aggregate_score": aggregate_score_5,
        "total_score_20": total_score_20,
        "label": label,
        "weight": weight,
        "confidence": confidence,
        "penalties_json_20": json.dumps(judged.get("penalties", []), ensure_ascii=False),
        "judge": "llm",
        "judge_meta": {"judge_name": "llm"},
        "notes": None,
    }
    return single_entry


def _aggregate_single_run_entry(single: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单轮打分结果 entry 转换为“聚合后形状”，
    使其尽量与 kto 脚本聚合 num_repeat>1 后的结果结构一致。
    """
    e = deepcopy(single)

    agg: Dict[str, Any] = {}

    # 这些字段在各轮中应当保持不变，直接拷贝
    stable_keys = {
        "candidate",
        "candidate_id",
        "model_name",
        "model_type",
        "judge",
        "judge_meta",
    }
    for k in list(e.keys()):
        if k in stable_keys:
            agg[k] = e.get(k)

    # ground_score / structure_score -> 列表
    for field in ("ground_score", "structure_score"):
        v = e.get(field)
        if v is None:
            agg[field] = []
        else:
            try:
                agg[field] = [float(v)]
            except Exception:
                agg[field] = []

    # aggregate_score / total_score_20 及对应 *_list
    for field in ("aggregate_score", "total_score_20"):
        v = e.get(field)
        if v is None:
            agg[field] = None
            agg[f"{field}_list"] = []
        else:
            try:
                val = float(v)
            except Exception:
                val = None
            agg[field] = val
            agg[f"{field}_list"] = [] if val is None else [val]

    # 其它标量字段：label / weight / confidence / notes 等直接拷贝
    for k in ("label", "weight", "confidence", "notes"):
        if k in e:
            agg[k] = e[k]

    # penalties_json_20：为每个 penalty 增加 repeat_idx=1
    pen_raw = e.get("penalties_json_20")
    combined_list: List[Dict[str, Any]] = []
    if pen_raw:
        try:
            items = json.loads(pen_raw)
            if isinstance(items, list):
                for d in items:
                    if isinstance(d, dict):
                        nd = dict(d)
                        nd["repeat_idx"] = 1
                        combined_list.append(nd)
        except Exception:
            pass
    agg["penalties_json_20"] = json.dumps(combined_list, ensure_ascii=False)

    # scores.ground / scores.structure 结构：score -> list，新增 min_score，raw_judge_output 合并 checks+confidence
    scores = e.get("scores") or {}
    if isinstance(scores, dict):
        out_scores: Dict[str, Any] = {}
        for dim in ("ground", "structure"):
            dim_src = scores.get(dim) or {}
            if not isinstance(dim_src, dict):
                continue
            dim_out: Dict[str, Any] = {}
            # max_score / prompt 直接拷贝
            if "max_score" in dim_src:
                dim_out["max_score"] = dim_src["max_score"]
            if "raw_judge_prompt" in dim_src:
                dim_out["raw_judge_prompt"] = dim_src["raw_judge_prompt"]

            score_val = dim_src.get("score")
            score_list: List[float] = []
            if score_val is not None:
                try:
                    score_list.append(float(score_val))
                except Exception:
                    pass
            dim_out["score"] = score_list
            dim_out["min_score"] = score_list[0] if score_list else None

    # raw_judge_output：checks 增加 repeat_idx、confidence 收集为列表
            raw_str = dim_src.get("raw_judge_output")
            merged_checks: List[Dict[str, Any]] = []
            conf_list: List[float] = []
            if raw_str:
                try:
                    jd = json.loads(raw_str)
                    checks = jd.get("checks") or []
                    if isinstance(checks, list):
                        for c in checks:
                            if isinstance(c, dict):
                                nc = dict(c)
                                nc["repeat_idx"] = 1
                                merged_checks.append(nc)
                    if "confidence" in jd:
                        try:
                            conf_val = float(jd.get("confidence"))
                            conf_list.append(conf_val)
                        except Exception:
                            pass
                except Exception:
                    pass
            dim_out["raw_judge_output"] = json.dumps(
                {"checks": merged_checks, "confidence": conf_list},
                ensure_ascii=False,
            )
            out_scores[dim] = dim_out

        if out_scores:
            agg["scores"] = out_scores

    # 确保 candidate_id / candidate 存在
    if "candidate_id" not in agg and "candidate_id" in e:
        agg["candidate_id"] = e["candidate_id"]
    if "candidate" not in agg and "candidate" in e:
        agg["candidate"] = e["candidate"]

    return agg


def _aggregate_value(per_run_values: List[Tuple[int, Any]]) -> Any:
    """
    与 KTO 主脚本中 _aggregate_value 保持一致的聚合规则：
    - 数值：取均值
    - 布尔：所有重复都是 True 才为 True
    - 字符串/列表/其他：按「第几次：内容」拼接
    - dict：递归聚合内部字段
    """
    non_none = [(idx, v) for idx, v in per_run_values if v is not None]
    if not non_none:
        return None

    first_val = non_none[0][1]

    # bool 需要优先判断（bool 是 int 的子类）
    if isinstance(first_val, bool):
        return all(bool(v) for _, v in non_none)

    # 数值：按重复次数取均值
    if isinstance(first_val, (int, float)) and not isinstance(first_val, bool):
        nums = [float(v) for _, v in non_none]
        return sum(nums) / len(nums) if nums else None

    # dict：对内部字段递归聚合
    if isinstance(first_val, dict):
        all_keys = set()
        for _, d in non_none:
            all_keys.update(d.keys())
        agg_dict: Dict[str, Any] = {}
        for k in all_keys:
            sub_vals = [(idx, d.get(k)) for idx, d in non_none]
            agg_dict[k] = _aggregate_value(sub_vals)
        return agg_dict

    # 其余（字符串 / list / 其他）一律转为文本并按“次数：内容”拼接
    parts = []
    for run_idx, v in non_none:
        parts.append(f"{run_idx + 1}: {str(v)}")
    return "\n".join(parts)


def _aggregate_candidate_repeats(
    single_run_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    对同一 candidate 的多次重复打分结果做聚合，
    逻辑对齐 src/scoring/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py
    中 _aggregate_jsonl_repeats 里 per-candidate 的聚合规则。
    """
    if not single_run_entries:
        return {}

    # entries 顺序即 run_idx 顺序
    entries = single_run_entries

    all_keys = set()
    for e in entries:
        all_keys.update(e.keys())

    agg_entry: Dict[str, Any] = {}
    stable_keys = {
        "candidate",
        "candidate_id",
        "model_name",
        "model_type",
        "judge",
        "judge_meta",
    }

    for k in all_keys:
        per_run_vals = [(idx, e.get(k)) for idx, e in enumerate(entries)]

        # 这些字段在各轮中应当保持不变，直接取第一轮的值
        if k in stable_keys:
            first_entry = entries[0]
            agg_entry[k] = first_entry.get(k)
            continue

        # ground_score / structure_score：合并为数值列表
        if k in {"ground_score", "structure_score"}:
            score_list: List[float] = []
            for _, e in enumerate(entries):
                v = e.get(k)
                if v is None:
                    continue
                try:
                    score_list.append(float(v))
                except Exception:
                    continue
            agg_entry[k] = score_list
            continue

        # aggregate_score / total_score_20：
        # - 字段本身为均值
        # - 另外增加 *_list 保存各轮数值
        if k in {"aggregate_score", "total_score_20"}:
            nums: List[float] = []
            for _, v in per_run_vals:
                if v is None:
                    continue
                try:
                    nums.append(float(v))
                except Exception:
                    continue

            if nums:
                if k == "aggregate_score":
                    agg_entry["aggregate_score_list"] = nums
                else:
                    agg_entry["total_score_20_list"] = nums

            agg_entry[k] = sum(nums) / len(nums) if nums else None
            continue

        # penalties_json_20：内部是字典列表，这里按 extend 风格拼接，并为每个字典增加 repeat_idx
        if k == "penalties_json_20":
            combined_list: List[Dict[str, Any]] = []
            for run_idx, v in per_run_vals:
                if not v:
                    continue
                try:
                    items = json.loads(v)
                except Exception:
                    continue
                if not isinstance(items, list):
                    continue
                for d in items:
                    if isinstance(d, dict):
                        new_d = dict(d)
                        # 使用 1-based 的 repeat_idx，便于和“第几次”对应
                        new_d["repeat_idx"] = run_idx + 1
                        combined_list.append(new_d)
            agg_entry[k] = json.dumps(combined_list, ensure_ascii=False)
            continue

        # 其余字段仍然走通用聚合逻辑
        agg_entry[k] = _aggregate_value(per_run_vals)

    # 确保 candidate_id / candidate 字段存在
    cid = agg_entry.get("candidate_id") or agg_entry.get("candidate")
    if "candidate_id" not in agg_entry:
        agg_entry["candidate_id"] = cid
    if "candidate" not in agg_entry:
        agg_entry["candidate"] = cid

    # 对 scores.ground / scores.structure 做特殊聚合：
    # - score：改为合并数值列表
    # - 新增 min_score：该维度在所有 repeat 中的最小值
    # - raw_judge_output：checks 打平并增加 repeat_idx，confidence 收集为列表
    scores_agg = agg_entry.get("scores")
    if isinstance(scores_agg, Dict):
        for dim in ("ground", "structure"):
            dim_obj = scores_agg.get(dim)
            if not isinstance(dim_obj, dict):
                continue

            # 1) 聚合 score 为数值列表，并计算 min_score
            score_list: List[float] = []
            for _, e in enumerate(entries):
                s = e.get("scores") or {}
                s_dim = s.get(dim) or {}
                v = s_dim.get("score")
                if v is None:
                    continue
                try:
                    score_list.append(float(v))
                except Exception:
                    continue
            if score_list:
                dim_obj["score"] = score_list
                dim_obj["min_score"] = min(score_list)
            else:
                dim_obj["score"] = []
                dim_obj["min_score"] = None

            # 2) 从各轮中收集原始 raw_judge_output
            per_run_raw: List[Tuple[int, str]] = []
            for run_idx, e in enumerate(entries):
                s = e.get("scores") or {}
                s_dim = s.get(dim) or {}
                raw_str = s_dim.get("raw_judge_output")
                if raw_str:
                    per_run_raw.append((run_idx, raw_str))

            if not per_run_raw:
                continue

            merged_checks: List[Dict[str, Any]] = []
            conf_list: List[float] = []
            for run_idx, raw_str in per_run_raw:
                try:
                    jd = json.loads(raw_str)
                except Exception:
                    continue
                # checks：列表里的每个 dict 增加 repeat_idx，然后拼接
                checks = jd.get("checks") or []
                if isinstance(checks, list):
                    for c in checks:
                        if isinstance(c, dict):
                            nc = dict(c)
                            nc["repeat_idx"] = run_idx + 1
                            merged_checks.append(nc)
                # confidence：不取均值，而是收集成列表
                if "confidence" in jd:
                    try:
                        conf_val = float(jd.get("confidence"))
                        conf_list.append(conf_val)
                    except Exception:
                        pass

            agg_raw = {
                "checks": merged_checks,
                "confidence": conf_list,
            }
            dim_obj["raw_judge_output"] = json.dumps(agg_raw, ensure_ascii=False)

    return agg_entry


def run_merge_scores(args: argparse.Namespace) -> None:
    # 加载环境变量
    try:
        load_env()
    except Exception:
        pass

    processed_dir = _build_processed_dir_from_raw(args.raw_data)

    # ==== 增量更新：若 merged/merger 已存在，优先使用作为基线 ====
    base_samples, base_results, meta = _load_existing_generated_and_judge_incremental(
        processed_dir
    )
    print(
        "[Info] 增量基线：\n"
        f"  - generated: {meta['generated_path']}\n"
        f"  - judge:     {meta['judge_path']}"
    )

    extra_files = _discover_extra_generated_files(
        processed_dir,
        explicit_files=args.extra_input_jsonl,
    )
    if not extra_files:
        print(f"[Info] 未发现需要合并的增量 generated_responses_*.jsonl，目录：{processed_dir}")
        return

    print("[Info] 将合并以下增量文件：")
    for p in extra_files:
        print(f"  - {p}")

    # ==== 初始化 LLM 客户端与参数 ====
    if OpenAI is None:
        raise ImportError("openai 包未安装，请先 pip install openai>=1.0")

    if not args.ground_api_key:
        raise ValueError(
            "GROUND judge 需要 API key，可通过 --ground_api_key 或环境变量 "
            "LLM_MODEL_GROUND_API_KEY / LLM_API_KEY / OPENAI_API_KEY 提供"
        )
    if not args.struct_api_key:
        raise ValueError(
            "STRUCT judge 需要 API key，可通过 --struct_api_key 或环境变量 "
            "LLM_MODEL_STRUCT_API_KEY / LLM_API_KEY / OPENAI_API_KEY 提供"
        )

    ground_client = OpenAI(api_key=args.ground_api_key, base_url=args.ground_base_url)
    struct_client = OpenAI(api_key=args.struct_api_key, base_url=args.struct_base_url)

    # ==== 收集所有新候选 ====
    # per sample_id -> List[candidate_dict]
    new_candidates_by_sample: Dict[str, List[Dict[str, Any]]] = {}

    # 已存在的 candidate_id 集合，用来去重
    existing_candidate_ids_by_sample: Dict[str, set[str]] = {}
    # 1) 来自已打分的 judge results（基础或 merger）
    for sid, results in base_results.items():
        s: set[str] = set()
        for r in results:
            cid = str(r.get("candidate_id") or r.get("candidate") or "").strip()
            if cid:
                s.add(cid)
        existing_candidate_ids_by_sample[sid] = s

    # 2) 来自已合并的 generated candidates（防止重复合并/重复打分）
    for sid, sample in base_samples.items():
        cands = sample.get("candidates") or []
        if not isinstance(cands, list):
            continue
        existed = existing_candidate_ids_by_sample.setdefault(sid, set())
        for c in cands:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("candidate_id") or c.get("candidate") or "").strip()
            if cid:
                existed.add(cid)

    for extra_path in extra_files:
        rows = _read_jsonl(extra_path)
        for row in rows:
            sid = str(row.get("sample_id") or "").strip()
            if not sid or sid not in base_samples:
                # 安全起见，只合并在基础文件中已存在的样本
                continue
            cands = row.get("candidates") or []
            if not isinstance(cands, list):
                continue
            existed = existing_candidate_ids_by_sample.setdefault(sid, set())
            target_list = new_candidates_by_sample.setdefault(sid, [])
            for c in cands:
                if not isinstance(c, dict):
                    continue
                cid = str(
                    c.get("candidate_id") or c.get("candidate") or ""
                ).strip()
                if not cid or cid in existed:
                    continue
                existed.add(cid)
                # 只对“新增待评估候选”的 response 做 think 标签清洗
                new_c = deepcopy(c)
                if "response" in new_c:
                    new_c["response"] = _strip_think_tags(str(new_c.get("response") or ""))
                target_list.append(new_c)

    # ==== 对新候选进行打分 ====
    new_results_by_sample: Dict[str, List[Dict[str, Any]]] = {}

    # ==== 对新候选进行打分：支持多次重复 + 并行 ====
    num_repeat = max(1, int(getattr(args, "num_repeat", 1)))
    workers = max(1, int(getattr(args, "workers", 8)))

    # 构建 candidate 级别的任务列表，便于在线程池中并行
    tasks: List[Tuple[str, int, Dict[str, Any], str, str, Dict[str, Any]]] = []
    for sid, cand_list in new_candidates_by_sample.items():
        if not cand_list:
            continue
        base_sample = base_samples.get(sid)
        if not base_sample:
            continue
        row, user_input, history_input = _build_row_for_kto(base_sample)
        for idx, cand in enumerate(cand_list):
            tasks.append((sid, idx, row, user_input, history_input, cand))

    if not tasks:
        print("[Info] 没有需要打分的新候选。")
        return

    print(
        f"[Info] 共有 {len(tasks)} 个新候选需要打分，"
        f"num_repeat={num_repeat}，workers={workers}"
    )

    def _score_one_candidate(
        task: Tuple[str, int, Dict[str, Any], str, str, Dict[str, Any]],
    ) -> Tuple[str, int, Dict[str, Any]] | None:
        sid, idx, row, user_input, history_input, cand = task
        cid = str(cand.get("candidate_id") or cand.get("candidate") or "").strip()
        model_type = str(cand.get("model_type") or "")
        model_name = str(cand.get("model_name") or "")
        answer = str(cand.get("response") or "")

        try:
            # 单次打分：保持与旧版本行为一致
            if num_repeat <= 1:
                local_args = deepcopy(args)
                setattr(local_args, "current_repeat_idx", 0)
                judged = kto_mod.judge_one_answer(
                    ground_client=ground_client,
                    struct_client=struct_client,
                    args=local_args,
                    user_input=user_input,
                    history_input=history_input,
                    answer=answer,
                    row=row,
                    progress_callback=None,
                    candidate_id=cid,
                    model_type=model_type,
                    model_name=model_name,
                )
                single_entry = _build_single_run_candidate_entry(
                    judged=judged,
                    row=row,
                    user_input=user_input,
                    history_input=history_input,
                    candidate_id=cid,
                    model_type=model_type,
                    model_name=model_name,
                    answer=answer,
                    args=local_args,
                )
                agg_entry = _aggregate_single_run_entry(single_entry)
                return sid, idx, agg_entry

            # 多次重复打分：对齐 KTO 主脚本的 num_repeat 语义与聚合逻辑
            single_run_entries: List[Dict[str, Any]] = []
            for repeat_idx in range(num_repeat):
                local_args = deepcopy(args)
                setattr(local_args, "current_repeat_idx", repeat_idx)
                judged = kto_mod.judge_one_answer(
                    ground_client=ground_client,
                    struct_client=struct_client,
                    args=local_args,
                    user_input=user_input,
                    history_input=history_input,
                    answer=answer,
                    row=row,
                    progress_callback=None,
                    candidate_id=cid,
                    model_type=model_type,
                    model_name=model_name,
                )
                single_entry = _build_single_run_candidate_entry(
                    judged=judged,
                    row=row,
                    user_input=user_input,
                    history_input=history_input,
                    candidate_id=cid,
                    model_type=model_type,
                    model_name=model_name,
                    answer=answer,
                    args=local_args,
                )
                single_run_entries.append(single_entry)

            agg_entry = _aggregate_candidate_repeats(single_run_entries)
            return sid, idx, agg_entry

        except Exception as e:  # pragma: no cover - 防御性日志
            print(
                f"[Error] 打分失败 sample_id={sid} candidate_id={cid} model={model_name}: {e}"
            )
            return None

    # candidate 级别的并行打分
    results_indexed: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_score_one_candidate, task): task for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            res = future.result()
            if not res:
                continue
            sid, idx, agg_entry = res
            results_indexed.setdefault(sid, []).append((idx, agg_entry))

    # 恢复到每个 sample_id 对应的 results 列表，并按 candidate 在输入中的顺序排序
    for sid, items in results_indexed.items():
        ordered = [entry for _, entry in sorted(items, key=lambda x: x[0])]
        if ordered:
            new_results_by_sample[sid] = ordered

    # ==== 写出合并后的 generated_responses_merged.jsonl ====
    merged_generated_rows: List[Dict[str, Any]] = []
    for sid, sample in base_samples.items():
        merged_sample = deepcopy(sample)
        extra_cands = new_candidates_by_sample.get(sid, [])
        if extra_cands:
            base_cands = list(sample.get("candidates") or [])
            merged_sample["candidates"] = base_cands + extra_cands
        merged_generated_rows.append(merged_sample)

    gen_merged_path = processed_dir / "generated_responses_merged.jsonl"
    _write_jsonl(gen_merged_path, merged_generated_rows)
    print(f"[Done] 写出合并后的 generated_responses 到: {gen_merged_path}")

    # ==== 写出合并后的 judge_results_kto_merger.jsonl ====
    merged_judge_rows: List[Dict[str, Any]] = []
    all_sample_ids = sorted(base_samples.keys(), key=lambda x: int(x) if x.isdigit() else x)
    for sid in all_sample_ids:
        base_res_list = base_results.get(sid, [])
        merged_res_list = list(base_res_list)
        new_res_list = new_results_by_sample.get(sid, [])
        if new_res_list:
            merged_res_list.extend(new_res_list)
        merged_judge_rows.append(
            {
                "sample_id": sid,
                "results": merged_res_list,
            }
        )

    judge_merged_path = processed_dir / "judge_results_kto_merger.jsonl"
    _write_jsonl(judge_merged_path, merged_judge_rows)
    print(f"[Done] 写出合并后的 judge 结果到: {judge_merged_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在已有 generated_responses.jsonl / judge_results_kto.jsonl 基础上，"
            "对增量 generated_responses_*.jsonl 中的新候选补充打分，并生成合并后的结果文件。"
        )
    )
    parser.add_argument(
        "--raw-data",
        required=True,
        help=(
            "本次实验的原始输入数据文件路径（例如 CSV），用于根据文件名自动推导 "
            "data/<文件名去扩展名>/processed/ 目录。"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发 worker 数（线程），用于对新候选并行打分。",
    )
    parser.add_argument(
        "--num_repeat",
        type=int,
        default=3,
        help="每个候选重复打分次数，与 KTO 主脚本默认设置保持一致。",
    )
    parser.add_argument(
        "--extra-input-jsonl",
        nargs="+",
        default=None,
        help=(
            "显式指定增量 generated_responses_*.jsonl 文件路径（相对路径默认基于 processed 目录）。"
            "若不指定，则自动从 processed 目录下发现所有 generated_responses_*.jsonl 并排除基础文件。"
        ),
    )

    # 与 kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py 中保持一致的参数
    parser.add_argument(
        "--ground_base_url",
        default=(
            os.environ.get("LLM_MODEL_GROUND_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://api.deepinfra.com/v1/openai"
        ),
        help="GROUND judge 使用的 base_url",
    )
    parser.add_argument(
        "--ground_api_key",
        default=(
            os.environ.get("LLM_MODEL_GROUND_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        ),
        help="GROUND judge 使用的 api_key",
    )
    parser.add_argument(
        "--struct_base_url",
        default=(
            os.environ.get("LLM_MODEL_STRUCT_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://api.deepinfra.com/v1/openai"
        ),
        help="STRUCT judge 使用的 base_url",
    )
    parser.add_argument(
        "--struct_api_key",
        default=(
            os.environ.get("LLM_MODEL_STRUCT_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        ),
        help="STRUCT judge 使用的 api_key",
    )
    parser.add_argument(
        "--ground_model",
        default=(
            os.environ.get("LLM_MODEL_GROUND_NAME")
            or os.environ.get("GROUND_MODEL")
            or "gpt-5-mini-2025-08-07"
        ),
    )
    parser.add_argument(
        "--struct_model",
        default=(
            os.environ.get("LLM_MODEL_STRUCT_NAME")
            or os.environ.get("STRUCT_MODEL")
            or "gpt-5-mini-2025-08-07"
        ),
    )

    # label/weight 相关参数，与 KTO 主脚本保持一致
    parser.add_argument("--threshold", type=float, default=12.0)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--w_min", type=float, default=0.1)
    parser.add_argument("--w_max", type=float, default=5.0)
    parser.add_argument("--allow_negative", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    # 先从 .env 加载环境变量，再解析命令行参数
    try:
        load_env()
    except Exception:
        # 若 python-dotenv 未安装或加载失败，不影响后续逻辑，
        # 仍可通过显式传参或已有环境变量提供 key
        pass

    args = parse_args()
    run_merge_scores(args)

"""
python src/scoring/merge_and_score_addtional_response.py \
  --raw-data data_diff_sample.csv \
  --workers 8 \
  --num_repeat 3 \
  --extra-input-jsonl generated_responses_qwen32_lora_sft_maxepochs30.jsonl generated_responses_qwen32_lora_kto_maxepochs30.jsonl generated_responses_qwen32_lora_sft_kto_maxepochs30.jsonl
"""