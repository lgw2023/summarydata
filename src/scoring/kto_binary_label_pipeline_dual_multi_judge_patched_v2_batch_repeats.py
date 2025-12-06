# -*- coding: utf-8 -*-
"""
kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch.py

Batch processing version with:
  - Multi-threading support (--workers)
  - Resume capability (skips already processed rows based on _source_row_index)
  - Streaming CSV output (saves progress in real-time)

Usage:
python kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch.py \
  --input_csv qa_dual.csv \
  --output_long_csv out_long.csv \
  --output_wide_csv out_wide.csv \
  --workers 10 \
  ...
"""

import os, re, json, time, argparse, math, sys
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
import pandas as pd
import tqdm
from string import Template
import concurrent.futures
import threading

# 允许作为脚本直接运行时也能找到项目根目录下的 src 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.env import load_env

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ===================== Scoring constants =====================
FIXED_MAX_SCORE = 20.0
ALLOWED_DEDUCTIONS = [20, 10, 5, 3]

STRICT_20 = {
    "EMPTY_OR_INCOMPLETE","ILLEGAL_CONTENT","SENSITIVE_ADVICE",
    "PERSONAL_DATA_MISMATCH","COURSE_LIB_MISSING","NUM_COMPARE_ERROR",
    "ARITH_ERROR","IRRELEVANT"
}
FIXED_5   = {"NO_MARKDOWN"}
FIXED_3   = {"REDUNDANT","GRAMMAR"}

LENIENT_RULES = {
    "CONTRADICT_KB_OR_EXPERT": (5, 10),
    "FACT_LOGIC_ISSUE": (5, 10),
    "PERSONAL_DATA_ANALYSIS_ISSUE": (3, 5),
    # validators may also set:
    "MISSING_CHART_TABLE": (5, 10),
    "MISSING_SERVICE": (10, 10),
    "HALLUCINATION_VALUE": (10, 10),
}

# 维度区分：哪些 rule 视为 Ground / Structure 相关，用于单独输出 0–5 分
GROUND_DIM_RULE_IDS = {
    "PERSONAL_DATA_MISMATCH",
    "COURSE_LIB_MISSING",
    "NUM_COMPARE_ERROR",
    "ARITH_ERROR",
    "CONTRADICT_KB_OR_EXPERT",
    "FACT_LOGIC_ISSUE",
    "IRRELEVANT",
    "MISSING_CHART_TABLE",
    "MISSING_SERVICE",
    "HALLUCINATION_VALUE",
}

STRUCT_DIM_RULE_IDS = {
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

# ===================== Utilities =====================
def _normalize_confidence(v):
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    s = str(v).strip().lower()
    if s.endswith("%"):
        try:
            return max(0.0, min(1.0, float(s[:-1])/100.0))
        except:
            return 0.5
    mapping = {
        "very high": 0.95, "high": 0.85,
        "medium": 0.60, "mid": 0.60,
        "low": 0.30, "very low": 0.10,
        "certain": 0.98, "uncertain": 0.40,
        "likely": 0.75, "unlikely": 0.25,
        "yes": 0.9, "no": 0.1,
        "true": 0.9, "false": 0.1,
    }
    if s in mapping:
        return mapping[s]
    try:
        return max(0.0, min(1.0, float(s)))
    except:
        return 0.5

def _coerce_penalty_score(x):
    try:
        v = float(x)
    except Exception:
        return None
    # snap to allowed buckets
    return float(min(ALLOWED_DEDUCTIONS, key=lambda a: abs(a - v)))

def _dedup_keep_max(penalties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = {}
    for p in penalties or []:
        rid = (p.get("rule_id") or "").upper()
        sc  = _coerce_penalty_score(p.get("score", 0)) or 0
        if sc <= 0:
            continue
        if rid not in merged or sc > merged[rid].get("score", 0):
            merged[rid] = {"rule_id": rid, "score": float(sc), "reason": p.get("reason",""), "excerpt": p.get("excerpt","")}
    return list(merged.values())

def enforce_and_recompute(judge_like: Dict[str, Any], allow_negative=False):
    penalties = _dedup_keep_max(judge_like.get("penalties", []))
    total = FIXED_MAX_SCORE - sum(p["score"] for p in penalties)
    if not allow_negative:
        total = max(0.0, total)
    out = dict(judge_like)
    out["scale"] = "fixed_20"
    out["max_score"] = FIXED_MAX_SCORE
    out["penalties"] = penalties
    out["total_score"] = float(total)
    out["confidence"] = _normalize_confidence(judge_like.get("confidence", 0.5))
    return out

def compute_label_and_weight(
    score: float,
    threshold: float = 12.0,
    min_score: float = 0.0,
    max_score: float = 20.0,
    delta: float = 0.5,
    gamma: float = 1.5,
    kappa: float = 0.5,
    confidence: float = 0.5,
    w_min: float = 0.1,
    w_max: float = 5.0,
) -> Dict[str, Any]:
    rng = max(1e-6, max_score - min_score)
    margin_raw = abs(score - threshold)
    margin = max(0.0, margin_raw - delta) / rng
    w = (margin ** gamma) * (1.0 + kappa * confidence)
    if w < w_min:
        w = w_min
    if w > w_max:
        w = w_max
    label = 1 if score >= threshold else 0
    return {"label": label, "weight": w, "threshold": threshold}

def guess_column(row_keys: List[str], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in row_keys:
            return name
    return None

# ===================== Validators =====================
def parse_data_json(row: dict) -> Dict[str, Any]:
    v = row.get("data")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return {}
    try:
        if isinstance(v, str):
            return json.loads(v)
        return dict(v)
    except Exception:
        return {}

def validator_sleep(answer: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Strict sleep checks: duration alignment and thresholded grading presence."""
    pens = []
    sleep = data.get("sleep") if isinstance(data, dict) else None
    if not isinstance(sleep, dict):
        return pens
    # duration check
    target_dur = sleep.get("duration_hours")
    start = sleep.get("start")
    end   = sleep.get("end")
    if start and end and (not target_dur):
        # naive cross-day handling
        try:
            # Try parse HH:MM or H:M text; this is a light validator (not datetime heavy)
            def _parse_hm(s):
                m = re.search(r"(\d{1,2}):(\d{2})", s)
                if not m: return None
                return int(m.group(1)), int(m.group(2))
            sh = _parse_hm(str(start)); eh = _parse_hm(str(end))
            if sh and eh:
                dur = (eh[0]*60+eh[1]) - (sh[0]*60+sh[1])
                if dur < 0: dur += 24*60
                target_dur = round(dur/60.0, 2)
        except:
            pass

    if target_dur is not None:
        # extract numeric duration in hours from answer
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|小时)", answer, flags=re.IGNORECASE)
        if m:
            try:
                ans_dur = float(m.group(1))
                if abs(ans_dur - float(target_dur)) > 0.25:  # >15 min gap
                    pens.append({"rule_id":"PERSONAL_DATA_MISMATCH","score":20,
                                 "reason":f"睡眠时长不匹配: ans={ans_dur}h vs data={target_dur}h（昨晚=昨天晚→今天早）",
                                 "excerpt": m.group(0)})
            except:
                pass
        else:
            pens.append({"rule_id":"PERSONAL_DATA_ANALYSIS_ISSUE","score":5,
                         "reason":"模块提供了睡眠时长信息，但答案未体现",
                         "excerpt":""})

    # score thresholding
    th = sleep.get("score_thresholds")
    scr = sleep.get("score")
    if isinstance(th, dict) and scr is not None:
        # expected grade
        def grade_by_thresholds(score: float, thresholds: Dict[str, Any]) -> Optional[str]:
            for k, rng in thresholds.items():
                try:
                    lo, hi = float(rng[0]), float(rng[1])
                    if lo <= score < hi or math.isclose(score, hi):
                        return k
                except Exception:
                    continue
            return None
        exp_grade = grade_by_thresholds(float(scr), th)
        if exp_grade:
            # require grade token in answer
            tokens = [exp_grade.lower(), "优","良","中","差","一般","poor","fair","good"]
            if not any(tok.lower() in answer.lower() for tok in tokens):
                pens.append({"rule_id":"PERSONAL_DATA_ANALYSIS_ISSUE","score":5,
                             "reason":f"已给 score/thresholds，应给出分级结论（期望包含 {exp_grade} 或等价中文）",
                             "excerpt":""})
    return pens

def validator_chart_table(answer: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    pens = []
    need = data.get("require_chart_or_table") if isinstance(data, dict) else None
    if not need:
        return pens
    has_table = bool(re.search(r"\n\|.+\|\n\|[-:\s|]+\|\n", answer))
    has_image = "![ " in answer or "![" in answer
    if not (has_table or has_image):
        pens.append({"rule_id":"MISSING_CHART_TABLE","score":10,
                     "reason":"要求提供图表/表格但回答缺失","excerpt":""})
    return pens

def validator_service(answer: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    pens = []
    services = data.get("services") if isinstance(data, dict) else None
    if not services:
        return pens
    missing = []
    for s in services if isinstance(services, list) else []:
        if isinstance(s, str) and s.strip():
            if s.lower() not in answer.lower():
                missing.append(s)
    if missing:
        pens.append({"rule_id":"MISSING_SERVICE","score":10,
                     "reason":f"回答未体现服务项: {', '.join(missing[:6])}",
                     "excerpt":""})
    return pens

def run_validators(answer: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    pens = []
    pens += validator_sleep(answer, data)
    pens += validator_chart_table(answer, data)
    pens += validator_service(answer, data)
    return pens

# ===================== Prompts (JSON-only checks) =====================
from score_prompt_v3 import GROUND_SYSTEM_PROMPT_TPL, GROUND_PROMPT_TPL
from score_prompt_v3 import STRUCT_SYSTEM_PROMPT_TPL, STRUCT_PROMPT_TPL


# ===================== LLM call =====================
def extract_json_from_text(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks and extra text."""
    if not text:
        raise ValueError("Empty response from LLM")
    
    text = text.strip()
    
    # Try to find JSON in markdown code blocks (handle nested braces by finding matching braces)
    json_block_pattern = r'```(?:json)?\s*(\{.*\})\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        # Validate it's valid JSON by trying to parse
        try:
            json.loads(candidate)
            return candidate
        except:
            pass
    
    # Try to find JSON object directly (from first { to matching last })
    brace_start = text.find('{')
    if brace_start >= 0:
        # Find matching closing brace by counting braces
        brace_count = 0
        brace_end = -1
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    brace_end = i
                    break
        
        if brace_end > brace_start:
            candidate = text[brace_start:brace_end+1]
            # Validate it's valid JSON
            try:
                json.loads(candidate)
                return candidate
            except:
                pass
    
    # If no valid JSON found, return original text (will fail in json.loads but we'll handle it)
    return text

def call_judge(
    client,
    model_name: str,
    prompt: str,
    system_prompt: str = "You must output a single JSON object exactly as instructed. No extra text.",
    retries: int = 10,
    temperature: float = 0.0,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    last_err = None
    last_text = None
    for a in range(retries):
        start_ts = time.time()
        try:
            # 日志：每次调用开始时输出关键信息（模型名、尝试次数、温度）
            # print(
            #     f"[INFO][call_judge] Calling model={model_name} "
            #     f"attempt={a+1}/{retries} temperature={temperature}",
            #     file=sys.stderr,
            # )

            # Try to use json_object format if supported (some APIs may not support it)
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
            except Exception as format_err:
                # Fallback if json_object format is not supported
                resp = client.chat.completions.create(
                    model=model_name,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
            
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Empty response from LLM")
            
            last_text = text
            
            # Extract JSON from text (handles markdown code blocks)
            json_text = extract_json_from_text(text)
            data = json.loads(json_text)
            checks = data.get("checks", [])
            conf = _normalize_confidence(data.get("confidence", 0.5))

            # 日志：成功解析出 JSON 后输出简要结果信息
            duration = time.time() - start_ts
            print(
                "[INFO][call_judge] Success "
                f"model={model_name} attempt={a+1}/{retries} "
                f"confidence={conf:.3f} checks_len={len(checks)} "
                f"elapsed={duration:.2f}s",
                file=sys.stderr,
            )

            # 成功完成一次 LLM 调用后，更新进度条（若提供）
            if progress_callback is not None:
                try:
                    progress_callback(1)
                except Exception:
                    # 进度条更新失败不应影响主流程
                    pass

            return {"checks": checks, "confidence": conf}
        except Exception as e:
            last_err = e
            duration = time.time() - start_ts
            error_msg = f"Attempt {a+1}/{retries} failed: {type(e).__name__}: {str(e)}"
            if last_text:
                # Log first 200 chars of response for debugging
                preview = last_text[:200] + ("..." if len(last_text) > 200 else "")
                error_msg += f"\nResponse preview: {preview}"
            # 日志：失败时同样输出模型名与耗时，方便排查
            error_msg = (
                f"[INFO][call_judge] Failed model={model_name} "
                f"attempt={a+1}/{retries} elapsed={duration:.2f}s\n"
                + error_msg
            )
            # 无论是哪种异常，只要本次调用未能成功解析出打分结构，就打印告警日志
            print(f"[WARN][call_judge] {error_msg}", file=sys.stderr)
            time.sleep(1.0 + a)
    
    final_error = f"judge failed after {retries} retries: {last_err}"
    if last_text:
        final_error += f"\nLast response (first 500 chars): {last_text[:500]}"
    # 所有重试结束仍未能成功解析出打分结构，打印最终错误日志
    print(f"[ERROR][call_judge] {final_error}", file=sys.stderr)
    raise RuntimeError(final_error)

def checks_to_penalties(checks: List[dict]) -> List[dict]:
    out = []
    for c in checks or []:
        rid = (c.get("rule_id") or "").upper()
        if not c.get("hit", False):
            continue
        sev = str(c.get("severity","")).lower()
        if rid in STRICT_20:
            sc = 20
        elif rid in FIXED_5:
            sc = 5
        elif rid in FIXED_3:
            sc = 3
        elif rid == "PERSONAL_DATA_ANALYSIS_ISSUE":
            sc = 5 if sev in ("5","major") else 3
        elif rid in {"CONTRADICT_KB_OR_EXPERT","FACT_LOGIC_ISSUE"}:
            sc = 10 if sev in ("major","严重","high") else 5
        elif rid in {"BURIED_CORE_ANSWER", "THIN_CONTENT"}:
            sc = 5
        elif rid in {"UNNATURAL_TONE", "BAD_MARKDOWN_USAGE", "LACK_VISUAL_AID"}:
            sc = 3
        else:
            # Unknown/unsupported rule -> ignore
            continue
        out.append({"rule_id": rid, "score": float(sc), "reason": c.get("reason",""), "excerpt": c.get("excerpt","")})
    return out

def minimal_programmatic_penalties(answer: str, data: dict) -> List[dict]:
    pens = []
    # No Markdown quick check
    has_md = ("#" in answer) or ("- " in answer) or ("* " in answer) or ("|" in answer and "\n|-" in answer)
    if not has_md:
        pens.append({"rule_id":"NO_MARKDOWN","score":5,"reason":"无明显 Markdown 结构","excerpt":""})
    # Sleep grade required when thresholds+score exist
    sleep = data.get("sleep") if isinstance(data, dict) else None
    if isinstance(sleep, dict):
        if ("score_thresholds" in sleep) and ("score" in sleep):
            tokens = ["poor","fair","good","优","良","中","差","一般"]
            if not any(tok.lower() in answer.lower() for tok in tokens):
                pens.append({"rule_id":"PERSONAL_DATA_ANALYSIS_ISSUE","score":5,
                             "reason":"模块给了score/thresholds，但答案未给出睡眠质量分级结论",
                             "excerpt":""})
    return pens

def zero_penalty_guard(all_pens: List[dict], answer: str, data: dict) -> List[dict]:
    if all_pens:
        return all_pens
    return minimal_programmatic_penalties(answer, data)

# ===================== Orchestration =====================
def merge_penalties(*pen_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    allp = []
    for pl in pen_lists:
        allp.extend(pl or [])
    return _dedup_keep_max(allp)

def build_modules_block(row: dict, data_json: Dict[str, Any], kb_text: str) -> str:
    def g(name):
        v = row.get(name)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)
    personal = g("data")
    course   = g("service")
    kb = []
    kb_in = g("rag")
    if kb_in: kb.append(kb_in)
    if kb_text: kb.append(kb_text)
    expert = g("suggest")

    block = []
    block.append("## 模块数据")
    block.append("[个人数据]\n" + (personal or ""))
    block.append("[课程库]\n" + (course or ""))
    if kb:
        block.append("[知识库知识]\n" + "\n".join([x for x in kb if x]))
    if expert:
        block.append("[专家建议]\n" + expert)
    if data_json:
        block.append("[data JSON]\n" + json.dumps(data_json, ensure_ascii=False))
    return "\n\n".join(block)

def build_modules_text(row: dict) -> Tuple[str, str]:
    """Backward-compat: returns kb_text (combined) and modules_block without data json (legacy)."""
    def g(name):
        v = row.get(name)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)
    kb = []
    if g("rag"): kb.append(g("rag"))
    if g("suggest"): kb.append(g("suggest"))
    kb_text = "\n".join([x for x in kb if x])
    personal_data = g("data")
    course_lib = g("service")
    modules_block = f"[个人数据]\n{personal_data}\n\n[课程库]\n{course_lib}"
    return kb_text, modules_block

def parse_context_string(context: str) -> dict:
    """Parse context string to extract module data."""
    result = {
        "data": "",
        "suggest": "",
        "rag": "",
        "service": "",
        "last_answer_phone": "",
        "query": ""
    }
    
    # Split by section markers
    sections = {
        "[个人数据]": "data",
        "[专家建议]": "suggest",
        "[知识库知识]": "rag",
        "[课程库]": "service",
        "[对话历史]": "last_answer_phone",
        "[用户提问]": "query"
    }
    
    current_section = None
    lines = context.split("\n")
    
    for line in lines:
        line_stripped = line.strip()
        # Check if this line is a section header
        found_section = None
        for header, key in sections.items():
            if header in line_stripped:
                found_section = key
                current_section = key
                # Extract content after the header on the same line if any
                remaining = line_stripped.replace(header, "").strip()
                if remaining:
                    result[current_section] = remaining
                else:
                    result[current_section] = ""
                break
        
        if found_section is None and current_section:
            # This is content for the current section
            if result[current_section]:
                result[current_section] += "\n" + line
            else:
                result[current_section] = line
    
    # Clean up "（无）" markers
    for key in result:
        if result[key] and result[key].strip() == "（无）":
            result[key] = ""
    
    # Extract query from last_answer_phone if it contains "user:"
    if result["last_answer_phone"]:
        # Check if it contains "assistant:" prefix
        if "assistant:" in result["last_answer_phone"]:
            result["last_answer_phone"] = result["last_answer_phone"].replace("assistant:", "").strip()
    
    return result

def judge_pair_ground(
    client,
    model,
    user_input: str,
    history_input: str,
    answer: str,
    row: dict,
    data_json: Dict[str, Any],
    kb_text: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    modules_block = build_modules_block(row, data_json, kb_text)
    prompt = GROUND_PROMPT_TPL.safe_substitute(
        input_data=user_input,
        history_input=history_input,
        answer=answer,
        modules_block=modules_block,
    )
    return call_judge(
        client,
        model,
        prompt,
        system_prompt=GROUND_SYSTEM_PROMPT_TPL,
        temperature=0.0,
        progress_callback=progress_callback,
    )


def judge_pair_struct(
    client,
    model,
    user_input: str,
    history_input: str,
    answer: str,
    row: dict,
    data_json: Dict[str, Any],
    kb_text: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    modules_block = build_modules_block(row, data_json, kb_text)
    prompt = STRUCT_PROMPT_TPL.safe_substitute(
        input_data=user_input,
        history_input=history_input,
        answer=answer,
        modules_block=modules_block,
    )
    return call_judge(
        client,
        model,
        prompt,
        system_prompt=STRUCT_SYSTEM_PROMPT_TPL,
        temperature=0.0,
        progress_callback=progress_callback,
    )


def judge_one_answer(
    ground_client,
    struct_client,
    args,
    user_input: str,
    history_input: str,
    answer: str,
    row: dict,
    progress_callback: Optional[Callable[[int], None]] = None,
    candidate_id = None,
    model_type = None,
    model_name = None,
) -> Dict[str, Any]:
    """
    对单条回复进行完整打分：
    - 统一合并 Ground / Structure / 程序校验的 penalties，得到 0–20 分总分；
    - 计算 KTO 所需的 label / weight；
    - 同时把中间结果（两个 judge 的原始输出等）挂在返回 dict 上，方便 JSONL 路径复用。
    """
    data_json = parse_data_json(row)
    kb_text, _ = build_modules_text(row)

    # 0) deterministic validators
    p_val = run_validators(answer, data_json)

    # 1) two judges -> checks（GROUND / STRUCT 各自使用独立的 client）
    jg = judge_pair_ground(
        ground_client,
        args.ground_model,
        user_input,
        history_input,
        answer,
        row,
        data_json,
        kb_text,
        progress_callback=progress_callback,
    )
    js = judge_pair_struct(
        struct_client,
        args.struct_model,
        user_input,
        history_input,
        answer,
        row,
        data_json,
        kb_text,
        progress_callback=progress_callback,
    )

    # 2) checks -> penalties (local mapping)
    p_g = checks_to_penalties(jg.get("checks", []))
    p_s = checks_to_penalties(js.get("checks", []))

    # 3) merge + zero-penalty guard（统一合并所有来源的 penalties）
    merged = merge_penalties(p_val, p_g, p_s)
    merged = zero_penalty_guard(merged, answer, data_json)

    # 4) recompute total + confidence (avg)
    conf = (jg.get("confidence", 0.5) + js.get("confidence", 0.5)) / 2.0
    final = enforce_and_recompute(
        {"penalties": merged, "confidence": conf},
        allow_negative=args.allow_negative,
    )
    # 5) 计算 KTO 所需的 label / weight
    lw = compute_label_and_weight(
        score=final["total_score"],
        threshold=args.threshold,
        min_score=0.0,
        max_score=20.0,
        delta=args.delta,
        gamma=args.gamma,
        kappa=args.kappa,
        confidence=final["confidence"],
        w_min=args.w_min,
        w_max=args.w_max,
    )

    # 调试打印：附带 sample_id / 行号 / 模型信息 / 当前 repeat 次数，便于排查
    try:
        meta_info = {
            "repeat_idx": getattr(args, "current_repeat_idx", None),
            "sample_id": row.get("sample_id"),
            "candidate_id": candidate_id,
            "model_type": model_type,
            "model_name": model_name,
            "merged": [p["score"] for p in merged],
            "penalties": [p["score"] for p in final.get("penalties")], 
            "confidence": conf,
            "total_score": final.get("total_score"),
            "threshold": lw.get("threshold"),
            "label": lw.get("label"),
            "weight": lw.get("weight"),
        }
        print("[INFO][judge_one_answer]", json.dumps(meta_info, ensure_ascii=False))
    except Exception as e:
        # 打印失败不影响主流程
        # print(f"[Error] judge_one_answer failed: {e}")
        pass

    # 把内部中间结果一并挂在返回 dict 上，方便后续 JSONL 封装使用
    final["_label"] = lw["label"]
    final["_weight"] = lw["weight"]
    final["_ground_judge"] = jg
    final["_struct_judge"] = js
    final["_p_val"] = p_val
    final["_p_g"] = p_g
    final["_p_s"] = p_s
    final["_data_json"] = data_json
    final["_kb_text"] = kb_text
    final["_user_input"] = user_input
    final["_answer"] = answer
    return final

def process_row(
    idx: int,
    row: dict,
    ground_client,
    struct_client,
    args,
    query_col: str,
    ans_a_col: str,
    ans_b_col: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[int, Dict, List[Dict]]:
    """Thread worker function to process a single row."""
    # 构造与 JSONL 路径一致的 user / history 文本
    raw_query = row.get(query_col)
    user_query = "" if (raw_query is None or (isinstance(raw_query, float) and pd.isna(raw_query))) else str(raw_query)
    user_input = f"user: {user_query}" if user_query else ""

    last_answer = row.get("last_answer_phone")
    if last_answer is None or (isinstance(last_answer, float) and pd.isna(last_answer)):
        history_input = ""
    else:
        history_input = f"assistant: {str(last_answer)}"

    ansA = "" if pd.isna(row.get(ans_a_col)) else str(row.get(ans_a_col))
    ansB = "" if pd.isna(row.get(ans_b_col)) else str(row.get(ans_b_col))

    try:
        # 注意：judge_one_answer 签名为
        # (ground_client, struct_client, args, user_input: str, history_input: str, answer: str, row: dict, progress_callback: Optional[Callable[[int], None]])
        resA = judge_one_answer(
            ground_client,
            struct_client,
            args,
            user_input,
            history_input,
            ansA,
            row,
            progress_callback=progress_callback,
        )
        resB = judge_one_answer(
            ground_client,
            struct_client,
            args,
            user_input,
            history_input,
            ansB,
            row,
            progress_callback=progress_callback,
        )
    except Exception as e:
        print(f"[Error] Row {idx} failed: {e}")
        # Return empty/error results or reraise. For batch processing, better to log and continue with partial result or empty.
        # We'll just return None to skip saving this row.
        return idx, None, None

    # long-form (one row per QA)
    base_keep = {k: row.get(k) for k in row.keys()}
    # Add source index to track progress
    base_keep["_source_row_index"] = idx
    
    recA = dict(base_keep)
    recA.update({
        "which":"A",
        "answer": ansA,
        "total_score": resA["total_score"],
        "confidence": resA["confidence"],
        "label": resA["_label"],
        "weight": resA["_weight"],
        "penalties_json": json.dumps(resA.get("penalties", []), ensure_ascii=False),
        "scale": "fixed_20",
        "max_score": 20.0,
    })
    recB = dict(base_keep)
    recB.update({
        "which":"B",
        "answer": ansB,
        "total_score": resB["total_score"],
        "confidence": resB["confidence"],
        "label": resB["_label"],
        "weight": resB["_weight"],
        "penalties_json": json.dumps(resB.get("penalties", []), ensure_ascii=False),
        "scale": "fixed_20",
        "max_score": 20.0,
    })
    
    long_rows_list = [recA, recB]

    # wide-form (one row per input)
    wide = {"_source_row_index": idx} # Key for resume logic
    # Copy base cols? Usually wide table is for summary. Let's keep base keys too if feasible, or just minimal.
    # Original script didn't copy base keys to wide, but it's safer to have them or at least the query.
    # Let's follow original minimal approach but add index.
    
    wide.update({
        "total_score_a": resA["total_score"],
        "confidence_a":  resA["confidence"],
        "label_a":       resA["_label"],
        "weight_a":      resA["_weight"],
        "penalties_json_a": json.dumps(resA.get("penalties", []), ensure_ascii=False),

        "total_score_b": resB["total_score"],
        "confidence_b":  resB["confidence"],
        "label_b":       resB["_label"],
        "weight_b":      resB["_weight"],
        "penalties_json_b": json.dumps(resB.get("penalties", []), ensure_ascii=False),
    })
    
    return idx, wide, long_rows_list

def process_jsonl_sample(
    sample: dict,
    ground_client,
    struct_client,
    args,
    progress_callback: Optional[Callable[[int], None]] = None,
    completed_candidate_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Process a single JSONL sample and return judge results for all candidates.

    原实现对 candidates 是串行处理，这里改为在 sample 内部对每个 candidate 并行打分，
    以便在单个 sample 有多路回答时也能利用多线程并行调用 LLM。
    """
    sample_id = sample.get("sample_id", "")
    context = sample.get("context", "")
    question = sample.get("question", "")
    candidates = sample.get("candidates", []) or []

    # 仅针对 JSONL 断点续跑：如果传入了已完成的 candidate_id 集合，则只对“未完成”的 candidates 继续打分
    if completed_candidate_ids:
        pending_candidates = []
        for c in candidates:
            cid = c.get("candidate_id", "") or c.get("candidate", "") or ""
            # 如果没有 id，则无法与历史结果对应，保守起见视为“尚未完成”，需要重新评估
            if (not cid) or (cid not in completed_candidate_ids):
                pending_candidates.append(c)
        candidates = pending_candidates

    # 若该 sample 下所有 candidate 都已完成，直接返回空结果（调用方会跳过写入）
    if not candidates:
        return {
            "sample_id": sample_id,
            "results": [],
        }
    
    # Parse context to extract module data
    parsed_context = parse_context_string(context)
    
    # Build row dict for judge_one_answer compatibility
    row = {
        "sample_id": sample_id,
        "data": parsed_context.get("data", ""),
        "suggest": parsed_context.get("suggest", ""),
        "rag": parsed_context.get("rag", ""),
        "service": parsed_context.get("service", ""),
        "last_answer_phone": parsed_context.get("last_answer_phone", ""),
    }
    
    # Build user_input from question and last_answer_phone
    user_input = f"user: {question}"
    history_input = ""
    if parsed_context.get("last_answer_phone"):
        history_input = f"assistant: {parsed_context['last_answer_phone']}"
    # user_input_parts = []
    # if parsed_context.get("last_answer_phone"):
    #     user_input_parts.append(f"assistant: {parsed_context['last_answer_phone']}")
    # user_input_parts.append(f"user: {question}")
    # user_input = "\n".join(user_input_parts)
    
    # Process each candidate：统一复用 judge_one_answer 的 0–20 分打分逻辑
    # 为了在单个 sample 内也能并行，我们在这里按 candidate 级别再开一层线程池。

    def _process_one_candidate(idx: int, candidate: dict):
        """单个 candidate 的打分逻辑，保持与原先循环体完全一致。"""
        candidate_id = candidate.get("candidate_id", "")
        model_type = candidate.get("model_type", "")
        model_name = candidate.get("model_name", "")
        response = candidate.get("response", "")

        try:
            # 直接调用 judge_one_answer，确保与 CSV 路径完全一致的单条回复打分逻辑
            judged = judge_one_answer(
                ground_client,
                struct_client,
                args,
                user_input,
                history_input,
                response,
                row,
                progress_callback=progress_callback,
                candidate_id=candidate_id,
                model_type=model_type,
                model_name=model_name,
            )

            total_score_20 = float(judged.get("total_score", 0.0))
            confidence = float(judged.get("confidence", 0.5))
            label = int(judged.get("_label", 0))
            weight = float(judged.get("_weight", 1.0))

            # ===================== Ground / Structure 各自 0–5 分 =====================
            # 从统一的 penalties 中，按 rule_id 划分到 Ground / Structure 两个维度，
            # 各自做「20 – 扣分 → 映射到 0–5」的变换，避免 ground / structure 分数完全相同。
            penalties_all = judged.get("penalties", []) or []

            def _dim_score_5(rule_set):
                total_deduction = 0.0
                for p in penalties_all:
                    rid = (p.get("rule_id") or "").upper()
                    if rid in rule_set:
                        try:
                            total_deduction += float(p.get("score", 0.0))
                        except Exception:
                            continue
                local_total_20 = FIXED_MAX_SCORE - total_deduction
                if not args.allow_negative:
                    local_total_20 = max(0.0, local_total_20)
                # 映射到 0–5
                return local_total_20 * 5.0 / FIXED_MAX_SCORE

            ground_score_5 = _dim_score_5(GROUND_DIM_RULE_IDS)
            structure_score_5 = _dim_score_5(STRUCT_DIM_RULE_IDS)

            # 用统一的 0–20 总分映射得到 0–5 aggregate_score
            aggregate_score_5 = total_score_20 / 20.0 * 5.0

            # 取出内部缓存的中间结果
            data_json = judged.get("_data_json", {}) or {}
            kb_text = judged.get("_kb_text", "") or ""
            jg = judged.get("_ground_judge", {}) or {}
            js = judged.get("_struct_judge", {}) or {}

            # 构造用于调试的 prompt 文本
            modules_block = build_modules_block(row, data_json, kb_text)
            ground_prompt = GROUND_PROMPT_TPL.safe_substitute(
                input_data=user_input,
                history_input=history_input,
                answer=response,
                modules_block=modules_block,
            )
            structure_prompt = STRUCT_PROMPT_TPL.safe_substitute(
                input_data=user_input,
                history_input=history_input,
                answer=response,
                modules_block=modules_block,
            )

            # 构建与 judge_results.jsonl 兼容的结果结构：
            # - ground/structure 分数目前复用同一套 0–5 分（来自统一的 0–20 总分）
            # - 顶层额外暴露 0–20 total_score / label / weight，方便与 CSV 路径对齐
            result_entry = {
                "candidate_id": candidate_id,
                "candidate": candidate_id,  # alias
                "model_type": model_type,
                "model_name": model_name,
                "scores": {
                    "ground": {
                        "score": ground_score_5,
                        "max_score": 5.0,
                        "raw_judge_output": json.dumps(jg, ensure_ascii=False, indent=2),
                        "raw_judge_prompt": ground_prompt,
                    },
                    "structure": {
                        "score": structure_score_5,
                        "max_score": 5.0,
                        "raw_judge_output": json.dumps(js, ensure_ascii=False, indent=2),
                        "raw_judge_prompt": structure_prompt,
                    },
                },
                "ground_score": ground_score_5,
                "structure_score": structure_score_5,
                "aggregate_score": aggregate_score_5,
                # 统一的 0–20 总分及 KTO 相关字段
                "total_score_20": total_score_20,
                "label": label,
                "weight": weight,
                "confidence": confidence,
                "penalties_json_20": json.dumps(judged.get("penalties", []), ensure_ascii=False),
                "judge": "llm",
                "judge_meta": {
                    "judge_name": "llm",
                },
                "notes": None,
            }
            return idx, result_entry

        except Exception as e:
            print(f"[Error] Failed to judge candidate {candidate_id} in sample {sample_id}: {e}")
            # Continue with other candidates
            return idx, None

    # 如果当前 sample 没有 candidates，直接返回空结果
    if not candidates:
        return {
            "sample_id": sample_id,
            "results": [],
        }

    # 在 sample 内部对所有 candidates 并行打分
    results_with_index = []

    # inner_workers 用于控制「每个 sample 内」的并行度：
    # - 未显式指定时，退回到 workers，保持与改造前相同的行为；
    # - 显式指定时，可以把总并发控制在：
    #     并发 LLM 调用 ≈ min(workers, num_samples) * min(inner_workers, candidates_per_sample)
    inner_workers = getattr(args, "inner_workers", None)
    if inner_workers is None or inner_workers <= 0:
        inner_workers = getattr(args, "workers", 16) or 1

    max_workers = min(inner_workers, len(candidates))
    if max_workers < 1:
        max_workers = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_one_candidate, idx, candidate): idx
            for idx, candidate in enumerate(candidates)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            try:
                idx, result_entry = future.result()
                if result_entry is not None:
                    results_with_index.append((idx, result_entry))
            except Exception as exc:
                # 理论上 _process_one_candidate 已经兜底，这里仅做额外保护
                idx = future_to_idx[future]
                print(f"[Error] Unexpected exception for candidate index {idx} in sample {sample_id}: {exc}")

    # 为了保持与原先行为一致，按 candidate 在输入中的顺序排序结果
    results_with_index.sort(key=lambda x: x[0])
    ordered_results = [r for _, r in results_with_index]

    return {
        "sample_id": sample_id,
        "results": ordered_results,
    }


def make_llm_progress(total: int, desc: str):
    """
    创建一个带线程锁的 tqdm 进度条，并返回：
    - progress_callback: 在多线程环境中安全更新进度条的回调
    - pbar: 原始 tqdm 对象，便于在调用方手动 close()
    """
    pbar = tqdm.tqdm(total=total, desc=desc)
    lock = threading.Lock()

    def progress_callback(n: int = 1) -> None:
        with lock:
            pbar.update(n)

    return progress_callback, pbar


# ===================== CLI =====================
# 默认 JSONL 输入/输出路径（基于项目根目录）
DEFAULT_INPUT_JSONL = os.path.join(
    PROJECT_ROOT, "data", "processed", "generated_responses.jsonl"
)
DEFAULT_OUTPUT_JSONL = os.path.join(
    PROJECT_ROOT, "data", "processed", "judge_results_kto.jsonl"
)


def make_repeat_path(base_path: str, suffix: str) -> str:
    """
    为带 repeat 后缀的文件生成路径：
    - 在原始文件同级目录下，新建 `<basename>_repeats/` 子目录
    - 文件名形如：`<basename>_<suffix><ext>`
      例如：
        base_path = /a/b/judge_results_kto.jsonl
        suffix    = repeat0
        -> /a/b/judge_results_kto_repeats/judge_results_kto_repeat0.jsonl
    """
    base_dir, base_name = os.path.split(base_path)
    root, ext = os.path.splitext(base_name)
    repeat_dir = os.path.join(base_dir, f"{root}_repeats")
    os.makedirs(repeat_dir, exist_ok=True)
    return os.path.join(repeat_dir, f"{root}_{suffix}{ext}")


def main():
    # 先尝试加载项目根目录下的 .env，填充环境变量
    try:
        load_env()
    except Exception:
        # 加载失败时保持静默，不影响后续从系统环境读取
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="Input CSV file (legacy format)")
    parser.add_argument(
        "--input_jsonl",
        help=(
            "Input JSONL file (generated_responses.jsonl format). "
            "若未指定且提供了 --raw-data，则自动使用 "
            "data/<原始文件名去扩展名>/processed/generated_responses.jsonl。"
        ),
    )
    parser.add_argument("--output_long_csv", help="Output long CSV file (legacy format)")
    parser.add_argument("--output_wide_csv", help="Output wide CSV file (legacy format)")
    parser.add_argument(
        "--output_jsonl",
        help=(
            "Output JSONL file (judge_results.jsonl format). "
            "若未指定：\n"
            "- 且提供了 --raw-data，则写入 data/<原始文件名去扩展名>/processed/judge_results_kto.jsonl；\n"
            "- 否则写在 --input_jsonl 同目录下的 judge_results_kto.jsonl。"
        ),
    )
    parser.add_argument(
        "--raw-data",
        dest="raw_data",
        help=(
            "本次实验的原始输入数据文件路径（例如 CSV），用于根据文件名自动推导 "
            "data/<文件名去扩展名>/processed/ 下的 JSONL 输入/输出路径。"
        ),
    )

    # New args
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent worker threads for outer-level parallelism (JSONL: samples, CSV: rows)",
    )
    parser.add_argument(
        "--inner_workers",
        type=int,
        default=None,
        help="Max concurrent candidates per sample (JSONL mode only). "
             "If not set, defaults to --workers. "
             "Effective LLM concurrency ≈ min(workers, num_samples) × min(inner_workers, candidates_per_sample).",
    )
    parser.add_argument("--num_repeat", type=int, default=3, help="Number of repetitions for the entire batch process")

    # API（GROUND / STRUCT 使用完全独立的配置，避免环境变量混用）
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
    

    # columns
    parser.add_argument("--query_col")
    parser.add_argument("--answer_a_col")
    parser.add_argument("--answer_b_col")

    # sampling / trial run
    parser.add_argument(
        "--head_n",
        type=int,
        default=0,
        help="只取前 N 行样本进行评估（0 表示使用全部数据）",
    )

    # label/weight
    parser.add_argument("--threshold", type=float, default=12.0)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--w_min", type=float, default=0.1)
    parser.add_argument("--w_max", type=float, default=5.0)
    parser.add_argument("--allow_negative", action="store_true")

    args = parser.parse_args()

    # 若提供了 --raw-data，则根据其文件名自动推导 JSONL 输入/输出路径：
    #   <PROJECT_ROOT>/data/<raw_data_stem>/processed/generated_responses.jsonl
    #   <PROJECT_ROOT>/data/<raw_data_stem>/processed/judge_results_kto.jsonl
    raw_data_value = getattr(args, "raw_data", None)
    if raw_data_value:
        raw_path = Path(raw_data_value)
        if not raw_path.is_absolute():
            raw_path = Path(PROJECT_ROOT) / raw_data_value
        run_name = raw_path.stem
        processed_dir = Path(PROJECT_ROOT) / "data" / run_name / "processed"
        # 若未显式指定 input_*，优先走 JSONL 模式，并使用推导出的 generated_responses.jsonl
        if not args.input_csv and not args.input_jsonl:
            args.input_jsonl = str(processed_dir / "generated_responses.jsonl")
        # 若未显式指定 output_jsonl，则默认写到同一 processed 目录下
        if not args.output_jsonl:
            args.output_jsonl = str(processed_dir / "judge_results_kto.jsonl")

    if OpenAI is None:
        raise ImportError("openai package not found. pip install openai>=1.0")
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

    # Prepare Clients (GROUND / STRUCT 各自独立)
    ground_client = OpenAI(api_key=args.ground_api_key, base_url=args.ground_base_url)
    struct_client = OpenAI(api_key=args.struct_api_key, base_url=args.struct_base_url)

    # Determine input format
    input_jsonl_mode = bool(args.input_jsonl)
    input_csv_mode = bool(args.input_csv)
    
    if not input_jsonl_mode and not input_csv_mode:
        raise ValueError("Must specify either --input_csv or --input_jsonl")
    if input_jsonl_mode and input_csv_mode:
        raise ValueError("Cannot specify both --input_csv and --input_jsonl")
    
    # Handle JSONL input mode
    if input_jsonl_mode:
        # 未显式传 --output_jsonl 且也未通过 --raw-data 推导时，
        # 默认写到 input_jsonl 同目录下的 judge_results_kto.jsonl
        if not args.output_jsonl:
            inp_path = Path(args.input_jsonl)
            args.output_jsonl = str(inp_path.with_name("judge_results_kto.jsonl"))

        input_file = args.input_jsonl
        base_output_file = args.output_jsonl

        # Read JSONL samples（所有 repeat 共享同一批样本）
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"[Warn] Skipping invalid JSON line: {e}")
                    continue

        # Optional: only keep first N samples for trial runs
        if getattr(args, "head_n", 0) and args.head_n > 0:
            samples = samples[:args.head_n]

        print(f"[Info] Loaded {len(samples)} samples from {input_file}")

        # 为 JSONL 模式记录输入样本的顺序，用于后续输出保持与输入一致的 sample_id 顺序
        sample_order = {}
        for idx, s in enumerate(samples):
            sid = s.get("sample_id", "")
            if sid and sid not in sample_order:
                sample_order[sid] = idx

        # 聚合辅助：对同一字段在多个 repeat 中的值做合并
        def _aggregate_value(per_run_values):
            """
            per_run_values: List[(run_idx, value)]
            规则：
            - 数值：按重复次数取均值
            - 布尔：所有重复都是 True 才为 True，否则 False
            - 字符串及其他类型：按“次数：内容”并用换行拼接
            - dict：递归按上述规则聚合其内部字段
            """
            non_none = [(idx, v) for idx, v in per_run_values if v is not None]
            if not non_none:
                return None

            first_val = non_none[0][1]

            # bool 需要优先判断（bool 是 int 的子类）
            if isinstance(first_val, bool):
                # 只要有一次为 False，则结果为 False
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
                agg_dict = {}
                for k in all_keys:
                    sub_vals = [(idx, d.get(k)) for idx, d in non_none]
                    agg_dict[k] = _aggregate_value(sub_vals)
                return agg_dict

            # 其余（字符串 / list / 其他）一律转为文本并按“次数：内容”拼接
            parts = []
            for run_idx, v in non_none:
                parts.append(f"{run_idx + 1}: {str(v)}")
            return "\n".join(parts)

        # 将多个 *_repeat{idx}.jsonl 聚合成一个基准文件：
        # - 数值字段：取均值
        # - 布尔字段：全 True 才 True
        # - 文本/其他：按“次数：内容”+换行拼接
        def _aggregate_jsonl_repeats(base_path: str, num_repeat: int):
            # 映射：sample_id -> candidate_id -> List[(run_idx, entry_dict)]
            agg_map = {}

            for repeat_idx in range(num_repeat):
                repeat_file = make_repeat_path(base_path, f"repeat{repeat_idx}")
                if not os.path.exists(repeat_file):
                    continue
                try:
                    with open(repeat_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            sample_id = obj.get("sample_id")
                            if sample_id is None:
                                continue
                            results = obj.get("results") or []
                            if sample_id not in agg_map:
                                agg_map[sample_id] = {}
                            for res in results:
                                cid = (
                                    res.get("candidate_id")
                                    or res.get("candidate")
                                    or ""
                                )
                                if cid not in agg_map[sample_id]:
                                    agg_map[sample_id][cid] = []
                                agg_map[sample_id][cid].append((repeat_idx, res))
                except Exception as e:
                    print(f"[Warn] Failed to read repeat file {repeat_file} for aggregation: {e}")

            if not agg_map:
                print(f"[Warn] No repeat JSONL files found for aggregation. Skip writing aggregate file: {base_path}")
                return

            # 写入聚合后的基准 JSONL 文件（按输入样本顺序或 sample_id 排序输出）
            def _sample_sort_key(sample_id):
                # 优先按原始 samples 中的顺序；若不存在，则尝试按数值/字符串排序
                if sample_order:
                    idx = sample_order.get(sample_id)
                    if idx is not None:
                        return idx
                try:
                    return int(sample_id)
                except Exception:
                    return sample_id

            with open(base_path, "w", encoding="utf-8") as out_f:
                for sample_id in sorted(agg_map.keys(), key=_sample_sort_key):
                    cand_map = agg_map[sample_id]
                    agg_results = []
                    for cid, entries in cand_map.items():
                        # 对该 sample_id + candidate_id 下的所有字段做聚合
                        all_keys = set()
                        for _, e in entries:
                            all_keys.update(e.keys())

                        agg_entry = {}
                        stable_keys = {
                            "candidate",
                            "candidate_id",
                            "model_name",
                            "model_type",
                            "judge",
                            "judge_meta",
                        }

                        for k in all_keys:
                            per_run_vals = [(idx, e.get(k)) for idx, e in entries]

                            # 这些字段在各轮中应当保持不变，直接取第一轮的值，避免被当作字符串拼接
                            if k in stable_keys:
                                first_entry = entries[0][1]
                                agg_entry[k] = first_entry.get(k)
                                continue

                            # 对于外层 ground_score / structure_score，改为合并数值列表
                            if k in {"ground_score", "structure_score"}:
                                score_list = []
                                for _, e in entries:
                                    v = e.get(k)
                                    if v is None:
                                        continue
                                    try:
                                        score_list.append(float(v))
                                    except Exception:
                                        continue
                                agg_entry[k] = score_list
                                continue

                            # 对于 aggregate_score / total_score_20：
                            # - 保持原字段为均值（沿用通用聚合逻辑）
                            # - 额外新增 aggregate_score_list / total_score_20_list，保存各轮数值列表
                            if k in {"aggregate_score", "total_score_20"}:
                                nums = []
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

                            # penalties_json_20：内部是字典列表，这里按 extend 风格拼接，并为每个字典增加 repeat_idx
                            if k == "penalties_json_20":
                                combined_list = []
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
                        if "candidate_id" not in agg_entry:
                            agg_entry["candidate_id"] = cid
                        if "candidate" not in agg_entry:
                            agg_entry["candidate"] = cid

                        # 对 scores.ground / scores.structure 做特殊聚合：
                        # - score：改为合并数值列表
                        # - 新增 min_score：该维度在所有 repeat 中的最小值
                        # - raw_judge_output：checks 打平并增加 repeat_idx，confidence 收集为列表
                        scores_agg = agg_entry.get("scores")
                        if isinstance(scores_agg, dict):
                            for dim in ("ground", "structure"):
                                dim_obj = scores_agg.get(dim)
                                if not isinstance(dim_obj, dict):
                                    continue

                                # 1) 聚合 score 为数值列表，并计算 min_score
                                score_list = []
                                for _, e in entries:
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
                                    # 若无有效得分，仍保证字段存在但为空
                                    dim_obj["score"] = []
                                    dim_obj["min_score"] = None

                                # 2) 从各轮中收集原始 raw_judge_output
                                per_run_raw = []
                                for run_idx, e in entries:
                                    s = e.get("scores") or {}
                                    s_dim = s.get(dim) or {}
                                    raw_str = s_dim.get("raw_judge_output")
                                    if raw_str:
                                        per_run_raw.append((run_idx, raw_str))

                                if not per_run_raw:
                                    continue

                                merged_checks = []
                                conf_list = []
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

                        agg_results.append(agg_entry)

                    out_line = {
                        "sample_id": sample_id,
                        "results": agg_results,
                    }
                    out_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")

            print(f"[Done] Aggregated {len(agg_map)} samples across {num_repeat} repeats into: {base_path}")

        # 支持 JSONL 模式下的多轮重复评估
        num_repeat = max(1, getattr(args, "num_repeat", 1))
        for repeat_idx in range(num_repeat):
            if num_repeat > 1:
                print(f"\n=== Starting JSONL Repeat Session {repeat_idx} (total {num_repeat}) ===")

            # 把当前 repeat 次数挂到 args 上，供下游调试打印使用
            setattr(args, "current_repeat_idx", repeat_idx)

            # 当 num_repeat == 1 时保持原有行为，直接写入基准文件；
            # 当 num_repeat > 1 时，为每一轮单独生成子目录中的 *_repeat{idx}.jsonl 文件，便于后处理与稳定性分析。
            if num_repeat == 1:
                output_file = base_output_file
            else:
                output_file = make_repeat_path(base_output_file, f"repeat{repeat_idx}")

            # Resume：每一轮根据当前轮的输出文件判断**每个 sample 下已经完成的 candidate**
            processed_candidates: Dict[str, set] = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                existing = json.loads(line)
                            except Exception:
                                continue
                            sid = existing.get("sample_id")
                            if not sid:
                                continue
                            results = existing.get("results") or []
                            for res in results:
                                if not isinstance(res, dict):
                                    continue
                                cid = (
                                    res.get("candidate_id")
                                    or res.get("candidate")
                                    or ""
                                )
                                if not cid:
                                    continue
                                # 仅在首次记录该 candidate 时打印一条“已完成、无需重跑”的日志
                                done_set = processed_candidates.setdefault(sid, set())
                                if cid not in done_set:
                                    done_set.add(cid)
                                    model_name = res.get("model_name") or ""
                                    print(
                                        "[Info][resume] Reuse completed result: "
                                        f"repeat={repeat_idx} sample_id={sid} "
                                        f"candidate_id={cid} model_name={model_name}"
                                    )
                    print(
                        f"[Info] Resuming: Found {sum(len(v) for v in processed_candidates.values())} "
                        f"processed candidates across {len(processed_candidates)} samples in {output_file}"
                    )
                except Exception as e:
                    print(f"[Warn] Could not read existing output file for resume: {e}. Starting fresh for this repeat.")

            # 为当前 repeat 构建待处理列表：仅针对“仍有未完成 candidate 的 sample”调度任务
            tasks_to_run = []
            total_pending_candidates = 0
            for sample in samples:
                sample_id = sample.get("sample_id", "")
                cands = sample.get("candidates", []) or []
                if not sample_id or not cands:
                    continue
                done_set = processed_candidates.get(sample_id, set())
                pending = 0
                for c in cands:
                    cid = (
                        c.get("candidate_id")
                        or c.get("candidate")
                        or ""
                    )
                    # 如果没有 id，只能整体判断：
                    # - 若该 sample 下历史结果中没有任何 candidate 记录，则认为本次需要整体评估；
                    # - 若已有记录，则认为这一批“无 id 候选”已评估过，避免重复。
                    if not cid:
                        if sample_id not in processed_candidates:
                            pending = len(cands)
                        break
                    if cid not in done_set:
                        pending += 1
                if pending > 0:
                    tasks_to_run.append(sample)
                    total_pending_candidates += pending

            print(
                f"[Info][Repeat {repeat_idx}] Total samples: {len(samples)}. "
                f"Samples to process in this repeat: {len(tasks_to_run)}. "
                f"Pending candidates: {total_pending_candidates}."
            )

            # 为本轮构建更精确的「LLM 调用」进度条：
            # 每个“尚未完成”的 candidate 会被 Ground / Structure 两个 judge 各调用一次 → 2 次 LLM 调用
            total_candidates = total_pending_candidates
            total_llm_calls = total_candidates * 2

            # 进度条与锁在主线程中创建，供各 worker 线程安全更新
            _progress_callback, llm_pbar = make_llm_progress(
                total_llm_calls,
                f"LLM calls (Repeat {repeat_idx})",
            )

            # Process with threading（按 sample 粒度并发），并在每个 sample 完成后立即写入一行 JSONL，
            # 避免整轮结束前进程异常导致已完成样本结果全部丢失。
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor, \
                 open(output_file, 'a', encoding='utf-8') as f:
                future_to_sample = {
                    executor.submit(
                        process_jsonl_sample,
                        sample,
                        ground_client,
                        struct_client,
                        args,
                        _progress_callback,
                        processed_candidates.get(sample.get("sample_id", ""), set()),
                    ): sample
                    for sample in tasks_to_run
                }

                # 样本级别的进度条：反映还有多少个 sample 尚未处理完
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(future_to_sample),
                    total=len(tasks_to_run),
                    desc=f"Judging JSONL (Repeat {repeat_idx}) [samples]",
                ):
                    try:
                        result = future.result()
                        if result and result.get("results"):
                            # 直接按完成顺序逐行写入，保证每个 sample 处理完就落盘
                            out_obj = {
                                "sample_id": result.get("sample_id", ""),
                                "results": result.get("results") or [],
                            }
                            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                            f.flush()
                    except Exception as exc:
                        sample = future_to_sample[future]
                        sample_id = sample.get("sample_id", "unknown")
                        print(f"[Error] Exception for sample {sample_id}: {exc}")

            llm_pbar.close()

            print(f"[Done] Finished JSONL Repeat {repeat_idx}. Results saved to: {output_file}")

        # 若 num_repeat > 1，则基于各轮 *_repeat{idx}.jsonl 生成一个聚合后的基准文件
        if num_repeat > 1:
            try:
                _aggregate_jsonl_repeats(base_output_file, num_repeat)
            except Exception as e:
                print(f"[Warn] Failed to aggregate JSONL repeats into base file {base_output_file}: {e}")

        return
    
    # Handle CSV input mode (legacy)
    input_file = args.input_csv
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        raise ValueError("Excel input is not supported. Please use CSV input.")
    else:
        df = pd.read_csv(input_file)

    # Optional: only keep first N rows for trial runs
    if getattr(args, "head_n", 0) and args.head_n > 0:
        df = df.head(args.head_n)

    row_keys = list(df.columns)
    query_col = args.query_col or guess_column(row_keys, ["query"])
    ans_a_col = args.answer_a_col or guess_column(row_keys, ["a_answer"])
    ans_b_col = args.answer_b_col or guess_column(row_keys, ["b_answer"])
    if not (query_col and ans_a_col and ans_b_col):
        raise ValueError(f"Cannot find required columns. Found keys: {row_keys}. Need one query + two answers (e.g., user_input/answer_a/answer_b).")

    # Resume Logic
    # NOTE: For long_csv, we change from excel to csv to support streaming append.
    # We do this check once before looping repeats.
    if args.output_long_csv and args.output_long_csv.endswith(".xlsx"):
        print("[WARN] Output long file ends in .xlsx but streaming write supports CSV only. Changing extension to .csv for safety.")
        args.output_long_csv = args.output_long_csv.replace(".xlsx", ".csv")

    base_wide = args.output_wide_csv
    base_long = args.output_long_csv
    all_rows = df.to_dict("records")

    for repeat_idx in range(args.num_repeat):
        print(f"\n=== Starting Repeat Session {repeat_idx} (total {args.num_repeat}) ===")

        # 把当前 repeat 次数挂到 args 上，供下游调试打印使用
        setattr(args, "current_repeat_idx", repeat_idx)

        # Construct filenames for this repeat：带 repeat 的 CSV 放到下一级子目录中
        cur_wide_csv = make_repeat_path(base_wide, f"repeat{repeat_idx}")
        cur_long_csv = make_repeat_path(base_long, f"repeat{repeat_idx}")

        processed_indices = set()
        if os.path.exists(cur_wide_csv):
            try:
                # Only read columns needed to check index
                existing_df = pd.read_csv(cur_wide_csv, usecols=["_source_row_index"])
                processed_indices = set(existing_df["_source_row_index"].unique())
                print(f"[Info] Resuming: Found {len(processed_indices)} processed rows in {cur_wide_csv}")
            except Exception as e:
                print(f"[Warn] Could not read existing output file for resume: {e}. Starting fresh or appending blindly.")
        
        tasks_to_run = []
        for i, row in enumerate(all_rows):
            if i not in processed_indices:
                tasks_to_run.append((i, row))
        
        print(f"[Info] Repeat {repeat_idx}: Total rows: {len(all_rows)}. To process: {len(tasks_to_run)}.")

        # Determine if we need to write headers
        wide_header_needed = not os.path.exists(cur_wide_csv)
        long_header_needed = not os.path.exists(cur_long_csv)

        # 对 CSV 模式同样增加基于 LLM 调用次数的精细进度条：
        # 每行包含 A/B 两个回答、每个回答需要 Ground / Structure 各一次 → 理论上 4 次 LLM 调用/行
        total_llm_calls_csv = len(tasks_to_run) * 4
        _progress_callback_csv, llm_pbar_csv = make_llm_progress(
            total_llm_calls_csv,
            f"LLM calls (Repeat {repeat_idx})",
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_row,
                    idx,
                    row,
                    ground_client,
                    struct_client,
                    args,
                    query_col,
                    ans_a_col,
                    ans_b_col,
                    _progress_callback_csv,
                ): idx
                for (idx, row) in tasks_to_run
            }

            # Iterate as they complete（样本级进度条）
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(tasks_to_run),
                desc=f"Judging (Repeat {repeat_idx}) [rows]",
            ):
                try:
                    idx, wide_res, long_res_list = future.result()
                    if wide_res is None:
                        continue  # Failed row

                    # Streaming Write - Wide
                    pd.DataFrame([wide_res]).to_csv(
                        cur_wide_csv,
                        mode='a',
                        header=wide_header_needed,
                        index=False,
                    )
                    wide_header_needed = False  # Only needed once

                    # Streaming Write - Long
                    if long_res_list:
                        pd.DataFrame(long_res_list).to_csv(
                            cur_long_csv,
                            mode='a',
                            header=long_header_needed,
                            index=False,
                        )
                        long_header_needed = False

                except Exception as exc:
                    idx = future_to_idx[future]
                    print(f"[Error] Exception for row {idx}: {exc}")

        llm_pbar_csv.close()

        print(f"[Done] Finished Repeat {repeat_idx}. Results saved to:\n  - {cur_wide_csv}\n  - {cur_long_csv}")

if __name__ == "__main__":
    main()

