from __future__ import annotations

from functools import lru_cache
import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .models import ValueType


_TIME_HHMM_RE = re.compile(r"\d{1,2}:\d{2}")


def _strip_leading_de(s: str) -> str:
    """
    清理中文里常见的前导“的”（例如：'的锻炼时长' -> '锻炼时长'）。
    """
    t = (s or "").strip()
    while t.startswith("的") and len(t) > 1:
        t = t[1:].lstrip()
    return t


def _normalize_time_cn_token(s: str) -> str:
    """
    把时间 token 规范化为 HH:mm（仅用于“时间点”语义，不涉及时长）。
    支持：
    - 20:33
    - 20时33分 / 20点33分 / 20时33
    """
    t = str(s or "").strip()
    if not t:
        return ""
    m = re.fullmatch(r"(?P<h>\d{1,2}):(?P<m>\d{1,2})", t)
    if m:
        try:
            hh = int(m.group("h"))
            mm = int(m.group("m"))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}"
        except Exception:
            return t
        return t
    m2 = re.fullmatch(r"(?P<h>\d{1,2})\s*(?:时|点)\s*(?P<m>\d{1,2})\s*(?:分)?", t)
    if m2:
        try:
            hh = int(m2.group("h"))
            mm = int(m2.group("m"))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}"
        except Exception:
            return t
        return t
    return t


def _normalize_value_and_unit(value_str: str, vt: ValueType, unit: str) -> tuple[str, str]:
    """
    规范化“数值+单位”：
    - Timestamp：单位固定为“无”
    - Int/Float：尽量把尾部单位从数值里剥离（数值保存纯数字字符串）
    - FloatRange：把区间内的单位剥离（如 96%-98% -> 96-98，单位=%）
    - Duration：
        - 配速/比率（含 “/”）：数值保留分子时长表现，单位仅保留分母语义（每公里/每米等）
        - 纯持续时长（不含 “/”）：统一保留原串，单位固定为“无”
    """
    s = (value_str or "").strip()
    u = (unit or "无").strip() or "无"

    # 时间点：单位必须是“无”
    if vt == "Timestamp":
        return s, "无"

    # 纯数值：剥离尾部单位
    if vt in ("Int", "Float"):
        if u != "无":
            s2 = re.sub(rf"\s*{re.escape(u)}\s*$", "", s).strip()
            return (s2 if s2 else s), u
        return s, u

    # 区间数值：剥离单位（常见：96%-98%、15分-17分）
    if vt == "FloatRange":
        if u != "无":
            s2 = s.replace(u, "").strip()
            # % 常见为粘连符号，补充去除
            if u == "%":
                s2 = s2.replace("%", "").strip()
            return (s2 if s2 else s), u
        return s, u

    # 时长：单一单位可剥离；复合时长不强行归一，避免破坏信息
    if vt == "Duration":
        # 配速/比率类 Duration：形如 "7分39秒/公里"、"7.8分钟/公里"、"0.5小时/公里"
        #
        # 统一目标：
        # - 数值：保留“时长表现”（如 21分47秒 / 7.8分钟 / 0.5小时）
        # - 单位：仅表达分母语义（如 每公里 / 每米），避免出现 "分钟/公里" 这类把“时间”塞进 unit 的写法
        #
        # 兼容：
        # - 历史 unit 可能仍是 "分钟/公里" / "小时/公里" / "秒/公里"；这里继续支持。
        if "/" in s:
            left_raw, right_raw = (p.strip() for p in s.split("/", 1))
            left_has_time = any(x in left_raw for x in ("小时", "分钟", "秒", "毫秒")) or ("分" in left_raw and "秒" in left_raw)
            right_has_dist = any(x in right_raw for x in ("公里", "千米", "米", "km", "KM", "Km"))
            if left_has_time and right_has_dist:
                # 若 unit 未给出（或为无），这里兜底推一个“每X”
                u2 = u
                if u2 == "无":
                    denom = "公里" if any(x in right_raw for x in ("公里", "千米", "km", "KM", "Km")) else "米"
                    u2 = f"每{denom}"

                left = left_raw
                # 兼容旧版：unit 是 "分钟/公里" 这类时，左侧末尾的“分钟/小时/秒”可剥离以避免重复
                if "/" in u2:
                    if u2.startswith("分钟/"):
                        left = re.sub(r"\s*分钟\s*$", "", left).strip() or left
                    elif u2.startswith("小时/"):
                        left = re.sub(r"\s*小时\s*$", "", left).strip() or left
                    elif u2.startswith("秒/"):
                        left = re.sub(r"\s*秒\s*$", "", left).strip() or left
                # 新版：unit 为 "每公里" 等，不剥离左侧时间单位词（否则会丢失“分钟”语义）
                return left, u2
        # 纯持续时长：统一不拆单位，避免表头出现 "(分钟)/(小时分钟)" 等。
        return s, "无"

    return s, u


def _attach_unit_to_value(value_str: str, unit: str) -> str:
    """
    把单位拼接回数值字符串（用于需要展示/存储“带单位”的场景）。

    约定：
    - unit == "无" 或 value 为空：直接返回 value
    - 若 value 已经以 unit 结尾（或 unit 本身为空）：不重复追加
    """
    v = (value_str or "").strip()
    u = (unit or "").strip()
    if not v:
        return v
    if (not u) or u == "无":
        return v
    # 特例：配速类 unit 采用 "每公里/每米" 表达时，回填到原文更自然的写法是 "xx/公里" 而不是 "xx每公里"
    if u.startswith("每") and len(u) > 1 and ("/" not in v):
        denom_raw = u[1:].strip()
        if denom_raw in ("公里", "千米", "km", "KM", "Km"):
            denom = "公里"
        elif denom_raw == "米":
            denom = "米"
        else:
            denom = ""
        if denom:
            suf = f"/{denom}"
            return v if v.endswith(suf) else f"{v}{suf}"
    return v if v.endswith(u) else f"{v}{u}"


def _join_raw_segments(segs: Sequence[str]) -> str:
    """
    将若干“原始句子片段”拼成一行，使用中文逗号作为默认分隔。
    """
    xs = [str(x).strip() for x in (segs or []) if str(x).strip()]
    return "，".join(xs).strip()


def _is_time_token(s: str) -> bool:
    return bool(_TIME_HHMM_RE.fullmatch(str(s or "").strip()))


def _is_missing_token(s: str) -> bool:
    t = str(s or "").strip()
    return (not t) or t in ("无", "None", "null", "NULL", "N/A", "NA")


_DATE_CN_YMD_RE = re.compile(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日")
_DATE_CN_MD_RE = re.compile(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})日")
_DATE_SEP_YMD_4_RE = re.compile(r"(?P<y>\d{4})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})")
_DATE_SEP_YMD_2_RE = re.compile(r"(?P<y>\d{2})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})")
_DATE_CN_YMD_OPT_DAY_RE = re.compile(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})(?:日)?")
_DATE_SEP_MD_RE = re.compile(r"(?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})")
_DATE_CN_MD_OPT_DAY_RE = re.compile(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})(?:日)?")


@lru_cache(maxsize=8192)
def _normalize_date_cn_cached(raw: str, default_year: int | None) -> str:
    # 注意：raw 需已 strip；该函数只做纯计算，便于缓存
    if not raw:
        return raw

    # 已是中文标准格式
    m0 = _DATE_CN_YMD_RE.fullmatch(raw)
    if m0:
        y, mo, da = int(m0.group("y")), int(m0.group("m")), int(m0.group("d"))
        return f"{y}年{mo:02d}月{da:02d}日"
    m0b = _DATE_CN_MD_RE.fullmatch(raw)
    if m0b:
        mo, da = int(m0b.group("m")), int(m0b.group("d"))
        return f"{mo:02d}月{da:02d}日"

    # YYYY/M/D or YYYY-M-D or YYYY.M.D
    m1 = _DATE_SEP_YMD_4_RE.fullmatch(raw)
    if m1:
        y, mo, da = int(m1.group("y")), int(m1.group("m")), int(m1.group("d"))
        return f"{y}年{mo:02d}月{da:02d}日"

    # YY/M/D or YY-M-D or YY.M.D（两位年份）
    m1b = _DATE_SEP_YMD_2_RE.fullmatch(raw)
    if m1b:
        y2, mo, da = int(m1b.group("y")), int(m1b.group("m")), int(m1b.group("d"))
        if default_year is not None:
            # 继承 default_year 的世纪（例如 default_year=2025，则 25 -> 2025）
            century = (int(default_year) // 100) * 100
            y = century + y2
        else:
            # 默认按 2000+YY（面向 20xx 数据）
            y = 2000 + y2
        return f"{y}年{mo:02d}月{da:02d}日"

    # YYYY年M月D日(可无“日”)
    m2 = _DATE_CN_YMD_OPT_DAY_RE.fullmatch(raw)
    if m2:
        y, mo, da = int(m2.group("y")), int(m2.group("m")), int(m2.group("d"))
        return f"{y}年{mo:02d}月{da:02d}日"

    # M/D or M-D or M.D
    m3 = _DATE_SEP_MD_RE.fullmatch(raw)
    if m3:
        mo, da = int(m3.group("m")), int(m3.group("d"))
        if default_year is not None:
            return f"{default_year}年{mo:02d}月{da:02d}日"
        return f"{mo:02d}月{da:02d}日"

    # M月D日(可无“日”)
    m4 = _DATE_CN_MD_OPT_DAY_RE.fullmatch(raw)
    if m4:
        mo, da = int(m4.group("m")), int(m4.group("d"))
        if default_year is not None:
            return f"{default_year}年{mo:02d}月{da:02d}日"
        return f"{mo:02d}月{da:02d}日"

    return raw


def _normalize_date_cn(s: str, *, default_year: int | None = None) -> str:
    """
    将常见日期字符串标准化为：
    - MM月DD日
    - YYYY年MM月DD日

    支持输入：
    - YYYY/M/D, YYYY-MM-DD, YYYY.MM.DD
    - YY/M/D, YY-MM-DD, YY.MM.DD（两位年份；若提供 default_year，则继承其世纪）
    - YYYY年M月D日（末尾日可省略）
    - M/D, M-D, M.D
    - M月D日（末尾日可省略）

    识别失败则原样返回（避免误改）。
    """
    raw = str(s or "").strip()
    return _normalize_date_cn_cached(raw, default_year)


def _normalize_date_or_range_cn(s: str) -> str:
    """
    标准化“单日期或日期范围”表达式：
    - 单日期：走 `_normalize_date_cn`
    - 范围：尝试按 (到|至|~|～|-|—) 切为两端日期并标准化，输出 "{A}~{B}"
      若左端含年份而右端不含，则右端继承左端年份。
    """
    raw = str(s or "").strip()
    if not raw:
        return raw

    # 先尝试当作单日期
    single = _normalize_date_cn(raw)
    # 若输入已经是标准格式 / 或被成功改写为标准格式，则直接返回
    if single != raw or re.fullmatch(r"(?:\d{4}年)?\d{2}月\d{2}日", single):
        return single

    m = re.fullmatch(r"\s*(?P<a>.+?)\s*(?P<sep>到|至|~|～|-|—)\s*(?P<b>.+?)\s*$", raw)
    if not m:
        return raw

    a_raw = (m.group("a") or "").strip()
    b_raw = (m.group("b") or "").strip()
    if not a_raw or not b_raw:
        return raw

    a_norm = _normalize_date_cn(a_raw)
    # 若 a_norm 含年份，则 b 允许继承年份
    y_default: int | None = None
    m_y = re.fullmatch(r"(?P<y>\d{4})年\d{2}月\d{2}日", a_norm)
    if m_y:
        y_default = int(m_y.group("y"))
    b_norm = _normalize_date_cn(b_raw, default_year=y_default)

    # 若两端都无法标准化，则不改
    if a_norm == a_raw and b_norm == b_raw:
        return raw
    return f"{a_norm}~{b_norm}"


def _format_date_range(st: str, ed: str) -> str:
    s = _normalize_date_cn(str(st or "").strip())
    e = _normalize_date_cn(str(ed or "").strip(), default_year=None)
    if not s and not e:
        return ""
    if s and (not e or e == s):
        return s
    if not s and e:
        return e
    return f"{s}~{e}"

__all__ = [
    "_strip_leading_de",
    "_normalize_time_cn_token",
    "_normalize_value_and_unit",
    "_attach_unit_to_value",
    "_join_raw_segments",
    "_is_time_token",
    "_is_missing_token",
    "_normalize_date_cn",
    "_normalize_date_or_range_cn",
    "_format_date_range",
]
