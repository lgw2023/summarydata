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


# ========= 特化预处理：饮食(摄入热量/三大营养素)日志 -> 单日期数值多项总结 =========
_MEAL_NUTRITION_LINE_RE = re.compile(
    r"^\s*(?P<date8>\d{8})\s*(?P<meal>[^：:\n\r]{1,20})\s*[：:]\s*(?P<body>.+?)\s*$"
)


def _yyyymmdd_to_ymd_slash(date8: str) -> str:
    """
    将 YYYYMMDD 转为 YYYY/M/D（用于喂给 `_normalize_date_cn` 支持的分隔符日期格式）。
    解析失败则原样返回。
    """
    t = str(date8 or "").strip()
    m = re.fullmatch(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})", t)
    if not m:
        return t
    try:
        y = int(m.group("y"))
        mo = int(m.group("m"))
        da = int(m.group("d"))
    except Exception:
        return t
    if not (1 <= mo <= 12 and 1 <= da <= 31):
        return t
    return f"{y}/{mo}/{da}"


def normalize_meal_nutrition_logs_to_single_date_value_multi_summary(text: str) -> str:
    """
    将“YYYYMMDD + 餐次 + 摄入热量/食物/脂肪/蛋白质/碳水化合物 ...”的日志文本，
    重写为当前解析器可识别的“单日期数值多项总结”句式，以便被：
    - aggregate_dataframe
    - aggregate_dataline
    - aggregate_time
    - aggregate_format
    等模块解析与聚合。

    输入示例（原始）：
      20251224早餐：摄入热量是223.00千卡,食物为包子（三鲜馅）,脂肪8.60克,蛋白质7.40克,碳水化合物29.10克

    输出示例（重写后）：
      2025/12/24 早餐包子（三鲜馅）摄入热量223.00千卡, 早餐包子（三鲜馅）脂肪8.60克, 早餐包子（三鲜馅）蛋白质7.40克, 早餐包子（三鲜馅）碳水化合物29.10克

    说明：
    - 本函数只做“文本层重写”，不直接构造数据类对象；
    - 重写后的文本可直接传给 `explode_newlines_and_route_to_dataclasses` / `get_patterns_all`。
    """
    raw = "" if text is None else str(text)
    if not raw.strip():
        return ""

    # 拆行：兼容复制粘贴的多行文本
    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out_lines: list[str] = []

    # 常见分隔：英文逗号/中文逗号
    split_re = re.compile(r"[，,]\s*")

    calories_re = re.compile(
        r"(?:^|.*?)(?P<name>摄入热量)\s*(?:是|为|:|：)?\s*(?P<val>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>千卡|大卡|kcal|KCAL|cal|CAL)?\s*$"
    )
    food_re = re.compile(r"(?:^|.*?)(?:食物|食物名称)\s*(?:为|是|:|：)?\s*(?P<food>.*)\s*$")
    # 通用“指标+数值+单位”片段（偏保守，避免误伤）
    metric_value_re = re.compile(
        r"^\s*(?P<metric>脂肪|蛋白质|碳水化合物|碳水|糖|膳食纤维|钠|胆固醇)\s*(?:为|是|:|：)?\s*(?P<val>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>克|g|毫克|mg)\s*$"
    )

    def _norm_unit(u: str) -> str:
        uu = (u or "").strip()
        if not uu:
            return ""
        if uu in ("大卡", "kcal", "KCAL"):
            return "千卡"
        if uu in ("cal", "CAL"):
            # 极少见：cal 更像 “卡路里”；这里仍折算到“千卡”展示更贴近现有体系，但不做数值换算（避免误改）
            return "千卡"
        if uu in ("g",):
            return "克"
        if uu in ("mg",):
            return "毫克"
        return uu

    def _sanitize_food_name(food0: str) -> str:
        """
        食物名清洗（非常保守）：
        - 避免食物名里带“330ml/500毫升/1L”等数字，导致后续“指标名+数值”解析把它误当成数值。
        - 仅剥离**末尾**的容量/重量/份数标注（如 65g / 55g*12枚 / 100ml*5 / 330ml 等）。
        - 其余位置的阿拉伯数字（如 “火锅(荤1素4)”）会被替换为中文数字（避免被当作“第一个数值”误切分）。
        """
        t = (food0 or "").strip()
        if not t:
            return t
        # 1) 剥离末尾容量/重量与“*份数”：
        # - 65g
        # - 55g*12枚
        # - 100ml*5
        # - 330ml / 1L / 1升 / 500毫升
        #
        # 注意：只剥离末尾，避免误伤 “B12/7喜” 这类在中间的品牌/型号信息。
        t2 = re.sub(
            r"\s*"
            r"\d+(?:\.\d+)?\s*"
            r"(?:ml|mL|ML|毫升|l|L|升|g|G|克|kg|KG|千克)"
            r"\s*"
            r"(?:\*\s*\d+\s*(?:枚|袋|包|盒|瓶|罐|片|支|个)?)?"
            r"\s*$",
            "",
            t,
        ).strip()
        if t2:
            t = t2

        # 2) 将剩余的阿拉伯数字替换为中文数字（逐位替换，简单稳定）
        digit_map = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }
        t3 = "".join(digit_map.get(ch, ch) for ch in t)
        return t3.strip() or t

    for ln in lines:
        if not ln:
            continue
        m = _MEAL_NUTRITION_LINE_RE.match(ln)
        if not m:
            # 无法识别：原样保留，避免丢信息（后续仍可由 router 其它分支/兜底处理）
            out_lines.append(ln)
            continue

        date8 = (m.group("date8") or "").strip()
        meal = (m.group("meal") or "").strip()
        body = (m.group("body") or "").strip()
        if not (date8 and meal and body):
            out_lines.append(ln)
            continue

        ymd = _yyyymmdd_to_ymd_slash(date8)

        # 解析 body 片段
        segs = [x.strip() for x in split_re.split(body) if x and x.strip()]
        food = ""
        calories_val = ""
        calories_unit = ""
        metrics: list[tuple[str, str, str]] = []  # (metric, val, unit)

        for seg in segs:
            if not seg:
                continue
            # 食物
            m_food = food_re.fullmatch(seg)
            if m_food and (food == ""):
                food = _sanitize_food_name((m_food.group("food") or "").strip())
                continue

            # 摄入热量
            m_cal = calories_re.fullmatch(seg)
            if m_cal and (calories_val == ""):
                calories_val = (m_cal.group("val") or "").strip()
                calories_unit = _norm_unit(m_cal.group("unit") or "千卡") or "千卡"
                continue

            # 其它营养素
            m_mv = metric_value_re.fullmatch(seg)
            if m_mv:
                metric = (m_mv.group("metric") or "").strip()
                val = (m_mv.group("val") or "").strip()
                unit = _norm_unit(m_mv.group("unit") or "")
                if metric and val:
                    metrics.append((metric, val, unit or "无"))
                continue

        # 构造“指标前缀”：餐次 + 食物名（若食物为空，则只用餐次）
        prefix = meal
        if food:
            prefix = f"{meal}{food}"

        rewritten: list[str] = []
        if calories_val:
            rewritten.append(f"{prefix}摄入热量{calories_val}{calories_unit}")
        for metric, val, unit in metrics:
            # 统一 grams 等单位在 value 中保留，便于下游推断单位
            if unit == "无":
                rewritten.append(f"{prefix}{metric}{val}")
            else:
                rewritten.append(f"{prefix}{metric}{val}{unit}")

        # 若没解析出任何指标，保留原行
        if not rewritten:
            out_lines.append(ln)
            continue

        # Router 提示：
        # - 解析路由在 “单日期数值多项总结” 的强特征判断里，会要求出现“平均/最高/最低...”等词（更像统计汇总）。
        # - 这类饮食日志本质上属于“同一日期的多指标汇总”，但天然不含这些统计词，容易被归到“单日期数值单项总结”。
        # - 这里追加一个**不会被解析为指标**的占位片段（无数字，因此不会生成记录），仅用于触发 router 更合适的分支。
        #   注意：占位片段不会出现在表格 rows 中，只会保留在原始文本 recover 中。
        rewritten.append("（平均）")

        # 目标句式：{date}{space}{seg1}, {seg2}, ...
        out_lines.append(f"{ymd} " + "，".join(rewritten))

    return "\n".join(out_lines).strip()


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
    "normalize_meal_nutrition_logs_to_single_date_value_multi_summary",
]
