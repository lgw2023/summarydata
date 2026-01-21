from __future__ import annotations

import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Iterable, Sequence

from .models import (
    NoDateValueSummaryRecord,
    NoTimestampTextSummaryRecord,
    PeriodTextSummaryRecord,
    PeriodValueCompareRecord,
    PeriodValueSingleSummaryRecord,
    PeriodValuemMultiSummaryRecord,
    PersonalDataPattern,
    SingleDateTextSummaryRecord,
    SingleDateValueMultiSummaryRecord,
    SingleDateValueSingleSummaryRecord,
    SingleMetricDetailRecord,
    SingleMetricStatsRecord,
    UnparsedRawPersonalData,
)
from .normalize import _attach_unit_to_value, _is_missing_token, _normalize_date_cn, _normalize_time_cn_token


@dataclass(frozen=True)
class _Event:
    start_date: str
    start_time: str
    end_date: str
    end_time: str
    sport_type: str
    metric: str
    value: str


_CN_YMD_RE = re.compile(r"(?P<y>\d{4})年(?P<m>\d{2})月(?P<d>\d{2})日")
_CN_MD_RE = re.compile(r"(?P<m>\d{2})月(?P<d>\d{2})日")
_SLASH_YMD_RE = re.compile(r"(?P<y>\d{2,4})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})")
_SLASH_MD_RE = re.compile(r"(?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})")


def _parse_date_parts(s: str) -> tuple[int | None, int | None, int | None]:
    """
    解析日期字符串，兼容中文 "YYYY年MM月DD日"/"MM月DD日" 和 slash "YYYY/M/D"/"M/D"。
    返回 (year, month, day)，解析失败返回 (None, None, None)。
    """
    raw = str(s or "").strip()
    if not raw:
        return (None, None, None)

    # 兼容部分上游混用口径：slash/dash/dot 日期后可能带一个“日”
    # 例如："2025/2/1日"、"2/1日"
    if raw.endswith("日") and ("年" not in raw) and any(sep in raw for sep in ("/", ".", "-")):
        raw = raw[:-1].strip()

    t = _normalize_date_cn(raw)
    m = _CN_YMD_RE.fullmatch(t)
    if m:
        return (int(m.group("y")), int(m.group("m")), int(m.group("d")))

    m2 = _CN_MD_RE.fullmatch(t)
    if m2:
        return (None, int(m2.group("m")), int(m2.group("d")))

    # 容忍 slash（或其它上游遗留），与旧逻辑保持一致：对 slash 用 raw 解析
    m3 = _SLASH_YMD_RE.fullmatch(raw)
    if m3:
        y_raw = m3.group("y")
        y = int(y_raw)
        if len(y_raw) == 2:
            y = 2000 + y
        return (y, int(m3.group("m")), int(m3.group("d")))

    m4 = _SLASH_MD_RE.fullmatch(raw)
    if m4:
        return (None, int(m4.group("m")), int(m4.group("d")))

    return (None, None, None)


def _date_to_slash(d: str) -> str:
    """
    将 normalize.py 输出的中文日期尽量转成 "YYYY/M/D"（或 "M/D"）。
    识别失败则原样返回。
    """
    raw = str(d or "").strip()
    if not raw:
        return ""
    y, mo, da = _parse_date_parts(raw)
    if y is not None and mo is not None and da is not None:
        return f"{y}/{mo}/{da}"
    if y is None and mo is not None and da is not None:
        return f"{mo}/{da}"
    return raw


def _dt_label(d: str, t: str) -> str:
    ds = _date_to_slash(d)
    ts = _normalize_time_cn_token(t) if t and t != "无" else ""
    return f"{ds} {ts}".strip() if ts else ds


# 指标后缀优先级（用于拆 sport_type 与排序）
_METRIC_SUFFIXES: list[str] = [
    # 运动类（更接近你示例里的输出顺序）
    "运动热量",
    "活动热量",
    "热量",
    "有氧训练压力",
    "无氧训练压力",
    "训练压力",
    "运动时长",
    "锻炼时长",
    "运动时间",
    "时长",
    "用时",
    "平均运动心率",
    "平均心率",
    "最大运动心率",
    "最大心率",
    "最小运动心率",
    "最小心率",
    "心率",
    "心率范围",
    # 运动表现
    "距离",
    "平均配速",
    "配速",
    "最快配速",
    "平均速度",
    "速度",
    "步频",
    "最大步频",
    "桨频",
    "步幅",
    "步数",
    # 跑步高级指标（部分设备/平台会给出）
    "最大摄氧量",
    "平均外翻幅度",
    "平均着地冲击",
    "平均摆动角度",
    "平均腾空时间",
    "平均触地时间",
    "平均触底腾空比",
    "泳姿",
    "平均划水频率",
    "个数",
    # 次数类（用于把“跑步运动次数/户外跑步次数”拆出运动类型）
    "运动次数",
    "次数",
]

_METRIC_RENAME: dict[str, str] = {
    # 让输出更贴近你给的样例（仅做轻量映射；无法保证所有源数据都应这样改）
    "热量": "运动热量",
    "用时": "运动时长",
    "时长": "运动时长",
    "运动时间": "运动时长",
    "平均心率": "平均运动心率",
    "最大心率": "最大运动心率",
    "最小心率": "最小运动心率",
    # 语义上更准确的是“平均配速”，但训练数据里通常只用“配速”这一指标名
    "平均配速": "配速",
    # “运动次数”更贴近自然语言：统一成“次数”
    "运动次数": "次数",
}

# ---------------------------------------------------------------------
# 从 prompts/domain.jsonl 读取 domain/subdomain 枚举（用于保证输出与 prompt 枚举一致）
# 若读取失败则回退到代码内的最小枚举。
# ---------------------------------------------------------------------
def _load_domain_taxonomy_from_prompts() -> tuple[set[str], set[str], set[str]]:
    """
    returns: (health_domains, sport_domains, sport_subdomains)
    """
    # repo_root = summarydata/
    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "prompts" / "domain.jsonl"
    if not p.exists():
        return set(), set(), set()

    txt = p.read_text(encoding="utf-8", errors="ignore")
    # 去掉注释行
    lines = [ln for ln in txt.splitlines() if not ln.strip().startswith("#")]
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return set(), set(), set()

    # 简单抽取文件中的 JSON 对象（该文件实际包含 2 个 JSON 对象）
    objs: list[str] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(cleaned[start : i + 1])
                    start = None
    if not objs:
        return set(), set(), set()

    import json

    health: set[str] = set()
    sport: set[str] = set()
    sub: set[str] = set()

    # 第一个对象：{"健康":[...],"运动":[...],"其他":[...]}
    try:
        obj0 = json.loads(objs[0])
        for k in (obj0.get("健康") or []):
            if isinstance(k, str) and k.strip():
                health.add(k.strip())
        for k in (obj0.get("运动") or []):
            if isinstance(k, str) and k.strip():
                sport.add(k.strip())
    except Exception:
        pass

    # 第二个对象：{"跑步":[...], "步行徒步":[...], ...}
    if len(objs) >= 2:
        try:
            obj1 = json.loads(objs[1])
            if isinstance(obj1, dict):
                for v in obj1.values():
                    if isinstance(v, list):
                        for x in v:
                            if isinstance(x, str) and x.strip():
                                sub.add(x.strip())
        except Exception:
            pass

    return health, sport, sub


_HEALTH_DOMAINS_SET, _SPORT_DOMAINS_SET, _SPORT_SUBDOMAINS_SET = _load_domain_taxonomy_from_prompts()

# 回退枚举（与 prompts/domain.jsonl 一致的最小集合）
_HEALTH_DOMAINS_FALLBACK: tuple[str, ...] = (
    "体温",
    "减脂",
    "心脏健康",
    "情绪健康",
    "生理健康",
    "血压",
    "血氧饱和度",
    "血糖",
    "睡眠",
    "午睡",
    "步数",
    "活力三环",
    "微体检",
    "饮食",
)
_SPORT_DOMAINS_FALLBACK: tuple[str, ...] = (
    "跑步",
    "骑行",
    "步行徒步",
    "游泳",
    "登山",
    "跳绳",
    "瑜伽",
    "普拉提",
    "划船机",
    "椭圆机",
    "潜水",
    "自由训练",
    "力量训练",
    "体能训练",
    "所有运动",
    "活动",
    "锻炼",
)


def _health_domains() -> set[str]:
    return _HEALTH_DOMAINS_SET or set(_HEALTH_DOMAINS_FALLBACK)


def _sport_domains() -> set[str]:
    return _SPORT_DOMAINS_SET or set(_SPORT_DOMAINS_FALLBACK)


def _sport_subdomains() -> set[str]:
    return _SPORT_SUBDOMAINS_SET


# ---------------------------------------------------------------------
# 健康指标域映射（用于把“非运动项目”的测量项从 sport_type=无 提升到健康 domain）
# 说明：
# - 目前 dataline 的输出字段名仍叫“运动类型”，但其语义更接近“domain/类别”。
# - 这里将常见体测/生理指标映射到 prompts/domain.jsonl 中的健康域枚举（如 微体检/血糖）。
# - 仅做轻量启发式；若后续引入更完整的 domain 分类器，可替换此处。
# ---------------------------------------------------------------------
_HEALTH_METRIC_TO_DOMAIN_AND_METRIC: dict[str, tuple[str, str]] = {
    # 体测/微体检（体成分秤等）
    "BMI": ("微体检", "BMI"),
    "Bmi": ("微体检", "BMI"),
    "体重": ("微体检", "体重"),
    "体重较上一次变化": ("微体检", "体重较上一次变化"),
    "脂肪率": ("微体检", "脂肪率"),
    "体脂率": ("微体检", "脂肪率"),
    "蛋白质": ("微体检", "蛋白质"),
    "骨骼肌量": ("微体检", "骨骼肌量"),
    "骨盐量": ("微体检", "骨盐量"),
    "基础代谢率": ("微体检", "基础代谢率"),
    "内脏脂肪等级": ("微体检", "内脏脂肪等级"),
    "去脂体重": ("微体检", "去脂体重"),
    "身体得分": ("微体检", "身体得分"),
    "身体年龄": ("微体检", "身体年龄"),
    "水分率": ("微体检", "水分率"),
    # 生理指标
    "血糖": ("血糖", "血糖"),
    # 心脏健康
    "静息心率": ("心脏健康", "静息心率"),
    "平均静息心率": ("心脏健康", "平均静息心率"),
    # 血氧饱和度
    "血氧": ("血氧饱和度", "血氧"),
    "平均血氧": ("血氧饱和度", "平均血氧"),
    "平均最大血氧": ("血氧饱和度", "平均最大血氧"),
    "平均最低血氧": ("血氧饱和度", "平均最低血氧"),
    # 体温（包含皮肤温度口径）
    "体温": ("体温", "体温"),
    "平均体温": ("体温", "平均体温"),
    "平均皮肤温度": ("体温", "平均皮肤温度"),
    # 微体检：兼容“最新一次BMI”这类口径
    "最新一次BMI": ("微体检", "最新一次BMI"),
}

# 一些历史/上游口径里的“类别前缀” -> prompts/domain.jsonl 的健康域
_LEGACY_CATEGORY_TO_DOMAIN: dict[str, str] = {
    # 活力/闭环
    # 语境统一：活动/锻炼/运动 都归到 “所有运动”
    "活动": "所有运动",
    "锻炼": "所有运动",
    "运动": "所有运动",
    "全闭环": "活力三环",
    "三环闭环": "活力三环",
    # 睡眠
    "清醒": "睡眠",
    "睡眠": "睡眠",
    "零星小睡": "午睡",
    # 情绪/压力
    "压力": "情绪健康",
    "平静": "情绪健康",
    "愉悦": "情绪健康",
    "不愉悦": "情绪健康",
    # 生理
    "心率": "心脏健康",
    "血氧": "血氧饱和度",
    # 运动/锻炼泛化（已在上方统一到“所有运动”）
}


def _maybe_map_health_metric(metric: str) -> tuple[str, str] | None:
    """
    若 metric 是明确的健康指标，则返回 (domain, canonical_metric)；否则返回 None。
    """
    m = str(metric or "").strip()
    if not m:
        return None
    # 容忍不同来源的空格/大小写/全角空格
    key = re.sub(r"\s+", "", m).replace("　", "")
    if not key:
        return None
    # 统一 BMI 的大小写（有些源会给 bmi/Bmi/BMI）
    if key.lower() == "bmi":
        key = "BMI"
    # 容忍“最新一次BMI/最近一次BMI”等前缀口径
    if key.endswith("BMI") and key in ("最新一次BMI", "最近一次BMI"):
        key = "最新一次BMI"
    hit = _HEALTH_METRIC_TO_DOMAIN_AND_METRIC.get(key)
    if hit:
        return hit

    # 兼容“缺失提示/数据占位”类字段：
    # - “血糖数据/血压数据/体温数据/心脏健康数据/血氧饱和度数据/生理健康数据 ...”
    # - “没有查询到血糖数据/未查询到血压数据 ...”
    #
    # 这类字段如果不提升到健康 domain，会在同日聚合时落到 “类型：无” 并与其它“无类型”混合。
    miss = re.search(r"(?:未|没有)查询到(?P<base>.+?)数据$", key)
    if miss:
        base_raw = miss.group("base").strip()
        base = _LEGACY_CATEGORY_TO_DOMAIN.get(base_raw, base_raw)
        if base in _health_domains():
            return (base, f"{base}数据")

    if key.endswith("数据") and len(key) > 2:
        base_raw = key[:-2].strip()
        base = _LEGACY_CATEGORY_TO_DOMAIN.get(base_raw, base_raw)
        if base in _health_domains():
            # 若发生 legacy -> domain 映射（如 血氧 -> 血氧饱和度），则同步调整指标名以保持一致
            metric2 = key if base == base_raw else f"{base}数据"
            return (base, metric2)
    return None


# 运动类型别名（用于将原始设备/口径统一到训练数据口径）
_SPORT_TYPE_ALIASES: dict[str, str] = {
    # 这里保持“划船机”原名更符合直觉（避免输出“划船桨”这种反直觉写法）
    # 汇总指标里常见“步行总距离/步行总热量/步行运动次数”，训练枚举中对应“步行徒步”
    "步行": "步行徒步",
}


def _normalize_sport_type(sport: str) -> str:
    s = str(sport or "").strip()
    if not s or s == "无":
        return "无"
    return _SPORT_TYPE_ALIASES.get(s, s)


# 周期/汇总类指标里常见的聚合词：可能出现在运动名前、运动名后、或运动名与指标后缀之间
# 兼容“最高/最低”这类口径（与“最大/最小”同义），避免“最高压力/最低压力”落到 sport_type=无
_AGG_WORDS: tuple[str, ...] = ("平均", "总", "最大", "最小", "最高", "最低", "最快", "最慢", "最长")

# 仅在“看起来确实是运动/子项”的情况下，才允许从指标名中拆出“运动类型”（避免误判成运动）
_SPORT_HINT_WORDS: tuple[str, ...] = (
    "跑步",
    "户外跑步",
    "室内跑步",
    "越野跑",
    "户外步行",
    "室内步行",
    "骑行",
    "户外骑行",
    "室内骑行",
    "游泳",
    "泳池游泳",
    "开放水域游泳",
    "跳绳",
    "划船",
    "划船机",
    "划船桨",
    "锻炼",
    "训练",
    "力量训练",
    "瑜伽",
    "健身",
    "徒步",
    "登山",
    "潜水",
    "椭圆机",
)


# 指标名里“前缀即运动类型”的常见场景（尤其是设备明细类 SingleMetricDetailRecord）：
# - 户外跑步心率范围 / 户外跑步最大摄氧量 / 户外跑步平均外翻幅度 / 户外跑步前脚掌触地次数 ...
# 这类指标若仅靠“后缀表”拆分，会把“前脚掌触地”误吸进 sport_type，或因缺少后缀导致 sport_type=无。
# 这里维护一份“可作为前缀的运动类型候选”，优先做 prefix split，以增强对未知指标组合的兼容性。
_NON_PREFIX_TYPES: set[str] = {
    # 过于泛化的类别词：不要作为“运动类型前缀”去拆指标名（否则会把“活动热量/锻炼时长”等拆坏）
    "活动",
    "锻炼",
    "运动",
    "训练",
    "健身",
    "所有运动",
}


def _sport_prefix_candidates() -> list[str]:
    """
    返回可用于“prefix split”的运动类型候选（按长度降序）。
    - 来源：domain 枚举 + hint words
    - 过滤：去掉过于泛化的类别词
    """
    cands: set[str] = set()
    for x in list(_sport_domains()) + list(_sport_subdomains()) + list(_SPORT_HINT_WORDS):
        t = _normalize_sport_type(x)
        if t and t != "无" and t not in _NON_PREFIX_TYPES:
            cands.add(t)
    # 更长的优先（避免“跑步”抢在“户外跑步/室内跑步”之前）
    return sorted(cands, key=lambda s: (-len(s), s))


_SPORT_PREFIX_CANDIDATES: list[str] = _sport_prefix_candidates()


def _split_by_sport_prefix(metric: str) -> tuple[str, str] | None:
    """
    若 metric 以某个运动类型候选作为前缀，则返回 (sport_type, metric_tail)；否则返回 None。
    """
    m = str(metric or "").strip()
    if not m:
        return None
    for sp in _SPORT_PREFIX_CANDIDATES:
        if m.startswith(sp) and len(m) > len(sp):
            rest = m[len(sp) :].strip()
            if rest:
                return (_normalize_sport_type(sp), rest)
    return None


def _looks_like_type_label(s: str) -> bool:
    t = str(s or "").strip()
    if not t or t == "无":
        return False
    # 健康域 / 运动域 / 运动子项（SubDomain）
    if t in _health_domains() or t in _sport_domains() or t in _sport_subdomains():
        return True
    return any(w in t for w in _SPORT_HINT_WORDS)


def _split_sport_and_metric(metric: str) -> tuple[str, str]:
    """
    尝试把 "户外跑步距离" -> ("户外跑步", "距离")。
    若无法拆分，则返回 ("无", metric)。
    """
    m = str(metric or "").strip()
    if not m:
        return ("无", "")

    # 某些上游/解析结果会把“为”误留在指标名末尾（例如 “平均室内跑步步幅为”），
    # 会导致后缀匹配失败（步幅/心率/配速等），从而无法推断 sport。
    # 这里做一次轻量清洗：只移除末尾的 “为/为：/为:”。
    m = re.sub(r"(?:为[:：]?)+\s*$", "", m).strip()
    if not m:
        return ("无", "")

    # 0) 优先处理“运动类型前缀”：
    #    例如：户外跑步心率范围 -> (户外跑步, 心率范围)
    #         户外跑步前脚掌触地次数 -> (户外跑步, 前脚掌触地次数)
    #    这样可以避免 suffix="次数" 时把“前脚掌触地”误吸进 sport_type。
    prefix_hit = _split_by_sport_prefix(m)
    if prefix_hit:
        sp, tail = prefix_hit
        # prefix split 后，对尾部指标名做一次“安全的”轻量 rename：
        # - 仅做“全匹配”映射（避免把“活动热量”误变成“活动运动热量”）
        # - 额外特例：总消耗热量 -> 总消耗运动热量
        tail2 = _METRIC_RENAME.get(tail, tail)
        if tail2 == tail:
            if tail.startswith("总消耗") and tail.endswith("热量") and (not tail.endswith("活动热量")):
                tail2 = f"{tail[: -len('热量')]}运动热量"
        return (sp, tail2)

    # 0) 指标名末尾聚合词后缀（“距离最长/速度最快/配速最慢/心率最高/心率最低 ...”）
    # 上游常见口径：把“聚合词”放在指标名末尾（例如 “骑行距离最长”），
    # 这不符合我们当前的 “<运动><聚合词><指标后缀>” 规则，导致 sport_type 拆不出来，从而落到“类型：无”。
    #
    # 处理方式：
    # - 先去掉末尾聚合词，对 base 再跑一次拆分；
    # - 若 base 能拆出 sport_type，则把聚合词拼回 metric（得到“距离最长”），并返回。
    for agg_tail in ("最长", "最短", "最快", "最慢", "最高", "最低"):
        if m.endswith(agg_tail) and len(m) > len(agg_tail):
            base = m[: -len(agg_tail)].strip()
            if base:
                sp_base, met_base = _split_sport_and_metric(base)
                if sp_base != "无" and met_base:
                    met2 = met_base if met_base.endswith(agg_tail) else f"{met_base}{agg_tail}"
                    return (sp_base, met2)

    # 0) 明确的健康指标：直接映射到健康 domain（避免输出“运动类型：无”）
    health_hit = _maybe_map_health_metric(m)
    if health_hit:
        domain, metric2 = health_hit
        return (domain, metric2)

    # 0.05) 缺失提示类字段：例如“心脏健康数据/没有查询到心脏健康数据”
    # 这类字段不应落到 sport_type=无，否则会在同日聚合时与其它“无类型”指标混在一行。
    if "心脏健康" in m:
        return ("心脏健康", m)

    # 0.1) 仅有“类别词”本身时（如：压力/愉悦/平静/不愉悦/血氧/睡眠等），也应能提升到健康域
    if m in ("压力", "平静", "愉悦", "不愉悦"):
        return ("情绪健康", m)
    # “最高压力/最低压力/平均压力”等统计口径：若不是“训练压力”语境，提升到情绪健康
    if m.endswith("压力") and ("训练" not in m):
        return ("情绪健康", m)
    # 睡眠相关（含“平均/最早/最晚/最低/最高”等统计口径）
    if any(
        k in m
        for k in (
            "睡眠",
            "入睡时间",
            "起床时间",
            "夜间睡眠时长",
            "睡眠时长",
        )
    ):
        return ("睡眠", m)
    # 睡眠结构/阶段相关（含“平均”前缀等）
    if any(x in m for x in ("浅睡", "深睡", "快速眼动", "REM")):
        return ("睡眠", m)
    # 步数汇总（避免把“日/平均日”误拆成运动类型）
    if m in ("日步数", "平均日步数"):
        return ("步数", m)
    # 体温/皮肤温度
    if ("体温" in m) or ("皮肤温度" in m):
        return ("体温", m)
    # 减脂相关（例如：减脂数据/燃脂时长等上游口径）
    if "减脂" in m or "燃脂" in m:
        return ("减脂", m)
    # 情绪健康相关（例如：情绪健康数据/情绪分布等；压力/愉悦等已在上方覆盖）
    if "情绪" in m:
        return ("情绪健康", m)
    # 静息心率
    if "静息心率" in m:
        return ("心脏健康", m)
    # 血氧（统一提升到 prompts/domain.jsonl 的“血氧饱和度”）
    if "血氧" in m:
        if m == "平均平均血氧":
            return ("血氧饱和度", "平均血氧")
        return ("血氧饱和度", m)

    # ---------------------------------------------------------------------
    # PeriodValueSingleSummaryRecord 的指标名里，经常直接包含“类别前缀 + 子项”
    # 例如：活动总热量 / 清醒总次数 / 心率过高提醒 / 疑似房颤 / 平均步数 ...
    # 这类字段并不是严格的“运动项目”，但训练数据口径希望能拆出“运动类型/子项”。
    # ---------------------------------------------------------------------
    # 1) “疑似房颤/疑似早搏” -> (房颤/早搏, 疑似)
    if m.startswith("疑似") and len(m) > 2:
        tail = m[2:].strip()
        if tail in ("房颤", "早搏"):
            # 归到健康域：心脏健康（prompts/domain.jsonl）
            return ("心脏健康", f"疑似{tail}")

    # 2) “心率过高提醒/心率过低提醒” -> (心率, 过高提醒/过低提醒)
    if m.startswith("心率") and len(m) > 2:
        rest = m[2:].strip()
        if rest in ("过高提醒", "过低提醒"):
            return ("心脏健康", m)

    # 3) “房颤/早搏” 单独出现时，按“次数”口径归一
    if m in ("房颤", "早搏"):
        return ("心脏健康", f"{m}次数")

    # 4) “平均平均血氧” 纠正为 “平均血氧”（已统一为血氧饱和度域；保留此处兼容旧行为）
    if m == "平均平均血氧":
        return ("血氧饱和度", "平均血氧")
    # 4.1) “平均最大血氧/平均最低血氧” -> (血氧饱和度, 原指标名)
    if m in ("平均最大血氧", "平均最低血氧"):
        return ("血氧饱和度", m)

    # 5) 常见可拆分前缀（不含聚合词时）
    _PREFIX_CATS: tuple[str, ...] = (
        "活动",
        "锻炼",
        "运动",
        "步数",
        "清醒",
        "零星小睡",
        "血氧",
        "压力",
        "平静",
        "愉悦",
        "不愉悦",
        "全闭环",
        "三环闭环",
        "椭圆机",
    )
    for cat in _PREFIX_CATS:
        if m.startswith(cat) and len(m) > len(cat):
            # 将健康类“前缀类别”映射到 prompts/domain.jsonl 的健康域，避免把健康项当成“运动类型”
            dom = _LEGACY_CATEGORY_TO_DOMAIN.get(cat, cat)
            if dom in _health_domains():
                return (dom, m)  # 保留完整指标名，避免丢信息
            # 语境规则：活动/锻炼/运动 都归到“所有运动”，但子项指标名需保留原字段前缀信息
            if dom == "所有运动" and cat in ("活动", "锻炼", "运动"):
                return (dom, m)
            if dom in _sport_domains() or dom in _sport_subdomains():
                return (dom, m[len(cat) :].strip())
            return (dom, m)

    # 6) 带聚合词前缀时（平均/总/最大/最小/最快/最慢/总计...）：如果聚合词后紧跟某类别，则提取类别
    #    - 平均步数 -> (步数, 平均步数)
    #    - 总步数 -> (步数, 总步数)
    #    - 平均活动热量 -> (活动, 平均活动热量)
    #    - 总计清醒次数 -> (清醒, 总计清醒次数)
    _AGG_PREFIXES: tuple[str, ...] = _AGG_WORDS + ("总计",)
    for agg in _AGG_PREFIXES:
        if m.startswith(agg) and len(m) > len(agg):
            rest = m[len(agg) :].strip()
            for cat in _PREFIX_CATS:
                if rest == cat:
                    dom = _LEGACY_CATEGORY_TO_DOMAIN.get(cat, cat)
                    return (dom, m)  # 子项保持原样，避免丢信息
                if rest.startswith(cat) and len(rest) > len(cat):
                    dom = _LEGACY_CATEGORY_TO_DOMAIN.get(cat, cat)
                    return (dom, m)  # 子项保持原样：更贴近训练数据口径（如“平均活动热量”）

    # 业务特例：某些指标本身包含运动语义，但不含“运动类型前缀”
    # 例如：平均潜水深度 -> (潜水, 平均潜水深度)
    if m == "平均潜水深度":
        return ("潜水", m)

    # 先匹配更长后缀，避免 "最大心率" 被 "心率" 吃掉
    for suf in sorted(_METRIC_SUFFIXES, key=lambda x: (-len(x), x)):
        if m == suf:
            return ("无", _METRIC_RENAME.get(suf, suf))
        if m.endswith(suf) and len(m) > len(suf):
            sport_raw = m[: -len(suf)].strip()

            # 2.x) “跑步总消耗热量/跑步总消耗运动热量” 这类口径：
            # - 旧逻辑会得到 sport="跑步总消耗"，metric="运动热量"
            # - 期望：sport="跑步"，metric="总消耗运动热量"
            if sport_raw.endswith("总消耗") and len(sport_raw) > len("总消耗"):
                sport = sport_raw[: -len("总消耗")].strip()
                if sport:
                    metric2_raw = f"总消耗{_METRIC_RENAME.get(suf, suf)}"
                    metric2 = _METRIC_RENAME.get(metric2_raw, metric2_raw)
                    return (_normalize_sport_type(sport), metric2)

            # 形如 “平均步数/总步数/最大心率” 中，sport_raw 可能恰好是聚合词本身；
            # 对“步数/血氧”等非传统运动项目的汇总，我们更希望把“类别”当作 sport_type。
            if sport_raw in _AGG_WORDS and suf in ("步数", "血氧"):
                dom = _LEGACY_CATEGORY_TO_DOMAIN.get(suf, suf)
                return (_normalize_sport_type(dom), m)

            # 1) 形如 "平均户外跑步速度"：聚合词在最前面
            for agg in _AGG_WORDS:
                if sport_raw.startswith(agg) and len(sport_raw) > len(agg):
                    sport = sport_raw[len(agg) :].strip()
                    metric2_raw = f"{agg}{suf}"
                    metric2 = _METRIC_RENAME.get(metric2_raw, metric2_raw)
                    return (_normalize_sport_type(sport or "无"), metric2)

            # 2) 形如 "跑步总距离" / "泳池游泳平均速度"：聚合词在运动名与后缀之间
            for agg in _AGG_WORDS:
                if sport_raw.endswith(agg) and len(sport_raw) > len(agg):
                    sport = sport_raw[: -len(agg)].strip()
                    metric2_raw = f"{agg}{suf}"
                    metric2 = _METRIC_RENAME.get(metric2_raw, metric2_raw)
                    return (_normalize_sport_type(sport or "无"), metric2)

            # 3) 普通：直接拆分
            sport = sport_raw
            metric2 = _METRIC_RENAME.get(suf, suf)
            return (_normalize_sport_type(sport or "无"), metric2)
    return ("无", _METRIC_RENAME.get(m, m))


def _metric_sort_key(metric: str) -> tuple[int, int, str]:
    m = str(metric or "").strip()
    if not m:
        return (9999, 9, "")
    # 用 _METRIC_SUFFIXES 的顺序作为偏好
    for i, suf in enumerate(_METRIC_SUFFIXES):
        if m == _METRIC_RENAME.get(suf, suf):
            return (i, 0, m)
    return (9999, 9, m)


def _parse_date_for_sort(s: str, *, default_year: int | None = None) -> tuple[int, int, int, str]:
    raw = str(s or "").strip()
    if not raw:
        return (9999, 99, 99, raw)
    y, mo, da = _parse_date_parts(raw)
    # 对缺少年份的日期（如 "4/22"、"4月22日"），可用样本内推断出的年份补齐，
    # 以避免“有年份的明细”被整体排到“无年份的汇总”前面，导致时间顺序不直观。
    if y is None and mo is not None and da is not None and default_year is not None:
        y = int(default_year)
    if y is not None and mo is not None and da is not None:
        return (y, mo, da, raw)
    if y is None and mo is not None and da is not None:
        return (9999, mo, da, raw)
    return (9999, 99, 99, raw)


def _parse_time_for_sort(s: str) -> tuple[int, int, str]:
    raw = str(s or "").strip()
    if not raw or raw == "无":
        return (99, 99, raw)
    m = re.fullmatch(r"(?P<h>\d{2}):(?P<m>\d{2})", _normalize_time_cn_token(raw))
    if m:
        return (int(m.group("h")), int(m.group("m")), raw)
    return (99, 99, raw)


def _event_sort_key(ev: _Event) -> tuple[int, int, int, int, int, str, str]:
    y, mo, da, _ = _parse_date_for_sort(ev.start_date)
    hh, mm, _ = _parse_time_for_sort(ev.start_time)
    return (y, mo, da, hh, mm, str(ev.sport_type or ""), str(ev.metric or ""))


def _safe_str(x: Any) -> str:
    return str(x if x is not None else "").strip()


def _value_with_unit_and_status(value: str, unit: str = "无", status: str = "") -> str:
    v = _safe_str(value)
    u = _safe_str(unit) or "无"
    st = _safe_str(status)

    v2 = _attach_unit_to_value(v, u) if (v and u and u != "无") else v
    if st and st not in ("无", "-"):
        v2 = f"{v2}{st}" if v2 else st
    return v2


def _format_time_range(sd: str, st: str, ed: str, et: str) -> str:
    """
    与旧逻辑保持一致的时间段格式化：
    - 若只存在一侧，则补齐为同一时间点
    - 若都不存在则用 "无时间~无时间"
    """
    left = _dt_label(sd, st)
    right = _dt_label(ed, et)

    if not left and right:
        left = right
    if left and not right:
        right = left
    if not left and not right:
        left = right = ""

    if left and right:
        # 若两侧时间戳相同，则无需输出为 “时间~时间”，直接输出单个时间点即可
        if left == right:
            return left
        return f"{left}~{right}"
    if left:
        return left
    return "无时间"


def _iter_aligned_rows(*cols: Sequence[Any]) -> Iterable[tuple[Any, ...]]:
    """
    将多个 list/sequence 对齐到同一长度，短的用 "" 补齐，返回逐行 tuple。
    用于消除大量 n=max(...) + i<len(...) 的模板代码。
    """
    seqs = [list(c or []) for c in cols]
    return zip_longest(*seqs, fillvalue="")


def _format_kv(metric: str, value: str) -> str:
    m = _safe_str(metric)
    v = _safe_str(value)
    if not m and not v:
        return ""
    if not m:
        return v
    if not v:
        return f"{m}: -"
    return f"{m}: {v}"


def _loose_line(obj: PersonalDataPattern) -> str:
    try:
        s = obj.recover_to_raw_data()
    except Exception:
        try:
            s = obj.format_print(max_items=4, max_len=200)
        except Exception:
            s = str(obj)
    return _safe_str(s).replace("\n", " ").replace("\r", " ").strip()


def _events_from_patterns(patterns: Sequence[PersonalDataPattern]) -> tuple[list[_Event], dict[str, list[str]]]:
    """
    生成两部分：
    - events：可按 "时间~时间 + 运动类型 + 指标kv" 重构的事件列表
    - grouped_raw_lines：不可重构的类型，按实体类型聚合为 raw 文本列表
    """
    objs = list(patterns or [])
    events: list[_Event] = []
    grouped_raw: dict[str, list[str]] = {}

    for obj in objs:
        et = _safe_str(getattr(obj, "实体类型", "")) or "未定义"

        # 这些类型要求“按类型聚合一行”
        if isinstance(
            obj,
            (
                PeriodValueCompareRecord,
                NoTimestampTextSummaryRecord,
                NoDateValueSummaryRecord,
                UnparsedRawPersonalData,
            ),
        ):
            grouped_raw.setdefault(et, []).append(_loose_line(obj))
            continue

        # 以下为“可重构事件”类型
        if isinstance(obj, SingleMetricDetailRecord):
            core = getattr(obj, "核心字段", None)
            metric_raw = _safe_str(getattr(core, "指标名称", "")) if core else ""
            unit = _safe_str(getattr(core, "单位", "")) if core else "无"
            sport, metric = _split_sport_and_metric(metric_raw)

            ds = list(getattr(obj, "日期列表", []) or [])
            ts = list(getattr(obj, "时间列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
            for d_raw, t_raw, v_raw in _iter_aligned_rows(ds, ts, vs):
                d = _safe_str(d_raw)
                t = _safe_str(t_raw)
                v = _safe_str(v_raw)
                if _is_missing_token(d) and _is_missing_token(v):
                    continue
                v2 = _value_with_unit_and_status(v, unit=unit)
                events.append(
                    _Event(
                        start_date=d,
                        start_time=t if t != "无" else "",
                        end_date=d,
                        end_time=t if t != "无" else "",
                        sport_type=sport or "无",
                        metric=metric,
                        value=v2,
                    )
                )
            continue

        if isinstance(obj, PeriodValueSingleSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            for st_raw, ed_raw, nm_raw, v_raw, u_raw in _iter_aligned_rows(starts, ends, names, vals, units):
                st = _safe_str(st_raw)
                ed = _safe_str(ed_raw)
                nm = _safe_str(nm_raw)
                v = _safe_str(v_raw)
                u = _safe_str(u_raw) or "无"
                if not (st or ed or nm or v):
                    continue
                v2 = _value_with_unit_and_status(v, unit=u)

                # 周期汇总类里常见 "跑步总距离/平均户外跑步速度"：
                # 尝试把运动类型从指标名中剥离，避免输出里运动类型带“总/平均”，以及子项重复携带运动名。
                sport_guess, metric_guess = _split_sport_and_metric(nm)
                if sport_guess != "无" and _looks_like_type_label(sport_guess):
                    sport2 = sport_guess
                    nm2 = metric_guess
                else:
                    sport2 = "无"
                    nm2 = nm
                events.append(
                    _Event(
                        start_date=st,
                        start_time="",
                        end_date=ed or st,
                        end_time="",
                        sport_type=sport2,
                        metric=nm2,
                        value=v2,
                    )
                )
            continue

        if isinstance(obj, PeriodValuemMultiSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            for st_raw, ed_raw, nm_raw, v_raw, u_raw, status_raw in _iter_aligned_rows(
                starts, ends, names, vals, units, sts
            ):
                st = _safe_str(st_raw)
                ed = _safe_str(ed_raw)
                nm = _safe_str(nm_raw)
                v = _safe_str(v_raw)
                u = _safe_str(u_raw) or "无"
                status = _safe_str(status_raw)
                if not (st or ed or nm or v or status):
                    continue
                v2 = _value_with_unit_and_status(v, unit=u, status=status)
                # 周期多项：也尝试从指标名推断健康域/运动域，避免后续“多数投票”被其它指标带偏
                sport_guess, metric_guess = _split_sport_and_metric(nm)
                metric_guess = _METRIC_RENAME.get(metric_guess, metric_guess)
                if sport_guess != "无" and _looks_like_type_label(sport_guess):
                    sport2 = sport_guess
                    nm2 = metric_guess
                else:
                    sport2 = "无"
                    nm2 = _METRIC_RENAME.get(nm, nm)
                events.append(
                    _Event(
                        start_date=st,
                        start_time="",
                        end_date=ed or st,
                        end_time="",
                        sport_type=sport2,
                        metric=nm2,
                        value=v2,
                    )
                )
            continue

        if isinstance(obj, PeriodTextSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            for st_raw, ed_raw, nm_raw, desc_raw in _iter_aligned_rows(starts, ends, names, descs):
                st = _safe_str(st_raw)
                ed = _safe_str(ed_raw)
                nm = _safe_str(nm_raw)
                desc = _safe_str(desc_raw)
                if not (st or ed or nm or desc):
                    continue
                # 周期文本：同样尝试从指标名推断 domain（例如：没有查询到体温数据/减脂数据）
                sport_guess, metric_guess = _split_sport_and_metric(nm)
                metric_guess = _METRIC_RENAME.get(metric_guess, metric_guess)
                if sport_guess != "无" and _looks_like_type_label(sport_guess):
                    sport2 = sport_guess
                    nm2 = metric_guess
                else:
                    sport2 = "无"
                    nm2 = _METRIC_RENAME.get(nm, nm)
                events.append(
                    _Event(
                        start_date=st,
                        start_time="",
                        end_date=ed or st,
                        end_time="",
                        sport_type=sport2,
                        metric=nm2,
                        value=desc,
                    )
                )
            continue

        if isinstance(obj, SingleMetricStatsRecord):
            # 明细：日期 + 数值；汇总：无日期 -> 归到同一“unknown 日期桶”（仍然视为可重构）
            core = getattr(obj, "核心字段", None)
            metric_base = _safe_str(getattr(core, "指标名称", "")) if core else ""
            unit = _safe_str(getattr(core, "单位", "")) if core else "无"

            ds = list(getattr(obj, "日期列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
            # 统计汇总行应该能拿到“该条数据”的首/末日期（避免输出无时间~无时间）
            ds_clean = [ _safe_str(x) for x in ds if _safe_str(x) and (not _is_missing_token(_safe_str(x))) ]
            summary_sd = ds_clean[0] if ds_clean else ""
            summary_ed = ds_clean[-1] if ds_clean else ""
            for d_raw, v_raw in _iter_aligned_rows(ds, vs):
                d = _safe_str(d_raw)
                v = _safe_str(v_raw)
                if not (d or v):
                    continue
                v2 = _value_with_unit_and_status(v, unit=unit)
                events.append(
                    _Event(
                        start_date=d,
                        start_time="",
                        end_date=d,
                        end_time="",
                        sport_type="无",
                        metric=metric_base or "指标",
                        value=v2,
                    )
                )

            sn = list(getattr(obj, "统计指标名称列表", []) or [])
            sv = list(getattr(obj, "统计数值列表", []) or [])
            ss = list(getattr(obj, "统计状态描述列表", []) or [])
            for nm_raw, v_raw, st_raw in _iter_aligned_rows(sn, sv, ss):
                nm = _safe_str(nm_raw) or metric_base or "统计项"
                v = _safe_str(v_raw)
                st = _safe_str(st_raw)
                if not (nm or v or st):
                    continue
                v2 = _value_with_unit_and_status(v, unit=unit, status=st)
                events.append(
                    _Event(
                        start_date=summary_sd,
                        start_time="",
                        end_date=summary_ed,
                        end_time="",
                        sport_type="无",
                        metric=nm,
                        value=v2,
                    )
                )
            continue

        if isinstance(obj, SingleDateValueSingleSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])

            for d_raw, nm_raw, v_raw, u_raw, st_raw in _iter_aligned_rows(ds, names, vals, units, sts):
                d = _safe_str(d_raw)
                nm = _safe_str(nm_raw)
                v = _safe_str(v_raw)
                u = _safe_str(u_raw) or "无"
                st = _safe_str(st_raw)
                if not (d or nm or v or st):
                    continue

                v2 = _value_with_unit_and_status(v, unit=u, status=st)

                # 尝试把健康/运动类型从指标名中拆出（例如：压力均值 -> 情绪健康；锻炼时长 -> 所有运动）
                sport_guess, metric_guess = _split_sport_and_metric(nm)
                metric_guess = _METRIC_RENAME.get(metric_guess, metric_guess)
                if sport_guess != "无" and _looks_like_type_label(sport_guess):
                    sport2 = sport_guess
                    nm2 = metric_guess
                else:
                    sport2 = "无"
                    nm2 = _METRIC_RENAME.get(nm, nm)

                events.append(
                    _Event(
                        start_date=d,
                        start_time="",
                        end_date=d,
                        end_time="",
                        sport_type=sport2,
                        metric=nm2,
                        value=v2,
                    )
                )
            continue

        if isinstance(obj, SingleDateValueMultiSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            for d_raw, nm_raw, v_raw, u_raw, st_raw in _iter_aligned_rows(ds, names, vals, units, sts):
                d = _safe_str(d_raw)
                nm = _safe_str(nm_raw)
                v = _safe_str(v_raw)
                u = _safe_str(u_raw) or "无"
                st = _safe_str(st_raw)
                if not (d or nm or v or st):
                    continue
                v2 = _value_with_unit_and_status(v, unit=u, status=st)
                # 单日期多项：尝试从指标名拆出健康域/运动域，避免被“活动/运动...”多数投票吞掉
                sport_guess, metric_guess = _split_sport_and_metric(nm)
                metric_guess = _METRIC_RENAME.get(metric_guess, metric_guess)
                if sport_guess != "无" and _looks_like_type_label(sport_guess):
                    sport2 = sport_guess
                    nm2 = metric_guess
                else:
                    sport2 = "无"
                    nm2 = _METRIC_RENAME.get(nm, nm)
                events.append(
                    _Event(
                        start_date=d,
                        start_time="",
                        end_date=d,
                        end_time="",
                        sport_type=sport2,
                        metric=nm2,
                        value=v2,
                    )
                )
            continue

        if isinstance(obj, SingleDateTextSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            for d_raw, nm_raw, st_raw in _iter_aligned_rows(ds, names, sts):
                d = _safe_str(d_raw)
                nm = _safe_str(nm_raw)
                st = _safe_str(st_raw)
                if not (d or nm or st):
                    continue
                # 单日期文本：例如“没有查询到体温数据/减脂数据/情绪健康数据”
                # 这些如果保持 sport_type=无，会在后续按同一日期合并时被其它“活动/运动...”指标投票成“活力三环”。
                sport_guess, metric_guess = _split_sport_and_metric(nm)
                metric_guess = _METRIC_RENAME.get(metric_guess, metric_guess)
                if sport_guess != "无" and _looks_like_type_label(sport_guess):
                    sport2 = sport_guess
                    nm2 = metric_guess
                else:
                    sport2 = "无"
                    nm2 = _METRIC_RENAME.get(nm, nm)
                events.append(
                    _Event(
                        start_date=d,
                        start_time="",
                        end_date=d,
                        end_time="",
                        sport_type=sport2,
                        metric=nm2,
                        value=st,
                    )
                )
            continue

        # 其它未知类型：按类型分组落一行（避免丢数据）
        grouped_raw.setdefault(et, []).append(_loose_line(obj))

    return events, grouped_raw


def aggregate_patterns_to_datalines(
    patterns: Sequence[PersonalDataPattern],
    *,
    include_unconstructable_types: bool = True,
    unconstructable_prefix_type: bool = True,
) -> list[str]:
    """
    构造“dataline”风格的训练数据文本行：

    - 可重构类型：按 (时间~时间, 运动类型) 聚合，行内用 “指标: 数值” 形式串联多个子项；
    - 不可按该样式重构的类型：按 实体类型 分组，每个类型输出一行（同类拼在同一行）。
    """
    events, grouped_raw = _events_from_patterns(patterns)

    # 推断“默认年份”：若样本里出现了显式年份（如 2025/4/22），则将缺少年份的日期（如 4/1）
    # 在排序时视为同一年，避免排序结果反直觉。
    years: list[int] = []
    for ev in events:
        for d in (ev.start_date, ev.end_date):
            y, _mo, _da = _parse_date_parts(_safe_str(d))
            if y is not None:
                years.append(int(y))
    default_year: int | None = Counter(years).most_common(1)[0][0] if years else None

    # ---------------------------------------------------------------------
    # 同日多时间点明细聚合（用于减少“同一天内多个时间戳逐条输出”的冗余）
    #
    # 目标格式（示例）：
    #   2025/4/22 健康类型：血氧饱和度，05:50的血氧饱和度为99.00%, 06:00的血氧饱和度为97.00%, ...
    #
    # 触发条件（尽量不影响旧用例）：
    # - 仅对“时间点事件”（start_date==end_date 且 start_time 非空，且 start_time==end_time）生效
    # - 且同一 (日期, 类型, 指标) 下存在 >=2 个不同时间点
    # ---------------------------------------------------------------------
    def _is_timepoint_event(ev: _Event) -> bool:
        sd = _safe_str(ev.start_date)
        ed = _safe_str(ev.end_date)
        st = _safe_str(ev.start_time)
        et = _safe_str(ev.end_time)
        if not sd or not ed or not st:
            return False
        if st == "无" or et == "无":
            return False
        return (sd == ed) and (st == et)

    # group timepoint events by (date, sport_type, metric)
    timepoint_groups: dict[tuple[str, str, str], list[_Event]] = {}
    for ev in events:
        if not _is_timepoint_event(ev):
            continue
        sd = _safe_str(ev.start_date)
        sport = _safe_str(ev.sport_type) or "无"
        metric = _safe_str(ev.metric)
        if not (sd and sport and metric):
            continue
        timepoint_groups.setdefault((sd, sport, metric), []).append(ev)

    # decide which groups are worth merging (>=2 valid distinct times)
    merge_timepoint_keys: set[tuple[str, str, str]] = set()
    for k, evs in timepoint_groups.items():
        times = [
            _normalize_time_cn_token(_safe_str(e.start_time))
            for e in evs
            if _safe_str(e.start_time) and _safe_str(e.start_time) != "无"
        ]
        distinct = {t for t in times if t}
        if len(distinct) >= 2:
            merge_timepoint_keys.add(k)

    # group events by (start/end + sport)
    groups: dict[tuple[str, str, str, str, str], list[_Event]] = {}
    for ev in events:
        sd = _safe_str(ev.start_date)
        st = _safe_str(ev.start_time)
        ed = _safe_str(ev.end_date) or sd
        et = _safe_str(ev.end_time) or st
        sport = _safe_str(ev.sport_type) or "无"

        # 若该事件属于“同日多时间点明细”的可合并组，则不再逐条进入旧的 (sd,st,ed,et,sport) 分组
        if _is_timepoint_event(ev):
            k3 = (sd, sport, _safe_str(ev.metric))
            if k3 in merge_timepoint_keys:
                continue

        key = (sd, st, ed, et, sport)
        groups.setdefault(key, []).append(ev)

    # 先构造“行级对象”，再统一排序：
    # - 先按 运动/健康 大类（运动类型优先，其次健康类型，最后其它类型）
    # - 再按 具体类型名（如：所有运动/睡眠/血氧饱和度...）
    # - 再按 起止时间（按时间升序）
    line_items: list[tuple[tuple[Any, ...], str]] = []

    # 0) 先输出“同日多时间点明细”合并行
    for (sd, sport, metric), evs in timepoint_groups.items():
        if (sd, sport, metric) not in merge_timepoint_keys:
            continue

        # time segs: sort by time asc; keep multiple values at same time in stable order
        def _timepoint_sort_key(e: _Event) -> tuple[int, int, str]:
            tt = _normalize_time_cn_token(_safe_str(e.start_time))
            return _parse_time_for_sort(tt)

        evs_sorted = sorted(evs, key=_timepoint_sort_key)
        segs: list[str] = []
        for e in evs_sorted:
            tt = _normalize_time_cn_token(_safe_str(e.start_time))
            vv = _safe_str(e.value) or "-"
            if not tt:
                continue
            # 统一为“HH:MM的指标为数值”（指标名保持原样，数值已包含单位）
            segs.append(f"{tt}的{metric}: {vv}")
        if not segs:
            continue

        sport_label = sport or "无"
        label_key = "类型"
        category_order = 2
        if sport_label in _health_domains():
            label_key = "健康类型"
            category_order = 1
        elif sport_label in _sport_domains() or sport_label in _sport_subdomains():
            label_key = "运动类型"
            category_order = 0

        date_label = _date_to_slash(sd)
        line = f"{date_label} {label_key}：{sport_label}， " + ", ".join(segs)

        # 排序：
        # - 先按 运动/健康 大类（运动类型优先，其次健康类型，最后其它类型）
        # - 再按 具体类型名（如：所有运动/睡眠/血氧饱和度...）
        # - 再按 起止时间（按时间升序）
        earliest_time = _normalize_time_cn_token(_safe_str(evs_sorted[0].start_time)) if evs_sorted else ""
        sort_key = (
            category_order,
            str(sport_label or ""),
            _parse_date_for_sort(sd, default_year=default_year),
            _parse_time_for_sort(earliest_time),
            _parse_date_for_sort(sd, default_year=default_year),
            _parse_time_for_sort(earliest_time),
        )
        line_items.append((sort_key, line.strip()))

    for (sd, st, ed, et, sport) in groups.keys():
        evs = groups.get((sd, st, ed, et, sport), [])
        if not evs:
            continue

        # 若某组 sport=无，但指标里能推断 sport，则用多数投票补齐
        sport2 = sport
        if (not sport2) or sport2 == "无":
            c = Counter(
                [
                    sp
                    for sp in [
                        _split_sport_and_metric(e.metric)[0]
                        for e in evs
                        if _split_sport_and_metric(e.metric)[0] != "无"
                    ]
                    if _looks_like_type_label(sp)
                ]
            )
            if c:
                sport2 = c.most_common(1)[0][0]

        time_range = _format_time_range(sd, st, ed, et)

        # 行内指标：同名后写覆盖（更接近“最新值”直觉）
        kv: dict[str, str] = {}
        for e in sorted(evs, key=lambda x: _metric_sort_key(x.metric)):
            k_metric = _safe_str(e.metric)
            v = _safe_str(e.value)
            if not k_metric:
                continue
            kv[k_metric] = v

        # 输出
        metric_segs = []
        for m in sorted(kv.keys(), key=_metric_sort_key):
            seg = _format_kv(m, kv[m])
            if seg:
                metric_segs.append(seg)

        sport_label = sport2 or "无"
        # 输出时区分“健康类型/运动类型”，避免健康项被展示成“运动类型”
        # 同时用于排序：运动类型优先，其次健康类型，最后其它
        label_key = "类型"
        category_order = 2
        if sport_label in _health_domains():
            label_key = "健康类型"
            category_order = 1
        elif sport_label in _sport_domains() or sport_label in _sport_subdomains():
            label_key = "运动类型"
            category_order = 0
        line = f"{time_range} {label_key}：{sport_label}"
        if metric_segs:
            line += "， " + ", ".join(metric_segs)
        sort_key = (
            category_order,
            str(sport_label or ""),
            _parse_date_for_sort(sd, default_year=default_year),
            _parse_time_for_sort(st),
            _parse_date_for_sort(ed, default_year=default_year),
            _parse_time_for_sort(et),
        )
        line_items.append((sort_key, line.strip()))

    line_items.sort(key=lambda x: x[0])

    # 展示层分隔（不改变原排序逻辑）：
    # - 大类（运动/健康/其它）切换：插入空行
    # - 同一大类内，具体类型（如：所有运动/睡眠/血氧饱和度...）切换：插入空行
    lines: list[str] = []
    prev_category: int | None = None
    prev_type: str | None = None
    for k, ln in line_items:
        # k[0] 是 category_order（0/1/2）
        cat = int(k[0]) if k else 2
        # k[1] 是具体类型名（sport_label）
        typ = str(k[1]) if (k and len(k) > 1) else ""

        if prev_category is not None:
            # 大类切换 or 具体类型切换：都插入空行
            if (cat != prev_category) or (typ != (prev_type or "")):
                lines.append("")
        lines.append(ln)
        prev_category = cat
        prev_type = typ

    if include_unconstructable_types and grouped_raw:
        lines.append("")
        for et in sorted(grouped_raw.keys()):
            items = [str(x).strip() for x in grouped_raw.get(et, []) if str(x).strip()]
            if not items:
                continue
            # 同类写在一行；条目之间用 "；" 分隔以减少歧义
            body = "；".join(items)
            if unconstructable_prefix_type:
                lines.append(f"数据类型：{et}，{body}".strip())
            else:
                lines.append(body.strip())

    return lines


def aggregate_patterns_to_dataline_text(
    patterns: Sequence[PersonalDataPattern],
    *,
    include_unconstructable_types: bool = True,
    unconstructable_prefix_type: bool = True,
) -> str:
    """
    `aggregate_patterns_to_datalines()` 的文本包装：用换行拼接输出。
    """
    lines = aggregate_patterns_to_datalines(
        patterns,
        include_unconstructable_types=include_unconstructable_types,
        unconstructable_prefix_type=unconstructable_prefix_type,
    )
    return ("\n".join(lines).rstrip() + "\n") if lines else ""


__all__ = ["aggregate_patterns_to_datalines", "aggregate_patterns_to_dataline_text"]

