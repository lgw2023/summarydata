from __future__ import annotations

"""
个人数据“数据样式(data pattern)”的数据类定义。

目标：
- 与 `scripts/analyze_personal_data.py` 中的样式 JSON 输出结构保持一致：
  style item 形如：{"实体类型": "...", "核心字段": {...}}
- 为以下实体类型分别提供数据类：
  - 单指标的明细记录
  - 周期数值单项总结
  - 周期文本总结
  - 周期数值对比记录
  - 周期数值多项总结
  - 无时间日期的文本总结
  - 无时间日期的数值总结
  - 单指标的明细汇总记录
- 额外提供“兜底类”：用于保存无法解析的数据样式，保留原始个人数据纯文本。

说明：
- 这里主要做“结构化承载”，不做强校验；强校验逻辑在 `analyze_personal_data.py` 中完成。
"""

from dataclasses import dataclass, field
from functools import lru_cache
import json
import re
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union

from .normalize import (
    _attach_unit_to_value,
    _format_date_range,
    _is_missing_token,
    _is_time_token,
    _join_raw_segments,
    _normalize_date_cn,
    _normalize_date_or_range_cn,
    _normalize_time_cn_token,
    _normalize_value_and_unit,
    _strip_leading_de,
)


# ========= 格式化打印（给 self-test / 人类阅读用）=========
def _shorten(s: Any) -> str:
    t = str(s if s is not None else "").replace("\n", " ").strip()
    return t


def _indent_lines(text: str, n: int = 2) -> str:
    pad = " " * max(0, int(n))
    return "\n".join((pad + ln) if ln else ln for ln in str(text).splitlines())


def _fmt_header(title: str) -> str:
    t = str(title).strip()
    return f"【{t}】" if t else "【】"


def _fmt_kv(key: str, value: Any) -> str:
    return f"- {key}：{_shorten(value)}"


def _fmt_list_preview(
    items: Sequence[Any],
    bullet: str = "- ",
) -> list[str]:
    xs = list(items or [])
    out: list[str] = []
    for i, x in enumerate(xs):
        out.append(f"{bullet}[{i}] {_shorten(x)}")
    return out


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _fmt_rows_table(
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str]
) -> str:
    """
    简易“表格”输出：对齐列宽，适合日志阅读（不追求 markdown）。
    """
    hs = [str(h) for h in headers]
    rs2 = [list(map(str, r)) for r in rows]
    # 计算列宽
    col_n = len(hs)
    widths = [len(hs[i]) for i in range(col_n)]
    for r in rs2:
        for i in range(min(col_n, len(r))):
            widths[i] = max(widths[i], len(r[i]))
    def _row_line(cols: Sequence[str]) -> str:
        cols2 = list(cols) + [""] * (col_n - len(cols))
        return "  ".join(cols2[i].ljust(widths[i]) for i in range(col_n))
    lines = [_row_line(hs), _row_line(["-" * w for w in widths])]
    for r in rs2:
        lines.append(_row_line(r))
    return "\n".join(lines)


ValueType = Literal["Int", "Float", "String", "Duration", "Timestamp", "FloatRange"]


# ========= 基类/工具 =========
@dataclass(frozen=True)
class PersonalDataPatternBase:
    """
    所有实体类型记录的共同接口。
    """

    实体类型: str

    def recover_to_raw_data(self) -> str:
        """
        尝试把数据类中的结构化字段“重新组织”，尽可能还原出原始个人数据文本。

        约定（优先级）：
        - 若对象本身携带 `原始个人数据` / `个人数据` 且非空：直接返回（这是最接近“原样”的还原）。
        - 否则按 `实体类型` 结合各类的列表字段进行拼接还原。
        - 若仍无法还原：退化为 `format_print()` 的简短输出（保证不会返回空串）。
        """
        raw0 = _safe_getattr(self, "原始个人数据", None)
        if isinstance(raw0, str) and raw0.strip():
            return raw0.strip()
        raw1 = _safe_getattr(self, "个人数据", None)
        if isinstance(raw1, str) and raw1.strip():
            return raw1.strip()

        et = str(_safe_getattr(self, "实体类型", "") or "").strip()
        try:
            if et == "未定义":
                # 兜底类：这里通常应当走 raw1（个人数据）；若没有则返回空串兜底
                return str(raw1 or "").strip()
            fn = _RECOVER_BY_ENTITY_TYPE.get(et)
            if fn is not None:
                return fn(self)
        except Exception:
            # 不让还原逻辑影响主流程
            pass

        # 最终兜底：返回简短可读文本，避免空串
        try:
            txt = self.format_print()
            return str(txt or "").strip() or et or str(self)
        except Exception:
            return et or str(self)

    def format_print(
        self,
    ) -> str:
        """
        面向人类阅读的格式化输出（用于 self-test / 日志 / 快速肉眼检查）。
        子类建议覆盖以输出更“领域友好”的结构。
        """
        lines = [_fmt_header(self.实体类型)]
        # 最小兜底：尽量列出常见字段（不直接 dump 整个 __dict__）
        for k in ("核心字段", "原始个人数据", "个人数据", "原因"):
            v = _safe_getattr(self, k, None)
            if v is None:
                continue
            if k in ("原始个人数据", "个人数据"):
                lines.append(_fmt_kv(k, v))
            elif k == "原因":
                lines.append(_fmt_kv(k, v))
            else:
                # 核心字段：只做浅层展示
                try:
                    core_dict = v.__dict__ if hasattr(v, "__dict__") else v
                except Exception:
                    core_dict = v
                lines.append("- 核心字段：")
                if isinstance(core_dict, Mapping):
                    for kk in ("指标名称", "开始日期", "结束日期", "日期", "时间", "单位", "数值类型", "状态描述"):
                        if kk in core_dict:
                            lines.append(_indent_lines(_fmt_kv(kk, core_dict.get(kk)), 2))
                else:
                    lines.append(_indent_lines(_shorten(core_dict), 2))
        return "\n".join(lines)


@dataclass(frozen=True)
class UnparsedRawPersonalData(PersonalDataPatternBase):
    """
    兜底：当无法将 LLM 输出解析成任何已知实体类型结构时，保存原始个人数据纯文本。
    """

    个人数据: str
    原因: str | None = None
    原始样式输出: Any | None = None

    def __init__(self, 个人数据: str, 原因: str | None = None, 原始样式输出: Any | None = None) -> None:
        object.__setattr__(self, "实体类型", "未定义")
        object.__setattr__(self, "个人数据", str(个人数据))
        object.__setattr__(self, "原因", 原因)
        object.__setattr__(self, "原始样式输出", 原始样式输出)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        原因: str | None = None,
        原始样式输出: Any | None = None,
    ) -> list["UnparsedRawPersonalData"]:
        """
        兜底解析入口：当你希望把“原始一行个人数据”显式标记为无法解析时使用。

        说明：
        - 与其它实体类型的 `from_raw_personal_data()` 形式保持一致，便于统一调用。
        - 该方法不会尝试解析任何结构，直接返回仅包含一个 `UnparsedRawPersonalData` 的列表。
        """
        raw = str(raw_line or "").strip()
        return [cls(个人数据=raw, 原因=原因, 原始样式输出=原始样式输出)]

    def format_print(self) -> str:  # type: ignore[override]
        lines = [_fmt_header("数据类型：未定义")]
        if self.原因:
            lines.append(_fmt_kv("原因", self.原因))
        lines.append(_fmt_kv("个人数据", self.个人数据))
        if self.原始样式输出 is not None:
            lines.append(_fmt_kv("原始样式输出", self.原始样式输出))
        return "\n".join(lines)


# ========= 单指标的明细记录 =========
@dataclass(frozen=True)
class SingleValueCore:
    """
    单指标的明细记录的“样式 core（metadata）”。

    注意：历史上该 core 只用于承载 LLM 输出的“字段格式/占位符”，不包含真实数值。
    真实一行个人数据往往包含多条记录（多组 日期/时间/数值），这些明细由 `SingleMetricDetailRecord`
    额外字段 `日期列表/时间列表/数值列表` 承载。
    """

    日期: str
    时间: str
    指标名称: str
    数值类型: ValueType
    单位: str


@dataclass(frozen=True)
class SingleMetricDetailRecord(PersonalDataPatternBase):
    核心字段: SingleValueCore
    # 一行个人数据中可能包含多条记录：用列表承载“明细”
    日期列表: list[str] = field(default_factory=list)
    时间列表: list[str] = field(default_factory=list)
    数值列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: SingleValueCore,
        *,
        日期列表: Sequence[str] | None = None,
        时间列表: Sequence[str] | None = None,
        数值列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "单指标的明细记录")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "日期列表", list(日期列表 or []))
        object.__setattr__(self, "时间列表", list(时间列表 or []))
        object.__setattr__(self, "数值列表", list(数值列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含多条明细列表），用于你后续真正做数据解析/训练数据构造。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "日期列表": list(self.日期列表),
            "时间列表": list(self.时间列表),
            "数值列表": list(self.数值列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(len(self.日期列表), len(self.时间列表), len(self.数值列表))

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）"),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        rows: list[list[Any]] = []
        n = max(len(self.日期列表), len(self.时间列表), len(self.数值列表))
        for i in range(n):
            d = self.日期列表[i] if i < len(self.日期列表) else ""
            t = self.时间列表[i] if i < len(self.时间列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            rows.append([d, t, v])
        if rows:
            lines.append("- 明细（日期 / 时间 / 数值）：")
            table = _fmt_rows_table(rows, headers=("日期", "时间", "数值"))
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
        数值类型: ValueType | None = None,
        单位: str | None = None,
        默认时间: str = "无",
    ) -> list["SingleMetricDetailRecord | UnparsedRawPersonalData"]:
        """
        从"单指标的明细记录"原始一行文本中抽取多条记录。

        支持常见格式（示例）：
        - 2025/2/1 06:07的户外跑步距离为：5.11千米,2025/2/2 06:46的...
        - 2025/1/24 05:52的跳绳平均速度为：125.00个/分钟,2025/1/25 05:39的跳绳个数为：10008.00（多指标）

        参数说明：
        - `指标名称`、`数值类型`、`单位`：仅在单指标情况下作为提示使用；多指标情况下会忽略（从数据中自动推断）

        返回值：
        - 如果只检测到一个指标，返回 `[SingleMetricDetailRecord]`
        - 如果检测到多个不同指标，返回 `list[SingleMetricDetailRecord]`（每个指标一个）
        - 如果解析失败，返回 `[UnparsedRawPersonalData]`
        """
        # 先尝试多指标解析
        parsed_list = _parse_single_value_line_multi(
            raw_line,
            默认时间=默认时间,
        )
        
        if isinstance(parsed_list, UnparsedRawPersonalData):
            # 多指标解析失败，尝试单指标解析（使用用户提供的参数）
            parsed = _parse_single_value_line(
                raw_line,
                指标名称=指标名称,
                数值类型=数值类型,
                单位=单位,
                默认时间=默认时间,
            )
            return [parsed]
        
        # 如果用户提供了参数，更新第一个记录的 core（如果与推断的不同）
        if len(parsed_list) == 1:
            record = parsed_list[0]
            if 指标名称 and record.核心字段.指标名称 != 指标名称:
                # 更新指标名称
                core = SingleValueCore(
                    日期=record.核心字段.日期,
                    时间=record.核心字段.时间,
                    指标名称=指标名称,
                    数值类型=数值类型 or record.核心字段.数值类型,
                    单位=单位 or record.核心字段.单位,
                )
                return [
                    SingleMetricDetailRecord(
                    核心字段=core,
                    日期列表=record.日期列表,
                    时间列表=record.时间列表,
                    数值列表=record.数值列表,
                    原始个人数据=record.原始个人数据,
                )
                ]
            return [record]
        
        return parsed_list


# ========= 周期数值单项总结 =========
@dataclass(frozen=True)
class PeriodSummaryCore:
    开始日期: str
    结束日期: str
    指标名称: str
    数值类型: ValueType
    单位: str


@dataclass(frozen=True)
class PeriodValueSingleSummaryRecord(PersonalDataPatternBase):
    核心字段: PeriodSummaryCore
    # 一行个人数据中可能包含多条“周期数值单项总结”：用列表承载“明细”
    开始日期列表: list[str] = field(default_factory=list)
    结束日期列表: list[str] = field(default_factory=list)
    指标名称列表: list[str] = field(default_factory=list)
    数值类型列表: list[ValueType] = field(default_factory=list)
    单位列表: list[str] = field(default_factory=list)
    数值列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: PeriodSummaryCore,
        *,
        开始日期列表: Sequence[str] | None = None,
        结束日期列表: Sequence[str] | None = None,
        指标名称列表: Sequence[str] | None = None,
        数值类型列表: Sequence[ValueType] | None = None,
        单位列表: Sequence[str] | None = None,
        数值列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "周期数值单项总结")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "开始日期列表", list(开始日期列表 or []))
        object.__setattr__(self, "结束日期列表", list(结束日期列表 or []))
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "数值类型列表", list(数值类型列表 or []))
        object.__setattr__(self, "单位列表", list(单位列表 or []))
        object.__setattr__(self, "数值列表", list(数值列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含明细列表），用于后续真正做数据解析/训练数据构造。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "开始日期列表": list(self.开始日期列表),
            "结束日期列表": list(self.结束日期列表),
            "指标名称列表": list(self.指标名称列表),
            "数值类型列表": list(self.数值类型列表),
            "单位列表": list(self.单位列表),
            "数值列表": list(self.数值列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(
            len(self.开始日期列表),
            len(self.结束日期列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
        )

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）"),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(
            len(self.开始日期列表),
            len(self.结束日期列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
        )
        rows: list[list[Any]] = []
        for i in range(n):
            st = self.开始日期列表[i] if i < len(self.开始日期列表) else ""
            ed = self.结束日期列表[i] if i < len(self.结束日期列表) else ""
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            vt = self.数值类型列表[i] if i < len(self.数值类型列表) else ""
            u = self.单位列表[i] if i < len(self.单位列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            rows.append([st, ed, nm, vt, u, v])
        if rows:
            lines.append("- 明细（开始/结束/指标/类型/单位/数值）：")
            table = _fmt_rows_table(
                rows,
                headers=("开始", "结束", "指标", "类型", "单位", "数值")
            )
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
        数值类型: ValueType | None = None,
        单位: str | None = None,
    ) -> list["PeriodValueSingleSummaryRecord | UnparsedRawPersonalData"]:
        """
        从“周期数值单项总结”原始一行文本中抽取 1~N 条汇总记录。

        支持常见格式（示例）：
        - 2025/2/1至2025/2/22的跑步总距离为201.27千米
        - 2025/2/1~2025/2/22的跑步总热量为12052.00千卡
        - 2025/2/1到2025/2/22的平均户外跑步心率为136.00次/分钟
        - 2025/2/1至2025/2/22的跑步总距离为201.27千米,跑步总热量为12052.00千卡（后段继承日期范围）

        约定：
        - `数值列表` 仅保存“纯数值”（如 201.27），不包含单位；单位会写入 `单位列表`。
        """
        parsed = _parse_period_summary_line(
            raw_line,
            指标名称=指标名称,
            数值类型=数值类型,
            单位=单位,
        )
        return [parsed]


# ========= 周期文本总结 =========
@dataclass(frozen=True)
class PeriodTextSummaryCore:
    开始日期: str
    结束日期: str
    指标名称: str
    状态描述: str  # 规则里要求严格为 "String"


@dataclass(frozen=True)
class PeriodTextSummaryRecord(PersonalDataPatternBase):
    核心字段: PeriodTextSummaryCore
    # 一行个人数据中可能包含多条“周期文本总结”：用列表承载“明细”
    开始日期列表: list[str] = field(default_factory=list)
    结束日期列表: list[str] = field(default_factory=list)
    指标名称列表: list[str] = field(default_factory=list)
    状态描述列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: PeriodTextSummaryCore,
        *,
        开始日期列表: Sequence[str] | None = None,
        结束日期列表: Sequence[str] | None = None,
        指标名称列表: Sequence[str] | None = None,
        状态描述列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "周期文本总结")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "开始日期列表", list(开始日期列表 or []))
        object.__setattr__(self, "结束日期列表", list(结束日期列表 or []))
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "状态描述列表", list(状态描述列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含明细列表）。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "开始日期列表": list(self.开始日期列表),
            "结束日期列表": list(self.结束日期列表),
            "指标名称列表": list(self.指标名称列表),
            "状态描述列表": list(self.状态描述列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(
            len(self.开始日期列表),
            len(self.结束日期列表),
            len(self.指标名称列表),
            len(self.状态描述列表),
        )

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", core.指标名称),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(
            len(self.开始日期列表),
            len(self.结束日期列表),
            len(self.指标名称列表),
            len(self.状态描述列表),
        )
        rows: list[list[Any]] = []
        for i in range(n):
            st = self.开始日期列表[i] if i < len(self.开始日期列表) else ""
            ed = self.结束日期列表[i] if i < len(self.结束日期列表) else ""
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            desc = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([st, ed, nm, desc])
        if rows:
            lines.append("- 明细（开始/结束/指标/状态描述）：")
            table = _fmt_rows_table(rows, headers=("开始", "结束", "指标", "状态描述"))
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
    ) -> list["PeriodTextSummaryRecord | UnparsedRawPersonalData"]:
        """
        从“周期文本总结”原始一行文本中抽取 1~N 条总结记录。

        支持常见格式（示例）：
        - 2024/1/1~2024/12/31锻炼时长偏低
        - 8/7 锻炼时长偏低
        - 2024/1/1至2024/12/31的锻炼时长偏低, 8/7 锻炼时长偏低（逗号分隔多条）
        """
        parsed = _parse_period_text_summary_line(raw_line, 指标名称=指标名称)
        return [parsed]


# ========= 周期数值对比记录 =========
@dataclass(frozen=True)
class PeriodValueCompareCore:
    日期范围1: str
    日期范围2: str
    指标名称: str
    数值类型: ValueType
    单位: str
    对比逻辑类型: str  # 规则里要求严格为 "String"
    差异数值类型: ValueType


@dataclass(frozen=True)
class PeriodValueCompareRecord(PersonalDataPatternBase):
    核心字段: PeriodValueCompareCore
    # 一行个人数据中可能包含多条“周期数值对比记录”：用列表承载“明细”
    日期范围1列表: list[str] = field(default_factory=list)
    数值1列表: list[str] = field(default_factory=list)
    日期范围2列表: list[str] = field(default_factory=list)
    数值2列表: list[str] = field(default_factory=list)
    差异数值列表: list[str] = field(default_factory=list)
    指标名称列表: list[str] = field(default_factory=list)
    数值类型列表: list[ValueType] = field(default_factory=list)
    单位列表: list[str] = field(default_factory=list)
    对比逻辑类型列表: list[str] = field(default_factory=list)
    差异数值类型列表: list[ValueType] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: PeriodValueCompareCore,
        *,
        日期范围1列表: Sequence[str] | None = None,
        数值1列表: Sequence[str] | None = None,
        日期范围2列表: Sequence[str] | None = None,
        数值2列表: Sequence[str] | None = None,
        差异数值列表: Sequence[str] | None = None,
        指标名称列表: Sequence[str] | None = None,
        数值类型列表: Sequence[ValueType] | None = None,
        单位列表: Sequence[str] | None = None,
        对比逻辑类型列表: Sequence[str] | None = None,
        差异数值类型列表: Sequence[ValueType] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "周期数值对比记录")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "日期范围1列表", list(日期范围1列表 or []))
        object.__setattr__(self, "数值1列表", list(数值1列表 or []))
        object.__setattr__(self, "日期范围2列表", list(日期范围2列表 or []))
        object.__setattr__(self, "数值2列表", list(数值2列表 or []))
        object.__setattr__(self, "差异数值列表", list(差异数值列表 or []))
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "数值类型列表", list(数值类型列表 or []))
        object.__setattr__(self, "单位列表", list(单位列表 or []))
        object.__setattr__(self, "对比逻辑类型列表", list(对比逻辑类型列表 or []))
        object.__setattr__(self, "差异数值类型列表", list(差异数值类型列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含明细列表），用于后续真正做数据解析/训练数据构造。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "日期范围1列表": list(self.日期范围1列表),
            "数值1列表": list(self.数值1列表),
            "日期范围2列表": list(self.日期范围2列表),
            "数值2列表": list(self.数值2列表),
            "差异数值列表": list(self.差异数值列表),
            "指标名称列表": list(self.指标名称列表),
            "数值类型列表": list(self.数值类型列表),
            "单位列表": list(self.单位列表),
            "对比逻辑类型列表": list(self.对比逻辑类型列表),
            "差异数值类型列表": list(self.差异数值类型列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(
            len(self.日期范围1列表),
            len(self.数值1列表),
            len(self.日期范围2列表),
            len(self.数值2列表),
            len(self.差异数值列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.对比逻辑类型列表),
            len(self.差异数值类型列表),
        )

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv(
                "指标名称",
                f"{core.指标名称}（{core.数值类型}，单位={core.单位}；差异类型={core.差异数值类型}）"
            ),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(
            len(self.日期范围1列表),
            len(self.数值1列表),
            len(self.日期范围2列表),
            len(self.数值2列表),
            len(self.差异数值列表),
            len(self.对比逻辑类型列表),
        )
        rows: list[list[Any]] = []
        for i in range(n):
            r1 = self.日期范围1列表[i] if i < len(self.日期范围1列表) else ""
            v1 = self.数值1列表[i] if i < len(self.数值1列表) else ""
            r2 = self.日期范围2列表[i] if i < len(self.日期范围2列表) else ""
            v2 = self.数值2列表[i] if i < len(self.数值2列表) else ""
            logic = self.对比逻辑类型列表[i] if i < len(self.对比逻辑类型列表) else ""
            dv = self.差异数值列表[i] if i < len(self.差异数值列表) else ""
            rows.append([r1, v1, r2, v2, logic, dv])
        if rows:
            lines.append("- 对比明细（范围1/值1/范围2/值2/1相较于2的逻辑/1相较于2的差异）：")
            table = _fmt_rows_table(
                rows,
                headers=("范围1", "值1", "范围2", "值2", "1相较于2的逻辑", "1相较于2的差异")
            )
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
        数值类型: ValueType | None = None,
        单位: str | None = None,
        对比逻辑类型: str | None = None,
        差异数值类型: ValueType | None = None,
    ) -> list["PeriodValueCompareRecord | UnparsedRawPersonalData"]:
        """
        从“周期数值对比记录”原始一行文本中抽取 1~N 组对比记录。

        支持常见格式（示例）：
        - 6/16~6/22的平均快速眼动比例为19.0%，6/23~6/26的平均快速眼动比例为22.0%，少3.0%
        - 6/16~6/22的平均睡眠时长为6小时2分钟，6/23~6/26的平均睡眠时长为5小时59分钟，多3分钟
        """
        parsed = _parse_period_value_compare_line(
            raw_line,
            指标名称=指标名称,
            数值类型=数值类型,
            单位=单位,
            对比逻辑类型=对比逻辑类型,
            差异数值类型=差异数值类型,
        )
        return [parsed]


# ========= 周期数值多项总结 =========
@dataclass(frozen=True)
class PeriodValueSummaryCore:
    开始日期: str
    结束日期: str
    指标名称: str
    数值类型: ValueType
    单位: str
    状态描述: str  # 规则里要求严格为 "String"


@dataclass(frozen=True)
class PeriodValuemMultiSummaryRecord(PersonalDataPatternBase):
    核心字段: PeriodValueSummaryCore
    # 一行个人数据中可能包含多条“周期数值多项总结”：用列表承载“明细”
    开始日期列表: list[str] = field(default_factory=list)
    结束日期列表: list[str] = field(default_factory=list)
    指标名称列表: list[str] = field(default_factory=list)
    数值类型列表: list[ValueType] = field(default_factory=list)
    单位列表: list[str] = field(default_factory=list)
    数值列表: list[str] = field(default_factory=list)
    状态描述列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: PeriodValueSummaryCore,
        *,
        开始日期列表: Sequence[str] | None = None,
        结束日期列表: Sequence[str] | None = None,
        指标名称列表: Sequence[str] | None = None,
        数值类型列表: Sequence[ValueType] | None = None,
        单位列表: Sequence[str] | None = None,
        数值列表: Sequence[str] | None = None,
        状态描述列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "周期数值多项总结")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "开始日期列表", list(开始日期列表 or []))
        object.__setattr__(self, "结束日期列表", list(结束日期列表 or []))
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "数值类型列表", list(数值类型列表 or []))
        object.__setattr__(self, "单位列表", list(单位列表 or []))
        object.__setattr__(self, "数值列表", list(数值列表 or []))
        object.__setattr__(self, "状态描述列表", list(状态描述列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含明细列表），用于后续真正做数据解析/训练数据构造。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "开始日期列表": list(self.开始日期列表),
            "结束日期列表": list(self.结束日期列表),
            "指标名称列表": list(self.指标名称列表),
            "数值类型列表": list(self.数值类型列表),
            "单位列表": list(self.单位列表),
            "数值列表": list(self.数值列表),
            "状态描述列表": list(self.状态描述列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(
            len(self.开始日期列表),
            len(self.结束日期列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
            len(self.状态描述列表),
        )

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）"),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(
            len(self.开始日期列表),
            len(self.结束日期列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
            len(self.状态描述列表),
        )
        rows: list[list[Any]] = []
        for i in range(n):
            st = self.开始日期列表[i] if i < len(self.开始日期列表) else ""
            ed = self.结束日期列表[i] if i < len(self.结束日期列表) else ""
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            vt = self.数值类型列表[i] if i < len(self.数值类型列表) else ""
            u = self.单位列表[i] if i < len(self.单位列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            desc = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([st, ed, nm, vt, u, v, desc])
        if rows:
            lines.append("- 明细（开始/结束/指标/类型/单位/数值/状态）：")
            table = _fmt_rows_table(
                rows,
                headers=("开始", "结束", "指标", "类型", "单位", "数值", "状态")
            )
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
        数值类型: ValueType | None = None,
        单位: str | None = None,
    ) -> list["PeriodValuemMultiSummaryRecord | UnparsedRawPersonalData"]:
        """
        从“周期数值多项总结”原始一行文本中抽取 1~N 条总结记录（通常是单日或日期范围）。

        支持常见格式（示例）：
        - 8月7日锻炼时长15分钟偏低
        - 4/18锻炼时长2小时49分钟正常
        - 2025/2/1~2025/2/7锻炼时长2小时49分钟正常, 4/18锻炼时长35分钟偏低（逗号分隔多条）
        """
        parsed = _parse_period_value_summary_line(
            raw_line,
            指标名称=指标名称,
            数值类型=数值类型,
            单位=单位,
        )
        return [parsed]


# ========= 单日期数值单项总结 =========
@dataclass(frozen=True)
class SingleDateValueSummaryCore:
    """
    单日期数值单项总结的“样式 core（metadata）”。

    设计保持与 `SingleValueCore` / `PeriodValueSummaryCore` 一致：
    - core 用于承载“字段占位/元信息”（日期格式、指标名称、数值类型、单位、状态描述占位）
    - 一行个人数据可能包含多条(指标名称, 数值, 状态描述)记录：由 record 的列表字段承载
    """

    指标名称: str
    日期: str
    数值类型: ValueType
    单位: str
    状态描述: str  # 规则里要求严格为 "String"


@dataclass(frozen=True)
class SingleDateValueSingleSummaryRecord(PersonalDataPatternBase):
    核心字段: SingleDateValueSummaryCore
    # 一行个人数据中可能包含多条“单日期数值单项总结”：用列表承载“明细”
    日期列表: list[str] = field(default_factory=list)
    指标名称列表: list[str] = field(default_factory=list)
    数值类型列表: list[ValueType] = field(default_factory=list)
    单位列表: list[str] = field(default_factory=list)
    数值列表: list[str] = field(default_factory=list)
    状态描述列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: SingleDateValueSummaryCore,
        *,
        日期列表: Sequence[str] | None = None,
        指标名称列表: Sequence[str] | None = None,
        数值类型列表: Sequence[ValueType] | None = None,
        单位列表: Sequence[str] | None = None,
        数值列表: Sequence[str] | None = None,
        状态描述列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "单日期数值单项总结")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "日期列表", list(日期列表 or []))
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "数值类型列表", list(数值类型列表 or []))
        object.__setattr__(self, "单位列表", list(单位列表 or []))
        object.__setattr__(self, "数值列表", list(数值列表 or []))
        object.__setattr__(self, "状态描述列表", list(状态描述列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含明细列表），用于后续真正做数据解析/训练数据构造。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "日期列表": list(self.日期列表),
            "指标名称列表": list(self.指标名称列表),
            "数值类型列表": list(self.数值类型列表),
            "单位列表": list(self.单位列表),
            "数值列表": list(self.数值列表),
            "状态描述列表": list(self.状态描述列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(
            len(self.日期列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
            len(self.状态描述列表),
        )

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）"),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(
            len(self.日期列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
            len(self.状态描述列表),
        )
        rows: list[list[Any]] = []
        for i in range(n):
            d = self.日期列表[i] if i < len(self.日期列表) else ""
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            vt = self.数值类型列表[i] if i < len(self.数值类型列表) else ""
            u = self.单位列表[i] if i < len(self.单位列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            desc = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([d, nm, vt, u, v, desc])
        if rows:
            lines.append("- 明细（日期/指标/类型/单位/数值/状态）：")
            table = _fmt_rows_table(
                rows,
                headers=("日期", "指标", "类型", "单位", "数值", "状态")
            )
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
        数值类型: ValueType | None = None,
        单位: str | None = None,
    ) -> list["SingleDateValueSingleSummaryRecord | UnparsedRawPersonalData"]:
        """
        从“单日期数值单项总结”原始一行文本中抽取 1~N 条(指标名称, 数值, 单位, 状态描述)记录。

        支持常见格式（示例）：
        - 7/21锻炼时长17分钟偏低
        - 4/23入睡时间01:40偏晚
        - 8月7日锻炼时长15分钟偏低, 活动热量213千卡偏低（后段继承日期）
        """
        parsed = _parse_single_date_value_summary_line(
            raw_line,
            指标名称=指标名称,
            数值类型=数值类型,
            单位=单位,
        )
        return [parsed]


# ========= 单日期文本总结 =========
@dataclass(frozen=True)
class SingleDateTextSummaryCore:
    """
    单日期文本总结的“样式 core（metadata）”。

    设计保持与 `SingleValueCore` / `SingleDateValueSummaryCore` 一致：
    - core 用于承载“字段占位/元信息”（日期格式、指标名称、状态描述占位）
    - 一行个人数据可能包含多条(指标名称, 状态描述)记录：由 record 的列表字段承载

    注意：
    - 出于与你现有 style/校验规则的一致性，这里沿用字段名 `时间` 表示“日期占位”（如 "Date (格式: MM/DD)"）。
    """

    指标名称: str
    时间: str
    状态描述: str  # 规则里要求严格为 "String"


@dataclass(frozen=True)
class SingleDateTextSummaryRecord(PersonalDataPatternBase):
    核心字段: SingleDateTextSummaryCore
    # 一行个人数据中可能包含多条“单日期文本总结”：用列表承载“明细”
    日期列表: list[str] = field(default_factory=list)
    指标名称列表: list[str] = field(default_factory=list)
    状态描述列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: SingleDateTextSummaryCore,
        *,
        日期列表: Sequence[str] | None = None,
        指标名称列表: Sequence[str] | None = None,
        状态描述列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "单日期文本总结")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "日期列表", list(日期列表 or []))
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "状态描述列表", list(状态描述列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含明细列表），用于后续真正做数据解析/训练数据构造。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "日期列表": list(self.日期列表),
            "指标名称列表": list(self.指标名称列表),
            "状态描述列表": list(self.状态描述列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(len(self.日期列表), len(self.指标名称列表), len(self.状态描述列表))

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", core.指标名称),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(len(self.日期列表), len(self.指标名称列表), len(self.状态描述列表))
        rows: list[list[Any]] = []
        for i in range(n):
            d = self.日期列表[i] if i < len(self.日期列表) else ""
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            desc = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([d, nm, desc])
        if rows:
            lines.append("- 明细（日期/指标/状态描述）：")
            table = _fmt_rows_table(rows, headers=("日期", "指标", "状态描述"))
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
    ) -> list["SingleDateTextSummaryRecord | UnparsedRawPersonalData"]:
        """
        从“单日期文本总结”原始一行文本中抽取 1~N 条(指标名称, 状态描述)记录。

        支持常见格式（示例）：
        - 8/2 睡眠得分中等，睡眠质量良好
        - 8月2日 睡眠得分中等, 睡眠质量良好（中文/英文逗号分隔）
        - 8/2 睡眠质量良好（仅一条记录时可不使用逗号）
        """
        parsed = _parse_single_date_text_summary_line(raw_line, 指标名称=指标名称)
        return [parsed]


# ========= 无时间日期的文本总结 =========
@dataclass(frozen=True)
class NoTimestampTextSummaryCore:
    指标名称: str
    状态描述: str  # 规则里要求严格为 "String"


@dataclass(frozen=True)
class NoTimestampTextSummaryRecord(PersonalDataPatternBase):
    核心字段: NoTimestampTextSummaryCore

    # 一行个人数据中可能包含多条“无时间日期的文本总结”：用列表承载“明细”
    指标名称列表: list[str] = field(default_factory=list)
    状态描述列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: NoTimestampTextSummaryCore,
        *,
        指标名称列表: Sequence[str] | None = None,
        状态描述列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "无时间日期的文本总结")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "状态描述列表", list(状态描述列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含多条明细列表）。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "指标名称列表": list(self.指标名称列表),
            "状态描述列表": list(self.状态描述列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(len(self.指标名称列表), len(self.状态描述列表))

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", core.指标名称),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(len(self.指标名称列表), len(self.状态描述列表))
        rows: list[list[Any]] = []
        for i in range(n):
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            desc = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([nm, desc])
        if rows:
            lines.append("- 明细（指标/状态描述）：")
            table = _fmt_rows_table(rows, headers=("指标", "状态描述"))
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
    ) -> list["NoTimestampTextSummaryRecord | UnparsedRawPersonalData"]:
        """
        从“无时间日期的文本总结”原始一行文本中抽取 1~N 条(指标名称, 状态描述)记录。

        支持常见格式（示例）：
        - 入睡时间欠规律,睡眠得分中等，睡眠质量良好,深睡连续性偏低
        """
        parsed = _parse_no_timestamp_text_summary_line(raw_line, 指标名称=指标名称)
        return [parsed]


# ========= 无时间日期的数值总结 =========
@dataclass(frozen=True)
class NoDateValueSummaryCore:
    指标名称: str
    数值类型: ValueType
    单位: str
    状态描述: str  # 规则里要求严格为 "String"


@dataclass(frozen=True)
class NoDateValueSummaryRecord(PersonalDataPatternBase):
    核心字段: NoDateValueSummaryCore

    # 一行个人数据中可能包含多条“无时间日期的数值总结”：用列表承载“明细”
    指标名称列表: list[str] = field(default_factory=list)
    数值类型列表: list[ValueType] = field(default_factory=list)
    单位列表: list[str] = field(default_factory=list)
    数值列表: list[str] = field(default_factory=list)
    状态描述列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: NoDateValueSummaryCore,
        *,
        指标名称列表: Sequence[str] | None = None,
        数值类型列表: Sequence[ValueType] | None = None,
        单位列表: Sequence[str] | None = None,
        数值列表: Sequence[str] | None = None,
        状态描述列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "无时间日期的数值总结")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "数值类型列表", list(数值类型列表 or []))
        object.__setattr__(self, "单位列表", list(单位列表 or []))
        object.__setattr__(self, "数值列表", list(数值列表 or []))
        object.__setattr__(self, "状态描述列表", list(状态描述列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含多条明细列表），用于后续真正做数据解析/训练数据构造。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "指标名称列表": list(self.指标名称列表),
            "数值类型列表": list(self.数值类型列表),
            "单位列表": list(self.单位列表),
            "数值列表": list(self.数值列表),
            "状态描述列表": list(self.状态描述列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
            len(self.状态描述列表),
        )

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）"),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
            len(self.状态描述列表),
        )
        rows: list[list[Any]] = []
        for i in range(n):
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            vt = self.数值类型列表[i] if i < len(self.数值类型列表) else ""
            u = self.单位列表[i] if i < len(self.单位列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            desc = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([nm, vt, u, v, desc])
        if rows:
            lines.append("- 明细（指标/类型/单位/数值/状态）：")
            table = _fmt_rows_table(rows, headers=("指标", "类型", "单位", "数值", "状态"))
            lines.append(_indent_lines(table, 2))
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
        数值类型: ValueType | None = None,
        单位: str | None = None,
    ) -> list["NoDateValueSummaryRecord | UnparsedRawPersonalData"]:
        """
        从“无时间日期的数值总结”原始一行文本中抽取 1~N 条(指标名称, 数值, 单位, 状态描述)记录。

        支持常见格式（示例）：
        - 平均压力均值36分正常，最低压力均值31分正常，最高压力均值49分正常
        - 平均血氧98%正常，最低血氧98%正常，最高血氧99%正常
        """
        parsed = _parse_no_date_value_summary_line(
            raw_line,
            指标名称=指标名称,
            数值类型=数值类型,
            单位=单位,
        )
        return [parsed]


# ========= 单指标的明细汇总记录 =========
@dataclass(frozen=True)
class StatsCompositeDataItem:
    日期: str
    数值类型: ValueType
    单位: str


@dataclass(frozen=True)
class StatsCompositeSummaryItem:
    指标名称: str
    数值类型: ValueType
    单位: str
    状态描述: str  # 规则里要求严格为 "String"


@dataclass(frozen=True)
class StatsCompositeCore:
    指标名称: str
    数值类型: ValueType
    单位: str
    数据列表: list[StatsCompositeDataItem]
    统计汇总描述: list[StatsCompositeSummaryItem]


@dataclass(frozen=True)
class SingleMetricStatsRecord(PersonalDataPatternBase):
    核心字段: StatsCompositeCore

    # 一行个人数据中的“明细/汇总”真实内容：用列表承载（与 SingleMetricDetailRecord 的设计保持一致）
    日期列表: list[str] = field(default_factory=list)
    数值列表: list[str] = field(default_factory=list)
    统计指标名称列表: list[str] = field(default_factory=list)
    统计数值列表: list[str] = field(default_factory=list)
    统计状态描述列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: StatsCompositeCore,
        *,
        日期列表: Sequence[str] | None = None,
        数值列表: Sequence[str] | None = None,
        统计指标名称列表: Sequence[str] | None = None,
        统计数值列表: Sequence[str] | None = None,
        统计状态描述列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "单指标的明细汇总记录")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "日期列表", list(日期列表 or []))
        object.__setattr__(self, "数值列表", list(数值列表 or []))
        object.__setattr__(self, "统计指标名称列表", list(统计指标名称列表 or []))
        object.__setattr__(self, "统计数值列表", list(统计数值列表 or []))
        object.__setattr__(self, "统计状态描述列表", list(统计状态描述列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        """
        输出“完整一行”的结构化结果（包含明细/汇总列表），用于后续真正做数据解析/训练数据构造。
        """
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": {
                "指标名称": self.核心字段.指标名称,
                "数值类型": self.核心字段.数值类型,
                "单位": self.核心字段.单位,
                "数据列表": [x.__dict__.copy() for x in self.核心字段.数据列表],
                "统计汇总描述": [x.__dict__.copy() for x in self.核心字段.统计汇总描述],
            },
            "日期列表": list(self.日期列表),
            "数值列表": list(self.数值列表),
            "统计指标名称列表": list(self.统计指标名称列表),
            "统计数值列表": list(self.统计数值列表),
            "统计状态描述列表": list(self.统计状态描述列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(
            len(self.日期列表),
            len(self.数值列表),
            len(self.统计指标名称列表),
            len(self.统计数值列表),
            len(self.统计状态描述列表),
        )

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）"),
            _fmt_kv("明细条数", len(self.日期列表)),
            _fmt_kv("汇总条数", len(self.统计指标名称列表)),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        # 明细表
        detail_rows: list[list[Any]] = []
        n_d = max(len(self.日期列表), len(self.数值列表))
        for i in range(n_d):
            d = self.日期列表[i] if i < len(self.日期列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            detail_rows.append([d, v])
        if detail_rows:
            lines.append("- 明细（日期/数值）：")
            lines.append(_indent_lines(_fmt_rows_table(detail_rows, headers=("日期", "数值")), 2))

        # 汇总表
        sum_rows: list[list[Any]] = []
        n_s = max(len(self.统计指标名称列表), len(self.统计数值列表), len(self.统计状态描述列表))
        for i in range(n_s):
            nm = self.统计指标名称列表[i] if i < len(self.统计指标名称列表) else ""
            v = self.统计数值列表[i] if i < len(self.统计数值列表) else ""
            st = self.统计状态描述列表[i] if i < len(self.统计状态描述列表) else ""
            sum_rows.append([nm, v, st])
        if sum_rows:
            lines.append("- 汇总（统计项/数值/状态）：")
            lines.append(
                _indent_lines(_fmt_rows_table(sum_rows, headers=("统计项", "数值", "状态")), 2)
            )
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
        数值类型: ValueType | None = None,
        单位: str | None = None,
    ) -> list["SingleMetricStatsRecord | UnparsedRawPersonalData"]:
        """
        从“单指标的明细汇总记录”原始一行文本中抽取：
        - 明细：[(日期, 数值)] 1~N 条
        - 汇总：[(统计指标名称, 数值, 状态描述)] 1~N 条

        支持常见格式（示例）：
        - 锻炼时长：[2月10日71分钟, 2月11日64分钟, ...]，平均锻炼时长69分钟正常，最低锻炼时长64分钟正常...
        """
        parsed = _parse_stats_composite_line(
            raw_line,
            指标名称=指标名称,
            数值类型=数值类型,
            单位=单位,
        )
        return [parsed]


@dataclass(frozen=True)
class SingleDateValueMultiSummaryCore:
    """
    新版“单日期数值多项总结”的样式 core（与 `predict_personal_data.py` 的强校验保持一致）：
      {"指标名称","日期","数值类型","单位","状态描述"}
    """

    指标名称: str
    日期: str
    数值类型: ValueType
    单位: str
    状态描述: str  # "String" 或 "无"


@dataclass(frozen=True)
class SingleDateValueMultiSummaryRecord(PersonalDataPatternBase):
    """
    单日期数值多项总结（新版样式）：
    - **一行原始个人数据**往往包含“同一日期 + 多个指标片段”，用中文/英文逗号分隔。
    - 本项目在“原始行解析”阶段更希望 **一个数据类对象即可概括整行**：
      `日期列表/指标名称列表/数值列表/单位列表/状态描述列表` 中每个索引代表一个片段。

    例：
      8月7日血氧饱和度96%-96%, 平均血氧饱和度96%正常, 最高血氧饱和度96%正常
    将被解析为 **一个** `SingleDateValueMultiSummaryRecord`（列表字段包含多条记录）。
    """

    核心字段: SingleDateValueMultiSummaryCore

    # 真实内容：用列表承载（与其它 summary record 保持一致）
    日期列表: list[str] = field(default_factory=list)
    指标名称列表: list[str] = field(default_factory=list)
    数值类型列表: list[ValueType] = field(default_factory=list)
    单位列表: list[str] = field(default_factory=list)
    数值列表: list[str] = field(default_factory=list)
    状态描述列表: list[str] = field(default_factory=list)
    原始个人数据: str | None = None

    def __init__(
        self,
        核心字段: SingleDateValueMultiSummaryCore,
        *,
        日期列表: Sequence[str] | None = None,
        指标名称列表: Sequence[str] | None = None,
        数值类型列表: Sequence[ValueType] | None = None,
        单位列表: Sequence[str] | None = None,
        数值列表: Sequence[str] | None = None,
        状态描述列表: Sequence[str] | None = None,
        原始个人数据: str | None = None,
    ) -> None:
        object.__setattr__(self, "实体类型", "单日期数值多项总结")
        object.__setattr__(self, "核心字段", 核心字段)
        object.__setattr__(self, "日期列表", list(日期列表 or []))
        object.__setattr__(self, "指标名称列表", list(指标名称列表 or []))
        object.__setattr__(self, "数值类型列表", list(数值类型列表 or []))
        object.__setattr__(self, "单位列表", list(单位列表 or []))
        object.__setattr__(self, "数值列表", list(数值列表 or []))
        object.__setattr__(self, "状态描述列表", list(状态描述列表 or []))
        object.__setattr__(self, "原始个人数据", 原始个人数据)

    def to_full_item(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "实体类型": self.实体类型,
            "核心字段": self.核心字段.__dict__.copy(),
            "日期列表": list(self.日期列表),
            "指标名称列表": list(self.指标名称列表),
            "数值类型列表": list(self.数值类型列表),
            "单位列表": list(self.单位列表),
            "数值列表": list(self.数值列表),
            "状态描述列表": list(self.状态描述列表),
        }
        if self.原始个人数据 is not None:
            out["原始个人数据"] = self.原始个人数据
        return out

    @property
    def 记录条数(self) -> int:
        return max(
            len(self.日期列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
            len(self.状态描述列表),
        )

    def format_print(self) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}；状态占位={core.状态描述}）"),
            _fmt_kv("记录条数", self.记录条数),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据))

        n = max(
            len(self.日期列表),
            len(self.指标名称列表),
            len(self.数值类型列表),
            len(self.单位列表),
            len(self.数值列表),
            len(self.状态描述列表),
        )
        rows: list[list[Any]] = []
        for i in range(n):
            d = self.日期列表[i] if i < len(self.日期列表) else ""
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            vt = self.数值类型列表[i] if i < len(self.数值类型列表) else ""
            u = self.单位列表[i] if i < len(self.单位列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            st = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([d, nm, vt, u, v, st])
        if rows:
            lines.append("- 明细（日期/指标/类型/单位/数值/状态）：")
            lines.append(
                _indent_lines(_fmt_rows_table(rows, headers=("日期", "指标", "类型", "单位", "数值", "状态")), 2)
            )
        return "\n".join(lines)

    @classmethod
    def from_raw_personal_data(
        cls,
        raw_line: str,
        *,
        指标名称: str | None = None,
        数值类型: ValueType | None = None,
        单位: str | None = None,
    ) -> list["SingleDateValueMultiSummaryRecord | UnparsedRawPersonalData"]:
        """
        从“单日期数值多项总结（新样式）”原始一行文本中抽取记录。

        支持常见格式（示例）：
        - 8月7日血氧饱和度96%-96%,平均血氧饱和度96%正常, 最高血氧饱和度96%正常, 最低血氧饱和度96%正常
        - 8月8日压力15分-15分,平均压力15分正常, 最高压力15分正常, 最低压力15分正常

        返回约定：
        - **始终返回长度为 1 的列表**（成功时为 `[SingleDateValueMultiSummaryRecord]`）
        - 解析失败时返回 `[UnparsedRawPersonalData]`
        """
        parsed = _parse_single_date_multi_value_summary_line(
            raw_line,
            指标名称=指标名称,
            数值类型=数值类型,
            单位=单位,
        )
        if isinstance(parsed, UnparsedRawPersonalData):
            return [parsed]
        return [parsed]


PersonalDataPattern = Union[
    SingleMetricDetailRecord,
    PeriodValueSingleSummaryRecord,
    PeriodTextSummaryRecord,
    PeriodValueCompareRecord,
    PeriodValuemMultiSummaryRecord,
    SingleDateValueSingleSummaryRecord,
    SingleDateTextSummaryRecord,
    NoTimestampTextSummaryRecord,
    NoDateValueSummaryRecord,
    SingleMetricStatsRecord,
    SingleDateValueMultiSummaryRecord,
    UnparsedRawPersonalData,
]


# ========= 解析：单指标的明细记录（从原始一行文本抽取多条记录） =========
_SPLIT_RE = re.compile(r"[，,]\s*")

# 常见：YYYY/M/D HH:mm的{指标}为：{数值}{单位}
_SEG_RE_1 = re.compile(
    r"^\s*(?P<date>(?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2})\s+"
    r"(?P<time>\d{1,2}:\d{2}|\d{1,2}\s*(?:时|点)\s*\d{1,2}\s*(?:分)?)的"
    r"(?P<name>.+?)为[:：]\s*(?P<val>.+?)\s*$"
)

# 宽松兜底：YYYY/M/D [HH:mm] ...（不强依赖“的/为：”）
_SEG_RE_2 = re.compile(
    # 只扩展“带年份”的日期支持 . / - 分隔；不扩展 MM.DD 以避免 79.0 误判日期
    r"^\s*(?P<date>(?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*"
    r"(?P<time>\d{1,2}:\d{2}|\d{1,2}\s*(?:时|点)\s*\d{1,2}\s*(?:分)?)?\s*"
    r"(?P<rest>.+?)\s*$"
)

_FIRST_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_TIME_HHMM_RE = re.compile(r"\d{1,2}:\d{2}")

# 值类型/单位推断热路径正则（尽量预编译）
_SIMPLE_PACE_RE = re.compile(
    r"\s*(?P<num>[-+]?\d+(?:\.\d+)?)\s*(?P<tunit>小时|分钟|秒|分)\s*/\s*(?P<dunit>公里|千米|米|km|KM|Km)\s*"
)
_VALUE_RANGE_RE = re.compile(
    r"\s*([-+]?\d+(?:\.\d+)?)\s*(?P<u1>[^\d\s]+)?\s*[-~～—]\s*([-+]?\d+(?:\.\d+)?)\s*(?P<u2>[^\d\s]+)?\s*"
)
_SIMPLE_TIME_RE = re.compile(r"\s*(?P<num>[-+]?\d+(?:\.\d+)?)\s*(?P<u>小时|分钟|秒|毫秒)\s*")
_INT_RE = re.compile(r"[-+]?\d+")


def _recover_single_metric_detail_record(obj: Any) -> str:
    core = _safe_getattr(obj, "核心字段", None)
    metric = str(_safe_getattr(core, "指标名称", "") or "").strip()
    unit = str(_safe_getattr(core, "单位", "") or "").strip() or "无"

    ds = list(_safe_getattr(obj, "日期列表", []) or [])
    ts = list(_safe_getattr(obj, "时间列表", []) or [])
    vs = list(_safe_getattr(obj, "数值列表", []) or [])
    n = max(len(ds), len(ts), len(vs))
    segs: list[str] = []
    for i in range(n):
        d = str(ds[i] if i < len(ds) else "").strip()
        t = str(ts[i] if i < len(ts) else "").strip()
        v = str(vs[i] if i < len(vs) else "").strip()
        if not d and not v:
            continue
        v2 = v
        if v2 and (unit and unit != "无"):
            v2 = _attach_unit_to_value(v2, unit)
        if d:
            if _is_time_token(t):
                # 典型：YYYY/MM/DD HH:mm的指标为：数值单位
                segs.append(f"{d} {t}的{metric}为：{v2}".strip())
            else:
                # 没有时间时尽量保持可读
                segs.append(f"{d}的{metric}为：{v2}".strip())
        else:
            segs.append(f"{metric}为：{v2}".strip())
    return _join_raw_segments(segs) or str(metric or "").strip()


def _recover_period_value_single_summary_record(obj: Any) -> str:
    starts = list(_safe_getattr(obj, "开始日期列表", []) or [])
    ends = list(_safe_getattr(obj, "结束日期列表", []) or [])
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    vals = list(_safe_getattr(obj, "数值列表", []) or [])
    units = list(_safe_getattr(obj, "单位列表", []) or [])

    n = max(len(starts), len(ends), len(names), len(vals), len(units))
    segs: list[str] = []
    prev_range: tuple[str, str] | None = None
    for i in range(n):
        st = str(starts[i] if i < len(starts) else "").strip()
        ed = str(ends[i] if i < len(ends) else "").strip()
        nm = str(names[i] if i < len(names) else "").strip()
        v = str(vals[i] if i < len(vals) else "").strip()
        u = str(units[i] if i < len(units) else "").strip() or "无"
        if not (st or ed or nm or v):
            continue
        v2 = _attach_unit_to_value(v, u)
        this_range = (st, ed)
        if prev_range is not None and this_range == prev_range and (st or ed):
            # 模拟“后段继承日期范围”
            segs.append(f"{nm}为{v2}".strip())
        else:
            rng = _format_date_range(st, ed)
            if rng:
                segs.append(f"{rng}的{nm}为{v2}".strip())
            else:
                segs.append(f"{nm}为{v2}".strip())
        prev_range = this_range
    return _join_raw_segments(segs)


def _recover_period_text_summary_record(obj: Any) -> str:
    starts = list(_safe_getattr(obj, "开始日期列表", []) or [])
    ends = list(_safe_getattr(obj, "结束日期列表", []) or [])
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    descs = list(_safe_getattr(obj, "状态描述列表", []) or [])
    n = max(len(starts), len(ends), len(names), len(descs))
    segs: list[str] = []
    for i in range(n):
        st = str(starts[i] if i < len(starts) else "").strip()
        ed = str(ends[i] if i < len(ends) else "").strip()
        nm = str(names[i] if i < len(names) else "").strip()
        desc = str(descs[i] if i < len(descs) else "").strip()
        if not (st or nm or desc):
            continue
        # 常见原始写法：日期(或范围)+指标+状态描述（不强制加“的”）
        rng = _format_date_range(st, ed)
        if rng:
            segs.append(f"{rng}{nm}{desc}".strip())
        else:
            segs.append(f"{nm}{desc}".strip())
    return _join_raw_segments(segs)


def _recover_period_value_compare_record(obj: Any) -> str:
    r1s = list(_safe_getattr(obj, "日期范围1列表", []) or [])
    v1s = list(_safe_getattr(obj, "数值1列表", []) or [])
    r2s = list(_safe_getattr(obj, "日期范围2列表", []) or [])
    v2s = list(_safe_getattr(obj, "数值2列表", []) or [])
    diffs = list(_safe_getattr(obj, "差异数值列表", []) or [])
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    logics = list(_safe_getattr(obj, "对比逻辑类型列表", []) or [])

    n = max(len(r1s), len(v1s), len(r2s), len(v2s), len(diffs), len(names), len(logics))
    segs: list[str] = []
    for i in range(n):
        r1 = str(r1s[i] if i < len(r1s) else "").strip()
        r2 = str(r2s[i] if i < len(r2s) else "").strip()
        nm = str(names[i] if i < len(names) else "").strip()
        v1 = str(v1s[i] if i < len(v1s) else "").strip()
        v2 = str(v2s[i] if i < len(v2s) else "").strip()
        dv = str(diffs[i] if i < len(diffs) else "").strip()
        lg = str(logics[i] if i < len(logics) else "").strip()
        lg2 = "" if _is_missing_token(lg) else lg
        if not (r1 and r2 and nm and v1 and v2 and dv):
            continue
        segs.append(f"{r1}的{nm}为{v1}，{r2}的{nm}为{v2}，{lg2}{dv}".strip("，"))
    return _join_raw_segments(segs)


def _recover_period_value_multi_summary_record(obj: Any) -> str:
    starts = list(_safe_getattr(obj, "开始日期列表", []) or [])
    ends = list(_safe_getattr(obj, "结束日期列表", []) or [])
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    vals = list(_safe_getattr(obj, "数值列表", []) or [])
    units = list(_safe_getattr(obj, "单位列表", []) or [])
    sts = list(_safe_getattr(obj, "状态描述列表", []) or [])

    n = max(len(starts), len(ends), len(names), len(vals), len(units), len(sts))
    segs: list[str] = []
    prev_range: tuple[str, str] | None = None
    for i in range(n):
        st = str(starts[i] if i < len(starts) else "").strip()
        ed = str(ends[i] if i < len(ends) else "").strip()
        nm = str(names[i] if i < len(names) else "").strip()
        v = str(vals[i] if i < len(vals) else "").strip()
        u = str(units[i] if i < len(units) else "").strip() or "无"
        s = str(sts[i] if i < len(sts) else "").strip()
        s2 = "" if _is_missing_token(s) else s
        if not (st or ed or nm or v):
            continue
        v2 = _attach_unit_to_value(v, u)
        this_range = (st, ed)
        if prev_range is not None and this_range == prev_range and (st or ed):
            segs.append(f"{nm}{v2}{s2}".strip())
        else:
            rng = _format_date_range(st, ed)
            if rng:
                segs.append(f"{rng}{nm}{v2}{s2}".strip())
            else:
                segs.append(f"{nm}{v2}{s2}".strip())
        prev_range = this_range
    return _join_raw_segments(segs)


def _recover_single_date_value_single_summary_record(obj: Any) -> str:
    ds = list(_safe_getattr(obj, "日期列表", []) or [])
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    vals = list(_safe_getattr(obj, "数值列表", []) or [])
    units = list(_safe_getattr(obj, "单位列表", []) or [])
    sts = list(_safe_getattr(obj, "状态描述列表", []) or [])

    n = max(len(ds), len(names), len(vals), len(units), len(sts))
    segs: list[str] = []
    prev_d: str | None = None
    for i in range(n):
        d = str(ds[i] if i < len(ds) else "").strip()
        nm = str(names[i] if i < len(names) else "").strip()
        v = str(vals[i] if i < len(vals) else "").strip()
        u = str(units[i] if i < len(units) else "").strip() or "无"
        s = str(sts[i] if i < len(sts) else "").strip()
        s2 = "" if _is_missing_token(s) else s
        if not (d or nm or v):
            continue
        v2 = _attach_unit_to_value(v, u)
        if prev_d is not None and d and d == prev_d:
            segs.append(f"{nm}{v2}{s2}".strip())
        else:
            segs.append(f"{d}{nm}{v2}{s2}".strip())
        prev_d = d or prev_d
    return _join_raw_segments(segs)


def _recover_single_date_text_summary_record(obj: Any) -> str:
    ds = list(_safe_getattr(obj, "日期列表", []) or [])
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    descs = list(_safe_getattr(obj, "状态描述列表", []) or [])
    n = max(len(ds), len(names), len(descs))
    segs: list[str] = []
    prev_d: str | None = None
    for i in range(n):
        d = str(ds[i] if i < len(ds) else "").strip()
        nm = str(names[i] if i < len(names) else "").strip()
        desc = str(descs[i] if i < len(descs) else "").strip()
        if not (d or nm or desc):
            continue
        if prev_d is not None and d and d == prev_d:
            segs.append(f"{nm}{desc}".strip())
        else:
            # 保持常见写法：8/2 睡眠质量良好（允许无空格）
            segs.append(f"{d}{nm}{desc}".strip())
        prev_d = d or prev_d
    return _join_raw_segments(segs)


def _recover_no_timestamp_text_summary_record(obj: Any) -> str:
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    descs = list(_safe_getattr(obj, "状态描述列表", []) or [])
    n = max(len(names), len(descs))
    segs: list[str] = []
    for i in range(n):
        nm = str(names[i] if i < len(names) else "").strip()
        desc = str(descs[i] if i < len(descs) else "").strip()
        if not (nm or desc):
            continue
        # 常见写法：指标+状态；若 desc 为空，则只输出指标
        segs.append(f"{nm}{desc}".strip())
    return _join_raw_segments(segs)


def _recover_no_date_value_summary_record(obj: Any) -> str:
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    vals = list(_safe_getattr(obj, "数值列表", []) or [])
    units = list(_safe_getattr(obj, "单位列表", []) or [])
    sts = list(_safe_getattr(obj, "状态描述列表", []) or [])
    n = max(len(names), len(vals), len(units), len(sts))
    segs: list[str] = []
    for i in range(n):
        nm = str(names[i] if i < len(names) else "").strip()
        v = str(vals[i] if i < len(vals) else "").strip()
        u = str(units[i] if i < len(units) else "").strip() or "无"
        s = str(sts[i] if i < len(sts) else "").strip()
        s2 = "" if _is_missing_token(s) else s
        if not (nm or v):
            continue
        v2 = _attach_unit_to_value(v, u)
        segs.append(f"{nm}{v2}{s2}".strip())
    return _join_raw_segments(segs)


def _recover_single_metric_stats_record(obj: Any) -> str:
    core = _safe_getattr(obj, "核心字段", None)
    metric = str(_safe_getattr(core, "指标名称", "") or "").strip()

    ds = list(_safe_getattr(obj, "日期列表", []) or [])
    vs = list(_safe_getattr(obj, "数值列表", []) or [])
    items: list[str] = []
    n_d = max(len(ds), len(vs))
    for i in range(n_d):
        d = str(ds[i] if i < len(ds) else "").strip()
        v = str(vs[i] if i < len(vs) else "").strip()
        if not (d and v):
            continue
        items.append(f"{d}{v}".strip())

    # 汇总
    sn = list(_safe_getattr(obj, "统计指标名称列表", []) or [])
    sv = list(_safe_getattr(obj, "统计数值列表", []) or [])
    ss = list(_safe_getattr(obj, "统计状态描述列表", []) or [])
    sum_segs: list[str] = []
    n_s = max(len(sn), len(sv), len(ss))
    for i in range(n_s):
        nm = str(sn[i] if i < len(sn) else "").strip()
        v = str(sv[i] if i < len(sv) else "").strip()
        st = str(ss[i] if i < len(ss) else "").strip()
        st2 = "" if _is_missing_token(st) else st
        if not nm:
            continue
        if v:
            sum_segs.append(f"{nm}{v}{st2}".strip())
        else:
            sum_segs.append(f"{nm}{st2}".strip())

    head = metric or "统计复合"
    body = f"{head}：[{_join_raw_segments(items)}]" if items else f"{head}：[]"
    if sum_segs:
        body = body + "，" + _join_raw_segments(sum_segs)
    return body.strip()


def _recover_single_date_value_multi_summary_record(obj: Any) -> str:
    ds = list(_safe_getattr(obj, "日期列表", []) or [])
    names = list(_safe_getattr(obj, "指标名称列表", []) or [])
    vals = list(_safe_getattr(obj, "数值列表", []) or [])
    units = list(_safe_getattr(obj, "单位列表", []) or [])
    sts = list(_safe_getattr(obj, "状态描述列表", []) or [])

    n = max(len(ds), len(names), len(vals), len(units), len(sts))
    segs: list[str] = []
    prev_d: str | None = None
    for i in range(n):
        d = str(ds[i] if i < len(ds) else "").strip()
        nm = str(names[i] if i < len(names) else "").strip()
        v = str(vals[i] if i < len(vals) else "").strip()
        u = str(units[i] if i < len(units) else "").strip() or "无"
        st = str(sts[i] if i < len(sts) else "").strip()
        st2 = "" if (_is_missing_token(st) or st == "无") else st
        if not (d or nm or v):
            continue
        v2 = _attach_unit_to_value(v, u)
        if prev_d is not None and d and d == prev_d:
            segs.append(f"{nm}{v2}{st2}".strip())
        else:
            segs.append(f"{d}{nm}{v2}{st2}".strip())
        prev_d = d or prev_d
    return _join_raw_segments(segs)


_RECOVER_BY_ENTITY_TYPE: dict[str, Callable[[Any], str]] = {
    "单指标的明细记录": _recover_single_metric_detail_record,
    "周期数值单项总结": _recover_period_value_single_summary_record,
    "周期文本总结": _recover_period_text_summary_record,
    "周期数值对比记录": _recover_period_value_compare_record,
    "周期数值多项总结": _recover_period_value_multi_summary_record,
    "单日期数值单项总结": _recover_single_date_value_single_summary_record,
    "单日期文本总结": _recover_single_date_text_summary_record,
    "无时间日期的文本总结": _recover_no_timestamp_text_summary_record,
    "无时间日期的数值总结": _recover_no_date_value_summary_record,
    "单指标的明细汇总记录": _recover_single_metric_stats_record,
    "单日期数值多项总结": _recover_single_date_value_multi_summary_record,
}


def _has_time_unit(s: str) -> bool:
    """
    是否包含“时长单位”。

    注意：
    - “分”在中文里既可能表示“分钟(缩写)”，也可能表示“分数/分值”（如 36分）。
    - 这里把“分”单独出现时视为“分值”，避免把压力/得分等误判为 Duration。
    - 当出现“秒”时，“分”更可能表示分钟（如 7分42秒）。
    """
    if "小时" in s or "分钟" in s or "秒" in s:
        return True
    # 仅在与“秒”共同出现时，才把“分”视为分钟缩写
    if "分" in s and "秒" in s:
        return True
    return False


def _has_distance_unit(s: str) -> bool:
    return any(x in s for x in ("公里", "千米", "米", "km", "KM", "Km"))


@lru_cache(maxsize=8192)
def _infer_value_type_from_value_str_cached(s: str) -> ValueType:
    # 先判“时间点”格式（例如 06:28）
    if _TIME_HHMM_RE.fullmatch(s):
        return "Timestamp"

    # “时间/距离”的配速类比值（如 7.80分钟/公里、7分42秒/公里）：
    # - 这类表达本质是“时长（分子） + 每X（分母）”，展示/还原时更希望保留时长语义；
    # - 因此统一按 Duration 处理，避免后续 value/unit 拆分时丢失“分钟/秒”等信息。
    m_simple_pace = _SIMPLE_PACE_RE.fullmatch(s)
    if m_simple_pace:
        return "Duration"

    # 判定“数值范围”（FloatRange），例如：
    # - 96%-98%
    # - 15分-17分
    # - 22-26
    # 注意：要求整段形如 <num><unit>? <sep> <num><unit>?，避免误伤日期等。
    m_range = _VALUE_RANGE_RE.fullmatch(s)
    if m_range:
        return "FloatRange"

    # 基于“单位/分子分母”判定：避免把 "公里/小时" 误判为 Duration
    unit = _infer_unit_from_value_str(s)
    if "/" in unit:
        left, right = (x.strip() for x in unit.split("/", 1))
        left_has_time, right_has_time = _has_time_unit(left), _has_time_unit(right)
        left_has_dist, right_has_dist = _has_distance_unit(left), _has_distance_unit(right)

        # 时间/距离（配速） => Duration
        if left_has_time and right_has_dist:
            # 重要：如果 s 仍是“纯数字 + 单一时间单位/距离单位”，应当走 Float/Int（上面已提前匹配）
            # 这里命中通常意味着“复合时长/距离”（如 7分42秒/公里），保留 Duration。
            return "Duration"
        # 距离/时间（速度）、次数/时间（心率）、步数/时间（步频） => Float（或 Int），这里统一先 Float
        if right_has_time and not left_has_time:
            # 若是整数形式，返回 Int
            m_int = _FIRST_NUMBER_RE.search(s)
            if m_int and _INT_RE.fullmatch(m_int.group(0)):
                return "Int"
            return "Float"

    # 不含分母的“时长描述”
    #
    # 重要：像 "0.47小时" / "15分钟" 这类“纯数值 + 单一时间单位”的表达，
    # 语义上仍是“持续时长”，应归类为 Duration（而不是 Float/Int）。
    # 复合时长（如 "2小时49分钟"、"7分42秒"）也同样是 Duration。
    m_simple_time = _SIMPLE_TIME_RE.fullmatch(s)
    if m_simple_time:
        return "Duration"

    if _has_time_unit(unit) or _has_time_unit(s):
        # 这里更偏向“复合时长”，不是“时间点”
        return "Duration"

    # 纯整数
    m = _FIRST_NUMBER_RE.search(s)
    if not m:
        return "String"
    num = m.group(0)
    return "Int" if _INT_RE.fullmatch(num) else "Float"


def _infer_value_type_from_value_str(value_str: str) -> ValueType:
    s = (value_str or "").strip()
    return _infer_value_type_from_value_str_cached(s)


@lru_cache(maxsize=8192)
def _infer_unit_from_value_str_cached(s: str) -> str:
    """
    从“数值+单位”的片段里粗略推断单位。
    - 5.11千米 -> 千米
    - 311.00千卡 -> 千卡
    - 7分42秒/公里 -> 分42秒/公里（这类更适合把整段当成“数值字符串”，单位仅作粗推断）
    """
    # 时间点（Timestamp）没有单位：避免 "01:40" 被推断成 ":40"
    if _TIME_HHMM_RE.fullmatch(s):
        return "无"

    # 重要：毫秒是独立单位，不能因为包含“秒”而被误归一成“秒”
    # 例如："10毫秒" 若被推成 unit="秒"，后续展示阶段会按 "秒" 剥离尾缀，导致值变成 "10毫"。
    if "毫秒" in s:
        # 若是 "10毫秒/xxx" 这类分母表达，仍走下面的 "/" 分支推断更合适；
        # 否则这里直接返回 "毫秒"。
        if "/" not in s:
            return "毫秒"

    # 配速（时间/距离）单位规范化：
    # - 7.80分钟/公里 -> 每公里
    # - 7分42秒/公里 -> 每公里（避免推成 "分42秒/公里"）
    #
    # 说明：
    # - 这里仅返回“分母语义”（每X），不把“分钟/小时/秒”放进 unit 里；
    # - 分子时间单位信息由 value_str 自身（或后续 Duration 规范化）保留；
    # - 这样在表格里会呈现：数值=21分47秒，单位=每公里，语义清晰且不破坏时长格式。
    if "/" in s:
        parts = [p.strip() for p in s.split("/", 1)]
        if len(parts) == 2:
            left_raw, right_raw = parts[0], parts[1]
            # 毫秒也算时间单位，但不要把 "毫秒" 当成 "秒"（上面已单独处理无分母的情况）
            left_has_time = any(x in left_raw for x in ("小时", "分钟", "秒", "毫秒")) or ("分" in left_raw and "秒" in left_raw)
            right_has_dist = any(x in right_raw for x in ("公里", "千米", "米", "km", "KM", "Km"))
            if left_has_time and right_has_dist:
                denom = "公里" if any(x in right_raw for x in ("公里", "千米", "km", "KM", "Km")) else "米"
                return f"每{denom}"

    # 优先处理“范围值”的单位推断：例如 96%-98% / 15分-17分 / 22-26
    # 目标：避免把单位推成 "%-98%" 之类带数字的错误结果。
    m_range = _VALUE_RANGE_RE.fullmatch(s)
    if m_range:
        u1 = (m_range.group("u1") or "").strip()
        u2 = (m_range.group("u2") or "").strip()
        # 若两侧单位不一致，优先左侧；否则取非空者；均空则无
        unit = u1 or u2
        return unit if unit else "无"

    # 特殊处理：不含分母(/) 的“复合时长”字符串，例如 "2小时49分钟"、"7分42秒"。
    # 目标：单位只保留单位词本身，不把中间的数字带进来（避免 "小时49分钟" 这种错误）。
    #
    # 注意：“36分”这类更可能是“分值”，不应被视为“分钟”；因此仅在明确出现 秒/分钟/小时 时触发该分支。
    if "/" not in s:
        has_hour = "小时" in s
        has_min_word = "分钟" in s
        # "毫秒" 不应触发 has_sec，否则会被误归一成 "秒"
        has_ms = "毫秒" in s
        has_sec = ("秒" in s) and (not has_ms)
        has_min_abbr = ("分" in s) and (not has_min_word) and has_sec  # 仅在与“秒”共同出现时，才把“分”视为分钟缩写

        if has_hour or has_min_word or has_sec or has_min_abbr or has_ms:
            unit_parts: list[str] = []
            if has_hour:
                unit_parts.append("小时")
            if has_min_word or has_min_abbr:
                unit_parts.append("分钟")
            if has_ms:
                unit_parts.append("毫秒")
            if has_sec:
                unit_parts.append("秒")
            if unit_parts:
                return "".join(unit_parts)

    m = _FIRST_NUMBER_RE.search(s)
    if not m:
        return "无"
    unit = s[m.end() :].strip()
    return unit if unit else "无"


def _infer_unit_from_value_str(value_str: str) -> str:
    s = (value_str or "").strip()
    return _infer_unit_from_value_str_cached(s)


def _extract_name_and_value_from_rest(rest: str) -> tuple[str, str]:
    """
    从“rest”里尽量拆出 指标名称 与 数值字符串。
    这是兜底逻辑：优先匹配 “{name}为：{val}”/“{name}为:{val}”，否则用“遇到第一个数字”分割。
    """
    s = (rest or "").strip()
    # 常见：xxx为：yyy / xxx为 yyy / xxx是 yyy / xxx是：yyy
    # 说明：
    # - “明细记录”经常出现 “...的体重是61.1千克” 这种句式，如果不识别“是”，会把“体重是”当指标名
    # - 这里允许 “为/是” 后面可选 “:” / "："
    m = re.search(r"^(?P<name>.+?)(?:为|是)\s*[:：]?\s*(?P<val>.+?)$", s)
    if m:
        return m.group("name").strip(), m.group("val").strip()

    # 否则：用第一个数字出现位置切分
    m2 = _FIRST_NUMBER_RE.search(s)
    if not m2:
        return s, ""
    name = s[: m2.start()].strip()
    val = s[m2.start() :].strip()
    return name, val


def _parse_single_value_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
    数值类型: ValueType | None = None,
    单位: str | None = None,
    默认时间: str = "无",
) -> SingleMetricDetailRecord | UnparsedRawPersonalData:
    """
    将“单指标的明细记录”的原始一行文本解析为 `SingleMetricDetailRecord`（多条明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为单指标的明细记录")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    # 同一行里若同时存在“缺失年份 / 两位年份”与“四位年份”，优先用四位年份做上下文推断（避免 25/6/2 被误判成 25月06日）
    default_year: int | None = None
    m_y = re.search(r"(?P<y>\d{4})[\/\.-]\d{1,2}[\/\.-]\d{1,2}", raw) or re.search(r"(?P<y>\d{4})年\d{1,2}月\d{1,2}", raw)
    if m_y:
        try:
            default_year = int(m_y.group("y"))
        except Exception:
            default_year = None
    dates: list[str] = []
    times: list[str] = []
    values: list[str] = []

    inferred_name: str | None = 指标名称
    inferred_unit: str | None = 单位
    inferred_type: ValueType | None = 数值类型

    for seg in segments:
        m1 = _SEG_RE_1.match(seg)
        if m1:
            d = m1.group("date").strip()
            t = _normalize_time_cn_token(m1.group("time").strip())
            nm = m1.group("name").strip()
            val = m1.group("val").strip()
        else:
            m2 = _SEG_RE_2.match(seg)
            if not m2:
                # 这段无法识别，跳过；最终如果完全没解析到任何条目，再整体兜底
                continue
            d = str(m2.group("date") or "").strip()
            t0 = str(m2.group("time") or "").strip()
            t = _normalize_time_cn_token(t0) if t0 else 默认时间
            nm, val = _extract_name_and_value_from_rest(str(m2.group("rest") or ""))

        if not d:
            continue

        # 指标名称：如果行内不一致，优先保留首次推断的，并继续收集明细
        if inferred_name is None and nm:
            inferred_name = nm

        if inferred_type is None and val:
            inferred_type = _infer_value_type_from_value_str(val)

        if inferred_unit is None and val:
            inferred_unit = _infer_unit_from_value_str(val)

        # 规范化数值：剥离单位以避免重复
        #
        # 重要：对于 Duration，`_infer_unit_from_value_str()` 可能会把
        # "58分钟48秒" 推断为单位 "分钟秒"（用于类型识别/粗推断），但展示层希望：
        # - 复合时长：单位=无，数值保留原串（避免列名出现 "(分钟秒)"）
        # 因此这里把 `_normalize_value_and_unit()` 返回的 unit_norm 写回 inferred_unit。
        val_normalized = val
        if val and inferred_type and inferred_unit:
            # 若该行被归一为“unitless duration”，则数值保持原串（不做单位剥离）
            if inferred_type == "Duration" and inferred_unit == "无":
                val_normalized = val
            else:
                val_normalized, unit_norm = _normalize_value_and_unit(val, inferred_type, inferred_unit)
                if inferred_type == "Duration" and unit_norm == "无":
                    inferred_unit = "无"

        dates.append(_normalize_date_cn(d, default_year=default_year))
        times.append(t or 默认时间)
        values.append(val_normalized)

    if not dates and not times and not values:
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(日期/时间/数值)记录")

    core = SingleValueCore(
        日期="Date (格式: YYYY/MM/DD)",
        时间="Time (格式: HH:mm)",
        指标名称=inferred_name or "",
        数值类型=inferred_type or "String",
        单位=inferred_unit or "无",
    )
    return SingleMetricDetailRecord(
        核心字段=core,
        日期列表=dates,
        时间列表=times,
        数值列表=values,
        原始个人数据=raw,
    )


def _parse_single_value_line_multi(
    raw_line: str,
    *,
    默认时间: str = "无",
) -> list[SingleMetricDetailRecord] | UnparsedRawPersonalData:
    """
    将"单指标的明细记录"的原始一行文本解析为多个 `SingleMetricDetailRecord`（按指标名称分组）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。

    此函数会检测一行数据中是否包含多个不同的指标，如果包含，会为每个指标创建一个独立的记录。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为单指标的明细记录")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]

    # 同一行里若同时存在“缺失年份 / 两位年份”与“四位年份”，优先用四位年份做上下文推断（避免 25/6/2 被误判成 25月06日）
    default_year: int | None = None
    m_y = re.search(r"(?P<y>\d{4})[\/\.-]\d{1,2}[\/\.-]\d{1,2}", raw) or re.search(r"(?P<y>\d{4})年\d{1,2}月\d{1,2}", raw)
    if m_y:
        try:
            default_year = int(m_y.group("y"))
        except Exception:
            default_year = None
    
    # 按指标名称分组存储记录
    metric_groups: dict[str, dict[str, list[str]]] = {}
    
    for seg in segments:
        m1 = _SEG_RE_1.match(seg)
        if m1:
            d = m1.group("date").strip()
            t = _normalize_time_cn_token(m1.group("time").strip())
            nm = m1.group("name").strip()
            val = m1.group("val").strip()
        else:
            m2 = _SEG_RE_2.match(seg)
            if not m2:
                continue
            d = str(m2.group("date") or "").strip()
            t0 = str(m2.group("time") or "").strip()
            t = _normalize_time_cn_token(t0) if t0 else 默认时间
            nm, val = _extract_name_and_value_from_rest(str(m2.group("rest") or ""))

        if not d or not nm:
            continue

        # 规范化指标名称（去除前导"的"）
        nm_normalized = _strip_leading_de(nm)
        
        # 按指标名称分组
        if nm_normalized not in metric_groups:
            metric_groups[nm_normalized] = {
                "dates": [],
                "times": [],
                "values": [],
            }
        
        metric_groups[nm_normalized]["dates"].append(_normalize_date_cn(d, default_year=default_year))
        metric_groups[nm_normalized]["times"].append(t or 默认时间)
        metric_groups[nm_normalized]["values"].append(val)

    if not metric_groups:
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(日期/时间/数值)记录")

        # 为每个指标创建一个 SingleMetricDetailRecord
    results: list[SingleMetricDetailRecord] = []
    for metric_name, group_data in metric_groups.items():
        dates = group_data["dates"]
        times = group_data["times"]
        raw_values = group_data["values"]
        
        # 推断数值类型和单位（从第一个值推断）
        inferred_type: ValueType = "String"
        inferred_unit: str = "无"
        if raw_values:
            inferred_type = _infer_value_type_from_value_str(raw_values[0])
            inferred_unit = _infer_unit_from_value_str(raw_values[0])
            # 对 Duration：让“复合时长 => 单位=无”的归一结果写回核心字段，
            # 避免列名出现 "(分钟秒)/(小时分钟秒)" 等推断单位。
            if inferred_type == "Duration":
                _v0, unit_norm0 = _normalize_value_and_unit(raw_values[0], inferred_type, inferred_unit)
                if unit_norm0 == "无":
                    inferred_unit = "无"
        
        # 规范化数值：剥离单位以避免重复
        values = []
        for val in raw_values:
            if not val:
                values.append(val)
                continue
            # 若该指标被归一为“unitless duration”，则数值保持原串（不做单位剥离），
            # 否则容易出现 unit=无 但 value 变成 "29" 这种语义丢失。
            if inferred_type == "Duration" and inferred_unit == "无":
                values.append(str(val).strip())
                continue
            if inferred_type and inferred_unit:
                val_normalized, _unit_norm = _normalize_value_and_unit(val, inferred_type, inferred_unit)
                values.append(val_normalized)
            else:
                values.append(val)
        
        core = SingleValueCore(
            日期="Date (格式: YYYY/MM/DD)",
            时间="Time (格式: HH:mm)",
            指标名称=metric_name,
            数值类型=inferred_type,
            单位=inferred_unit,
        )
        results.append(
            SingleMetricDetailRecord(
                核心字段=core,
                日期列表=dates,
                时间列表=times,
                数值列表=values,
                原始个人数据=raw,  # 保留原始数据以便追溯
            )
        )
    
    return results


# ========= 解析：周期数值单项总结（从原始一行文本抽取 1~N 条汇总记录） =========
# 形如：{开始日期}[到｜至｜~]{结束日期}的{指标名称}为{数值}{单位}
_PERIOD_SUMMARY_SEG_RE_1 = re.compile(
    # 兼容 “2025/2/1日到2025年6月16日...” 这种混用口径：slash 日期后可能跟一个“日”
    r"^\s*(?P<start>(?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}(?:日)?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*"
    r"(?P<sep>到|至|~|～|-|—)\s*"
    r"(?P<end>(?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}(?:日)?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*"
    # 注意：不能把时间值里的 ":"（如 23:20）当作 “name/val” 分隔符；因此对 ":"/ "：" 做负向数字前缀约束。
    r"(?:的)?(?P<name>.+?)\s*(?:为|(?<!\d)[:：])\s*(?P<val>.+?)\s*$"
)

# 无 “为/：” 分隔符版本：{开始日期}[到｜至｜~]{结束日期}(的)?{rest}
# 后续会用 `_extract_name_and_value_from_rest()` 从 rest 里按“第一个数字”切分 name/val。
_PERIOD_SUMMARY_SEG_RE_3 = re.compile(
    # 兼容 “2025/2/1日到...”：slash 日期后可带“日”
    r"^\s*(?P<start>(?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}(?:日)?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*"
    r"(?P<sep>到|至|~|～|-|—)\s*"
    r"(?P<end>(?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}(?:日)?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*"
    r"(?:的)?\s*(?P<rest>.+?)\s*$"
)

# 允许“继承日期范围”的后半段：{指标名称}为{数值}{单位}
_PERIOD_SUMMARY_SEG_RE_2 = re.compile(r"^\s*(?P<name>.+?)\s*(?:为|(?<!\d)[:：])\s*(?P<val>.+?)\s*$")


def _parse_period_summary_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
    数值类型: ValueType | None = None,
    单位: str | None = None,
) -> PeriodValueSingleSummaryRecord | UnparsedRawPersonalData:
    """
    将“周期数值单项总结”的原始一行文本解析为 `PeriodValueSingleSummaryRecord`（多条明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为周期数值单项总结")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    starts: list[str] = []
    ends: list[str] = []
    names: list[str] = []
    types: list[ValueType] = []
    units: list[str] = []
    values: list[str] = []

    inferred_name: str | None = 指标名称
    inferred_type: ValueType | None = 数值类型
    inferred_unit: str | None = 单位
    inferred_start: str | None = None
    inferred_end: str | None = None

    for seg in segments:
        m1 = _PERIOD_SUMMARY_SEG_RE_1.match(seg)
        if m1:
            st = m1.group("start").strip()
            ed = m1.group("end").strip()
            nm = m1.group("name").strip()
            val_raw = m1.group("val").strip()
            inferred_start, inferred_end = st, ed
        else:
            # 1) 先尝试“无为/：分隔符”的周期汇总头段
            m3 = _PERIOD_SUMMARY_SEG_RE_3.match(seg)
            if m3:
                st = str(m3.group("start") or "").strip()
                ed = str(m3.group("end") or "").strip()
                rest = str(m3.group("rest") or "").strip()
                nm, val_raw = _extract_name_and_value_from_rest(rest)
                inferred_start, inferred_end = st, ed
            else:
                # 2) 再尝试“继承日期范围”的后半段：{指标名称}为{数值}{单位}
                m2 = _PERIOD_SUMMARY_SEG_RE_2.match(seg)
                if not m2:
                    continue
                if not inferred_start or not inferred_end:
                    # 没有日期范围可继承：跳过；最终若完全无记录再兜底
                    continue
                st, ed = inferred_start, inferred_end
                nm = m2.group("name").strip()
                val_raw = m2.group("val").strip()

        if not st or not ed:
            continue
        st2 = _normalize_date_cn(st)
        m_y = re.fullmatch(r"(?P<y>\d{4})年\d{2}月\d{2}日", st2)
        y_default: int | None = int(m_y.group("y")) if m_y else None
        ed2 = _normalize_date_cn(ed, default_year=y_default)

        if inferred_name is None and nm:
            inferred_name = nm
        # 推断类型/单位，并做“数值+单位”规范化：
        # - Int/Float：数值列表存纯数字（单位进单位列表）
        # - FloatRange：数值列表存区间数字（单位进单位列表）
        # - Timestamp：单位固定“无”
        # - Duration：单一单位可剥离成纯数字；复合时长保留原串，单位置“无”（避免重复/丢信息）
        t = _infer_value_type_from_value_str(val_raw) if val_raw else (inferred_type or "String")
        u = _infer_unit_from_value_str(val_raw) if val_raw else (inferred_unit or "无")
        val_norm, unit_norm = _normalize_value_and_unit(val_raw, t, u)

        if inferred_type is None and val_raw:
            inferred_type = t
        if inferred_unit is None and val_raw:
            inferred_unit = unit_norm

        starts.append(st2)
        ends.append(ed2)
        names.append(nm)
        types.append(t)
        units.append(unit_norm)
        values.append(val_norm)

    if not (starts or ends or names or values):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(开始日期/结束日期/指标名称/数值)记录")

    core = PeriodSummaryCore(
        开始日期="Date (格式: YYYY/MM/DD)",
        结束日期="Date (格式: YYYY/MM/DD)",
        指标名称=inferred_name or (names[0] if names else ""),
        数值类型=inferred_type or (types[0] if types else "String"),
        单位=inferred_unit or (units[0] if units else "无"),
    )
    return PeriodValueSingleSummaryRecord(
        核心字段=core,
        开始日期列表=starts,
        结束日期列表=ends,
        指标名称列表=names,
        数值类型列表=types,
        单位列表=units,
        数值列表=values,
        原始个人数据=raw,
    )


# ========= 解析：周期文本总结（从原始一行文本抽取 1~N 条总结记录） =========
_PERIOD_TEXT_SUMMARY_SEG_RE = re.compile(
    r"^\s*(?P<start>\d{4}/\d{1,2}/\d{1,2}(?:日)?|\d{2}/\d{1,2}/\d{1,2}(?:日)?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*"
    r"(?:(?P<sep>到|至|~|～|-|—)\s*"
    r"(?P<end>\d{4}/\d{1,2}/\d{1,2}(?:日)?|\d{2}/\d{1,2}/\d{1,2}(?:日)?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*)?"
    r"(?:的)?\s*(?P<rest>.+?)\s*$"
)


def _extract_metric_and_status_from_text(rest: str) -> tuple[str, str]:
    """
    从“周期文本总结”的 rest 文本里尽量拆出 指标名称 与 状态描述。

    优先规则：如果以常见状态描述词结尾（偏低/正常/...），则把结尾状态描述词作为 状态描述，其余作为 指标名称。
    兜底：用空白分割，取最后一段作为状态描述；仍不行则把整段当作指标名称，状态描述置空。
    """
    s = (rest or "").strip()
    if not s:
        return "", ""

    # 先处理“没有查询/没有查询到/暂无数据/无数据/查无...”这类“状态词在前”的句式：
    #   - 7/21~7/26没有查询到减脂数据
    #   - 6/20~8/26暂无生理健康数据
    # 期望拆为：
    #   指标名称=减脂数据, 状态描述=没有查询到
    #   指标名称=生理健康数据, 状态描述=暂无
    #
    # 同时兼容“状态词在后”的写法：
    #   - 减脂数据没有查询到
    _no_result_prefix = (
        "没有查询到",
        "没有查询",
        "未查询到",
        "未查询",
        "查无",
        "未找到",
        "没有数据",
        "无数据",
        "暂无数据",
        "暂无",
    )
    s0 = s.strip(" ：:，,。；;")
    for st in _no_result_prefix:
        if s0.startswith(st) and len(s0) > len(st):
            metric = _strip_leading_de(s0[len(st) :].strip(" ：:，,。；;"))
            status = st
            if metric:
                return metric, status
        if s0.endswith(st) and len(s0) > len(st):
            metric = _strip_leading_de(s0[: -len(st)].strip(" ：:，,。；;"))
            status = st
            if metric:
                return metric, status

    # 先处理“...是...”结构（例如：占比最高的情绪是愉悦）
    m_is = re.match(r"^(?P<metric>.+?)是(?P<status>.+?)$", s)
    if m_is:
        metric = _strip_leading_de((m_is.group("metric") or "").strip(" ：:，,"))
        status = (m_is.group("status") or "").strip(" ：:，,")
        if metric and status:
            return metric, status

    # 常见状态描述/评价词（按“更长优先”排序）
    #
    # 说明：
    # - `SingleDateTextSummaryRecord` 的样例里，状态描述经常是后缀形式：
    #   - 入睡时间晚 / 起床时间早
    #   - 夜间睡眠时长偏长
    #   - BMI偏重 / 脂肪率标准
    # - 若后缀表不覆盖这些词，会导致“指标名称=整段、状态描述=整段”的重复输出。
    # - 这里通过补齐后缀词表解决，并保持“长词优先”以避免把 “偏晚” 误切成 “偏 + 晚”。
    status_suffixes = sorted(
        {
            "欠规律",
            "不规律",
            # “待改善/待提升”类（文本总结里很常见，例如：睡眠质量待改善）
            "有待改善",
            "需要改善",
            "需改善",
            "待改善",
            "有待提升",
            "需要提升",
            "需提升",
            "待提升",
            "偏晚",
            "晚",
            "偏早",
            "早",
            "偏低",
            "偏高",
            # 允许更短的“高/低”结尾（例如：睡眠得分低）
            "低",
            "高",
            "偏少",
            "偏多",
            "偏长",
            "偏短",
            "过少",
            "过多",
            "过长",
            "过短",
            "正常",
            "中等",
            "一般",
            "良好",
            "标准",
            "较低",
            "较高",
            "不足",
            "过低",
            "过高",
            "不佳",
            "较差",
            # 体重/体脂等“状态词”
            "偏重",
            "超重",
            "肥胖",
            "偏瘦",
            "偏胖",
            "偏轻",
        },
        key=len,
        reverse=True,
    )
    for suf in status_suffixes:
        if s.endswith(suf) and len(s) > len(suf):
            metric = _strip_leading_de(s[: -len(suf)].strip(" ：:，,"))
            status = suf
            return metric, status

    parts = [p for p in re.split(r"\s+", s) if p]
    if len(parts) >= 2:
        metric = _strip_leading_de(" ".join(parts[:-1]).strip())
        status = parts[-1].strip()
        return metric, status

    return _strip_leading_de(s), ""


def _parse_period_text_summary_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
) -> PeriodTextSummaryRecord | UnparsedRawPersonalData:
    """
    将“周期文本总结”的原始一行文本解析为 `PeriodTextSummaryRecord`（多条明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为周期文本总结")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    starts: list[str] = []
    ends: list[str] = []
    names: list[str] = []
    descs: list[str] = []

    inferred_name: str | None = 指标名称
    last_range: tuple[str, str] | None = None

    def _append_one(st2: str, ed2: str, rest: str) -> None:
        nonlocal inferred_name
        r = (rest or "").strip()
        if not (st2 and ed2 and r):
            return
        nm, desc = _extract_metric_and_status_from_text(r)
        if inferred_name is None and nm:
            inferred_name = nm
        starts.append(st2)
        ends.append(ed2)
        names.append(nm)
        descs.append(desc if desc else r)

    for seg in segments:
        m = _PERIOD_TEXT_SUMMARY_SEG_RE.match(seg)
        if m:
            st = str(m.group("start") or "").strip()
            ed = str(m.group("end") or "").strip() or st  # 单日期：结束日期=开始日期
            rest = str(m.group("rest") or "").strip()
            if not st or not rest:
                continue

            st2 = _normalize_date_cn(st)
            m_y = re.fullmatch(r"(?P<y>\d{4})年\d{2}月\d{2}日", st2)
            y_default: int | None = int(m_y.group("y")) if m_y else None
            ed2 = _normalize_date_cn(ed, default_year=y_default)
            last_range = (st2, ed2)

            _append_one(st2, ed2, rest)
            continue

        # 兼容“逗号后段继承上一段日期范围”的写法，例如：
        # - 4/1~4/22 睡眠得分中等，睡眠质量良好
        # - 2024/1/1~2024/12/31锻炼时长偏低, 运动频率偏低
        if last_range is not None:
            st2, ed2 = last_range
            _append_one(st2, ed2, seg)

    if not (starts or ends or names or descs):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(开始日期/结束日期/指标名称/状态描述)记录")

    core = PeriodTextSummaryCore(
        开始日期="Date (格式: YYYY/MM/DD)",
        结束日期="Date (格式: YYYY/MM/DD)",
        指标名称=inferred_name or (names[0] if names else ""),
        状态描述="String",
    )
    return PeriodTextSummaryRecord(
        核心字段=core,
        开始日期列表=starts,
        结束日期列表=ends,
        指标名称列表=names,
        状态描述列表=descs,
        原始个人数据=raw,
    )


# ========= 解析：周期数值对比记录（从原始一行文本抽取 1~N 组对比记录） =========
# 一组对比常见为三段，用逗号分隔：
#   {日期范围1}的{指标名称}为{数值1}，{日期范围2}的{指标名称}为{数值2}，{少/多/...}{差异数值}
_PVC_DATE_OR_RANGE_RE = (
    r"(?:"
    r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}"
    r"|\d{4}年\d{1,2}月\d{1,2}日"
    r"|\d{1,2}/\d{1,2}"
    r"|\d{1,2}月\d{1,2}(?:日)?"
    r")"
    r"(?:\s*(?:到|至|~|～|-|—)\s*"
    r"(?:"
    r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}"
    r"|\d{4}年\d{1,2}月\d{1,2}日"
    r"|\d{1,2}/\d{1,2}"
    r"|\d{1,2}月\d{1,2}(?:日)?"
    r"))?"
)

# 兼容真实数据里“无的”的写法，例如：
# - 2025/2/27~2025/3/5平均浅睡比例为49.7
# - 2025/3/6的平均浅睡比例为56
_PVC_CLAUSE_RE = re.compile(
    rf"^\s*(?P<range>{_PVC_DATE_OR_RANGE_RE})\s*(?:的)?\s*(?P<name>.+?)(?:为|[:：])\s*(?P<val>.+?)\s*$"
)
_PVC_DIFF_RE = re.compile(
    r"^\s*(?P<logic>少|多|高|低|增加|减少|提升|下降|降低|升高|早|晚|提前|延后|推迟|延迟)?\s*(?P<diff>.+?)\s*$"
)


def _parse_period_value_compare_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
    数值类型: ValueType | None = None,
    单位: str | None = None,
    对比逻辑类型: str | None = None,
    差异数值类型: ValueType | None = None,
) -> PeriodValueCompareRecord | UnparsedRawPersonalData:
    """
    将“周期数值对比记录”的原始一行文本解析为 `PeriodValueCompareRecord`（多组明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为周期数值对比记录")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]

    range1s: list[str] = []
    val1s: list[str] = []
    range2s: list[str] = []
    val2s: list[str] = []
    diff_vals: list[str] = []
    names: list[str] = []
    types: list[ValueType] = []
    units: list[str] = []
    logics: list[str] = []
    diff_types: list[ValueType] = []

    inferred_name: str | None = 指标名称
    inferred_type: ValueType | None = 数值类型
    inferred_unit: str | None = 单位
    inferred_logic: str | None = 对比逻辑类型
    inferred_diff_type: ValueType | None = 差异数值类型

    def _clean_tail(s: str) -> str:
        return (s or "").strip().strip("。；;")

    # 以 3 段为一组进行解析（允许一行有多组对比）
    i = 0
    while i + 2 < len(segments):
        c1, c2, c3 = segments[i], segments[i + 1], segments[i + 2]
        m1 = _PVC_CLAUSE_RE.match(c1)
        m2 = _PVC_CLAUSE_RE.match(c2)
        m3 = _PVC_DIFF_RE.match(c3)
        if not (m1 and m2 and m3):
            break

        r1 = _normalize_date_or_range_cn(_clean_tail(m1.group("range")))
        n1 = _strip_leading_de(_clean_tail(m1.group("name")))
        v1 = _clean_tail(m1.group("val"))
        r2 = _normalize_date_or_range_cn(_clean_tail(m2.group("range")))
        n2 = _strip_leading_de(_clean_tail(m2.group("name")))
        v2 = _clean_tail(m2.group("val"))

        logic = _clean_tail(str(m3.group("logic") or "")) or "无"
        diff_v = _clean_tail(str(m3.group("diff") or ""))

        if not (r1 and v1 and r2 and v2 and diff_v):
            break

        # 指标名称：优先取第一个子句的；第二个若不同，仍按 n1 记（但 names 列表里保留 n1）
        if inferred_name is None and n1:
            inferred_name = n1

        # 数值类型/单位：从 v1 推断（通常 v1/v2 单位一致）
        t = _infer_value_type_from_value_str(v1) if v1 else (inferred_type or "String")
        u = _infer_unit_from_value_str(v1) if v1 else (inferred_unit or "无")
        dt = _infer_value_type_from_value_str(diff_v) if diff_v else (inferred_diff_type or "String")

        if inferred_type is None:
            inferred_type = t
        if inferred_unit is None:
            inferred_unit = u
        if inferred_logic is None and logic and logic != "无":
            inferred_logic = logic
        if inferred_diff_type is None:
            inferred_diff_type = dt

        range1s.append(r1)
        val1s.append(v1)
        range2s.append(r2)
        val2s.append(v2)
        diff_vals.append(diff_v)
        names.append(n1 or n2 or (inferred_name or ""))
        types.append(t)
        units.append(u)
        logics.append(logic)
        diff_types.append(dt)

        i += 3

    if not (range1s and val1s and range2s and val2s and diff_vals):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(日期范围1/数值1/日期范围2/数值2/差异数值)记录")

    core = PeriodValueCompareCore(
        日期范围1="DateRange (格式: YYYY/MM/DD~YYYY/MM/DD)",
        日期范围2="DateRange (格式: YYYY/MM/DD~YYYY/MM/DD)",
        指标名称=inferred_name or (names[0] if names else ""),
        数值类型=inferred_type or (types[0] if types else "String"),
        单位=inferred_unit or (units[0] if units else "无"),
        对比逻辑类型=inferred_logic or "String",
        差异数值类型=inferred_diff_type or (diff_types[0] if diff_types else "String"),
    )
    return PeriodValueCompareRecord(
        核心字段=core,
        日期范围1列表=range1s,
        数值1列表=val1s,
        日期范围2列表=range2s,
        数值2列表=val2s,
        差异数值列表=diff_vals,
        指标名称列表=names,
        数值类型列表=types,
        单位列表=units,
        对比逻辑类型列表=logics,
        差异数值类型列表=diff_types,
        原始个人数据=raw,
    )


# ========= 解析：周期数值多项总结（从原始一行文本抽取 1~N 条总结记录） =========
# 形如：
#   {日期}[~到至-]{日期} {指标名称}{数值}{状态描述}
#   8月7日锻炼时长15分钟偏低
#   4/18锻炼时长2小时49分钟正常
_PERIOD_VALUE_SUMMARY_SEG_RE = re.compile(
    r"^\s*(?P<start>\d{4}/\d{1,2}/\d{1,2}(?:日)?|\d{2}/\d{1,2}/\d{1,2}(?:日)?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*"
    r"(?:(?P<sep>到|至|~|～|-|—)\s*"
    r"(?P<end>\d{4}/\d{1,2}/\d{1,2}(?:日)?|\d{2}/\d{1,2}/\d{1,2}(?:日)?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}(?:日)?|\d{1,2}月\d{1,2}日)\s*)?"
    r"(?:的)?\s*(?P<rest>.+?)\s*$"
)


def _extract_metric_value_status_from_value_summary_rest(rest: str) -> tuple[str, str, str]:
    """
    从“周期数值多项总结”的 rest 文本里尽量拆出：
    - 指标名称（metric）
    - 数值字符串（value）
    - 状态描述（status，可为空）

    规则：
    - 优先用“第一个数字出现位置”来切分 指标名称 与 (数值+状态描述)
    - 再尝试把尾部常见状态描述词剥离出来（偏低/正常/...）
    """
    s = (rest or "").strip()
    if not s:
        return "", "", ""

    m_num = _FIRST_NUMBER_RE.search(s)
    if not m_num:
        # 没有数值：整段当指标名称
        return s, "", ""

    metric = s[: m_num.start()].strip(" ：:，,")
    metric = _strip_leading_de(metric)
    tail = s[m_num.start() :].strip(" ：:，,")
    if not tail:
        return metric, "", ""

    # 常见状态描述/评价词（按“更长优先”排序）
    status_suffixes = sorted(
        {
            "欠规律",
            "不规律",
            "偏晚",
            "偏早",
            "偏长",
            "偏短",
            "偏低",
            "偏高",
            "偏少",
            "偏多",
            "偏弱",
            "偏强",
            "过少",
            "过多",
            "过长",
            "过短",
            "正常",
            "中等",
            "一般",
            "良好",
            "达标",
            "优秀",
            "较低",
            "较高",
            "不足",
            "过低",
            "过高",
            "不佳",
            "较差",
        },
        key=len,
        reverse=True,
    )
    for suf in status_suffixes:
        if tail.endswith(suf) and len(tail) > len(suf):
            value = tail[: -len(suf)].strip(" ：:，,")
            status = suf
            return metric, value, status

    # 没有识别到状态描述词：全部视为 value
    return metric, tail, ""


def _parse_period_value_summary_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
    数值类型: ValueType | None = None,
    单位: str | None = None,
) -> PeriodValuemMultiSummaryRecord | UnparsedRawPersonalData:
    """
    将“周期数值多项总结”的原始一行文本解析为 `PeriodValuemMultiSummaryRecord`（多条明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为周期数值多项总结")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    starts: list[str] = []
    ends: list[str] = []
    names: list[str] = []
    types: list[ValueType] = []
    units: list[str] = []
    values: list[str] = []
    statuses: list[str] = []

    inferred_name: str | None = 指标名称
    inferred_type: ValueType | None = 数值类型
    inferred_unit: str | None = 单位
    inferred_start: str | None = None
    inferred_end: str | None = None

    for seg in segments:
        m = _PERIOD_VALUE_SUMMARY_SEG_RE.match(seg)
        if m:
            st = str(m.group("start") or "").strip()
            ed = str(m.group("end") or "").strip() or st  # 单日期：结束日期=开始日期
            rest = str(m.group("rest") or "").strip()
            if not st or not rest:
                continue
            inferred_start, inferred_end = st, ed
        else:
            # 允许后续片段不带日期：继承首段日期范围（与 PeriodValueSingleSummaryRecord 保持一致）
            if not inferred_start:
                continue
            st = inferred_start
            ed = inferred_end or inferred_start
            rest = str(seg or "").strip()
            if not rest:
                continue

        nm, val, status = _extract_metric_value_status_from_value_summary_rest(rest)
        if inferred_name is None and nm:
            inferred_name = nm
        if inferred_type is None and val:
            inferred_type = _infer_value_type_from_value_str(val)
        if inferred_unit is None and val:
            inferred_unit = _infer_unit_from_value_str(val)

        st2 = _normalize_date_cn(st)
        m_y = re.fullmatch(r"(?P<y>\d{4})年\d{2}月\d{2}日", st2)
        y_default: int | None = int(m_y.group("y")) if m_y else None
        ed2 = _normalize_date_cn(ed, default_year=y_default)
        starts.append(st2)
        ends.append(ed2)
        names.append(nm)
        t = _infer_value_type_from_value_str(val) if val else (inferred_type or "String")
        u = _infer_unit_from_value_str(val) if val else (inferred_unit or "无")
        val2, u2 = _normalize_value_and_unit(val, t, u)
        types.append(t)
        units.append(u2)
        values.append(val2)
        statuses.append(status if status else "无")

    if not (starts or ends or names or values):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(开始日期/结束日期/指标名称/数值)记录")

    core = PeriodValueSummaryCore(
        开始日期="Date (格式: YYYY/MM/DD)",
        结束日期="Date (格式: YYYY/MM/DD)",
        指标名称=inferred_name or (names[0] if names else ""),
        数值类型=inferred_type or (types[0] if types else "String"),
        单位=inferred_unit or (units[0] if units else "无"),
        状态描述="String",
    )
    return PeriodValuemMultiSummaryRecord(
        核心字段=core,
        开始日期列表=starts,
        结束日期列表=ends,
        指标名称列表=names,
        数值类型列表=types,
        单位列表=units,
        数值列表=values,
        状态描述列表=statuses,
        原始个人数据=raw,
    )


# ========= 解析：单日期数值单项总结（从原始一行文本抽取 1~N 条记录） =========
_SINGLE_DATE_VALUE_SUMMARY_SEG_RE = re.compile(
    r"^\s*(?P<date>\d{4}/\d{1,2}/\d{1,2}|\d{2}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(?P<rest>.+?)\s*$"
)

_SINGLE_DATE_VALUE_SUMMARY_PREFIX_SEG_RE = re.compile(
    r"^\s*(?P<prefix>.+?)\s*[:：]\s*(?P<date>\d{4}/\d{1,2}/\d{1,2}|\d{2}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(?P<rest>.+?)\s*$"
)

_DATE_TOKEN_RE = re.compile(
    r"\d{4}/\d{1,2}/\d{1,2}|\d{2}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日"
)
_DETAIL_DATE_TIME_RE = re.compile(r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}\s+\d{1,2}:\d{2}")


def _inject_commas_for_concatenated_date_prefixed_segments(raw_line: str) -> str:
    """
    防御性预处理：处理“同一行里被异常拼接了多段、每段都以日期开头”的情况。

    典型脏数据：
      4/23的活动总热量为150.00千卡4/23的锻炼总时长为26分钟

    处理策略：
    - 仅当检测到同一行出现 2+ 个“日期 token”时尝试修复
    - 严格避开“日期范围”（如 3/23~4/23）与“明细记录(YYYY/MM/DD HH:mm ...)”以避免改变原有结果
    - 只在“后续日期 token 前面不是分隔符/空白/范围符号”时插入中文逗号 `，`
    """
    s = (raw_line or "").strip()
    if not s:
        return s

    # 明细记录（带年份+时间点）不要动：这类通常由“单指标的明细记录”解析器处理
    if _DETAIL_DATE_TIME_RE.search(s):
        return s

    ms = list(_DATE_TOKEN_RE.finditer(s))
    if len(ms) <= 1:
        return s

    delim_chars = set("，,;；、\n\t ")
    range_prefix_chars = set("~～-—")

    pieces: list[str] = []
    last = 0
    changed = False
    for i, m in enumerate(ms):
        if i == 0:
            continue
        idx = m.start()
        if idx <= 0:
            continue
        prev = s[idx - 1]

        # 已有分隔符/空白：说明本来就能切段，不插入
        if prev in delim_chars:
            continue
        # 日期范围：3/23~4/23 / 3/23-4/23 / 3/23～4/23
        if prev in range_prefix_chars:
            continue
        # 中文日期范围：3/23至4/23 / 3/23到4/23
        if prev in ("至", "到"):
            continue

        pieces.append(s[last:idx])
        pieces.append("，")
        last = idx
        changed = True

    if not changed:
        return s
    pieces.append(s[last:])
    return "".join(pieces)


def _parse_single_date_value_summary_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
    数值类型: ValueType | None = None,
    单位: str | None = None,
) -> SingleDateValueSingleSummaryRecord | UnparsedRawPersonalData:
    """
    将“单日期数值单项总结”的原始一行文本解析为 `SingleDateValueSingleSummaryRecord`（多条明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为单日期数值单项总结")

    # 防御性：处理“多段日期前缀被拼接成一行”的脏数据
    raw = _inject_commas_for_concatenated_date_prefixed_segments(raw)

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    dates: list[str] = []
    names: list[str] = []
    types: list[ValueType] = []
    units: list[str] = []
    values: list[str] = []
    statuses: list[str] = []

    inferred_name: str | None = 指标名称
    inferred_type: ValueType | None = 数值类型
    inferred_unit: str | None = 单位
    inferred_date: str | None = None

    def _clean_tail(s: str) -> str:
        return (s or "").strip().strip("。；;")

    for seg in segments:
        seg2 = _clean_tail(seg)
        if not seg2:
            continue

        d: str | None = None
        rest: str | None = None

        # 形态 A：日期开头（原逻辑）
        m1 = _SINGLE_DATE_VALUE_SUMMARY_SEG_RE.match(seg2)
        if m1:
            d = str(m1.group("date") or "").strip()
            rest = str(m1.group("rest") or "").strip()
            # 防止把“日期范围”误判为单日期（例如 8/2~8/8 ...）
            if re.match(
                r"^\s*((?:\d{4}|\d{2})[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
                seg2,
            ):
                continue
            if d:
                inferred_date = d
        else:
            # 形态 B：指标在前 + 冒号 + 日期（新增）
            m2 = _SINGLE_DATE_VALUE_SUMMARY_PREFIX_SEG_RE.match(seg2)
            if m2:
                prefix = str(m2.group("prefix") or "").strip()
                d = str(m2.group("date") or "").strip()
                rest_after_date = str(m2.group("rest") or "").strip()
                # 防止把“指标：日期范围 ...”误判为单日期
                if re.match(
                    r"^\s*.+?[:：]\s*(?:\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
                    seg2,
                ):
                    continue
                # 让后续拆解逻辑拿到指标名：把 prefix 拼回 rest
                rest = (prefix + rest_after_date) if (prefix and not rest_after_date.startswith(prefix)) else rest_after_date
                if d:
                    inferred_date = d
            else:
                # 允许后续片段不带日期：继承首段日期
                if not inferred_date:
                    continue
                d = inferred_date
                rest = seg2

        if not d or not rest:
            continue

        nm, val, status = _extract_metric_value_status_from_value_summary_rest(rest)
        if inferred_name is None and nm:
            inferred_name = nm
        if inferred_type is None and val:
            inferred_type = _infer_value_type_from_value_str(val)
        if inferred_unit is None and val:
            inferred_unit = _infer_unit_from_value_str(val)

        n = nm if nm else (inferred_name or "")
        t = _infer_value_type_from_value_str(val) if val else (inferred_type or "String")
        u = _infer_unit_from_value_str(val) if val else (inferred_unit or "无")
        val2, u2 = _normalize_value_and_unit(val, t, u)

        dates.append(_normalize_date_cn(d))
        names.append(n)
        types.append(t)
        units.append(u2)
        values.append(val2)
        statuses.append(status if status else "无")

    if not (dates and names and values):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(日期/指标名称/数值)记录")

    core = SingleDateValueSummaryCore(
        指标名称=inferred_name or (names[0] if names else ""),
        日期="Date (格式: MM月DD日)",
        数值类型=inferred_type or (types[0] if types else "String"),
        单位=inferred_unit or (units[0] if units else "无"),
        状态描述="String",
    )
    return SingleDateValueSingleSummaryRecord(
        核心字段=core,
        日期列表=dates,
        指标名称列表=names,
        数值类型列表=types,
        单位列表=units,
        数值列表=values,
        状态描述列表=statuses,
        原始个人数据=raw,
    )


# ========= 解析：单日期文本总结（从原始一行文本抽取 1~N 条记录） =========
_SINGLE_DATE_TEXT_SUMMARY_SEG_RE = re.compile(
    r"^\s*(?P<date>\d{4}/\d{1,2}/\d{1,2}|\d{2}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(?P<rest>.+?)\s*$"
)


def _parse_single_date_text_summary_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
) -> SingleDateTextSummaryRecord | UnparsedRawPersonalData:
    """
    将“单日期文本总结”的原始一行文本解析为 `SingleDateTextSummaryRecord`（多条明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为单日期文本总结")

    # 防御性：处理“多段日期前缀被拼接成一行”的脏数据
    raw = _inject_commas_for_concatenated_date_prefixed_segments(raw)

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    dates: list[str] = []
    names: list[str] = []
    descs: list[str] = []

    inferred_name: str | None = 指标名称
    inferred_date: str | None = None

    def _clean_tail(s: str) -> str:
        return (s or "").strip().strip("。；;")

    for seg in segments:
        seg2 = _clean_tail(seg)
        if not seg2:
            continue

        m = _SINGLE_DATE_TEXT_SUMMARY_SEG_RE.match(seg2)
        if m:
            d = str(m.group("date") or "").strip()
            rest = str(m.group("rest") or "").strip()
            # 防止把“日期范围”误判为单日期（例如 8/2~8/8 ...）
            if re.match(
                r"^\s*(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
                seg2,
            ):
                continue
            if d:
                inferred_date = d
        else:
            # 允许后续片段不带日期：继承首段日期
            if not inferred_date:
                continue
            d = inferred_date
            rest = seg2

        if not d or not rest:
            continue

        nm, desc = _extract_metric_and_status_from_text(rest)
        if inferred_name is None and nm:
            inferred_name = nm

        n = nm if nm else (inferred_name or "")
        dates.append(_normalize_date_cn(d))
        names.append(n)
        descs.append(desc if desc else rest)

    if not (dates and names and descs):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(日期/指标名称/状态描述)记录")

    core = SingleDateTextSummaryCore(
        指标名称=inferred_name or (names[0] if names else ""),
        时间="Date (格式: MM/DD)",
        状态描述="String",
    )
    return SingleDateTextSummaryRecord(
        核心字段=core,
        日期列表=dates,
        指标名称列表=names,
        状态描述列表=descs,
        原始个人数据=raw,
    )


# ========= 解析：无时间日期的文本总结（从原始一行文本抽取 1~N 条记录） =========
def _parse_no_timestamp_text_summary_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
) -> NoTimestampTextSummaryRecord | UnparsedRawPersonalData:
    """
    将“无时间日期的文本总结”的原始一行文本解析为 `NoTimestampTextSummaryRecord`（多条明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为无时间日期的文本总结")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    names: list[str] = []
    descs: list[str] = []

    inferred_name: str | None = 指标名称

    def _clean_tail(s: str) -> str:
        return (s or "").strip().strip("。；;")

    for seg in segments:
        seg2 = _clean_tail(seg)
        if not seg2:
            continue
        nm, desc = _extract_metric_and_status_from_text(seg2)
        if inferred_name is None and nm:
            inferred_name = nm
        names.append(nm if nm else (inferred_name or ""))
        descs.append(desc if desc else seg2)

    if not (names or descs):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(指标名称/状态描述)记录")

    core = NoTimestampTextSummaryCore(
        指标名称=inferred_name or (names[0] if names else ""),
        状态描述="String",
    )
    return NoTimestampTextSummaryRecord(
        核心字段=core,
        指标名称列表=names,
        状态描述列表=descs,
        原始个人数据=raw,
    )


# ========= 解析：无时间日期的数值总结（从原始一行文本抽取 1~N 条记录） =========
def _parse_no_date_value_summary_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
    数值类型: ValueType | None = None,
    单位: str | None = None,
) -> NoDateValueSummaryRecord | UnparsedRawPersonalData:
    """
    将“无时间日期的数值总结”的原始一行文本解析为 `NoDateValueSummaryRecord`（多条明细用列表承载）。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为无时间日期的数值总结")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    names: list[str] = []
    types: list[ValueType] = []
    units: list[str] = []
    values: list[str] = []
    statuses: list[str] = []

    inferred_name: str | None = 指标名称
    inferred_type: ValueType | None = 数值类型
    inferred_unit: str | None = 单位

    def _clean_tail(s: str) -> str:
        return (s or "").strip().strip("。；;")

    for seg in segments:
        seg2 = _clean_tail(seg)
        if not seg2:
            continue

        # 这里复用“周期数值多项总结”的拆解逻辑：以第一个数字为分界，尾部剥离常见状态描述词
        nm, val, status = _extract_metric_value_status_from_value_summary_rest(seg2)
        if inferred_name is None and nm:
            inferred_name = nm
        if inferred_type is None and val:
            inferred_type = _infer_value_type_from_value_str(val)
        if inferred_unit is None and val:
            inferred_unit = _infer_unit_from_value_str(val)

        n = nm if nm else (inferred_name or "")
        t = _infer_value_type_from_value_str(val) if val else (inferred_type or "String")
        u = _infer_unit_from_value_str(val) if val else (inferred_unit or "无")
        val2, u2 = _normalize_value_and_unit(val, t, u)

        names.append(n)
        types.append(t)
        units.append(u2)
        values.append(val2)
        statuses.append(status if status else "无")

    if not (names or values):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(指标名称/数值/单位/状态描述)记录")

    core = NoDateValueSummaryCore(
        指标名称=inferred_name or (names[0] if names else ""),
        数值类型=inferred_type or (types[0] if types else "String"),
        单位=inferred_unit or (units[0] if units else "无"),
        状态描述="String",
    )
    return NoDateValueSummaryRecord(
        核心字段=core,
        指标名称列表=names,
        数值类型列表=types,
        单位列表=units,
        数值列表=values,
        状态描述列表=statuses,
        原始个人数据=raw,
    )


# ========= 解析：单指标的明细汇总记录（从原始一行文本抽取明细+汇总） =========
_STATS_COMP_HEAD_RE = re.compile(
    r"^\s*(?P<metric>.+?)\s*[:：]\s*(?P<rest>.+?)\s*$"
)


def _find_bracket_span(s: str) -> tuple[int, int] | None:
    """
    在字符串里寻找第一对 [] 或 【】 的跨度，返回 (start_idx, end_idx)（包含括号本身）。
    找不到则返回 None。
    """
    if not s:
        return None
    pairs = [("[", "]"), ("【", "】")]
    for l, r in pairs:
        i = s.find(l)
        if i >= 0:
            j = s.find(r, i + 1)
            if j >= 0:
                return i, j
    return None


_STATS_ITEM_RE = re.compile(
    r"^\s*(?P<date>\d{4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(?P<val>.+?)\s*$"
)

# 特化脏数据拦截：
# - 形如：
#   活动心率：[5月10日107-170, 平均145次/分钟, 5月11日76-118, 平均97次/分钟]，
#   平均活动心率121次/分钟，最低活动心率76次/分钟，最高活动心率170次/分钟
# - 该类数据虽“看起来”可归为 SingleMetricStatsRecord，但明细列表中混入“平均xx次/分钟”片段，
#   会造成结构歧义与后续推断风险；这里按用户约定直接丢到 UnparsedRawPersonalData。
_DIRTY_ACTIVITY_HEART_RATE_RANGE_AVG_RE = re.compile(
    r"活动心率\s*[:：]\s*[\[【][^\]】]*"
    r"(?:\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2})\s*\d+\s*-\s*\d+[^\]】]*"
    # 明细列表内混入“平均xx(次/分钟)?”，单位可能缺失
    r"平均\s*\d+(?:\s*(?:次\s*/\s*分钟|bpm|BPM))?[^\]】]*[\]】]"
    # 末尾汇总必须包含“平均活动心率xx(次/分钟)?”，单位也可能缺失
    r"[^\n]*平均\s*活动心率\s*\d+(?:\s*(?:次\s*/\s*分钟|bpm|BPM))?"
)


def _is_unitless_duration_metric(metric_name: str) -> bool:
    """
    某些指标更适合“单位不单列展示”，而应当保留在 value 字符串中（更贴近原始表达）。

    目前仅针对：锻炼时长。
    - 表头不要出现 “锻炼时长 (分钟)”
    - 数值尽量保持/补齐为 “xx小时xx分钟 / xx分钟 ...” 的样式
    """
    nm = _strip_leading_de(str(metric_name or "").strip())
    return nm == "锻炼时长"


def _normalize_unitless_duration_value_for_display(v: str) -> str:
    """
    仅用于“锻炼时长”这类 unitless-duration 指标，把 value 变成更稳定的可读样式：
    - 纯数字：视作分钟 -> "29" => "29分钟"
    - "1小时23" 这类缺少“分钟”尾巴：补齐 -> "1小时23分钟"
    - 其它（已包含 小时/分钟/秒 等单位词）：尽量原样保留
    """
    t = str(v or "").strip()
    if not t:
        return t
    if t in ("-", "—", "无", "None", "null", "NULL", "N/A", "NA"):
        return t

    # 纯数字：默认分钟
    if re.fullmatch(r"\d+", t):
        return f"{t}分钟"

    # 常见缺尾巴：1小时23 -> 1小时23分钟
    m = re.fullmatch(r"\s*(?P<h>\d+)\s*小时\s*(?P<m>\d+)\s*$", t)
    if m:
        return f"{m.group('h')}小时{m.group('m')}分钟"

    return t


def _parse_stats_composite_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
    数值类型: ValueType | None = None,
    单位: str | None = None,
) -> SingleMetricStatsRecord | UnparsedRawPersonalData:
    """
    将“单指标的明细汇总记录”的原始一行文本解析为 `SingleMetricStatsRecord`。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为单指标的明细汇总记录")

    # 用户允许的特化规则：活动心率“数字范围 + 平均”脏数据直接进入 Unparsed。
    if _DIRTY_ACTIVITY_HEART_RATE_RANGE_AVG_RE.search(raw):
        return UnparsedRawPersonalData(个人数据=raw, 原因="活动心率(范围+平均)脏数据：按特化规则不解析")

    m_head = _STATS_COMP_HEAD_RE.match(raw)
    if not m_head:
        return UnparsedRawPersonalData(个人数据=raw, 原因="未找到形如“指标名称: ...”的头部结构")

    metric = str(m_head.group("metric") or "").strip()
    rest = str(m_head.group("rest") or "").strip()
    if not metric or not rest:
        return UnparsedRawPersonalData(个人数据=raw, 原因="头部解析失败：指标名称或剩余内容为空")

    inferred_name: str = 指标名称 or metric
    inferred_type: ValueType | None = 数值类型
    inferred_unit: str | None = 单位
    force_unitless_duration = _is_unitless_duration_metric(inferred_name)

    # 先解析 [] / 【】 里的明细列表
    span = _find_bracket_span(rest)
    if not span:
        return UnparsedRawPersonalData(个人数据=raw, 原因="未找到明细列表的括号结构（[] 或 【】）")
    l_i, r_i = span
    inside = rest[l_i + 1 : r_i].strip()
    tail = rest[r_i + 1 :].strip()
    # 去掉 tail 前面的分隔符
    tail = tail.lstrip("，,；; ").strip()

    dates: list[str] = []
    values: list[str] = []
    # 记录“明细/汇总”中出现的单位线索，用于在解析结束后做“整条数据级别”的单位修正
    unit_hints: list[str] = []

    if inside:
        def _split_inside_items(s: str) -> list[str]:
            """
            stats-composite 明细列表切段：
            - 正常数据：用逗号分隔 => 直接按 `_SPLIT_RE` 切
            - 脏数据：多个“日期前缀片段”被空格/无分隔拼接（如：'2月2日22:22 2月5日00:12 ...'）
              这类若不切段，会导致 `_infer_unit_from_value_str()` 把 ':22 2月5日00:12 ...' 误当成“单位”，
              从而污染 core.unit 与表头列名。
            """
            t = (s or "").strip()
            if not t:
                return []

            # 1) 常规：按逗号切
            xs = [x.strip() for x in _SPLIT_RE.split(t) if x and x.strip()]
            if len(xs) >= 2:
                return xs

            # 2) 防御：若同一段里出现 2+ 个日期 token，则按“日期 token 的起止”切段
            ms = list(_DATE_TOKEN_RE.finditer(t))
            if len(ms) <= 1:
                return xs

            segs: list[str] = []
            for i, m in enumerate(ms):
                st = m.start()
                ed = ms[i + 1].start() if i + 1 < len(ms) else len(t)
                seg = t[st:ed].strip().strip("，,;；、 ")
                if seg:
                    segs.append(seg)
            return segs if segs else xs

        items = _split_inside_items(inside)
        for it in items:
            m_it = _STATS_ITEM_RE.match(it)
            if not m_it:
                continue
            d = str(m_it.group("date") or "").strip()
            v = str(m_it.group("val") or "").strip()
            if not d or not v:
                continue

            if force_unitless_duration:
                v = _normalize_unitless_duration_value_for_display(v)

            if inferred_type is None and v:
                inferred_type = _infer_value_type_from_value_str(v)
            if not force_unitless_duration:
                vt0: ValueType = inferred_type or _infer_value_type_from_value_str(v)
                u0 = _infer_unit_from_value_str(v)
                v_norm, u_norm = _normalize_value_and_unit(v, vt0, u0)
                # 记录归一后的值/单位
                v = v_norm
                if inferred_unit is None and v:
                    inferred_unit = u_norm
                if v:
                    unit_hints.append(u_norm)

            dates.append(_normalize_date_cn(d))
            values.append(v)

    # 再解析括号后的汇总描述：多段用逗号分隔
    sum_names: list[str] = []
    sum_values: list[str] = []
    sum_statuses: list[str] = []
    if tail:
        segs = [x.strip() for x in _SPLIT_RE.split(tail) if x and x.strip()]
        for seg in segs:
            nm, val, status = _extract_metric_value_status_from_value_summary_rest(seg)
            if not nm and seg:
                nm = seg
            if force_unitless_duration and val:
                val = _normalize_unitless_duration_value_for_display(val)
            if inferred_type is None and val:
                inferred_type = _infer_value_type_from_value_str(val)
            if not force_unitless_duration and val:
                vt0: ValueType = inferred_type or _infer_value_type_from_value_str(val)
                u0 = _infer_unit_from_value_str(val)
                _v_norm, u_norm = _normalize_value_and_unit(val, vt0, u0)
                if inferred_unit is None:
                    inferred_unit = u_norm
                unit_hints.append(u_norm)
            # 汇总数值也做一次规范化：
            # - 先把“值/单位”拆开，避免重复单位
            # - 再把单位拼接回去，保证与“明细数值列表”一样带单位（便于人读/对齐展示）
            t = _infer_value_type_from_value_str(val) if val else (inferred_type or "String")
            u = _infer_unit_from_value_str(val) if val else (inferred_unit or "无")
            val2, _u2 = _normalize_value_and_unit(val, t, u)
            sum_names.append(nm)
            sum_values.append(_attach_unit_to_value(val2, _u2))
            sum_statuses.append(status if status else "无")

    if not (dates and values) and not (sum_names or sum_values or sum_statuses):
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(日期/数值/统计汇总)记录")

    # core：用于 style 输出的 metadata（占位符 + 推断出的数值类型/单位）
    core_type: ValueType = inferred_type or "String"
    # 单位推断（二次修正）：统计复合明细常出现“同一指标混合不同单位样式”。
    # 例如：快速眼动时长既可能写作“36分钟”，也可能写作“1小时5分钟”。
    # 旧逻辑仅在 inferred_unit is None 时用第一条明细锁定单位，导致整列表头固定为“分钟”，
    # 后续在表格透视阶段剥离“分钟”时，会把“1小时5分钟”错误显示为“1小时5”。
    #
    # 这里基于整条数据的 unit_hints 做一次“升级选择”：
    # - 若出现任何“小时+分钟”的复合时长，则优先使用 "小时分钟"
    # - 其次按出现情况选择 "小时"/"分钟"/"秒"
    if not force_unitless_duration:
        hints = [str(x or "").strip() for x in unit_hints if str(x or "").strip() and str(x).strip() != "无"]
        # 去掉分母单位（如 分钟/公里），避免误判为“复合时长”
        hints_no_ratio = [h for h in hints if "/" not in h]
        if hints_no_ratio:
            has_h = any("小时" in h for h in hints_no_ratio)
            has_m = any("分钟" in h for h in hints_no_ratio)
            has_s = any("秒" in h for h in hints_no_ratio)
            # 出现“小时+分钟”（无论是否同一段）=> 统一为 小时分钟
            if has_h and has_m:
                inferred_unit = "小时分钟"
            elif any(("小时" in h and "分钟" in h) for h in hints_no_ratio):
                inferred_unit = "小时分钟"
            elif inferred_unit in (None, "", "无"):
                if has_h:
                    inferred_unit = "小时"
                elif has_m:
                    inferred_unit = "分钟"
                elif has_s:
                    inferred_unit = "秒"
        # 对纯持续时长：统一 unitless（避免列名出现 "(小时分钟)" 等）
        if (inferred_type or "String") == "Duration" and (not inferred_unit or ("/" not in str(inferred_unit))):
            inferred_unit = "无"

    # 兜底：部分指标（如“睡眠得分/评分”）真实数据常省略“分”，这里给一个更合理的默认单位
    if (inferred_unit is None or inferred_unit == "无") and any(x in inferred_name for x in ("得分", "评分")):
        inferred_unit = "分"
    core_unit: str = "无" if force_unitless_duration else (inferred_unit or "无")
    core = StatsCompositeCore(
        指标名称=inferred_name,
        数值类型=core_type,
        单位=core_unit,
        数据列表=[StatsCompositeDataItem(日期="Date (格式: YYYY/MM/DD)", 数值类型=core_type, 单位=core_unit)],
        统计汇总描述=[
            StatsCompositeSummaryItem(指标名称="String", 数值类型=core_type, 单位=core_unit, 状态描述="String")
        ],
    )

    return SingleMetricStatsRecord(
        核心字段=core,
        日期列表=dates,
        数值列表=values,
        统计指标名称列表=sum_names,
        统计数值列表=sum_values,
        统计状态描述列表=sum_statuses,
        原始个人数据=raw,
    )


# ========= 解析：单日期数值多项总结（新版：每个逗号片段一条记录） =========
_SINGLE_DATE_HEAD_RE = re.compile(
    r"^\s*(?P<date>\d{4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(?P<rest>.+?)\s*$"
)


def _parse_single_date_multi_value_summary_line(
    raw_line: str,
    *,
    指标名称: str | None = None,
    数值类型: ValueType | None = None,
    单位: str | None = None,
) -> SingleDateValueMultiSummaryRecord | UnparsedRawPersonalData:
    """
    将“单日期数值多项总结（新样式）”的原始一行文本解析为 **一个** `SingleDateValueMultiSummaryRecord`。

    约定（与 `predict_personal_data.py` 的新样式对齐）：
    - 输入通常形如：{日期}{指标}{数值}{单位},{指标}{数值}{单位}{状态描述}, ...
    - 本函数会解析每个逗号片段，但最终把所有片段合并到同一个对象的列表字段中。
    - core 的 `状态描述`：若任一片段包含状态词（如 正常/偏低），则 core.状态描述 = "String"，否则为 "无"。

    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为单日期数值多项总结")

    segments = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
    if not segments:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行或仅包含分隔符，无法解析为单日期数值多项总结")

    first = segments[0]
    m_head = _SINGLE_DATE_HEAD_RE.match(first)
    if not m_head:
        return UnparsedRawPersonalData(个人数据=raw, 原因="未找到形如“日期 + 指标 + 数值”的头部结构")

    date_str = _normalize_date_cn(str(m_head.group("date") or "").strip())
    rest0 = str(m_head.group("rest") or "").strip()
    if not date_str or not rest0:
        return UnparsedRawPersonalData(个人数据=raw, 原因="头部解析失败：日期或剩余内容为空")

    def _prefer_float_for_percent(vt: ValueType, unit0: str) -> ValueType:
        # 与 `predict_personal_data.py` 示例保持一致：百分比即使为整数也按 Float 输出（例如 96%）
        u = (unit0 or "").strip()
        if u == "%" and vt == "Int":
            return "Float"
        return vt

    def _parse_segment_to_one_record(seg_rest: str, *, is_first: bool) -> SingleDateValueMultiSummaryRecord | None:
        seg_rest2 = (seg_rest or "").strip()
        if not seg_rest2:
            return None

        nm, val_raw, status = _extract_metric_value_status_from_value_summary_rest(seg_rest2)
        if not nm or not val_raw:
            return None

        # 允许后续片段只写“平均/最高/最低...”而不写具体指标：
        # 例如：4/18活动心率107-192, 平均161次/分钟
        # 此时 nm 会被解析为 "平均"，需要补全为 "平均活动心率"（更符合表格展示与单位推断）。
        _stat_only_prefixes = ("平均", "最高", "最低", "最大", "最小", "最晚", "最早", "最长", "最短", "总计", "累计")
        if (not is_first) and nm in _stat_only_prefixes and "first_metric_name" in locals():
            # mypy/pyright: locals() 检查仅为运行期兜底；first_metric_name 在外层会定义
            pass

        # 允许对“首段”用调用方提供的提示（与其它 record 的接口一致）
        vt0: ValueType = (数值类型 if (is_first and 数值类型) else _infer_value_type_from_value_str(val_raw))  # type: ignore[assignment]
        unit0: str = (单位 if (is_first and 单位) else _infer_unit_from_value_str(val_raw))  # type: ignore[assignment]
        vt0 = _prefer_float_for_percent(vt0, unit0)

        # 规范化 value/unit：
        # - Float/Int/Duration 等：尽量把单位从 value 中剥离（单位进入单位列表）
        # - FloatRange：为更贴近原始展示（如 96%-96%），保留 value 原串，同时单位仍记为 "%"
        val_norm, unit_norm = _normalize_value_and_unit(val_raw, vt0, unit0)
        if vt0 == "FloatRange":
            val_norm = val_raw.strip()
            unit_norm = unit0 or unit_norm

        status_norm = (status or "").strip() or "无"
        core_status_placeholder = "String" if (status_norm and status_norm != "无") else "无"

        return SingleDateValueMultiSummaryRecord(
            核心字段=SingleDateValueMultiSummaryCore(
                指标名称=(指标名称 if (is_first and 指标名称) else nm) or nm,
                日期="Date (格式: MM月DD日)",
                数值类型=vt0,
                单位=str(unit_norm or "无"),
                状态描述=core_status_placeholder,
            ),
            日期列表=[date_str],
            指标名称列表=[(指标名称 if (is_first and 指标名称) else nm) or nm],
            数值类型列表=[vt0],
            单位列表=[str(unit_norm or "无")],
            数值列表=[str(val_norm)],
            状态描述列表=[status_norm],
            原始个人数据=raw,
        )

    # 先按“每段 -> 一个 record”的方式解析（复用已有逻辑），再合并为一个 record。
    tmp: list[SingleDateValueMultiSummaryRecord] = []
    first_rec = _parse_segment_to_one_record(rest0, is_first=True)
    if first_rec is not None:
        tmp.append(first_rec)
    # 记录“主指标名称”（用于后续补全 "平均/最高/最低" 等省略写法）
    first_metric_name = ""
    if first_rec is not None:
        first_metric_name = str(getattr(first_rec.核心字段, "指标名称", "") or "").strip()
    for seg in segments[1:]:
        # 对后续段，若仅包含“平均/最高/最低”等前缀，尝试补全为 "前缀 + 主指标"
        # 这样可以：
        # - 修复表格中出现 “指标=平均” 这种不完整展示
        # - 让单位推断/补齐更准确（例如活动心率的范围值 107-192）
        seg2 = seg
        nm2, val2, st2 = _extract_metric_value_status_from_value_summary_rest(seg2)
        if nm2 in ("平均", "最高", "最低", "最大", "最小", "最晚", "最早", "最长", "最短", "总计", "累计") and first_metric_name:
            # 仅当原片段确实没有写出主指标名时才补全：用最小侵入方式把指标名插入到片段文本里
            # 例： "平均161次/分钟" -> "平均活动心率161次/分钟"
            seg2 = f"{nm2}{first_metric_name}{val2}{st2}".strip()
        rec = _parse_segment_to_one_record(seg2, is_first=False)
        if rec is not None:
            tmp.append(rec)

    if not tmp:
        return UnparsedRawPersonalData(个人数据=raw, 原因="未能从该行中解析出任何(日期/指标名称/数值)记录")

    # 合并：把每段的单条列表字段串起来
    dates: list[str] = []
    names: list[str] = []
    types: list[ValueType] = []
    units: list[str] = []
    values: list[str] = []
    statuses: list[str] = []
    any_status = False
    for r in tmp:
        # 每个 r 应该只有 1 条明细（单段）
        dates.extend(list(getattr(r, "日期列表", []) or []))
        names.extend(list(getattr(r, "指标名称列表", []) or []))
        types.extend(list(getattr(r, "数值类型列表", []) or []))
        units.extend(list(getattr(r, "单位列表", []) or []))
        values.extend(list(getattr(r, "数值列表", []) or []))
        sts = list(getattr(r, "状态描述列表", []) or [])
        statuses.extend(sts)
        if any((str(x or "").strip() and str(x).strip() != "无") for x in sts):
            any_status = True

    # 单位补齐（解析侧修复）：
    # - 同一条“单日期多项总结”里，经常出现：
    #   "活动心率107-192, 平均活动心率161次/分钟"
    #   第一段范围值没有单位，导致 unit="无"；但实际上应与后续段一致为 "次/分钟"。
    # - 这里按“去掉统计前缀后的主指标名”分组，把缺失单位补齐。
    _stat_prefixes2 = ("平均", "最高", "最低", "最大", "最小", "最晚", "最早", "最长", "最短", "总计", "累计")

    def _base_metric_name(nm: str) -> str:
        t = str(nm or "").strip()
        for p in _stat_prefixes2:
            if t.startswith(p) and len(t) > len(p):
                return t[len(p) :].strip()
        return t

    unit_by_base: dict[str, str] = {}
    for nm, u in zip(names, units):
        ub = str(u or "").strip() or "无"
        if ub != "无":
            base = _base_metric_name(nm)
            if base and base not in unit_by_base:
                unit_by_base[base] = ub

    for i in range(len(units)):
        u0 = str(units[i] or "").strip() or "无"
        if u0 != "无":
            continue
        base = _base_metric_name(names[i] if i < len(names) else "")
        if not base:
            continue
        ub = unit_by_base.get(base)
        if not ub:
            continue
        # 只对“看起来是数值/区间”的条目补齐，避免误给纯文本状态类塞单位
        v0 = str(values[i] if i < len(values) else "").strip()
        if re.search(r"\d", v0):
            units[i] = ub

    # core：以首段为主（更符合“整体指标名称”直觉），状态占位取“是否存在状态”
    first_core = tmp[0].核心字段
    # 合并后单位修复：
    # - 解析过程中“首段”可能缺失单位（如 "血氧91-99"），导致 first_core.单位="无"
    # - 但同一条记录的后续段可能提供了明确单位（如 "平均血氧98%"），上面的补齐逻辑也会把 units[0] 补成 "%"
    # - 这里优先用“同主指标名”的 unit_by_base 纠正 core.单位，避免出现 core=无 但 单位列表=% 的不一致
    core_unit = str(getattr(first_core, "单位", "") or "无").strip() or "无"
    if core_unit == "无":
        base0 = _base_metric_name(str(getattr(first_core, "指标名称", "") or "").strip())
        if base0:
            core_unit = unit_by_base.get(base0, core_unit)
        # 再兜底：若仍无，但 units 里已有非无单位，则取首个非无
        if core_unit == "无":
            for u in units:
                uu = str(u or "").strip() or "无"
                if uu != "无":
                    core_unit = uu
                    break
    merged_core = SingleDateValueMultiSummaryCore(
        指标名称=str(getattr(first_core, "指标名称", "") or ""),
        日期=str(getattr(first_core, "日期", "") or "Date (格式: MM月DD日)"),
        数值类型=getattr(first_core, "数值类型", "String"),  # type: ignore[arg-type]
        单位=core_unit,
        状态描述=("String" if any_status else "无"),
    )

    return SingleDateValueMultiSummaryRecord(
        核心字段=merged_core,
        日期列表=dates,
        指标名称列表=names,
        数值类型列表=types,
        单位列表=units,
        数值列表=values,
        状态描述列表=statuses,
        原始个人数据=raw,
    )


__all__ = [
    # base
    "ValueType",
    "PersonalDataPatternBase",
    "UnparsedRawPersonalData",
    # entity types
    "SingleValueCore",
    "SingleMetricDetailRecord",
    "PeriodSummaryCore",
    "PeriodValueSingleSummaryRecord",
    "PeriodTextSummaryCore",
    "PeriodTextSummaryRecord",
    "PeriodValueCompareCore",
    "PeriodValueCompareRecord",
    "PeriodValueSummaryCore",
    "PeriodValuemMultiSummaryRecord",
    "SingleDateValueSummaryCore",
    "SingleDateValueSingleSummaryRecord",
    "SingleDateTextSummaryCore",
    "SingleDateTextSummaryRecord",
    "NoTimestampTextSummaryCore",
    "NoTimestampTextSummaryRecord",
    "NoDateValueSummaryCore",
    "NoDateValueSummaryRecord",
    "StatsCompositeDataItem",
    "StatsCompositeSummaryItem",
    "StatsCompositeCore",
    "SingleMetricStatsRecord",
    "SingleDateValueMultiSummaryCore",
    "SingleDateValueMultiSummaryRecord",
    # parsing helpers (models 内的低层解析器)
    "PersonalDataPattern",
    "_indent_lines",
    "_parse_single_value_line",
    "_parse_single_value_line_multi",
    "_parse_period_summary_line",
    "_parse_period_text_summary_line",
    "_parse_period_value_compare_line",
    "_parse_period_value_summary_line",
    "_parse_single_date_value_summary_line",
    "_parse_single_date_text_summary_line",
    "_parse_no_timestamp_text_summary_line",
    "_parse_no_date_value_summary_line",
    "_parse_stats_composite_line",
    "_parse_single_date_multi_value_summary_line",
]


