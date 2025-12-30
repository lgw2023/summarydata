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
  - 单指标的统计复合记录
- 额外提供“兜底类”：用于保存无法解析的数据样式，保留原始个人数据纯文本。

说明：
- 这里主要做“结构化承载”，不做强校验；强校验逻辑在 `analyze_personal_data.py` 中完成。
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union


# ========= 格式化打印（给 self-test / 人类阅读用）=========
def _shorten(s: Any, max_len: int = 120) -> str:
    t = str(s if s is not None else "").replace("\n", " ").strip()
    if max_len <= 0:
        return ""
    return t if len(t) <= max_len else (t[: max_len - 1] + "…")


def _indent_lines(text: str, n: int = 2) -> str:
    pad = " " * max(0, int(n))
    return "\n".join((pad + ln) if ln else ln for ln in str(text).splitlines())


def _fmt_header(title: str) -> str:
    t = str(title).strip()
    return f"【{t}】" if t else "【】"


def _fmt_kv(key: str, value: Any, *, max_len: int = 120) -> str:
    return f"- {key}：{_shorten(value, max_len=max_len)}"


def _fmt_list_preview(
    items: Sequence[Any],
    *,
    max_items: int = 6,
    max_len: int = 120,
    bullet: str = "- ",
) -> list[str]:
    xs = list(items or [])
    out: list[str] = []
    for i, x in enumerate(xs[: max(0, int(max_items))]):
        out.append(f"{bullet}[{i}] {_shorten(x, max_len=max_len)}")
    if len(xs) > max_items:
        out.append(f"{bullet}…（剩余 {len(xs) - max_items} 项已省略）")
    return out


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _fmt_rows_table(
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str],
    *,
    max_rows: int = 8,
    max_cell_len: int = 60,
) -> str:
    """
    简易“表格”输出：对齐列宽 + 截断单元格，适合日志阅读（不追求 markdown）。
    """
    hs = [str(h) for h in headers]
    rs = [list(map(str, r)) for r in rows[: max(0, int(max_rows))]]
    # 截断
    rs2: list[list[str]] = []
    for r in rs:
        rs2.append([_shorten(c, max_len=max_cell_len) for c in r])
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
    if len(rows) > max_rows:
        lines.append(f"…（剩余 {len(rows) - max_rows} 行已省略）")
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
            if et == "单指标的明细记录":
                return _recover_single_metric_detail_record(self)
            if et == "周期数值单项总结":
                return _recover_period_value_single_summary_record(self)
            if et == "周期文本总结":
                return _recover_period_text_summary_record(self)
            if et == "周期数值对比记录":
                return _recover_period_value_compare_record(self)
            if et == "周期数值多项总结":
                return _recover_period_value_multi_summary_record(self)
            if et == "单日期数值单项总结":
                return _recover_single_date_value_single_summary_record(self)
            if et == "单日期文本总结":
                return _recover_single_date_text_summary_record(self)
            if et == "无时间日期的文本总结":
                return _recover_no_timestamp_text_summary_record(self)
            if et == "无时间日期的数值总结":
                return _recover_no_date_value_summary_record(self)
            if et == "单指标的统计复合记录":
                return _recover_single_metric_stats_record(self)
            if et == "单日期数值多项总结":
                return _recover_single_date_value_multi_summary_record(self)
            if et == "未定义":
                # 兜底类：这里通常应当走 raw1（个人数据）；若没有则返回空串兜底
                return str(raw1 or "").strip()
        except Exception:
            # 不让还原逻辑影响主流程
            pass

        # 最终兜底：返回简短可读文本，避免空串
        try:
            txt = self.format_print(max_items=6, max_len=160)
            return str(txt or "").strip() or et or str(self)
        except Exception:
            return et or str(self)

    def format_print(
        self,
        *,
        max_items: int = 8,
        max_len: int = 120,
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
                lines.append(_fmt_kv(k, v, max_len=max_len))
            elif k == "原因":
                lines.append(_fmt_kv(k, v, max_len=max_len))
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
                            lines.append(_indent_lines(_fmt_kv(kk, core_dict.get(kk), max_len=max_len), 2))
                else:
                    lines.append(_indent_lines(_shorten(core_dict, max_len=max_len), 2))
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

    def format_print(self, *, max_items: int = 8, max_len: int = 220) -> str:  # type: ignore[override]
        lines = [_fmt_header("数据类型：未定义")]
        if self.原因:
            lines.append(_fmt_kv("原因", self.原因, max_len=max_len))
        lines.append(_fmt_kv("个人数据", self.个人数据, max_len=max_len))
        if self.原始样式输出 is not None:
            lines.append(_fmt_kv("原始样式输出(截断)", self.原始样式输出, max_len=max_len))
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

    def format_print(self, *, max_items: int = 8, max_len: int = 120) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）", max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

        rows: list[list[Any]] = []
        n = max(len(self.日期列表), len(self.时间列表), len(self.数值列表))
        for i in range(n):
            d = self.日期列表[i] if i < len(self.日期列表) else ""
            t = self.时间列表[i] if i < len(self.时间列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            rows.append([d, t, v])
        if rows:
            lines.append("- 明细（日期 / 时间 / 数值）：")
            table = _fmt_rows_table(rows, headers=("日期", "时间", "数值"), max_rows=max_items, max_cell_len=max_len)
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

    def format_print(self, *, max_items: int = 8, max_len: int = 120) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）", max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

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
                headers=("开始", "结束", "指标", "类型", "单位", "数值"),
                max_rows=max_items,
                max_cell_len=max_len,
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

    def format_print(self, *, max_items: int = 8, max_len: int = 140) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", core.指标名称, max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

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
            table = _fmt_rows_table(rows, headers=("开始", "结束", "指标", "状态描述"), max_rows=max_items, max_cell_len=max_len)
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

    def format_print(self, *, max_items: int = 8, max_len: int = 140) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv(
                "指标名称",
                f"{core.指标名称}（{core.数值类型}，单位={core.单位}；差异类型={core.差异数值类型}）",
                max_len=max_len,
            ),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

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
                headers=("范围1", "值1", "范围2", "值2", "1相较于2的逻辑", "1相较于2的差异"),
                max_rows=max_items,
                max_cell_len=max_len,
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

    def format_print(self, *, max_items: int = 8, max_len: int = 140) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）", max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

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
                headers=("开始", "结束", "指标", "类型", "单位", "数值", "状态"),
                max_rows=max_items,
                max_cell_len=max_len,
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

    def format_print(self, *, max_items: int = 8, max_len: int = 140) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）", max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

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
                headers=("日期", "指标", "类型", "单位", "数值", "状态"),
                max_rows=max_items,
                max_cell_len=max_len,
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

    def format_print(self, *, max_items: int = 8, max_len: int = 160) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", core.指标名称, max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

        n = max(len(self.日期列表), len(self.指标名称列表), len(self.状态描述列表))
        rows: list[list[Any]] = []
        for i in range(n):
            d = self.日期列表[i] if i < len(self.日期列表) else ""
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            desc = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([d, nm, desc])
        if rows:
            lines.append("- 明细（日期/指标/状态描述）：")
            table = _fmt_rows_table(rows, headers=("日期", "指标", "状态描述"), max_rows=max_items, max_cell_len=max_len)
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

    def format_print(self, *, max_items: int = 10, max_len: int = 160) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", core.指标名称, max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

        n = max(len(self.指标名称列表), len(self.状态描述列表))
        rows: list[list[Any]] = []
        for i in range(n):
            nm = self.指标名称列表[i] if i < len(self.指标名称列表) else ""
            desc = self.状态描述列表[i] if i < len(self.状态描述列表) else ""
            rows.append([nm, desc])
        if rows:
            lines.append("- 明细（指标/状态描述）：")
            table = _fmt_rows_table(rows, headers=("指标", "状态描述"), max_rows=max_items, max_cell_len=max_len)
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

    def format_print(self, *, max_items: int = 10, max_len: int = 140) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）", max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

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
            table = _fmt_rows_table(rows, headers=("指标", "类型", "单位", "数值", "状态"), max_rows=max_items, max_cell_len=max_len)
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


# ========= 单指标的统计复合记录 =========
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
        object.__setattr__(self, "实体类型", "单指标的统计复合记录")
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

    def format_print(self, *, max_items: int = 8, max_len: int = 140) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}）", max_len=max_len),
            _fmt_kv("明细条数", len(self.日期列表), max_len=max_len),
            _fmt_kv("汇总条数", len(self.统计指标名称列表), max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

        # 明细表
        detail_rows: list[list[Any]] = []
        n_d = max(len(self.日期列表), len(self.数值列表))
        for i in range(n_d):
            d = self.日期列表[i] if i < len(self.日期列表) else ""
            v = self.数值列表[i] if i < len(self.数值列表) else ""
            detail_rows.append([d, v])
        if detail_rows:
            lines.append("- 明细（日期/数值）：")
            lines.append(_indent_lines(_fmt_rows_table(detail_rows, headers=("日期", "数值"), max_rows=max_items, max_cell_len=max_len), 2))

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
                _indent_lines(_fmt_rows_table(sum_rows, headers=("统计项", "数值", "状态"), max_rows=max_items, max_cell_len=max_len), 2)
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
        从“单指标的统计复合记录”原始一行文本中抽取：
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

    def format_print(self, *, max_items: int = 8, max_len: int = 140) -> str:  # type: ignore[override]
        core = self.核心字段
        lines = [
            _fmt_header(f"数据类型：{self.实体类型}"),
            _fmt_kv("指标名称", f"{core.指标名称}（{core.数值类型}，单位={core.单位}；状态占位={core.状态描述}）", max_len=max_len),
            _fmt_kv("记录条数", self.记录条数, max_len=max_len),
        ]
        if self.原始个人数据:
            lines.append(_fmt_kv("原始个人数据", self.原始个人数据, max_len=max_len * 2))

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
                _indent_lines(_fmt_rows_table(rows, headers=("日期", "指标", "类型", "单位", "数值", "状态"), max_rows=max_items, max_cell_len=max_len), 2)
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


def parse_style_item_to_dataclass(
    item: Mapping[str, Any],
    *,
    raw_personal_data: str | None = None,
) -> PersonalDataPattern:
    """
    将单个 style item(dict) 转为对应的数据类。
    解析失败时返回 `UnparsedRawPersonalData`（优先使用 raw_personal_data 作为原始文本）。
    """
    et = str(item.get("实体类型", "")).strip()
    core: Any = item.get("核心字段")
    if not isinstance(core, Mapping):
        # 容错：用户/日志样例里可能把 core 直接平铺到了 item 顶层（缺少 "核心字段" 包裹）。
        # 由于 `predict_personal_data.py` 的强校验要求 item 只能有 {"实体类型","核心字段"}，
        # 这种结构一般不会来自模型真实输出，但这里做兼容以方便调试/迁移。
        if et == "单日期数值多项总结" and all(
            k in item for k in ("指标名称", "日期", "数值类型", "单位", "状态描述")
        ):
            core = item
        else:
            return UnparsedRawPersonalData(
                个人数据=raw_personal_data or "",
                原因="核心字段 缺失或不是对象(dict)",
                原始样式输出=dict(item),
            )

    try:
        if et == "单指标的明细记录":
            # style 输出（来自 LLM）通常只有 metadata，不包含真实“数值/多条记录”；
            # 因此这里将明细列表留空，仅把 raw_personal_data 透传，便于后续需要时二次解析。
            return SingleMetricDetailRecord(
                核心字段=SingleValueCore(
                    日期=str(core.get("日期", "")),
                    时间=str(core.get("时间", "")),
                    指标名称=str(core.get("指标名称", "")),
                    数值类型=core.get("数值类型"),  # type: ignore[assignment]
                    单位=str(core.get("单位", "")),
                ),
                原始个人数据=raw_personal_data,
            )
        if et == "周期数值单项总结":
            return PeriodValueSingleSummaryRecord(
                核心字段=PeriodSummaryCore(
                    开始日期=str(core.get("开始日期", "")),
                    结束日期=str(core.get("结束日期", "")),
                    指标名称=str(core.get("指标名称", "")),
                    数值类型=core.get("数值类型"),  # type: ignore[assignment]
                    单位=str(core.get("单位", "")),
                ),
                原始个人数据=raw_personal_data,
            )
        if et == "周期文本总结":
            return PeriodTextSummaryRecord(
                核心字段=PeriodTextSummaryCore(
                    开始日期=str(core.get("开始日期", "")),
                    结束日期=str(core.get("结束日期", "")),
                    指标名称=str(core.get("指标名称", "")),
                    状态描述=str(core.get("状态描述", "")),
                )
                ,
                原始个人数据=raw_personal_data,
            )
        if et == "周期数值对比记录":
            return PeriodValueCompareRecord(
                核心字段=PeriodValueCompareCore(
                    日期范围1=str(core.get("日期范围1", "")),
                    日期范围2=str(core.get("日期范围2", "")),
                    指标名称=str(core.get("指标名称", "")),
                    数值类型=core.get("数值类型"),  # type: ignore[assignment]
                    单位=str(core.get("单位", "")),
                    对比逻辑类型=str(core.get("对比逻辑类型", "")),
                    差异数值类型=core.get("差异数值类型"),  # type: ignore[assignment]
                ),
                原始个人数据=raw_personal_data,
            )
        if et == "周期数值多项总结":
            return PeriodValuemMultiSummaryRecord(
                核心字段=PeriodValueSummaryCore(
                    开始日期=str(core.get("开始日期", "")),
                    结束日期=str(core.get("结束日期", "")),
                    指标名称=str(core.get("指标名称", "")),
                    数值类型=core.get("数值类型"),  # type: ignore[assignment]
                    单位=str(core.get("单位", "")),
                    状态描述=str(core.get("状态描述", "")),
                )
                ,
                原始个人数据=raw_personal_data,
            )
        if et == "单日期数值单项总结":
            return SingleDateValueSingleSummaryRecord(
                核心字段=SingleDateValueSummaryCore(
                    指标名称=str(core.get("指标名称", "")),
                    日期=str(core.get("日期", "")),
                    数值类型=core.get("数值类型"),  # type: ignore[assignment]
                    单位=str(core.get("单位", "")),
                    状态描述=str(core.get("状态描述", "")),
                ),
                原始个人数据=raw_personal_data,
            )
        if et == "单日期文本总结":
            return SingleDateTextSummaryRecord(
                核心字段=SingleDateTextSummaryCore(
                    指标名称=str(core.get("指标名称", "")),
                    时间=str(core.get("时间", "")),
                    状态描述=str(core.get("状态描述", "")),
                ),
                原始个人数据=raw_personal_data,
            )
        if et == "无时间日期的文本总结":
            return NoTimestampTextSummaryRecord(
                核心字段=NoTimestampTextSummaryCore(
                    指标名称=str(core.get("指标名称", "")),
                    状态描述=str(core.get("状态描述", "")),
                )
                ,
                原始个人数据=raw_personal_data,
            )
        if et == "无时间日期的数值总结":
            return NoDateValueSummaryRecord(
                核心字段=NoDateValueSummaryCore(
                    指标名称=str(core.get("指标名称", "")),
                    数值类型=core.get("数值类型"),  # type: ignore[assignment]
                    单位=str(core.get("单位", "")),
                    状态描述=str(core.get("状态描述", "")),
                )
                ,
                原始个人数据=raw_personal_data,
            )
        if et == "单指标的统计复合记录":
            data_list_raw = core.get("数据列表")
            sum_list_raw = core.get("统计汇总描述")
            data_list: list[StatsCompositeDataItem] = []
            sum_list: list[StatsCompositeSummaryItem] = []

            if isinstance(data_list_raw, Sequence):
                for x in data_list_raw:
                    if isinstance(x, Mapping):
                        data_list.append(
                            StatsCompositeDataItem(
                                日期=str(x.get("日期", "")),
                                数值类型=x.get("数值类型"),  # type: ignore[assignment]
                                单位=str(x.get("单位", "")),
                            )
                        )
            if isinstance(sum_list_raw, Sequence):
                for x in sum_list_raw:
                    if isinstance(x, Mapping):
                        sum_list.append(
                            StatsCompositeSummaryItem(
                                指标名称=str(x.get("指标名称", "")),
                                数值类型=x.get("数值类型"),  # type: ignore[assignment]
                                单位=str(x.get("单位", "")),
                                状态描述=str(x.get("状态描述", "")),
                            )
                        )

            # 兼容历史 style：core 里可能没有“数值类型/单位”，则从列表里兜底推断
            core_value_type = core.get("数值类型")
            if core_value_type is None:
                core_value_type = data_list[0].数值类型 if data_list else (sum_list[0].数值类型 if sum_list else "String")
            core_unit = core.get("单位")
            if core_unit is None:
                core_unit = data_list[0].单位 if data_list else (sum_list[0].单位 if sum_list else "无")

            return SingleMetricStatsRecord(
                核心字段=StatsCompositeCore(
                    指标名称=str(core.get("指标名称", "")),
                    数值类型=core_value_type,  # type: ignore[arg-type]
                    单位=str(core_unit),
                    数据列表=data_list,
                    统计汇总描述=sum_list,
                )
            )

        if et == "单日期数值多项总结":
            # 新版样式：core 键严格为 {"指标名称","日期","数值类型","单位","状态描述"}
            return SingleDateValueMultiSummaryRecord(
                核心字段=SingleDateValueMultiSummaryCore(
                    指标名称=str(core.get("指标名称", "")),
                    日期=str(core.get("日期", "")),
                    数值类型=core.get("数值类型"),  # type: ignore[assignment]
                    单位=str(core.get("单位", "")),
                    状态描述=str(core.get("状态描述", "")),
                ),
                原始个人数据=raw_personal_data,
            )

        # 未定义实体类型：走兜底
        return UnparsedRawPersonalData(
            个人数据=raw_personal_data or "",
            原因=f"未定义实体类型：{et!r}",
            原始样式输出=dict(item),
        )
    except Exception as e:
        return UnparsedRawPersonalData(
            个人数据=raw_personal_data or "",
            原因=f"解析 style item 异常：{type(e).__name__}: {e}",
            原始样式输出=dict(item),
        )


def parse_style_to_dataclasses(
    style: Any,
    *,
    raw_personal_data: str,
) -> list[PersonalDataPattern]:
    """
    将一行个人数据对应的 style(list) 转为数据类列表。
    - style 必须为 list[dict]；否则返回仅包含一个兜底类。
    """
    if not isinstance(style, list):
        return [
            UnparsedRawPersonalData(
                个人数据=raw_personal_data,
                原因=f"style 不是 list：实际类型={type(style).__name__}",
                原始样式输出=style,
            )
        ]
    out: list[PersonalDataPattern] = []
    for it in style:
        if isinstance(it, Mapping):
            out.append(parse_style_item_to_dataclass(it, raw_personal_data=raw_personal_data))
        else:
            out.append(
                UnparsedRawPersonalData(
                    个人数据=raw_personal_data,
                    原因=f"style item 不是 dict：实际类型={type(it).__name__}",
                    原始样式输出=it,
                )
            )
    return out


def route_raw_personal_data_to_dataclass(
    raw_line: str,
    *,
    strict_uncovered_to_unparsed: bool = True,
) -> list[PersonalDataPattern]:
    """
    Router：输入“一条个人数据长字符串”，自动判定数据类型并分发到对应的数据类解析器。

    设计目标：
    - 用户不提供实体类型时，尽量自动识别（基于特征 + 多候选打分）。
    - **统一返回 list[PersonalDataPattern]**：
      - 对于常规情况返回长度为 1 的列表
      - 对于“单指标的明细记录”的多指标场景，可能返回多个记录对象
      - 解析失败时，返回仅包含一个 `UnparsedRawPersonalData` 的列表，保留原始文本与失败原因。
    """

    raw = str(raw_line or "").strip()
    if not raw:
        return [UnparsedRawPersonalData(个人数据=raw, 原因="空行，router 无法判定数据类型")]

    # 注意：这里有一些“当前体系不覆盖”的句式（来自金标准 self-test），
    # 这些句式即使能被某些解析器“形式上解析出数值/单位”，语义也会被误归类。
    #
    # 因此提供一个开关：
    # - strict_uncovered_to_unparsed=True（默认）：命中这些句式就直接 Unparsed（保持金标准一致）
    # - strict_uncovered_to_unparsed=False：不强制兜底，继续尝试其它数据类（用于探索/扩展覆盖）
    _uncovered_hints: list[str] = []

    # 常见“状态/评价”后缀：用于 router 判别
    # 说明：
    # - 如果一句话带有“正常/偏晚/偏低/欠规律...”等状态词，通常更像“周期数值多项总结/单日期数值单项总结”
    #   而不是“周期数值单项总结”（后者往往是“累计/总计/合计/平均...为XXX单位”的纯汇总值）。
    _STATUS_SUFFIX_RE = re.compile(
        r"(欠规律|不规律|偏晚|偏早|偏低|偏高|偏少|偏多|偏长|偏短|过少|过多|过长|过短|正常|中等|一般|良好|标准|达标|优秀|较低|较高|不足|过低|过高|不佳|较差|偏重|超重|肥胖|偏瘦|偏胖|偏轻)\s*$"
    )

    # 先处理已知“当前体系不覆盖/应兜底”的句式（来自金标准 test_UnparsedRawPersonalData）
    # - 目标差距类：锻炼时长XX分钟，距离目标XX分钟还差XX分钟
    # - 达到目标类：活动热量424千卡，达到目标270千卡
    # - 情绪占比类：8/8占比最高的情绪是不愉悦
    # - 经期/预测经期类：经期为4/7~4/19，共13天；1/25是经期第2天
    if ("距离目标" in raw and "还差" in raw) or ("占比最高的情绪" in raw and "是" in raw):
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为未覆盖的句式，按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：距离目标/情绪占比最高（历史上按金标准走兜底）")
    if "达到目标" in raw:
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为达到目标类句式（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：达到目标（历史上按金标准走兜底）")
    # - 标签/称号类：4/1~4/15 活力达人（非指标总结，按金标准走兜底）
    if "活力达人" in raw:
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为标签/称号类句式（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：标签/称号（活力达人）")
    if re.search(r"(预测)?经期", raw):
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为经期/预测经期相关句式（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：经期/预测经期")
    # - 单独的“时间跨度阈值”短句：超过两年（缺少指标语义，按金标准走兜底）
    if re.fullmatch(r"\s*超过\s*[一二两三四五六七八九十0-9]+(?:\.\d+)?\s*年\s*", raw):
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为孤立的时间跨度阈值短句（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：孤立时间跨度阈值（例如“超过两年”）")
    # - 目标完成度/KPI 类：步数目标完成天数10天，完成率32%（当前未覆盖，按金标准走兜底）
    if ("目标完成天数" in raw) or ("完成率" in raw and "目标" in raw):
        if strict_uncovered_to_unparsed:
            return [UnparsedRawPersonalData(个人数据=raw, 原因="router 识别为目标完成度/完成率类句式（当前未覆盖），按金标准走兜底")]
        _uncovered_hints.append("疑似未覆盖句式：目标完成度/完成率")

    def _has_date_token(s: str) -> bool:
        # 仅做“是否含日期线索”的粗判定（不要过重正则）
        return bool(
            re.search(
                # 只扩展“带年份的日期”支持 . / - 分隔，避免把 79.0 这类小数误判为日期
                r"(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2})|(\d{1,2}/\d{1,2})|(\d{1,2}月\d{1,2}(?:日)?)",
                s,
            )
        )

    def _has_time_token(s: str) -> bool:
        # 注意：Python 的 \w 会把中文也算作“单词字符”，导致 \b 在 "06:07的" 这类场景失效；
        # 因此这里用更稳的“直接搜索时间形态”。
        return bool(re.search(r"\d{1,2}:\d{2}", s))

    def _is_duration_like_value(s: str) -> bool:
        """
        判断字符串 s 是否更像“时长值”而不是“比值/速率”。

        背景：
        - router 在区分“周期数值单项总结”与“周期数值多项总结”时，会把“平均/最长/最短...”+时长
          的组合偏向解释为“周期数值多项总结”（例如：平均入睡时间23:20 / 最长睡眠时长7小时）。
        - 但像“平均配速 7.80分钟/公里”“平均心率 136次/分钟”这类 **含分母(/)** 的比值，
          虽然包含“分钟”等时间单位，但语义并不是“时长”，更符合“周期数值单项总结”的写法。
        """
        t = (s or "").strip()
        if not t:
            return False
        # 含分母(/)通常代表比值/速率，而非“时长”
        if "/" in t:
            return False
        return _has_time_unit(t)

    def _looks_like_stats_composite(s: str) -> bool:
        # 形如：指标名称：[...]
        if not _STATS_COMP_HEAD_RE.match(s):
            return False
        m = _STATS_COMP_HEAD_RE.match(s)
        if not m:
            return False
        rest = str(m.group("rest") or "")
        return _find_bracket_span(rest) is not None

    def _looks_like_single_date_stats_composite(s: str) -> bool:
        # 形如：8月8日压力15分-15分, 平均..., 最高..., 最低...
        if not _SINGLE_DATE_HEAD_RE.match(s):
            return False
        # 如果是“日期范围”（如 6/16~6/22），不要误判为单日期统计复合
        if re.match(
            r"^\s*(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
            s,
        ):
            return False
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        if len(segs) < 2:
            return False
        # 若整句更像“周期数值对比记录”（两段为/：子句 + 差异段），不要误判为统计复合
        # 例如：2025/3/6的平均浅睡比例为56，2025/2/27~2025/3/5平均浅睡比例为49.7，多6.3%
        if _looks_like_period_value_compare(s):
            return False
        # 有明显统计汇总词更可信
        return any(
            any(key in seg for key in ("平均", "最高", "最低", "最短", "最长", "最大", "最小"))
            for seg in segs[1:]
        )

    def _looks_like_single_date_value_summary(s: str) -> bool:
        # 形如：7/21锻炼时长17分钟偏低 / 4/23入睡时间01:40偏晚
        # - 必须是“单日期”而不是“日期范围”
        # - 必须包含数值（数字），否则更可能是“单日期文本总结”/“周期文本总结”
        # - 允许数值本身为时间点（如 01:40）；但要避免把“单指标的明细记录(YYYY/MM/DD HH:mm ...)”误判进来
        if re.search(r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}\s+\d{1,2}:\d{2}", s):
            return False
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        if not segs:
            return False
        first = segs[0]
        m = _SINGLE_DATE_HEAD_RE.match(first)
        if not m:
            return False
        # 防止把“日期范围”误判为单日期
        if re.match(
            r"^\s*(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
            first,
        ):
            return False
        rest = str(m.group("rest") or "")
        return bool(_FIRST_NUMBER_RE.search(rest))

    def _looks_like_single_date_text_summary(s: str) -> bool:
        # 形如：8/2 睡眠得分中等，睡眠质量良好
        # - 必须是“单日期”而不是“日期范围”
        # - 不应包含数值（数字）；否则更可能是“单日期数值单项总结”
        # - 不应包含 HH:mm 时间点；否则更可能是“单日期数值单项总结/周期数值多项总结”
        if _has_time_token(s):
            return False
        if re.search(r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}\s+\d{1,2}:\d{2}", s):
            return False
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        if not segs:
            return False
        first = segs[0]
        m = _SINGLE_DATE_HEAD_RE.match(first)
        if not m:
            return False
        # 防止把“日期范围”误判为单日期
        if re.match(
            r"^\s*(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
            first,
        ):
            return False
        rest = str(m.group("rest") or "").strip()
        if not rest:
            return False
        return not bool(_FIRST_NUMBER_RE.search(rest))

    def _looks_like_period_value_compare(s: str) -> bool:
        # 形如：A的...为..., B的...为..., 少/多...
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        if len(segs) < 3:
            return False

        m1 = _PVC_CLAUSE_RE.match(segs[0])
        m2 = _PVC_CLAUSE_RE.match(segs[1])
        if not (m1 and m2):
            return False

        # 关键：对比记录的日期段，允许“单日 vs 周期”或“周期 vs 周期”，因此不强制两段都有范围分隔符。
        # 但至少应出现一次范围分隔符，或者两段日期本身不同（避免把单日统计复合误判为对比）。
        r1 = str(m1.group("range") or "")
        r2 = str(m2.group("range") or "")
        range_sep_re = re.compile(r"(到|至|~|～|-|—)")
        if not (range_sep_re.search(r1) or range_sep_re.search(r2) or (r1.strip() and r2.strip() and r1.strip() != r2.strip())):
            return False

        # 第三段应当是“差异描述”，而不是 또 一个 “...的...为...” 子句
        third = segs[2]
        if _PVC_CLAUSE_RE.match(third):
            return False

        # 差异段需要包含对比逻辑词（少/多/高/低/增加/减少/提升/下降... 或 早/晚/提前/延后 等）
        if not re.search(r"(少|多|高|低|增加|减少|提升|下降|降低|升高|早|晚|提前|延后|推迟|延迟)", third):
            return False

        # 且差异段通常包含数字（如 3.0% / 6分钟）
        if not _FIRST_NUMBER_RE.search(third):
            return False

        return True

    def _looks_like_single_value(s: str) -> bool:
        # 形如：YYYY/M/D HH:mm的xxx为：...
        return bool(re.search(r"\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}\s+\d{1,2}:\d{2}", s))

    def _is_all_unparsed(objs: list[PersonalDataPattern]) -> bool:
        return all(isinstance(x, UnparsedRawPersonalData) for x in objs)

    def _safe_call(
        name: str,
        fn: Callable[[str], list[PersonalDataPattern]],
    ) -> list[PersonalDataPattern]:
        try:
            out = fn(raw)
            if not out:
                return [UnparsedRawPersonalData(个人数据=raw, 原因=f"router 调用解析器异常：{name}: 返回空列表")]
            return out
        except Exception as e:
            return [UnparsedRawPersonalData(个人数据=raw, 原因=f"router 调用解析器异常：{name}: {type(e).__name__}: {e}")]

    # 统一“尝试某个解析器”的入口：
    # - 记录已尝试的解析器，避免在不同分支重复调用
    # - 解析成功（返回不全是 Unparsed）则立即返回
    _tried: set[str] = set()

    def _try_parser(
        key: str,
        display_name: str,
        fn: Callable[[str], list[PersonalDataPattern]],
    ) -> list[PersonalDataPattern] | None:
        if key in _tried:
            return None
        _tried.add(key)
        objs = _safe_call(display_name, fn)
        return None if _is_all_unparsed(objs) else objs

    def _looks_like_period_summary(s: str) -> bool:
        # 周期汇总：通常存在“日期范围 + (的)? + 指标 + 为/： + 数值”
        #
        # 但真实数据/金标准里也大量出现“不带 为/：分隔符”的写法，例如：
        # - 12月1日到12月31日总计清醒次数31次
        # - 6/1~6/30累计跑步距离201.27千米
        #
        # 这类句式如果不特殊处理，会被 _looks_like_period_value_summary 误判为“周期数值多项总结”。
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        summary_hint_keywords = (
            "总计",
            "总共",
            "合计",
            "累计",
            "总和",
            "共计",
            "共",
            "平均",
            "最大",
            "最小",
            "最高",
            "最低",
        )
        # “总量/累计”更像周期汇总；而“平均/最长/最短/最高/最低...”在没有总量信号时，
        # 很多是“周期数值多项总结”的多条统计项（尤其当值是时长/时间点）。
        total_hint_keywords = ("总计", "总共", "合计", "累计", "总和", "共计", "总")
        stats_only_keywords = ("平均", "最长", "最短", "最大", "最小", "最高", "最低")
        for seg in segs:
            if _PERIOD_SUMMARY_SEG_RE_1.match(seg):
                # 若“为/：”后面带明显状态词（如 "23:20正常" / "15分钟偏低"），更像“周期数值多项总结”，避免误判为汇总。
                m1 = _PERIOD_SUMMARY_SEG_RE_1.match(seg)
                if m1:
                    val = str(m1.group("val") or "").strip()
                    nm = str(m1.group("name") or "").strip()
                    # 若 val 本身是“时间点”(HH:mm)，通常是睡眠/入睡时间这类统计，更像“周期数值多项总结”
                    # 例如：平均入睡时间23:20 / 最晚入睡时间01:30
                    if val and _has_time_token(val):
                        continue
                    if val and _STATUS_SUFFIX_RE.search(val):
                        continue
                return True
            m3 = _PERIOD_SUMMARY_SEG_RE_3.match(seg)
            if not m3:
                continue
            rest = str(m3.group("rest") or "").strip()
            if not rest:
                continue
            # 必须含数字，否则更像“周期文本总结”
            if not _FIRST_NUMBER_RE.search(rest):
                continue
            # 若包含“时间点”(HH:mm)，更像“周期数值多项总结”，避免被误判成周期汇总
            # 例如：平均零星小睡入睡时间13:00
            if _has_time_token(rest):
                continue
            # 若 rest 结尾出现明显“状态/评价词”，更像“周期数值多项总结”（例如：平均入睡时间23:20正常）
            if _STATUS_SUFFIX_RE.search(rest):
                continue
            # 有“汇总信号词”更强烈指向“周期汇总”
            if any(k in rest for k in summary_hint_keywords):
                return True
        return False

    def _looks_like_period_value_summary(s: str) -> bool:
        # 周期数值多项总结：日期(或日期范围)+rest，且 rest 内有数值；并且不应带 HH:mm 时间点
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        for seg in segs:
            m = _PERIOD_VALUE_SUMMARY_SEG_RE.match(seg)
            if not m:
                continue
            rest = str(m.group("rest") or "")
            if _FIRST_NUMBER_RE.search(rest):
                return True
        return False

    def _looks_like_period_text_summary(s: str) -> bool:
        # 周期文本总结：日期(或日期范围)+rest，且 rest 大概率不含数值（更偏文本状态）
        if _has_time_token(s):
            return False
        segs = [x.strip() for x in _SPLIT_RE.split(s) if x and x.strip()]
        for seg in segs:
            m = _PERIOD_TEXT_SUMMARY_SEG_RE.match(seg)
            if not m:
                continue
            rest = str(m.group("rest") or "")
            if rest and (not _FIRST_NUMBER_RE.search(rest)):
                return True
        return False

    has_date = _has_date_token(raw)

    # 1) 强特征：多日期统计复合（带“指标: [..]”）
    if _looks_like_stats_composite(raw):
        got = _try_parser(
            "单指标的统计复合记录",
            "单指标的统计复合记录",
            lambda s: SingleMetricStatsRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 2) 强特征：周期对比（三段：A、B、差异）
    if _looks_like_period_value_compare(raw):
        got = _try_parser(
            "周期数值对比记录",
            "周期数值对比记录",
            lambda s: PeriodValueCompareRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 3) 强特征：单日期统计复合（日期开头 + 统计汇总词）
    if _looks_like_single_date_stats_composite(raw):
        got = _try_parser(
            "单日期数值多项总结",
            "单日期数值多项总结",
            lambda s: SingleDateValueMultiSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 3.5) 强特征：单日期数值单项总结（单日期 + 数值 + 状态）
    if _looks_like_single_date_value_summary(raw):
        got = _try_parser(
            "单日期数值单项总结",
            "单日期数值单项总结",
            lambda s: SingleDateValueSingleSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 3.6) 强特征：单日期文本总结（单日期 + 文本状态；无数值）
    if _looks_like_single_date_text_summary(raw):
        got = _try_parser(
            "单日期文本总结",
            "单日期文本总结",
            lambda s: SingleDateTextSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 4) 强特征：单次记录（含 YYYY/MM/DD + HH:mm）
    if _looks_like_single_value(raw):
        got = _try_parser(
            "单指标的明细记录",
            "单指标的明细记录",
            lambda s: SingleMetricDetailRecord.from_raw_personal_data(s),
        )
        if got is not None:
            # 关键变更：多指标情况下，直接返回全部记录（不再只取第一个）
            return got

    # 5) 周期汇总（日期范围 + 为 + 数值）
    if _looks_like_period_summary(raw):
        # 关键定义（按你的描述）：
        # - 周期数值多项总结：一条记录里 >=1 个指标记录；可能带状态词；且常见为“逗号分隔多条指标”，后段可继承日期范围
        # - 周期数值单项总结：一条记录结构中只有 1 个指标+数字，且没有状态词
        segs = [x.strip() for x in _SPLIT_RE.split(raw) if x and x.strip()]
        seg_has_status = any(_STATUS_SUFFIX_RE.search(x.strip()) for x in segs)
        seg_with_num_cnt = sum(1 for x in segs if _FIRST_NUMBER_RE.search(x))

        # 多指标 or 有状态：优先走“多项总结”
        if seg_has_status or seg_with_num_cnt >= 2:
            got = _try_parser(
                "周期数值多项总结",
                "周期数值多项总结",
                lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s),
            )
            if got is not None:
                return got

        # 否则才尝试“单项总结”（严格单指标、无状态）
        got_single = _try_parser(
            "周期数值单项总结",
            "周期数值单项总结",
            lambda s: PeriodValueSingleSummaryRecord.from_raw_personal_data(s),
        )
        if got_single is not None:
            obj = got_single[0]
            # 若解析出多条指标，按定义应归入“多项总结”
            try:
                if getattr(obj, "记录条数", 1) > 1:
                    got2 = _try_parser(
                        "周期数值多项总结",
                        "周期数值多项总结(由单项降级)",
                        lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s),
                    )
                    if got2 is not None:
                        return got2
            except Exception:
                pass
            return got_single

    # 6) 仍然有日期线索：在 周期数值多项总结 vs 周期文本总结 之间做更谨慎的选择
    if has_date:
        pv_ok = _looks_like_period_value_summary(raw)
        pt_ok = _looks_like_period_text_summary(raw)

        if pv_ok and not pt_ok:
            got = _try_parser(
                "周期数值多项总结",
                "周期数值多项总结",
                lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s),
            )
            if got is not None:
                return got
        if pt_ok and not pv_ok:
            got = _try_parser(
                "周期文本总结",
                "周期文本总结",
                lambda s: PeriodTextSummaryRecord.from_raw_personal_data(s),
            )
            if got is not None:
                return got

        # 两者都可能：都试一遍，做二选一（尽量避免把“文本总结”误判成“数值总结”）
        # 这里故意不走 _try_parser：需要拿到两边结果做比较（但仍会把 key 记为 tried，避免后面重复扫描）
        _tried.add("周期数值多项总结")
        _tried.add("周期文本总结")
        objs_v = _safe_call("周期数值多项总结", lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s))
        objs_t = _safe_call("周期文本总结", lambda s: PeriodTextSummaryRecord.from_raw_personal_data(s))
        v_bad, t_bad = _is_all_unparsed(objs_v), _is_all_unparsed(objs_t)
        if not v_bad and t_bad:
            return objs_v
        if not t_bad and v_bad:
            return objs_t
        if not v_bad and not t_bad:
            obj_v = objs_v[0]
            obj_t = objs_t[0]
            # 以“是否解析到了数值列表且包含数字”作为更强信号
            v_vals = getattr(obj_v, "数值列表", [])
            t_descs = getattr(obj_t, "状态描述列表", [])
            v_has_num = any(_FIRST_NUMBER_RE.search(str(x or "")) for x in (v_vals or []))
            t_has_num = any(_FIRST_NUMBER_RE.search(str(x or "")) for x in (t_descs or []))
            if v_has_num and not t_has_num:
                return objs_v
            if t_has_num and not v_has_num:
                return objs_t
            # 默认：更偏向文本总结（更保守）
            return objs_t

        # 日期线索但都失败：给一个最后的兜底尝试（可能是边界格式）
        got = _try_parser(
            "周期文本总结",
            "周期文本总结(兜底)",
            lambda s: PeriodTextSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got
        got = _try_parser(
            "周期数值多项总结",
            "周期数值多项总结(兜底)",
            lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 7) 无日期线索：NoDateValueSummary vs NoTimestampTextSummary
    has_digit = bool(_FIRST_NUMBER_RE.search(raw))
    if has_digit:
        got = _try_parser(
            "无时间日期的数值总结",
            "无时间日期的数值总结",
            lambda s: NoDateValueSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got
        got = _try_parser(
            "无时间日期的文本总结",
            "无时间日期的文本总结",
            lambda s: NoTimestampTextSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got
    else:
        got = _try_parser(
            "无时间日期的文本总结",
            "无时间日期的文本总结",
            lambda s: NoTimestampTextSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got
        got = _try_parser(
            "无时间日期的数值总结",
            "无时间日期的数值总结",
            lambda s: NoDateValueSummaryRecord.from_raw_personal_data(s),
        )
        if got is not None:
            return got

    # 8) 兜底策略：把剩余未尝试的解析器全部跑一遍
    # 目的：确保“只要存在一个可兼容的数据类，就不会过早 Unparsed”。
    _ALL_PARSERS: list[tuple[str, str, Callable[[str], list[PersonalDataPattern]]]] = [
        ("单指标的统计复合记录", "单指标的统计复合记录(全量兜底)", lambda s: SingleMetricStatsRecord.from_raw_personal_data(s)),
        ("周期数值对比记录", "周期数值对比记录(全量兜底)", lambda s: PeriodValueCompareRecord.from_raw_personal_data(s)),
        ("单日期数值多项总结", "单日期数值多项总结(全量兜底)", lambda s: SingleDateValueMultiSummaryRecord.from_raw_personal_data(s)),
        ("单日期数值单项总结", "单日期数值单项总结(全量兜底)", lambda s: SingleDateValueSingleSummaryRecord.from_raw_personal_data(s)),
        ("单日期文本总结", "单日期文本总结(全量兜底)", lambda s: SingleDateTextSummaryRecord.from_raw_personal_data(s)),
        ("单指标的明细记录", "单指标的明细记录(全量兜底)", lambda s: SingleMetricDetailRecord.from_raw_personal_data(s)),
        ("周期数值单项总结", "周期数值单项总结(全量兜底)", lambda s: PeriodValueSingleSummaryRecord.from_raw_personal_data(s)),
        ("周期数值多项总结", "周期数值多项总结(全量兜底)", lambda s: PeriodValuemMultiSummaryRecord.from_raw_personal_data(s)),
        ("周期文本总结", "周期文本总结(全量兜底)", lambda s: PeriodTextSummaryRecord.from_raw_personal_data(s)),
        ("无时间日期的数值总结", "无时间日期的数值总结(全量兜底)", lambda s: NoDateValueSummaryRecord.from_raw_personal_data(s)),
        ("无时间日期的文本总结", "无时间日期的文本总结(全量兜底)", lambda s: NoTimestampTextSummaryRecord.from_raw_personal_data(s)),
    ]
    for key, disp, fn in _ALL_PARSERS:
        got = _try_parser(key, disp, fn)
        if got is not None:
            return got

    reason = "router 未能判定：所有数据类解析器均不兼容/解析失败"
    if _uncovered_hints:
        reason = reason + "；提示：" + "；".join(_uncovered_hints)
    return [UnparsedRawPersonalData(个人数据=raw, 原因=reason)]


def explode_newlines_and_route_to_dataclasses(
    text: str,
    *,
    strict_uncovered_to_unparsed: bool = True,
    keep_empty: bool = False,
    normalize_escaped_newlines: bool = True,
    max_rounds: int = 50,
) -> list[PersonalDataPattern]:
    """
    将一段文本按“换行裂变”拆成多行，然后对每一行做数据类分发与加载，最终返回总的数据类列表。

    设计意图：
    - 支持输入是一整段多行文本（例如复制粘贴的日志/对话/多条个人数据拼在一起）
    - 对 `\\n` / `\\r\\n` 这类“字面量转义换行”做可选归一化
    - 反复执行 splitlines() 直到不再产生新的换行（防御极端/脏数据）

    参数：
    - strict_uncovered_to_unparsed: 透传给 `route_raw_personal_data_to_dataclass`，默认 True（保持金标准策略）
    - keep_empty: 是否保留空行（默认 False，空行直接丢弃）
    - normalize_escaped_newlines: 是否把字面量 "\\n"/"\\r\\n"/"\\r" 转成真实换行（默认 True）
    - max_rounds: 最大裂变轮数（防御性参数，避免意外死循环）
    """
    raw_text = "" if text is None else str(text)
    if normalize_escaped_newlines and raw_text:
        # 注意顺序：先处理 \r\n，再处理单独的 \r 与 \n
        raw_text = raw_text.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")

    # “裂变”：反复 splitlines() + flatten，直到稳定
    chunks: list[str] = [raw_text]
    rounds = max(1, int(max_rounds))
    for _ in range(rounds):
        changed = False
        new_chunks: list[str] = []
        for ch in chunks:
            s = "" if ch is None else str(ch)
            parts = s.splitlines()
            if len(parts) <= 1:
                new_chunks.append(s)
                continue
            changed = True
            new_chunks.extend(parts)
        chunks = new_chunks
        if not changed:
            break

    # 清洗：strip + 去空
    lines: list[str] = []
    for ch in chunks:
        t = (ch or "").strip()
        if not t and not keep_empty:
            continue
        lines.append(t)

    # 对每一行路由并聚合
    out: list[PersonalDataPattern] = []
    for ln in lines:
        if (not ln) and (not keep_empty):
            continue
        out.extend(
            route_raw_personal_data_to_dataclass(
                ln,
                strict_uncovered_to_unparsed=strict_uncovered_to_unparsed,
            )
        )
    return out


def aggregate_patterns_to_formatted_text(
    patterns: Sequence[PersonalDataPattern],
    *,
    max_rows_per_table: int = 300,
    max_cell_len: int = 120,
    include_loose_lines: bool = True,
    loose_line_max_len: int = 200,
) -> str:
    """
    聚合/汇总一个数据类列表，并输出为可读文本（Markdown 表格）。

    目标：
    - “同类数据”尽量按 `实体类型` 聚合为表格/rows
    - 对无法抽取为表格行的对象，输出一个精简单行，并追加在主体输出之后，保证信息不丢失

    参数：
    - max_rows_per_table: 每个实体类型最多输出多少行（防止极端大文本）
    - max_cell_len: 单元格截断长度（仅影响格式化展示，不改数据）
    - include_loose_lines: 是否在主体输出后追加“零散/无法聚合”的单行列表
    - loose_line_max_len: 零散单行的截断长度
    """
    objs = list(patterns or [])
    if not objs:
        return ""

    def _safe_str(x: Any) -> str:
        return str(x if x is not None else "").strip()

    def _cell(v: Any) -> str:
        s = _safe_str(v)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        # markdown 表格需要转义管道符
        s = s.replace("|", r"\|")
        return _shorten(s, max_len=max_cell_len)

    def _pick_first_non_empty(*xs: Any) -> str:
        for x in xs:
            s = _safe_str(x)
            if s:
                return s
        return ""

    def _rows_from_obj(obj: PersonalDataPattern) -> list[dict[str, str]] | None:
        """
        把一个数据类对象尽量展开成 rows（用于 Markdown 表格聚合）。
        返回：
        - list[dict]：可聚合
        - None：不可聚合（走 loose line）
        """
        et = _safe_str(getattr(obj, "实体类型", ""))
        if not et:
            return None

        # 兜底类：不聚合
        if isinstance(obj, UnparsedRawPersonalData):
            return None

        # 单指标的明细记录：按(日期/时间/数值)展开
        if isinstance(obj, SingleMetricDetailRecord):
            core = getattr(obj, "核心字段", None)
            metric = _safe_str(getattr(core, "指标名称", "")) if core else ""
            unit = _safe_str(getattr(core, "单位", "")) if core else ""
            ds = list(getattr(obj, "日期列表", []) or [])
            ts = list(getattr(obj, "时间列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
            n = max(len(ds), len(ts), len(vs))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                d = _safe_str(ds[i] if i < len(ds) else "")
                t = _safe_str(ts[i] if i < len(ts) else "")
                v = _safe_str(vs[i] if i < len(vs) else "")
                v2 = _attach_unit_to_value(v, unit) if (v and unit and unit != "无") else v
                rows.append(
                    {
                        "日期": d,
                        "时间": t,
                        "指标": metric,
                        "数值": v2,
                        "单位": unit or "无",
                    }
                )
            return rows

        # 周期数值单项总结
        if isinstance(obj, PeriodValueSingleSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            n = max(len(starts), len(ends), len(names), len(vals), len(units))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                st = _safe_str(starts[i] if i < len(starts) else "")
                ed = _safe_str(ends[i] if i < len(ends) else "")
                nm = _safe_str(names[i] if i < len(names) else "")
                v = _safe_str(vals[i] if i < len(vals) else "")
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                # 数值列保持“纯数值”，单位单独放在“单位”列（便于后续再聚合/透视）
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                v = _trim_number_like(v)
                rows.append(
                    {
                        "开始": st,
                        "结束": ed,
                        "指标": nm,
                        "数值": v,
                        "单位": u,
                    }
                )
            return rows

        # 周期文本总结
        if isinstance(obj, PeriodTextSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(starts), len(ends), len(names), len(descs))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                rows.append(
                    {
                        "开始": _safe_str(starts[i] if i < len(starts) else ""),
                        "结束": _safe_str(ends[i] if i < len(ends) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "状态": _safe_str(descs[i] if i < len(descs) else ""),
                    }
                )
            return rows

        # 周期数值对比记录
        if isinstance(obj, PeriodValueCompareRecord):
            r1s = list(getattr(obj, "日期范围1列表", []) or [])
            v1s = list(getattr(obj, "数值1列表", []) or [])
            r2s = list(getattr(obj, "日期范围2列表", []) or [])
            v2s = list(getattr(obj, "数值2列表", []) or [])
            diffs = list(getattr(obj, "差异数值列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            logics = list(getattr(obj, "对比逻辑类型列表", []) or [])
            n = max(len(r1s), len(v1s), len(r2s), len(v2s), len(diffs), len(names), len(logics))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                rows.append(
                    {
                        "范围1": _safe_str(r1s[i] if i < len(r1s) else ""),
                        "值1": _safe_str(v1s[i] if i < len(v1s) else ""),
                        "范围2": _safe_str(r2s[i] if i < len(r2s) else ""),
                        "值2": _safe_str(v2s[i] if i < len(v2s) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "逻辑": _safe_str(logics[i] if i < len(logics) else ""),
                        "差异": _safe_str(diffs[i] if i < len(diffs) else ""),
                    }
                )
            return rows

        # 周期数值多项总结
        if isinstance(obj, PeriodValuemMultiSummaryRecord):
            starts = list(getattr(obj, "开始日期列表", []) or [])
            ends = list(getattr(obj, "结束日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(starts), len(ends), len(names), len(vals), len(units), len(sts))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []

            def _strip_unit_from_value_for_table(v: str, unit: str) -> str:
                """
                周期数值多项总结的 Markdown 表格展示：
                - “单位”已单列展示，因此“数值”列应尽量保持为纯数值/纯区间
                - 兼容区间值：如 "96%-96%" -> "96-96"
                """
                t = _safe_str(v)
                u = _safe_str(unit) or "无"
                if (not t) or (not u) or u == "无":
                    return t
                # 区间：两侧分别剥离单位（避免只剥离尾部导致残留）
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
                # 普通：剥离尾部单位
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                return t2 if t2 else t

            for i in range(n):
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                v = _safe_str(vals[i] if i < len(vals) else "")
                # 数值列保持“纯数值”，单位单独放在“单位”列（避免出现“数值=0.6次 + 单位=次”的重复展示）
                v = _trim_number_like(_strip_unit_from_value_for_table(v, u))
                rows.append(
                    {
                        "开始": _safe_str(starts[i] if i < len(starts) else ""),
                        "结束": _safe_str(ends[i] if i < len(ends) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "数值": v,
                        "单位": u,
                        "状态": _safe_str(sts[i] if i < len(sts) else ""),
                    }
                )
            return rows

        # 单日期数值单项总结
        if isinstance(obj, SingleDateValueSingleSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(ds), len(names), len(vals), len(units), len(sts))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                v = _safe_str(vals[i] if i < len(vals) else "")
                # 数值列保持“纯数值”，单位单独放在“单位”列（避免出现“数值=233千卡 + 单位=千卡”的重复展示）
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                v = _trim_number_like(v)
                rows.append(
                    {
                        "日期": _safe_str(ds[i] if i < len(ds) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "数值": v,
                        "单位": u,
                        "状态": _safe_str(sts[i] if i < len(sts) else ""),
                    }
                )
            return rows

        # 单日期文本总结
        if isinstance(obj, SingleDateTextSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(ds), len(names), len(descs))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                rows.append(
                    {
                        "日期": _safe_str(ds[i] if i < len(ds) else ""),
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "状态": _safe_str(descs[i] if i < len(descs) else ""),
                    }
                )
            return rows

        # 无时间日期的文本总结
        if isinstance(obj, NoTimestampTextSummaryRecord):
            names = list(getattr(obj, "指标名称列表", []) or [])
            descs = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(names), len(descs))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                rows.append({"指标": _safe_str(names[i] if i < len(names) else ""), "状态": _safe_str(descs[i] if i < len(descs) else "")})
            return rows

        # 无时间日期的数值总结
        if isinstance(obj, NoDateValueSummaryRecord):
            names = list(getattr(obj, "指标名称列表", []) or [])
            vals = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])
            n = max(len(names), len(vals), len(units), len(sts))
            if n <= 0:
                return None
            rows: list[dict[str, str]] = []
            for i in range(n):
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                v = _safe_str(vals[i] if i < len(vals) else "")
                # 数值列保持“纯数值”，单位单独放在“单位”列
                if v and u and u != "无":
                    v = re.sub(rf"{re.escape(u)}\s*$", "", v).strip() or v
                v = _trim_number_like(v)
                rows.append(
                    {
                        "指标": _safe_str(names[i] if i < len(names) else ""),
                        "数值": v,
                        "单位": u,
                        "状态": _safe_str(sts[i] if i < len(sts) else ""),
                    }
                )
            return rows

        # 单指标的统计复合记录：输出两类行（明细/汇总）
        if isinstance(obj, SingleMetricStatsRecord):
            core = getattr(obj, "核心字段", None)
            metric = _safe_str(getattr(core, "指标名称", "")) if core else ""
            unit = _safe_str(getattr(core, "单位", "")) if core else "无"
            # 明细
            ds = list(getattr(obj, "日期列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
            # 汇总
            sn = list(getattr(obj, "统计指标名称列表", []) or [])
            sv = list(getattr(obj, "统计数值列表", []) or [])
            ss = list(getattr(obj, "统计状态描述列表", []) or [])

            rows: list[dict[str, str]] = []
            n_d = max(len(ds), len(vs))
            for i in range(n_d):
                d = _safe_str(ds[i] if i < len(ds) else "")
                v = _safe_str(vs[i] if i < len(vs) else "")
                if not (d or v):
                    continue
                rows.append({"类别": "明细", "日期": d, "指标": metric, "数值": _attach_unit_to_value(v, unit), "单位": unit})

            n_s = max(len(sn), len(sv), len(ss))
            for i in range(n_s):
                nm = _safe_str(sn[i] if i < len(sn) else "")
                v = _safe_str(sv[i] if i < len(sv) else "")
                st = _safe_str(ss[i] if i < len(ss) else "")
                if not (nm or v or st):
                    continue
                rows.append({"类别": "汇总", "日期": "", "指标": nm or metric, "数值": v, "单位": unit, "状态": st})

            return rows if rows else None

        # 单日期数值多项总结（新版样式）：每个对象本身就是“一条指标记录”
        if isinstance(obj, SingleDateValueMultiSummaryRecord):
            ds = list(getattr(obj, "日期列表", []) or [])
            names = list(getattr(obj, "指标名称列表", []) or [])
            vs = list(getattr(obj, "数值列表", []) or [])
            units = list(getattr(obj, "单位列表", []) or [])
            sts = list(getattr(obj, "状态描述列表", []) or [])

            n = max(len(ds), len(names), len(vs), len(units), len(sts))
            if n <= 0:
                return None

            def _strip_unit_from_value_for_table(v: str, unit: str) -> str:
                """
                单日期多项总结的 Markdown 表格展示：
                - “单位”已单列展示，因此“数值”列应尽量保持为纯数值/纯区间
                - 修复区间值：如 "37°C-37°C" / "96%-96%" -> "37-37" / "96-96"
                """
                t = _safe_str(v)
                u = _safe_str(unit) or "无"
                if (not t) or (not u) or u == "无":
                    return t
                # 区间：两侧分别剥离单位（避免只剥离尾部导致残留）
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
                # 普通：剥离尾部单位
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                return t2 if t2 else t

            rows: list[dict[str, str]] = []
            for i in range(n):
                d = _safe_str(ds[i] if i < len(ds) else "")
                nm = _safe_str(names[i] if i < len(names) else "")
                v = _safe_str(vs[i] if i < len(vs) else "")
                u = _safe_str(units[i] if i < len(units) else "") or "无"
                st = _safe_str(sts[i] if i < len(sts) else "")
                rows.append(
                    {
                        "日期": d,
                        "指标": nm,
                        # 单位列已存在，数值列避免重复拼接单位
                        "数值": _trim_number_like(_strip_unit_from_value_for_table(v, u)),
                        "单位": u,
                        "状态": st,
                    }
                )
            return rows

        # 其它已知类型：当前不做强行表格化（走 loose line）
        return None

    def _loose_line(obj: PersonalDataPattern) -> str:
        try:
            s = obj.recover_to_raw_data()
        except Exception:
            try:
                s = obj.format_print(max_items=4, max_len=loose_line_max_len)
            except Exception:
                s = str(obj)
        s2 = _safe_str(s).replace("\n", " ").replace("\r", " ").strip()
        return _shorten(s2, max_len=loose_line_max_len)

    def _trim_number_like(s: str) -> str:
        """
        把看起来像数字的字符串做轻量格式化：
        - "7.70" -> "7.7"
        - "10.00" -> "10.0"（保留至少 1 位小数，尽量贴近你现有示例）
        - "5" -> "5"
        其它内容原样返回（如 "-" / "01:40" / "2小时49分钟"）。
        """
        t = (s or "").strip()
        if not t:
            return t
        if t in ("-", "—", "无", "None", "null", "NULL", "N/A", "NA"):
            return t
        # 只处理纯数字（含可选小数点）
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t):
            return t
        if "." not in t:
            return t
        a, b = t.split(".", 1)
        b2 = b.rstrip("0")
        if b2 == "":
            # 10.00 -> 10.0
            return f"{a}.0"
        return f"{a}.{b2}"

    def _pivot_single_metric_detail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        将 “单指标的明细记录” 的长表 rows：
          日期/时间/指标/数值/单位
        透视成宽表：
          日期/时间/运动类型 + 各指标列
        """
        if not rows:
            return rows

        # 指标名常见后缀（按长度降序，避免 “心率” 抢走 “最大心率”）
        # 指标名常见后缀及其排列组合增强，考虑更多“最大、最快、平均、最小、最低、最高”等前缀与项目的组合
        _stat_prefixes = [
            "最大",
            "最高",
            "最小",
            "最低",
            "平均",
            "最快",
            "最慢",
            "总"
        ]
        _base_metrics = [
            "心率",
            "步频",
            "配速",
            "速度",
            "步幅",
            "步数",
            "距离",
            "用时",
            "时长",
            "热量",
            "次数",
            "训练压力",
            "无氧训练压力",
            "有氧训练压力",
        ]
        # 构造前缀+基础指标的各种组合（按长度降序: 更长优先匹配）
        metric_suffixes = sorted(
            set(
                list(_base_metrics)
                + [f"{prefix}{metric}" for prefix in _stat_prefixes for metric in _base_metrics]
            ),
            key=lambda x: -len(x)
        )

        def _split_activity_and_metric(full_metric: str) -> tuple[str, str]:
            m = (full_metric or "").strip()
            for suf in metric_suffixes:
                if m.endswith(suf) and len(m) > len(suf):
                    return m[: -len(suf)].strip(), suf
            return "", m

        def _col_name(metric_name: str, unit: str) -> str:
            u = (unit or "").strip() or "无"
            if not metric_name:
                return "数值"
            if (not u) or u == "无":
                return metric_name
            return f"{metric_name} ({u})"

        def _strip_unit_from_value(v: str, unit: str) -> str:
            t = (v or "").strip()
            u = (unit or "").strip() or "无"
            if not t:
                return t
            # 特例：配速/比率类（单位形如 "分钟/公里"、"小时/公里"、"秒/公里"）
            # 原始值常写作 "7分52秒/公里" 或 "7.8分钟/公里"：
            # - 表头已显示 (分钟/公里)，单元格里再带 "/公里" 会显得“单位重复”
            # - 同时，解析阶段可能会把单位推断为 "分钟/公里"，但值里只包含 "/公里"（不完全等于单位串），
            #   仅靠“尾部匹配单位”无法剥离。
            if u != "无" and "/" in u and "/" in t and u.startswith(("分钟/", "小时/", "秒/")):
                left = t.split("/", 1)[0].strip()
                # 若左侧包含“分钟/小时/秒”字样，去掉以保持与表头一致（表头已经体现单位）
                if u.startswith("分钟/"):
                    left2 = re.sub(r"\s*分钟\s*$", "", left).strip()
                    left = left2 if left2 else left
                elif u.startswith("小时/"):
                    left2 = re.sub(r"\s*小时\s*$", "", left).strip()
                    left = left2 if left2 else left
                elif u.startswith("秒/"):
                    left2 = re.sub(r"\s*秒\s*$", "", left).strip()
                    left = left2 if left2 else left
                return left
            # 区间值：形如 "37°C-37°C" / "97%-97%" / "17分-17分"
            # 之前仅做“尾部单位剥离”，会导致 "37°C-37°C" -> "37°C-37"（只去掉后半段单位）。
            # 这里对分隔符两侧分别剥离单位，避免表头已带单位时单元格仍残留单位。
            if u and u != "无":
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
            if u and u != "无":
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                t = t2 if t2 else t
            return t

        # group by (日期, 时间, 运动类型)
        groups: dict[tuple[str, str, str], dict[str, str]] = {}
        # 记录列名出现顺序（稳定）
        col_seen: list[str] = []
        col_seen_set: set[str] = set()

        for r in rows:
            d = _safe_str(r.get("日期", ""))
            t = _safe_str(r.get("时间", ""))
            full_metric = _safe_str(r.get("指标", ""))
            unit = _safe_str(r.get("单位", "")) or "无"
            val_with_unit = _safe_str(r.get("数值", ""))

            act, metric_name = _split_activity_and_metric(full_metric)
            col = _col_name(metric_name or full_metric, unit)

            v0 = _strip_unit_from_value(val_with_unit, unit)
            v0 = _trim_number_like(v0)
            if _is_missing_token(v0):
                v0 = "-"

            key = (d, t, act)
            row = groups.get(key)
            if row is None:
                row = {"日期": d, "时间": t, "运动类型": act}
                groups[key] = row
            # 同一 cell 多次出现：后写覆盖（更接近“最新值”直觉）
            row[col] = v0

            if col not in col_seen_set:
                col_seen.append(col)
                col_seen_set.add(col)

        def _parse_date_for_sort(s: str) -> tuple[int, int, int, str]:
            """
            将日期字符串解析成可排序的 (Y, M, D, raw)。
            支持：
            - YYYY/M/D, YYYY-MM-DD, YYYY.MM.DD
            - 兜底返回 (9999, 99, 99, raw)
            """
            raw = (s or "").strip()
            if not raw:
                return (9999, 99, 99, raw)
            m = re.fullmatch(r"(?P<y>\d{4})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m:
                return (int(m.group("y")), int(m.group("m")), int(m.group("d")), raw)
            # 其它格式暂不强行推断年（月/日、x月x日等），避免误排序；留给 raw 稳定兜底
            return (9999, 99, 99, raw)

        def _parse_time_for_sort(s: str) -> tuple[int, int, str]:
            raw = (s or "").strip()
            if not raw:
                return (99, 99, raw)
            m = re.fullmatch(r"(?P<h>\d{1,2}):(?P<mi>\d{2})", raw)
            if m:
                return (int(m.group("h")), int(m.group("mi")), raw)
            return (99, 99, raw)

        # 输出按 “真实日期/时间” 排序（修复 2/10 排在 2/2 前的问题）
        def _group_sort_key(k: tuple[str, str, str]) -> tuple[int, int, int, int, int, str, str, str]:
            d, t, act = k
            y, mo, da, d_raw = _parse_date_for_sort(d)
            hh, mm, t_raw = _parse_time_for_sort(t)
            # y/mo/da/hh/mm 为主排序；raw 保证同类场景稳定；act 最后
            return (y, mo, da, hh, mm, d_raw, t_raw, act)

        out = [groups[k] for k in sorted(groups.keys(), key=_group_sort_key)]
        # 确保每行都有所有列（缺失填 "-"，更接近你示例里的 "-"）
        for row in out:
            for col in col_seen:
                if col not in row:
                    row[col] = "-"
        return out

    def _compact_period_value_single_summary_rows(rows: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        """
        周期数值单项总结：若全部行的(开始,结束)一致，则把日期范围挪到标题里，并去掉重复列。
        返回：(title_suffix, new_rows)
        """
        if not rows:
            return None, rows
        starts = {(_safe_str(r.get("开始", ""))) for r in rows if _safe_str(r.get("开始", ""))}
        ends = {(_safe_str(r.get("结束", ""))) for r in rows if _safe_str(r.get("结束", ""))}
        if len(starts) == 1 and len(ends) == 1:
            st = next(iter(starts))
            ed = next(iter(ends))
            new_rows: list[dict[str, str]] = []
            for r in rows:
                r2 = dict(r)
                r2.pop("开始", None)
                r2.pop("结束", None)
                new_rows.append(r2)
            return f"{st}~{ed}", new_rows
        return None, rows

    def _compact_rows_by_same_date_range(
        rows: list[dict[str, str]],
        *,
        start_key: str = "开始",
        end_key: str = "结束",
        drop_keys: tuple[str, str] | None = None,
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        通用"日期范围上移标题"压缩：
        - 若 rows 中所有非空的 (start_key, end_key) 仅有一组取值，则：
          - 返回 title_suffix = "start~end"（若 start==end，则返回 start）
          - 并从每行里删除 start_key/end_key（避免表格重复列）
        - 否则原样返回。
        """
        if not rows:
            return None, rows

        sk = str(start_key or "").strip() or "开始"
        ek = str(end_key or "").strip() or "结束"
        dk = drop_keys if drop_keys is not None else (sk, ek)

        starts = {(_safe_str(r.get(sk, ""))) for r in rows if _safe_str(r.get(sk, ""))}
        ends = {(_safe_str(r.get(ek, ""))) for r in rows if _safe_str(r.get(ek, ""))}
        if len(starts) == 1 and len(ends) == 1:
            st = next(iter(starts))
            ed = next(iter(ends))
            suffix = st if (st and st == ed) else f"{st}~{ed}"
            new_rows: list[dict[str, str]] = []
            for r in rows:
                r2 = dict(r)
                for k in dk:
                    r2.pop(k, None)
                new_rows.append(r2)
            return suffix, new_rows
        return None, rows

    def _pivot_stats_composite_detail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        将"单指标的统计复合记录"的明细部分透视成宽表：
          类别/日期/指标/数值/单位
        透视成：
          日期 + 各指标列
        """
        if not rows:
            return rows
        
        # 只处理明细行
        detail_rows = [r for r in rows if _safe_str(r.get("类别", "")) == "明细"]
        if not detail_rows:
            return rows
        
        def _col_name(metric_name: str, unit: str) -> str:
            u = (unit or "").strip() or "无"
            if not metric_name:
                return "数值"
            if (not u) or u == "无":
                return metric_name
            return f"{metric_name} ({u})"
        
        def _strip_unit_from_value(v: str, unit: str) -> str:
            t = (v or "").strip()
            u = (unit or "").strip() or "无"
            if not t:
                return t
            # 区间值：形如 "37°C-37°C" / "97%-97%" / "17分-17分"
            # 之前仅做“尾部单位剥离”，会导致 "37°C-37°C" -> "37°C-37"（只去掉后半段单位）。
            if u and u != "无":
                for sep in ("-", "～", "~", "—", "−"):
                    if sep in t and not t.startswith(sep):
                        parts = [p.strip() for p in t.split(sep)]
                        if len(parts) == 2 and (any(ch.isdigit() for ch in parts[0]) and any(ch.isdigit() for ch in parts[1])):
                            p0 = re.sub(rf"{re.escape(u)}\s*$", "", parts[0]).strip() or parts[0]
                            p1 = re.sub(rf"{re.escape(u)}\s*$", "", parts[1]).strip() or parts[1]
                            return f"{p0}{sep}{p1}".strip()
            if u and u != "无":
                t2 = re.sub(rf"{re.escape(u)}\s*$", "", t).strip()
                t = t2 if t2 else t
            return t
        
        # group by 日期
        groups: dict[str, dict[str, str]] = {}
        # 记录列名出现顺序（稳定）
        col_seen: list[str] = []
        col_seen_set: set[str] = set()
        
        for r in detail_rows:
            d = _safe_str(r.get("日期", ""))
            metric = _safe_str(r.get("指标", ""))
            unit = _safe_str(r.get("单位", "")) or "无"
            val_with_unit = _safe_str(r.get("数值", ""))
            
            col = _col_name(metric, unit)
            v0 = _strip_unit_from_value(val_with_unit, unit)
            v0 = _trim_number_like(v0)
            if _is_missing_token(v0):
                v0 = "-"
            
            row = groups.get(d)
            if row is None:
                row = {"日期": d}
                groups[d] = row
            # 同一 cell 多次出现：后写覆盖（更接近"最新值"直觉）
            row[col] = v0
            
            if col not in col_seen_set:
                col_seen.append(col)
                col_seen_set.add(col)
        
        def _parse_date_for_sort(s: str) -> tuple[int, int, int, str]:
            """
            将日期字符串解析成可排序的 (Y, M, D, raw)。
            支持：
            - YYYY/M/D, YYYY-MM-DD, YYYY.MM.DD
            - M/D, MM/DD
            - x月x日
            - 兜底返回 (9999, 99, 99, raw)
            """
            raw = (s or "").strip()
            if not raw:
                return (9999, 99, 99, raw)
            # YYYY/M/D
            m = re.fullmatch(r"(?P<y>\d{4})[\/\.-](?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m:
                return (int(m.group("y")), int(m.group("m")), int(m.group("d")), raw)
            # M/D
            m = re.fullmatch(r"(?P<m>\d{1,2})[\/\.-](?P<d>\d{1,2})", raw)
            if m:
                return (9999, int(m.group("m")), int(m.group("d")), raw)
            # x月x日
            m = re.fullmatch(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})日", raw)
            if m:
                return (9999, int(m.group("m")), int(m.group("d")), raw)
            return (9999, 99, 99, raw)
        
        # 输出按日期排序
        def _date_sort_key(d: str) -> tuple[int, int, int, str]:
            y, mo, da, d_raw = _parse_date_for_sort(d)
            return (y, mo, da, d_raw)
        
        out = [groups[k] for k in sorted(groups.keys(), key=_date_sort_key)]
        # 确保每行都有所有列（缺失填 "-"）
        for row in out:
            for col in col_seen:
                if col not in row:
                    row[col] = "-"
        
        # 只返回透视后的明细行（汇总行由调用方单独处理）
        return out
    
    def _format_stats_composite_summary_as_sentences(rows: list[dict[str, str]]) -> list[str]:
        """
        将"单指标的统计复合记录"的汇总部分转换为句子列表。
        """
        summary_rows = [r for r in rows if _safe_str(r.get("类别", "")) == "汇总"]
        if not summary_rows:
            return []
        
        sentences: list[str] = []
        for r in summary_rows:
            metric = _safe_str(r.get("指标", ""))
            value = _safe_str(r.get("数值", ""))
            status = _safe_str(r.get("状态", ""))
            
            if not metric and not value:
                continue
            
            parts = []
            if metric:
                parts.append(metric)
            if value:
                parts.append(value)
            if status and status not in ("无", "-", ""):
                parts.append(status)
            
            if parts:
                sentences.append("".join(parts))
        
        return sentences
    
    def _compact_rows_by_same_date(
        rows: list[dict[str, str]],
        *,
        date_key: str = "日期",
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        单日期类型"日期上移标题"压缩：
        - 若 rows 中所有非空的 date_key 仅有一个取值，则：
          - 返回 title_suffix = 该日期值
          - 并从每行里删除 date_key（避免表格重复列）
        - 否则原样返回。
        """
        if not rows:
            return None, rows

        dk = str(date_key or "").strip() or "日期"
        dates = {(_safe_str(r.get(dk, ""))) for r in rows if _safe_str(r.get(dk, ""))}
        if len(dates) == 1:
            date_val = next(iter(dates))
            new_rows: list[dict[str, str]] = []
            for r in rows:
                r2 = dict(r)
                r2.pop(dk, None)
                new_rows.append(r2)
            return date_val, new_rows
        return None, rows

    def _group_rows_by_date(
        rows: list[dict[str, str]],
        *,
        date_key: str = "日期",
    ) -> list[tuple[str, list[dict[str, str]]]]:
        """
        按日期分组行：
        - 返回 [(date1, rows1), (date2, rows2), ...]
        - 每个日期组的行中，date_key 列会被删除
        """
        if not rows:
            return []

        dk = str(date_key or "").strip() or "日期"
        groups: dict[str, list[dict[str, str]]] = {}
        empty_date_rows: list[dict[str, str]] = []

        for r in rows:
            date_val = _safe_str(r.get(dk, ""))
            if date_val:
                if date_val not in groups:
                    groups[date_val] = []
                r2 = dict(r)
                r2.pop(dk, None)
                groups[date_val].append(r2)
            else:
                r2 = dict(r)
                r2.pop(dk, None)
                empty_date_rows.append(r2)

        result: list[tuple[str, list[dict[str, str]]]] = []
        # 按日期排序（保持稳定顺序）
        for date_val in sorted(groups.keys()):
            result.append((date_val, groups[date_val]))
        # 空日期的行放在最后
        if empty_date_rows:
            result.append(("", empty_date_rows))

        return result

    # 1) 聚合为 tables
    tables: dict[str, list[dict[str, str]]] = {}
    loose: list[str] = []
    # 单独处理"单指标的统计复合记录"，保留对象边界信息
    stats_composite_objs: list[PersonalDataPattern] = []
    for obj in objs:
        et = _safe_str(getattr(obj, "实体类型", "")) or "未定义"
        if et == "单指标的统计复合记录":
            stats_composite_objs.append(obj)
            continue
        rows = _rows_from_obj(obj)
        if rows is None:
            if include_loose_lines:
                loose.append(_loose_line(obj))
            continue
        tables.setdefault(et, []).extend(rows)

    # 1.5) 二次聚合（更高级的表格）
    # - 单指标明细：长表 -> 宽表（按 日期/时间/运动类型 pivot）
    # - 周期数值单项总结：若日期范围一致，去掉重复列
    # - 周期文本总结 / 周期数值多项总结：若日期范围一致，去掉重复列（与单项总结一致）
    # - 单日期文本总结 / 单日期数值多项总结 / 单日期数值单项总结：若日期一致，去掉重复列
    # - 单指标的统计复合记录：明细部分透视成宽表，汇总部分按对象分组单独处理
    table_title_suffix: dict[str, str] = {}
    stats_composite_summary_sentences_by_obj: dict[str, list[list[str]]] = {}
    
    # 处理"单指标的统计复合记录"：分别处理每个对象
    if stats_composite_objs:
        all_detail_rows: list[dict[str, str]] = []
        summary_sentences_by_obj: list[list[str]] = []
        
        for obj in stats_composite_objs:
            rows = _rows_from_obj(obj)
            if rows is None:
                if include_loose_lines:
                    loose.append(_loose_line(obj))
                continue
            # 提取该对象的汇总句子
            obj_summary_sentences = _format_stats_composite_summary_as_sentences(rows)
            summary_sentences_by_obj.append(obj_summary_sentences)
            # 收集明细行（用于后续透视）
            detail_rows = [r for r in rows if _safe_str(r.get("类别", "")) == "明细"]
            all_detail_rows.extend(detail_rows)
        
        # 透视所有明细行
        if all_detail_rows:
            tables["单指标的统计复合记录"] = _pivot_stats_composite_detail_rows(all_detail_rows)
            stats_composite_summary_sentences_by_obj["单指标的统计复合记录"] = summary_sentences_by_obj
    
    if "单指标的明细记录" in tables:
        tables["单指标的明细记录"] = _pivot_single_metric_detail_rows(tables["单指标的明细记录"])
    if "周期数值单项总结" in tables:
        suf, new_rows = _compact_period_value_single_summary_rows(tables["周期数值单项总结"])
        tables["周期数值单项总结"] = new_rows
        if suf:
            table_title_suffix["周期数值单项总结"] = suf
    if "周期文本总结" in tables:
        suf, new_rows = _compact_rows_by_same_date_range(tables["周期文本总结"])
        tables["周期文本总结"] = new_rows
        if suf:
            table_title_suffix["周期文本总结"] = suf
    if "周期数值多项总结" in tables:
        suf, new_rows = _compact_rows_by_same_date_range(tables["周期数值多项总结"])
        tables["周期数值多项总结"] = new_rows
        if suf:
            table_title_suffix["周期数值多项总结"] = suf
    # 单日期文本总结、单日期数值多项总结和单日期数值单项总结：按日期分组，每个日期一个表格
    # 这三个类型不在这里处理，而是在输出时按日期分组

    # 2) 截断表格行数
    max_n = max(0, int(max_rows_per_table))
    if max_n > 0:
        for k in list(tables.keys()):
            if len(tables[k]) > max_n:
                tables[k] = tables[k][:max_n]

    # 输出：markdown
    parts: list[str] = []
    for et, rows in tables.items():
        if not rows:
            continue

        # 单日期文本总结、单日期数值多项总结和单日期数值单项总结：按日期分组输出
        if et in ("单日期文本总结", "单日期数值多项总结", "单日期数值单项总结"):
            date_groups = _group_rows_by_date(rows)
            for date_val, group_rows in date_groups:
                if not group_rows:
                    continue
                # 列顺序：优先常见字段
                prefer_cols = [
                    "类别",
                    "日期",
                    "时间",
                    "运动类型",
                    "开始",
                    "结束",
                    "范围1",
                    "值1",
                    "范围2",
                    "值2",
                    "指标",
                    "数值",
                    "单位",
                    "状态",
                    "逻辑",
                    "差异",
                ]

                all_cols: list[str] = []
                seen: set[str] = set()
                # 先按 prefer_cols 收集
                for c in prefer_cols:
                    if any(c in r for r in group_rows):
                        all_cols.append(c)
                        seen.add(c)
                # 再补齐剩余列（稳定排序）
                more = sorted({c for r in group_rows for c in r.keys()} - seen)
                all_cols.extend(more)

                title = et
                if date_val:
                    title = f"{et}（{date_val}）"
                parts.append(f"### {title}")
                header = "| " + " | ".join(all_cols) + " |"
                sep = "| " + " | ".join(["---"] * len(all_cols)) + " |"
                parts.append(header)
                parts.append(sep)
                for r in group_rows:
                    parts.append("| " + " | ".join(_cell(r.get(c, "")) for c in all_cols) + " |")
                parts.append("")  # 空行分隔
            continue

        # 列顺序：优先常见字段
        prefer_cols = [
            "类别",
            "日期",
            "时间",
            "运动类型",
            "开始",
            "结束",
            "范围1",
            "值1",
            "范围2",
            "值2",
            "指标",
            "数值",
            "单位",
            "状态",
            "逻辑",
            "差异",
        ]

        # "单指标的明细记录(宽表)"：把动态指标列按更自然的顺序输出
        if et == "单指标的明细记录":
            metric_order = [
                "距离",
                "用时",
                "时长",
                "配速",
                "最快配速",
                "速度",
                "热量",
                "心率",
                "最大心率",
                "步频",
                "最大步频",
                "步幅",
            ]

            def _metric_col_sort_key(c: str) -> tuple[int, str]:
                # c 形如 "距离 (千米)" / "步幅" 等
                name = c.split(" (", 1)[0].strip()
                if name in ("日期", "时间", "运动类型"):
                    return (-1, name)
                if name in metric_order:
                    return (metric_order.index(name), c)
                return (9999, c)

            # 把所有"非基础列"视作指标列并按 metric_order 排序
            base_cols = {"日期", "时间", "运动类型"}
            metric_cols = sorted({c for r in rows for c in r.keys()} - set(prefer_cols) - base_cols, key=_metric_col_sort_key)
            # 保证基础列优先，其次指标列（动态），最后其它列
            prefer_cols = ["日期", "时间", "运动类型"] + metric_cols
            # 兼容：若还有其它列（非常规），后面仍会走 more 补齐
        
        # "单指标的统计复合记录(宽表)"：把动态指标列按更自然的顺序输出
        if et == "单指标的统计复合记录":
            metric_order = [
                "运动次数",
                "运动时长",
                "运动热量",
                "活动小时数",
                "活动热量",
                "锻炼时长",
                "距离",
                "用时",
                "时长",
                "配速",
                "最快配速",
                "速度",
                "热量",
                "心率",
                "最大心率",
                "步频",
                "最大步频",
                "步幅",
            ]

            def _metric_col_sort_key(c: str) -> tuple[int, str]:
                # c 形如 "运动次数 (次)" / "运动时长" 等
                name = c.split(" (", 1)[0].strip()
                if name == "日期":
                    return (-1, name)
                if name in metric_order:
                    return (metric_order.index(name), c)
                return (9999, c)

            # 把所有"非基础列"视作指标列并按 metric_order 排序
            base_cols = {"日期"}
            metric_cols = sorted({c for r in rows for c in r.keys()} - set(prefer_cols) - base_cols, key=_metric_col_sort_key)
            # 保证基础列优先，其次指标列（动态），最后其它列
            prefer_cols = ["日期"] + metric_cols
            # 兼容：若还有其它列（非常规），后面仍会走 more 补齐

        all_cols: list[str] = []
        seen: set[str] = set()
        # 先按 prefer_cols 收集
        for c in prefer_cols:
            if any(c in r for r in rows):
                all_cols.append(c)
                seen.add(c)
        # 再补齐剩余列（稳定排序）
        more = sorted({c for r in rows for c in r.keys()} - seen)
        all_cols.extend(more)

        title = et
        if et in table_title_suffix:
            title = f"{et}（{table_title_suffix[et]}）"
        parts.append(f"### {title}")
        header = "| " + " | ".join(all_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(all_cols)) + " |"
        parts.append(header)
        parts.append(sep)
        for r in rows:
            parts.append("| " + " | ".join(_cell(r.get(c, "")) for c in all_cols) + " |")
        parts.append("")  # 空行分隔
        
        # 单指标的统计复合记录：在表格后输出汇总句子（按对象分组）
        if et == "单指标的统计复合记录" and et in stats_composite_summary_sentences_by_obj:
            obj_summary_groups = stats_composite_summary_sentences_by_obj[et]
            for obj_summary_sentences in obj_summary_groups:
                if obj_summary_sentences:
                    # 每个对象的汇总句子用逗号连接，以句号结尾
                    parts.append("，".join(obj_summary_sentences) + "。")
            if any(obj_summary_groups):
                parts.append("")  # 空行分隔

    if include_loose_lines and loose:
        parts.append("### 零散/无法聚合")
        for ln in loose:
            parts.append(f"- {ln}")

    return "\n".join(parts).rstrip() + "\n"


# ========= 解析：单指标的明细记录（从原始一行文本抽取多条记录） =========
_SPLIT_RE = re.compile(r"[，,]\s*")

# 常见：YYYY/M/D HH:mm的{指标}为：{数值}{单位}
_SEG_RE_1 = re.compile(
    r"^\s*(?P<date>\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2})\s+(?P<time>\d{1,2}:\d{2})的(?P<name>.+?)为[:：]\s*(?P<val>.+?)\s*$"
)

# 宽松兜底：YYYY/M/D [HH:mm] ...（不强依赖“的/为：”）
_SEG_RE_2 = re.compile(
    # 只扩展“带年份”的日期支持 . / - 分隔；不扩展 MM.DD 以避免 79.0 误判日期
    r"^\s*(?P<date>\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(?P<time>\d{1,2}:\d{2})?\s*(?P<rest>.+?)\s*$"
)

_FIRST_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _strip_leading_de(s: str) -> str:
    """
    清理中文里常见的前导“的”（例如：'的锻炼时长' -> '锻炼时长'）。
    """
    t = (s or "").strip()
    while t.startswith("的") and len(t) > 1:
        t = t[1:].lstrip()
    return t


def _normalize_value_and_unit(value_str: str, vt: ValueType, unit: str) -> tuple[str, str]:
    """
    规范化“数值+单位”：
    - Timestamp：单位固定为“无”
    - Int/Float：尽量把尾部单位从数值里剥离（数值保存纯数字字符串）
    - FloatRange：把区间内的单位剥离（如 96%-98% -> 96-98，单位=%）
    - Duration：
        - 单一单位（如 15分钟/0.47小时/30秒）：剥离单位，保留纯数字，单位保留
        - 复合时长（如 2小时49分钟/7分42秒）：保留原串，单位置为“无”（避免重复与误归一）
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
        # 这类表达的“单位”应当保留为 "分钟/公里"（或 "小时/公里"），不应被当作“复合时长”而置为 "无"。
        if u != "无" and "/" in u and "/" in s:
            left, _right = (p.strip() for p in s.split("/", 1))
            # 若左侧是“纯数值 + 时间单位词”，可剥离时间单位词以保持 value 更干净
            if u.startswith("分钟/"):
                left = re.sub(r"\s*分钟\s*$", "", left).strip() or left
            elif u.startswith("小时/"):
                left = re.sub(r"\s*小时\s*$", "", left).strip() or left
            elif u.startswith("秒/"):
                left = re.sub(r"\s*秒\s*$", "", left).strip() or left
            return left, u

        m_simple = re.fullmatch(r"\s*(?P<num>[-+]?\d+(?:\.\d+)?)\s*(?P<u>小时|分钟|秒)\s*", s)
        if m_simple:
            return m_simple.group("num"), m_simple.group("u")
        # 复合：保留原串，单位置“无”以避免重复
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
    return v if v.endswith(u) else f"{v}{u}"


def _join_raw_segments(segs: Sequence[str]) -> str:
    """
    将若干“原始句子片段”拼成一行，使用中文逗号作为默认分隔。
    """
    xs = [str(x).strip() for x in (segs or []) if str(x).strip()]
    return "，".join(xs).strip()


def _is_time_token(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}", str(s or "").strip()))


def _is_missing_token(s: str) -> bool:
    t = str(s or "").strip()
    return (not t) or t in ("无", "None", "null", "NULL", "N/A", "NA")


def _format_date_range(st: str, ed: str) -> str:
    s = str(st or "").strip()
    e = str(ed or "").strip()
    if not s and not e:
        return ""
    if s and (not e or e == s):
        return s
    if not s and e:
        return e
    return f"{s}~{e}"


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
        if st and ed and ed != st:
            segs.append(f"{st}~{ed}{nm}{desc}".strip())
        elif st:
            segs.append(f"{st}{nm}{desc}".strip())
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


def _infer_value_type_from_value_str(value_str: str) -> ValueType:
    s = (value_str or "").strip()
    # 先判“时间点”格式（例如 06:28）
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        return "Timestamp"

    # “时间/距离”的配速类比值（如 7.80分钟/公里）：
    # - 纯数字 + (小时|分钟|秒|分)/(公里|千米|米|km) 更适合作为可计算的数值 Float/Int
    # - 若是复合时长（如 7分42秒/公里、1小时02分钟/公里）则仍按 Duration（更像“时长字符串”）处理
    m_simple_pace = re.fullmatch(
        r"\s*(?P<num>[-+]?\d+(?:\.\d+)?)\s*(?P<tunit>小时|分钟|秒|分)\s*/\s*(?P<dunit>公里|千米|米|km|KM|Km)\s*",
        s,
    )
    if m_simple_pace:
        num = m_simple_pace.group("num")
        return "Int" if re.fullmatch(r"[-+]?\d+", num) else "Float"

    # 判定“数值范围”（FloatRange），例如：
    # - 96%-98%
    # - 15分-17分
    # - 22-26
    # 注意：要求整段形如 <num><unit>? <sep> <num><unit>?，避免误伤日期等。
    m_range = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?)\s*([^\d\s]+)?\s*[-~～—]\s*([-+]?\d+(?:\.\d+)?)\s*([^\d\s]+)?\s*",
        s,
    )
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
            if m_int and re.fullmatch(r"[-+]?\d+", m_int.group(0)):
                return "Int"
            return "Float"

    # 不含分母的“时长描述”
    #
    # 重要：像 "0.47小时" / "15分钟" 这类“纯数值 + 单一时间单位”的表达，
    # 语义上仍是“持续时长”，应归类为 Duration（而不是 Float/Int）。
    # 复合时长（如 "2小时49分钟"、"7分42秒"）也同样是 Duration。
    m_simple_time = re.fullmatch(r"\s*(?P<num>[-+]?\d+(?:\.\d+)?)\s*(?P<u>小时|分钟|秒)\s*", s)
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
    return "Int" if re.fullmatch(r"[-+]?\d+", num) else "Float"


def _infer_unit_from_value_str(value_str: str) -> str:
    """
    从“数值+单位”的片段里粗略推断单位。
    - 5.11千米 -> 千米
    - 311.00千卡 -> 千卡
    - 7分42秒/公里 -> 分42秒/公里（这类更适合把整段当成“数值字符串”，单位仅作粗推断）
    """
    s = (value_str or "").strip()

    # 时间点（Timestamp）没有单位：避免 "01:40" 被推断成 ":40"
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        return "无"

    # 配速（时间/距离）单位规范化：
    # - 7.80分钟/公里 -> 分钟/公里
    # - 7分42秒/公里 -> 分钟/公里（避免推成 "分42秒/公里"）
    # 说明：这里只做“单位词”级别规范化，不改写原始 value_str 的表现形式。
    if "/" in s:
        parts = [p.strip() for p in s.split("/", 1)]
        if len(parts) == 2:
            left_raw, right_raw = parts[0], parts[1]
            left_has_time = any(x in left_raw for x in ("小时", "分钟", "秒")) or ("分" in left_raw and "秒" in left_raw)
            right_has_dist = any(x in right_raw for x in ("公里", "千米", "米", "km", "KM", "Km"))
            if left_has_time and right_has_dist:
                denom = "公里" if any(x in right_raw for x in ("公里", "千米", "km", "KM", "Km")) else "米"
                # 若含“秒”，通常对应“分秒/公里”，仍统一成 “分钟/公里” 更利于 downstream 使用
                if "小时" in left_raw and ("分钟" not in left_raw and "分" not in left_raw and "秒" not in left_raw):
                    return f"小时/{denom}"
                return f"分钟/{denom}"

    # 优先处理“范围值”的单位推断：例如 96%-98% / 15分-17分 / 22-26
    # 目标：避免把单位推成 "%-98%" 之类带数字的错误结果。
    m_range = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?)\s*(?P<u1>[^\d\s]+)?\s*[-~～—]\s*([-+]?\d+(?:\.\d+)?)\s*(?P<u2>[^\d\s]+)?\s*",
        s,
    )
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
        has_sec = "秒" in s
        has_min_abbr = ("分" in s) and (not has_min_word) and has_sec  # 仅在与“秒”共同出现时，才把“分”视为分钟缩写

        if has_hour or has_min_word or has_sec or has_min_abbr:
            unit_parts: list[str] = []
            if has_hour:
                unit_parts.append("小时")
            if has_min_word or has_min_abbr:
                unit_parts.append("分钟")
            if has_sec:
                unit_parts.append("秒")
            if unit_parts:
                return "".join(unit_parts)

    m = _FIRST_NUMBER_RE.search(s)
    if not m:
        return "无"
    unit = s[m.end() :].strip()
    return unit if unit else "无"


def _extract_name_and_value_from_rest(rest: str) -> tuple[str, str]:
    """
    从“rest”里尽量拆出 指标名称 与 数值字符串。
    这是兜底逻辑：优先匹配 “{name}为：{val}”/“{name}为:{val}”，否则用“遇到第一个数字”分割。
    """
    s = (rest or "").strip()
    # 常见：xxx为：yyy
    m = re.search(r"^(?P<name>.+?)为[:：]\s*(?P<val>.+?)$", s)
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
            t = m1.group("time").strip()
            nm = m1.group("name").strip()
            val = m1.group("val").strip()
        else:
            m2 = _SEG_RE_2.match(seg)
            if not m2:
                # 这段无法识别，跳过；最终如果完全没解析到任何条目，再整体兜底
                continue
            d = str(m2.group("date") or "").strip()
            t = str(m2.group("time") or "").strip() or 默认时间
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
        val_normalized = val
        if val and inferred_type and inferred_unit:
            val_normalized, _ = _normalize_value_and_unit(val, inferred_type, inferred_unit)

        dates.append(d)
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
    
    # 按指标名称分组存储记录
    metric_groups: dict[str, dict[str, list[str]]] = {}
    
    for seg in segments:
        m1 = _SEG_RE_1.match(seg)
        if m1:
            d = m1.group("date").strip()
            t = m1.group("time").strip()
            nm = m1.group("name").strip()
            val = m1.group("val").strip()
        else:
            m2 = _SEG_RE_2.match(seg)
            if not m2:
                continue
            d = str(m2.group("date") or "").strip()
            t = str(m2.group("time") or "").strip() or 默认时间
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
        
        metric_groups[nm_normalized]["dates"].append(d)
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
        
        # 规范化数值：剥离单位以避免重复
        values = []
        for val in raw_values:
            if val and inferred_type and inferred_unit:
                val_normalized, _ = _normalize_value_and_unit(val, inferred_type, inferred_unit)
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
    r"^\s*(?P<start>\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*"
    r"(?P<sep>到|至|~|～|-|—)\s*"
    r"(?P<end>\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*"
    # 注意：不能把时间值里的 ":"（如 23:20）当作 “name/val” 分隔符；因此对 ":"/ "：" 做负向数字前缀约束。
    r"(?:的)?(?P<name>.+?)\s*(?:为|(?<!\d)[:：])\s*(?P<val>.+?)\s*$"
)

# 无 “为/：” 分隔符版本：{开始日期}[到｜至｜~]{结束日期}(的)?{rest}
# 后续会用 `_extract_name_and_value_from_rest()` 从 rest 里按“第一个数字”切分 name/val。
_PERIOD_SUMMARY_SEG_RE_3 = re.compile(
    r"^\s*(?P<start>\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*"
    r"(?P<sep>到|至|~|～|-|—)\s*"
    r"(?P<end>\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*"
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

        starts.append(st)
        ends.append(ed)
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
    r"^\s*(?P<start>\d{4}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*"
    r"(?:(?P<sep>到|至|~|～|-|—)\s*"
    r"(?P<end>\d{4}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*)?"
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
            "偏晚",
            "晚",
            "偏早",
            "早",
            "偏低",
            "偏高",
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

    for seg in segments:
        m = _PERIOD_TEXT_SUMMARY_SEG_RE.match(seg)
        if not m:
            continue
        st = str(m.group("start") or "").strip()
        ed = str(m.group("end") or "").strip() or st  # 单日期：结束日期=开始日期
        rest = str(m.group("rest") or "").strip()
        if not st or not rest:
            continue

        nm, desc = _extract_metric_and_status_from_text(rest)
        if inferred_name is None and nm:
            inferred_name = nm

        starts.append(st)
        ends.append(ed)
        names.append(nm)
        descs.append(desc if desc else rest)

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

        r1 = _clean_tail(m1.group("range"))
        n1 = _strip_leading_de(_clean_tail(m1.group("name")))
        v1 = _clean_tail(m1.group("val"))
        r2 = _clean_tail(m2.group("range"))
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
    r"^\s*(?P<start>\d{4}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*"
    r"(?:(?P<sep>到|至|~|～|-|—)\s*"
    r"(?P<end>\d{4}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*)?"
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

        starts.append(st)
        ends.append(ed)
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
    r"^\s*(?P<date>\d{4}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(?P<rest>.+?)\s*$"
)


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

        m = _SINGLE_DATE_VALUE_SUMMARY_SEG_RE.match(seg2)
        if m:
            d = str(m.group("date") or "").strip()
            rest = str(m.group("rest") or "").strip()
            # 防止把“日期范围”误判为单日期（例如 8/2~8/8 ...）
            if re.match(
                r"^\s*(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
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

        dates.append(d)
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
    r"^\s*(?P<date>\d{4}/\d{1,2}/\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(?P<rest>.+?)\s*$"
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
                r"^\s*(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日)\s*(到|至|~|～|-|—)",
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
        dates.append(d)
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


# ========= 解析：单指标的统计复合记录（从原始一行文本抽取明细+汇总） =========
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
    将“单指标的统计复合记录”的原始一行文本解析为 `SingleMetricStatsRecord`。
    若无法解析到任何记录，返回 `UnparsedRawPersonalData`。
    """
    raw = str(raw_line or "").strip()
    if not raw:
        return UnparsedRawPersonalData(个人数据=raw, 原因="空行，无法解析为单指标的统计复合记录")

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

    if inside:
        items = [x.strip() for x in _SPLIT_RE.split(inside) if x and x.strip()]
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
            if (not force_unitless_duration) and inferred_unit is None and v:
                inferred_unit = _infer_unit_from_value_str(v)

            dates.append(d)
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
            if (not force_unitless_duration) and inferred_unit is None and val:
                inferred_unit = _infer_unit_from_value_str(val)
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

    date_str = str(m_head.group("date") or "").strip()
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
    merged_core = SingleDateValueMultiSummaryCore(
        指标名称=str(getattr(first_core, "指标名称", "") or ""),
        日期=str(getattr(first_core, "日期", "") or "Date (格式: MM月DD日)"),
        数值类型=getattr(first_core, "数值类型", "String"),  # type: ignore[arg-type]
        单位=str(getattr(first_core, "单位", "") or "无"),
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
    # 8 entity types
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
    # parsing helpers
    "PersonalDataPattern",
    "parse_style_item_to_dataclass",
    "parse_style_to_dataclasses",
    "route_raw_personal_data_to_dataclass",
    "explode_newlines_and_route_to_dataclasses",
    "aggregate_patterns_to_formatted_text",
    # raw-line parsing helpers
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

