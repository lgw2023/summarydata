from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_clean import *  # type: ignore  # noqa: F403
else:
    from . import *  # type: ignore  # noqa: F403


def _self_test_aggregate_patterns_to_dataline_text(
    lines: Sequence[str],
    *,
    max_cases: Optional[int] = 999,
    print_preview: bool = True,
) -> None:
    """
    self-test：验证
    - explode_newlines_and_route_to_dataclasses()
    - aggregate_patterns_to_dataline_text()
    的端到端行为（从原始多行文本 -> 数据类列表 -> dataline 行文本）。
    """
    xs = list(lines or [])
    if max_cases is not None:
        xs = xs[: max(0, int(max_cases))]

    def _short(s: object, n: int = 260) -> str:
        t = str(s if s is not None else "").replace("\n", " ").strip()
        if n <= 0:
            return ""
        return t if len(t) <= n else (t[: n - 1] + "…")

    ok = 0
    skipped_empty = 0
    for i, text in enumerate(xs):
        raw = (text or "").strip()
        if not raw:
            skipped_empty += 1
            continue
        patterns = explode_newlines_and_route_to_dataclasses(raw)
        out = aggregate_patterns_to_dataline_text(patterns)

        if not isinstance(out, str) or not out.strip():
            raise AssertionError(f"[self-test][dataline] 输出为空/非字符串：case#{i} raw={_short(raw)}")

        # 结构断言：至少包含一行；
        # - 可重构行：应包含 “运动类型：/健康类型：/类型：”
        # - 不可重构类型分组行：通常以 “数据类型：” 开头
        lines_out = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not lines_out:
            raise AssertionError(f"[self-test][dataline] 输出无有效行：case#{i} raw={_short(raw)} out={_short(out)}")
        if not any(
            ("运动类型：" in ln)
            or ("健康类型：" in ln)
            or ("类型：" in ln)
            or ln.startswith("数据类型：")
            for ln in lines_out
        ):
            raise AssertionError(f"[self-test][dataline] 输出结构异常（未见运动类型/数据类型前缀）：case#{i} raw={_short(raw)} out={_short(out)}")

        ok += 1
        if print_preview:
            print(f"  - case#{i} 解析对象数={len(patterns)}")
            print(f"    原始=\"\"\"\n{raw}\n\"\"\"")
            # 绿色打印
            print("\033[92m" + f"    输出=\"\"\"\n{out}\n\"\"\"" + "\033[0m")

    if skipped_empty:
        print(f"[self-test] aggregate_patterns_to_dataline_text 跳过空样本：{skipped_empty}")
    print(f"[self-test] aggregate_patterns_to_dataline_text 通过：{ok}/{len(xs) - skipped_empty}")



def _self_test_jump_rope_avg_speed_and_count() -> None:
    """
    回归用例：
    - "跳绳平均速度" 不应被拆成 sport="跳绳平均", metric="速度"
    - "跳绳个数" 不应落到 sport="无"
    期望：sport="跳绳"，指标包含 "平均速度" 与 "个数"
    """
    try:
        from ._personal_data_class_test_data import test_SingleMetricDetailRecord  # type: ignore
    except Exception:
        from src.data_clean._personal_data_class_test_data import test_SingleMetricDetailRecord  # type: ignore

    raw = next(
        x
        for x in list(test_SingleMetricDetailRecord)
        if ("跳绳平均速度为" in x) and ("跳绳个数为" in x)
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    # 核心断言：运动类型应为“跳绳”，且同一行可同时包含平均速度与个数
    expect_prefix = "2025/1/24 05:52 运动类型：跳绳"
    if expect_prefix not in out:
        raise AssertionError(f"[self-test][jump_rope] 未找到期望行前缀：{expect_prefix!r}\n输出前600={out[:600]!r}")
    if ("平均速度:" not in out) or ("个数:" not in out):
        raise AssertionError(f"[self-test][jump_rope] 未看到“平均速度/个数”字段：\n输出前800={out[:800]!r}")
    if ("运动类型：跳绳平均" in out) or ("运动类型：无， 跳绳个数" in out):
        raise AssertionError(f"[self-test][jump_rope] 仍存在错误拆分痕迹：\n输出前900={out[:900]!r}")


def _self_test_rowing_machine_stroke_rate() -> None:
    """
    回归用例：
    - "划船机桨频" 应被拆分为 sport="划船机"，metric="桨频"
    期望：运动类型为“划船机”，且指标名为“桨频”（而非“划船机桨频”）
    """
    try:
        from ._personal_data_class_test_data import test_SingleMetricDetailRecord  # type: ignore
    except Exception:
        from src.data_clean._personal_data_class_test_data import test_SingleMetricDetailRecord  # type: ignore

    raw = next(x for x in list(test_SingleMetricDetailRecord) if "划船机桨频为" in x)
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    # 核心断言：两条时间点都应解析为“划船机 + 桨频”
    expect_1 = "2025/2/17 20:35 运动类型：划船机， 桨频: 18.00"
    expect_2 = "2025/2/19 06:14 运动类型：划船机， 桨频: 28.00"
    if expect_1 not in out or expect_2 not in out:
        raise AssertionError(
            f"[self-test][rowing] 未找到期望输出行：\n"
            f"- expect_1={expect_1!r}\n"
            f"- expect_2={expect_2!r}\n"
            f"输出前900={out[:900]!r}"
        )
    if ("运动类型：无" in out) or ("划船机桨频:" in out):
        raise AssertionError(f"[self-test][rowing] 仍存在错误拆分痕迹：\n输出前900={out[:900]!r}")


def _self_test_activity_calories_not_in_sport() -> None:
    """
    回归用例：
    - "泳池游泳活动热量" 不应被拆成 sport="泳池游泳活动", metric="热量"
    期望：sport="泳池游泳"，指标为“活动热量”（保留原字段信息，避免与运动热量混淆）
    """
    try:
        from ._personal_data_class_test_data import test_SingleMetricDetailRecord  # type: ignore
    except Exception:
        from src.data_clean._personal_data_class_test_data import test_SingleMetricDetailRecord  # type: ignore

    raw = next(x for x in list(test_SingleMetricDetailRecord) if "泳池游泳活动热量为" in x)
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    if "运动类型：泳池游泳活动" in out:
        raise AssertionError(f"[self-test][activity_calories] sport 错误包含“活动”：\n输出前900={out[:900]!r}")
    if "运动类型：泳池游泳" not in out:
        raise AssertionError(f"[self-test][activity_calories] 未出现期望 sport=泳池游泳：\n输出前900={out[:900]!r}")
    if "活动热量:" not in out:
        raise AssertionError(f"[self-test][activity_calories] 未看到“活动热量”字段：\n输出前900={out[:900]!r}")


def _self_test_emotion_pressure_and_heart_health_not_mixed() -> None:
    """
    回归用例（对应你提到的 case#19）：
    - “最高压力/最低压力”应归到“情绪健康”，不应落到 sport_type=无
    - “心脏健康数据: 没有查询到”应归到“心脏健康”
    - 二者不应在同一条 “类型：无” 行内混合输出
    """
    raw = "\n".join(
        [
            "8/10 活动热量过少",
            "8/10 锻炼时长偏低",
            "8/11 入睡时间偏晚",
            "8/11 夜间睡眠时长偏短",
            "8/11 睡眠得分中等，睡眠质量良好",
            "8/10没有查询到体温数据",
            "7/10~9/10没有查询到生理健康数据",
            "8/10没有查询到心脏健康数据",
            "8月10日血氧饱和度97%-97%,平均血氧饱和度97%正常, 最高血氧饱和度97%正常, 最低血氧饱和度97%正常",
            "8月10日压力53分-56分,平均压力55分正常, 最高压力56分正常, 最低压力56分正常",
            "8月11日深睡时长2小时36分钟",
            "8月11日浅睡时长1小时53分钟",
            "8月11日快速眼动时长1小时23分钟",
            "8月10日活动热量237千卡偏低",
            "8月10日锻炼时长7分钟偏低",
            "8月11日入睡时间1:30偏晚",
            "8月11日睡眠得分86分中等",
            "8月11日起床时间7:25",
            "8月11日深睡连续性100分正常",
            "8月11日快速眼动比例24%正常",
            "8月11日清醒次数1次正常",
            "8月10日活动小时数11小时正常",
            "8月11日睡眠时长5小时52分钟偏短",
            "8月11日深睡比例44%正常",
            "8月11日浅睡比例32%正常",
            "8月10日零星小睡入睡时间为12:57",
            "8月10日零星小睡时长为35分钟",
            "2025/8/10 14:19的体重为：58千克,",
            "8/10的锻炼总时长为7分钟",
            "8/10的活动总小时数为11小时",
            "8/10的活动总热量为237千卡",
            "活动热量237千卡，距离目标270千卡还差33千卡",
            "活动小时数11小时，距离目标12小时还差1小时",
            "锻炼时长7分钟，距离目标25分钟还差18分钟",
            "8/10占比最高的情绪是不愉悦",
        ]
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    # 1) 心脏健康缺失提示应归到“心脏健康”
    if ("健康类型：心脏健康" not in out) or ("心脏健康数据: 没有查询到" not in out):
        raise AssertionError(f"[self-test][pressure_heart] 未看到期望的心脏健康行：\n输出前1200={out[:1200]!r}")

    # 2) 压力统计项（最高/最低）应归到“情绪健康”
    if ("健康类型：情绪健康" not in out) or ("最高压力:" not in out) or ("最低压力:" not in out):
        raise AssertionError(f"[self-test][pressure_heart] 未看到期望的情绪健康压力统计项：\n输出前1200={out[:1200]!r}")

    # 3) 不应出现 “类型：无， 心脏健康数据 ... 最高/最低压力 ...” 这种混合行
    if "类型：无， 心脏健康数据:" in out:
        raise AssertionError(f"[self-test][pressure_heart] 心脏健康仍落到类型：无：\n输出前1200={out[:1200]!r}")


def _self_test_sport_line_grouping() -> None:
    """
    基础回归：
    - dataline 输出应能同时包含“运动类型/健康类型”两大类行，并且不返回空串
    - 这是一个偏“冒烟测试”的兜底检查：避免聚合层改动导致整段输出退化为原文/空文本
    """
    raw = "\n".join(
        [
            "2025/4/22 05:50的户外跑步距离为：5.11千米,",
            "2025/4/22 06:00的血氧饱和度为：97.00%,",
        ]
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)
    if not isinstance(out, str) or not out.strip():
        raise AssertionError(f"[self-test][sport_grouping] 输出为空：out={out!r}")
    if "运动类型：" not in out:
        raise AssertionError(f"[self-test][sport_grouping] 未看到运动类型行：\nout={out!r}")
    if "健康类型：" not in out:
        raise AssertionError(f"[self-test][sport_grouping] 未看到健康类型行：\nout={out!r}")


def _self_test_steps_swim_style_and_dive_depth() -> None:
    """
    回归用例（批量覆盖你提到的类似问题）：
    - 室内/户外 跑步/步行 的“步数”应能拆 sport + metric
    - 开放水域游泳/泳池游泳 的“泳姿”应能拆 sport + metric
    - 平均潜水深度 应解析为 sport="潜水"
    """
    try:
        from ._personal_data_class_test_data import test_SingleMetricDetailRecord  # type: ignore
    except Exception:
        from src.data_clean._personal_data_class_test_data import test_SingleMetricDetailRecord  # type: ignore

    # 1) 步数：室内跑步（1394.00步）、户外跑步（6056.00步）
    raw_indoor_run_steps = next(x for x in list(test_SingleMetricDetailRecord) if "室内跑步步数为：1394.00步" in x)
    raw_outdoor_run_steps = next(x for x in list(test_SingleMetricDetailRecord) if "户外跑步步数为：6056.00步" in x)

    # 2) 步数：室内步行（多条）
    raw_indoor_walk_steps = next(x for x in list(test_SingleMetricDetailRecord) if "室内步行步数为：6165.00步" in x)
    # 3) 步数：户外步行（多条）
    raw_outdoor_walk_steps = next(x for x in list(test_SingleMetricDetailRecord) if x.startswith("2025/1/24 20:11的户外步行步数为：5317.00步"))

    # 4) 泳姿：开放水域游泳 / 泳池游泳
    raw_open_water_style = next(x for x in list(test_SingleMetricDetailRecord) if "开放水域游泳泳姿为：蛙泳" in x)
    raw_pool_style = next(x for x in list(test_SingleMetricDetailRecord) if "泳池游泳泳姿为：蝶泳" in x)

    # 5) 平均划水频率：泳池游泳
    raw_pool_stroke_rate = next(x for x in list(test_SingleMetricDetailRecord) if "泳池游泳平均划水频率为：14.00" in x)

    # 6) 平均配速：泳池游泳（回归：不要把“平均”吞进运动类型）
    raw_pool_avg_pace = next(x for x in list(test_SingleMetricDetailRecord) if "泳池游泳平均配速为：28分43秒/公里" in x)

    # 7) 平均潜水深度
    raw_avg_dive_depth = next(x for x in list(test_SingleMetricDetailRecord) if "平均潜水深度为：21.90米" in x)

    raws = [
        raw_indoor_run_steps,
        raw_outdoor_run_steps,
        raw_indoor_walk_steps,
        raw_outdoor_walk_steps,
        raw_open_water_style,
        raw_pool_style,
        raw_pool_stroke_rate,
        raw_pool_avg_pace,
        raw_avg_dive_depth,
    ]

    for raw in raws:
        patterns = explode_newlines_and_route_to_dataclasses(raw)
        out = aggregate_patterns_to_dataline_text(patterns)

        if "运动类型：无" in out:
            raise AssertionError(f"[self-test][batch_fix] 仍出现运动类型=无：raw={raw!r}\n输出前900={out[:900]!r}")

    # 更强断言：关键行内容应出现（避免只是恰好被别的规则遮蔽）
    out_indoor_run = aggregate_patterns_to_dataline_text(explode_newlines_and_route_to_dataclasses(raw_indoor_run_steps))
    if "2025/2/6 18:44 运动类型：室内跑步， 步数: 1394.00步" not in out_indoor_run:
        raise AssertionError(f"[self-test][steps] 室内跑步步数输出不匹配：\n输出={out_indoor_run!r}")

    out_open_style = aggregate_patterns_to_dataline_text(explode_newlines_and_route_to_dataclasses(raw_open_water_style))
    if "2025/1/20 19:00 运动类型：开放水域游泳， 泳姿: 蛙泳" not in out_open_style:
        raise AssertionError(f"[self-test][style] 开放水域游泳泳姿输出不匹配：\n输出={out_open_style!r}")

    out_pool_avg_pace = aggregate_patterns_to_dataline_text(explode_newlines_and_route_to_dataclasses(raw_pool_avg_pace))
    if "2025/1/25 16:59 运动类型：泳池游泳， 配速: 28分43秒/公里" not in out_pool_avg_pace:
        raise AssertionError(f"[self-test][pace] 泳池游泳平均配速输出不匹配：\n输出={out_pool_avg_pace!r}")
    if "运动类型：泳池游泳平均" in out_pool_avg_pace:
        raise AssertionError(f"[self-test][pace] 仍存在错误 sport='泳池游泳平均'：\n输出={out_pool_avg_pace!r}")

    out_dive = aggregate_patterns_to_dataline_text(explode_newlines_and_route_to_dataclasses(raw_avg_dive_depth))
    if "2025/2/24 02:16 运动类型：潜水， 平均潜水深度: 21.90米" not in out_dive:
        raise AssertionError(f"[self-test][dive] 平均潜水深度输出不匹配：\n输出={out_dive!r}")


def _self_test_period_summary_sport_and_metric_split() -> None:
    """
    回归用例（对应你反馈的 case#0/#3/#4/#5/#6/#7）：
    - 周期汇总类指标名里携带“平均/总”等聚合词时，不应被吸进运动类型
    - 子项指标名不应重复携带运动名（应输出为：总距离/平均速度/平均步频/总时长...）
    """
    raws = [
        (
            "2025/2/1至2025/2/22的跑步总距离为201.27千米",
            "2025/2/1~2025/2/22 运动类型：跑步， 总距离: 201.27千米",
            ["运动类型：跑步总", "跑步总距离:"],
        ),
        (
            "2025/2/1至2025/2/22的平均户外跑步速度为8.19公里/小时",
            "2025/2/1~2025/2/22 运动类型：户外跑步， 平均速度: 8.19公里/小时",
            ["运动类型：平均户外跑步", "平均户外跑步速度:"],
        ),
        (
            "2025/2/1至2025/2/22的平均越野跑心率为0.00次/分钟",
            "2025/2/1~2025/2/22 运动类型：越野跑， 平均运动心率: 0.00次/分钟",
            ["运动类型：平均越野跑", "平均越野跑心率:"],
        ),
        (
            "2025/2/1至2025/2/22的平均户外跑步步频为173.28步/分钟",
            "2025/2/1~2025/2/22 运动类型：户外跑步， 平均步频: 173.28步/分钟",
            ["运动类型：平均户外跑步", "平均户外跑步步频:"],
        ),
        (
            "2025/2/1至2025/2/22的平均户外跑步步幅为0.00",
            "2025/2/1~2025/2/22 运动类型：户外跑步， 平均步幅: 0.00",
            ["运动类型：平均户外跑步", "平均户外跑步步幅:"],
        ),
        (
            "4/1~4/30的锻炼总时长为1404.00分钟",
            "4/1~4/30 运动类型：所有运动， 锻炼总时长: 1404.00分钟",
            ["运动类型：锻炼总", " 总时长:"],
        ),
    ]

    for raw, expect, forbid_subs in raws:
        patterns = explode_newlines_and_route_to_dataclasses(raw)
        out = aggregate_patterns_to_dataline_text(patterns)
        if expect not in out:
            raise AssertionError(f"[self-test][period_split] 输出不匹配：\nraw={raw!r}\nexpect={expect!r}\nout={out!r}")
        for bad in forbid_subs:
            if bad in out:
                raise AssertionError(f"[self-test][period_split] 仍存在错误痕迹：bad={bad!r}\nraw={raw!r}\nout={out!r}")


def _self_test_health_sleep_blood_oxygen_bmi_and_walk_summary() -> None:
    """
    回归用例（对应你反馈的“类型：无，但其实可进一步解析”的场景）：
    - 步行汇总类：步行总距离/总热量/总时长/运动次数/最长距离 -> 运动类型应归到“步行徒步”
    - 睡眠类：入睡时间/睡眠得分/睡眠数据/浅睡比例 -> 健康类型应归到“睡眠”
    - 静息心率 -> 健康类型应归到“心脏健康”
    - 血氧相关 -> 健康类型应归到“血氧饱和度”，并修正“平均平均血氧”
    - 最新一次BMI -> 健康类型应归到“微体检”
    """
    cases = [
        (
            "2025/1/1至2025/1/31的步行总距离为21.90千米",
            "2025/1/1~2025/1/31 运动类型：步行徒步， 总距离: 21.90千米",
        ),
        (
            "2025/1/1至2025/1/31的步行总热量为1640.00千卡",
            "2025/1/1~2025/1/31 运动类型：步行徒步， 总热量: 1640.00千卡",
        ),
        (
            "2025/1/1至2025/1/31的步行总时长为4.81小时",
            "2025/1/1~2025/1/31 运动类型：步行徒步， 总时长: 4.81小时",
        ),
        (
            "2025/1/1至2025/1/31的步行运动次数为10.00次",
            "2025/1/1~2025/1/31 运动类型：步行徒步， 次数: 10.00次",
        ),
        (
            "2025/1/1至2025/1/31的步行最长距离为4.63千米",
            "2025/1/1~2025/1/31 运动类型：步行徒步， 最长距离: 4.63千米",
        ),
        (
            "24/1/1~24/3/4入睡时间欠规律",
            "2024/1/1~2024/3/4 健康类型：睡眠， 入睡时间: 欠规律",
        ),
        (
            "24/1/1~24/3/4睡眠得分中等，睡眠质量良好",
            "2024/1/1~2024/3/4 健康类型：睡眠， 睡眠得分: 中等",
        ),
        (
            "25/1/1~25/3/4未查询到睡眠数据",
            "2025/1/1~2025/3/4 健康类型：睡眠， 睡眠数据: 未查询到",
        ),
        (
            "2/17-2025/2/23浅睡比例偏高",
            "2/17~2025/2/23 健康类型：睡眠， 浅睡比例: 偏高",
        ),
        (
            "2025/2/12~2025/2/19静息心率偏低",
            "2025/2/12~2025/2/19 健康类型：心脏健康， 静息心率: 偏低",
        ),
        (
            "2024/1/1~2024/12/31的平均最低血氧为97.53%",
            "2024/1/1~2024/12/31 健康类型：血氧饱和度， 平均最低血氧: 97.53%",
        ),
        (
            "2024/1/1~2024/12/31的平均平均血氧为97.59%",
            "2024/1/1~2024/12/31 健康类型：血氧饱和度， 平均血氧: 97.59%",
        ),
        (
            "2024/1/1~2024/12/31的平均最大血氧为97.70%",
            "2024/1/1~2024/12/31 健康类型：血氧饱和度， 平均最大血氧: 97.70%",
        ),
        (
            "1/19~1/25 最新一次BMI偏重",
            "1/19~1/25 健康类型：微体检， 最新一次BMI: 偏重",
        ),
    ]

    for raw, expect in cases:
        patterns = explode_newlines_and_route_to_dataclasses(raw)
        out = aggregate_patterns_to_dataline_text(patterns)
        if expect not in out:
            raise AssertionError(f"[self-test][health_sleep_walk] 输出不匹配：\nraw={raw!r}\nexpect={expect!r}\nout={out!r}")
        if "类型：无" in out:
            raise AssertionError(f"[self-test][health_sleep_walk] 仍出现“类型：无”：\nraw={raw!r}\nout={out!r}")


def _self_test_single_date_value_single_summary_to_dataline() -> None:
    """
    回归用例：
    - SingleDateValueSingleSummaryRecord（单日期数值单项总结）应可转成 dataline，
      不应再落到 “数据类型：单日期数值单项总结” 的不可重构分组行。
    """
    raw = "4/23压力均值78分偏高"
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    if "数据类型：单日期数值单项总结" in out:
        raise AssertionError(f"[self-test][single_date_value_single] 仍被当作不可重构类型：\nraw={raw!r}\nout={out!r}")
    if "健康类型：情绪健康" not in out:
        raise AssertionError(f"[self-test][single_date_value_single] 未能推断健康类型=情绪健康：\nraw={raw!r}\nout={out!r}")
    if "压力均值:" not in out:
        raise AssertionError(f"[self-test][single_date_value_single] 未看到指标名输出：\nraw={raw!r}\nout={out!r}")


def _self_test_metric_suffix_trailing_wei_should_split() -> None:
    """
    回归用例：
    - 指标名末尾残留“为”（例如 “平均室内跑步步幅为”）时，仍应能正常拆出运动类型与指标后缀。
    该问题会导致输出出现“类型：无， 平均室内跑步步幅为: ...”。
    """
    cases = [
        (
            "2025/2/20至2025/2/20的平均室内跑步步幅为0.00",
            "2025/2/20 运动类型：室内跑步， 平均步幅: 0.00",
        ),
        (
            "2025/2/20至2025/2/20的平均室内跑步心率为149.00次/分钟",
            "2025/2/20 运动类型：室内跑步， 平均运动心率: 149.00次/分钟",
        ),
        (
            "2025/2/20至2025/2/20的跑步总时长为0.13小时",
            "2025/2/20 运动类型：跑步， 总时长: 0.13小时",
        ),
        (
            "2025/2/20至2025/2/20的跑步总热量为51.00千卡",
            "2025/2/20 运动类型：跑步， 总热量: 51.00千卡",
        ),
        (
            "2025/2/20至2025/2/20的室内跑步次数为1.00次",
            "2025/2/20 运动类型：室内跑步， 次数: 1.00次",
        ),
        (
            "2025/2/20至2025/2/20的平均室内跑步配速为7.21分钟/公里",
            "2025/2/20 运动类型：室内跑步， 配速: 7.21分钟/公里",
        ),
        (
            "2025/2/20至2025/2/20的平均室内跑步步频为140.00步/分钟",
            "2025/2/20 运动类型：室内跑步， 平均步频: 140.00步/分钟",
        ),
        (
            "2025/2/20至2025/2/20的跑步运动次数为1.00次",
            "2025/2/20 运动类型：跑步， 次数: 1.00次",
        ),
        (
            "2025/2/20至2025/2/20的跑步总距离为1.11千米",
            "2025/2/20 运动类型：跑步， 总距离: 1.11千米",
        ),
        (
            "2025/2/20至2025/2/20的跑步平均配速为7.21分钟/公里",
            "2025/2/20 运动类型：跑步， 配速: 7.21分钟/公里",
        ),
        (
            "2025/2/20至2025/2/20的平均室内跑步速度为8.33公里/小时",
            "2025/2/20 运动类型：室内跑步， 平均速度: 8.33公里/小时",
        ),
    ]

    for raw, expect in cases:
        patterns = explode_newlines_and_route_to_dataclasses(raw)
        out = aggregate_patterns_to_dataline_text(patterns)
        if expect not in out:
            raise AssertionError(f"[self-test][trailing_wei] 输出不匹配：\nraw={raw!r}\nexpect={expect!r}\nout={out!r}")
        if "类型：无" in out:
            raise AssertionError(f"[self-test][trailing_wei] 仍出现类型=无：\nraw={raw!r}\nout={out!r}")
        if "为:" in out or "为：" in out:
            raise AssertionError(f"[self-test][trailing_wei] 仍残留“为:”在指标名：\nraw={raw!r}\nout={out!r}")


def _self_test_merge_multi_timepoints_same_day_same_metric() -> None:
    """
    回归：同一天同一指标存在多个时间点时，不应逐条刷屏输出，
    而应合并为“日期 类型，HH:MM的指标为数值, HH:MM的指标为数值, ...”。
    """
    raw = "\n".join(
        [
            "2025/4/22 05:50的血氧饱和度为：99.00%,",
            "2025/4/22 06:00的血氧饱和度为：97.00%,",
            "2025/4/22 06:10的血氧饱和度为：97.00%,",
        ]
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    expect = (
        "2025/4/22 健康类型：血氧饱和度， "
        "05:50的血氧饱和度为99.00%, 06:00的血氧饱和度为97.00%, 06:10的血氧饱和度为97.00%"
    )
    if expect not in out:
        raise AssertionError(f"[self-test][timepoints_merge] 输出不匹配：\nraw={raw!r}\nexpect={expect!r}\nout={out!r}")

    # 不应再出现逐条的“YYYY/M/D HH:MM 健康类型：...， 指标: ...”
    if "2025/4/22 05:50 健康类型：血氧饱和度" in out:
        raise AssertionError(f"[self-test][timepoints_merge] 仍逐条输出（05:50）：\nout={out!r}")
    if "2025/4/22 06:00 健康类型：血氧饱和度" in out:
        raise AssertionError(f"[self-test][timepoints_merge] 仍逐条输出（06:00）：\nout={out!r}")


def _self_test_sort_by_real_date_within_same_type() -> None:
    """
    回归用例（按你的排序规则不变）：
    - 仍然保持：大类 -> 具体类型名 -> 起止时间
    - 仅修复“日期排序要按真实日期（2/1,2/2,2/10...），不能按数值/字符串导致 2/10 排到 2/2 前面”等问题
    """
    raw = "\n".join(
        [
            "2/1静息心率68次/分钟正常",
            "2/10静息心率63次/分钟正常",
            "2/2静息心率79次/分钟正常",
        ]
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)
    lines_out = [ln.strip() for ln in out.splitlines() if ln.strip()]

    # 三条都应归到“心脏健康”，并按日期升序：2/1, 2/2, 2/10
    idx_21 = next((i for i, ln in enumerate(lines_out) if ln.startswith("2/1 ") and "健康类型：心脏健康" in ln), None)
    idx_22 = next((i for i, ln in enumerate(lines_out) if ln.startswith("2/2 ") and "健康类型：心脏健康" in ln), None)
    idx_210 = next((i for i, ln in enumerate(lines_out) if ln.startswith("2/10 ") and "健康类型：心脏健康" in ln), None)
    if idx_21 is None or idx_22 is None or idx_210 is None:
        raise AssertionError(
            f"[self-test][sort_real_date_within_type] 未找到期望的输出行：\n"
            f"- idx_21={idx_21}\n"
            f"- idx_22={idx_22}\n"
            f"- idx_210={idx_210}\n"
            f"out_head={out[:900]!r}"
        )
    if not (idx_21 < idx_22 < idx_210):
        raise AssertionError(
            f"[self-test][sort_real_date_within_type] 同一类型内日期未按升序排序："
            f"idx_21={idx_21}, idx_22={idx_22}, idx_210={idx_210}\n"
            f"out={out!r}"
        )


def _self_test_single_metric_stats_summary_should_have_date_range() -> None:
    """
    回归用例（对应你反馈的 case#635）：
    - SingleMetricStatsRecord 的“统计汇总”最后一行不应输出“无时间~无时间”，
      而应从该条数据的明细日期列表中取首/末日期作为起止时间。
    """
    raw = (
        "零星小睡时长：[5月27日33分钟,5月28日25分钟,5月29日31分钟,5月30日20分钟,5月31日37分钟,"
        "6月1日36分钟,6月2日25分钟,6月3日23分钟,6月4日32分钟,6月5日1小时14分钟,6月6日1小时16分钟,"
        "6月7日21分钟,6月8日37分钟,6月9日33分钟,6月10日38分钟,6月11日32分钟,6月12日24分钟,"
        "6月13日38分钟,6月14日21分钟,6月15日36分钟,6月16日24分钟,6月17日43分钟,6月18日24分钟,"
        "6月19日38分钟,6月20日21分钟,6月21日36分钟,6月22日24分钟,6月23日43分钟,6月24日41分钟,"
        "6月25日33分钟,6月26日32分钟] , 平均零星小睡时长34分钟正常, 最高零星小睡时长1小时16分钟偏长, 最低零星小睡时长20分钟正常"
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    if "无时间~无时间" in out:
        raise AssertionError(f"[self-test][single_metric_stats_summary_range] 仍出现无时间：\nraw={raw!r}\nout={out!r}")
    if "5/27~6/26 健康类型：午睡" not in out:
        raise AssertionError(
            f"[self-test][single_metric_stats_summary_range] 未找到期望的汇总日期范围行：\n"
            f"expect_substr={'5/27~6/26 健康类型：午睡'!r}\n"
            f"out_head={out[:1200]!r}"
        )


def _self_test_missing_blood_pressure_and_sugar_should_be_domain() -> None:
    """
    回归用例（对应你反馈的 case#107）：
    - “血糖数据/血压数据: 没有查询到” 即使查不到数据，也应归入对应健康 domain
    - 不应落到 “类型：无” 的合并行里
    """
    raw = "\n".join(
        [
            "8/8没有查询到血糖数据",
            "8/8没有查询到血压数据",
        ]
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    if "8/8 健康类型：血糖" not in out or "血糖数据:" not in out:
        raise AssertionError(
            f"[self-test][missing_blood] 未归类到血糖 domain：\nraw={raw!r}\nout={out!r}"
        )
    if "8/8 健康类型：血压" not in out or "血压数据:" not in out:
        raise AssertionError(
            f"[self-test][missing_blood] 未归类到血压 domain：\nraw={raw!r}\nout={out!r}"
        )
    if ("类型：无" in out) and (("血糖" in out) or ("血压" in out)):
        raise AssertionError(
            f"[self-test][missing_blood] 仍出现“血糖/血压”落到类型=无：\nout={out!r}"
        )


def _self_test_period_metric_suffix_then_agg_tail_should_split() -> None:
    """
    回归用例（对应你反馈的 case#157）：
    - “2/8~2/22的骑行距离最长为3.20千米” 这种“指标后缀 + 聚合词尾缀（最长）”的口径，
      应能拆出运动类型=骑行，而不应落到“类型：无”。
    """
    raw = "\n".join(
        [
            "2/8~2/22的骑行距离最长为3.20千米",
        ]
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    if "2/8~2/22 类型：无" in out:
        raise AssertionError(f"[self-test][period_suffix_then_agg_tail] 仍出现类型=无：\nraw={raw!r}\nout={out!r}")
    if ("2/8~2/22 运动类型：骑行" not in out) or ("距离最长:" not in out and "距离最长：" not in out):
        raise AssertionError(
            f"[self-test][period_suffix_then_agg_tail] 未得到期望的骑行拆分：\n"
            f"expect_substr={'2/8~2/22 运动类型：骑行'!r}\n"
            f"out={out!r}"
        )


def _self_test_running_advanced_metrics_should_split_prefix() -> None:
    """
    回归用例（对应你反馈的 case#151）：
    - 户外/室内跑步的“心率范围/最大摄氧量/平均外翻幅度/平均着地冲击/平均摆动角度/平均腾空时间/平均触地时间/平均触底腾空比”
      应能拆成 sport=户外跑步(or 室内跑步) + metric=对应指标名（不再携带“户外跑步/室内跑步”前缀）
    - “户外跑步前脚掌触地次数/后脚掌触底次数”不应把“前脚掌触地/后脚掌触底”误吸进运动类型
    """
    raw = "\n".join(
        [
            "2025/6/26 18:09的户外跑步前脚掌触地次数为：4次,",
            "2025/6/26 18:09的户外跑步后脚掌触底次数为：200次,",
            "2025/6/26 18:09的户外跑步心率范围为：85-162次/分钟,",
            "2025/6/26 18:09的户外跑步最大摄氧量为：44ml/(kg.min),",
            "2025/6/26 18:09的户外跑步平均外翻幅度为：16度,",
            "2025/6/26 18:09的户外跑步平均着地冲击为：7g,",
            "2025/6/26 18:09的户外跑步平均摆动角度为：70度,",
            "2025/6/26 18:09的户外跑步平均腾空时间为：10毫秒,",
            "2025/6/26 18:09的户外跑步平均触地时间为：535毫秒,",
            "2025/6/26 18:09的户外跑步平均触底腾空比为：1.8,",
        ]
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    # 核心断言：落到户外跑步行，而不是“类型：无”或把“前脚掌触地”当运动类型
    if "2025/6/26 18:09 运动类型：户外跑步" not in out:
        raise AssertionError(f"[self-test][run_adv] 未找到期望的户外跑步行：\nout={out!r}")
    if "类型：无" in out:
        raise AssertionError(f"[self-test][run_adv] 仍出现类型=无：\nout={out!r}")
    if ("运动类型：户外跑步前脚掌触地" in out) or ("运动类型：户外跑步后脚掌触底" in out):
        raise AssertionError(f"[self-test][run_adv] 前脚掌/后脚掌被误吸进运动类型：\nout={out!r}")

    # 指标名不应再携带“户外跑步”前缀
    must_have = [
        "前脚掌触地次数:",
        "后脚掌触底次数:",
        "心率范围:",
        "最大摄氧量:",
        "平均外翻幅度:",
        "平均着地冲击:",
        "平均摆动角度:",
        "平均腾空时间:",
        "平均触地时间:",
        "平均触底腾空比:",
    ]
    for k in must_have:
        if k not in out:
            raise AssertionError(f"[self-test][run_adv] 未看到期望字段 {k!r}：\nout={out!r}")
    if ("户外跑步心率范围:" in out) or ("户外跑步最大摄氧量:" in out):
        raise AssertionError(f"[self-test][run_adv] 指标名仍残留“户外跑步”前缀：\nout={out!r}")


def _self_test_total_consume_calories_should_not_become_type() -> None:
    """
    回归用例（对应你反馈的口径）：
    - “2/9~8/8的跑步总消耗热量为3523千卡”
      不应输出为 “类型：跑步总消耗， 运动热量: ...”
      期望：运动类型=跑步，指标=总消耗运动热量
    """
    raw = "2/9~8/8的跑步总消耗热量为3523千卡"
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    if "类型：跑步总消耗" in out:
        raise AssertionError(f"[self-test][consume] 仍出现错误类型“跑步总消耗”：\nout={out!r}")
    if "2/9~8/8 运动类型：跑步" not in out:
        raise AssertionError(f"[self-test][consume] 未出现期望 sport=跑步：\nout={out!r}")
    if "总消耗运动热量:" not in out:
        raise AssertionError(f"[self-test][consume] 未出现期望指标“总消耗运动热量”：\nout={out!r}")


def _self_test_dirty_cn_full_datetime_without_de_should_parse_as_detail() -> None:
    """
    回归用例（对应你反馈的脏数据）：
    - 形如：2025年02月16日19时17分户外跑步距离为5.52千米
    - 特点：带“YYYY年MM月DD日HH时mm分”，但缺少“的”（与原先明细口径不一致）

    期望：
    - router 能把每一行都识别为 SingleMetricDetailRecord（明细记录）
    - 聚合输出包含归一化后的时间点与运动类型（2025/2/16 19:17 运动类型：户外跑步）
    - 至少保留关键指标：距离、步数（避免信息被误判/丢失）
    """
    raw = "\n".join(
        [
            "2025年02月16日19时17分户外跑步最快配速为6分0秒/公里",
            "2025年02月16日19时17分户外跑步配速为6分27秒/公里",
            "2025年02月16日19时17分户外跑步步数为6306步",
            "2025年02月16日19时17分户外跑步热量为391千卡",
            "2025年02月16日19时17分户外跑步最大步频为197步/分钟",
            "2025年02月16日19时17分户外跑步步频为176步/分钟",
            "2025年02月16日19时17分户外跑步用时为35分38秒",
            "2025年02月16日19时17分户外跑步最大心率为163次/分钟",
            "2025年02月16日19时17分户外跑步心率为142次/分钟",
            "2025年02月16日19时17分户外跑步距离为5.52千米",
            "2025年02月16日19时17分户外跑步速度为9.29公里/小时",
        ]
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    if not patterns:
        raise AssertionError("[self-test][dirty_cn_datetime] router 返回空列表")
    if any(isinstance(x, UnparsedRawPersonalData) for x in patterns):
        bad = next(x for x in patterns if isinstance(x, UnparsedRawPersonalData))
        assert isinstance(bad, UnparsedRawPersonalData)
        raise AssertionError(f"[self-test][dirty_cn_datetime] 仍存在未解析：原因={bad.原因!r} 数据={bad.个人数据!r}")
    if not all(isinstance(x, SingleMetricDetailRecord) for x in patterns):
        raise AssertionError(
            f"[self-test][dirty_cn_datetime] 类型不一致：{[type(x).__name__ for x in patterns]!r}"
        )

    out = aggregate_patterns_to_dataline_text(patterns)
    if "2025/2/16 19:17 运动类型：户外跑步" not in out:
        raise AssertionError(
            f"[self-test][dirty_cn_datetime] 未看到期望的时间点+运动类型行：\n"
            f"expect_substr={'2025/2/16 19:17 运动类型：户外跑步'!r}\n"
            f"out_head={out[:900]!r}"
        )
    for k in ("距离:", "步数:"):
        if k not in out:
            raise AssertionError(f"[self-test][dirty_cn_datetime] 未看到关键字段 {k!r}：\nout={out!r}")


def _self_test_period_range_with_slash_date_and_trailing_day_should_parse() -> None:
    """
    回归用例（对应你反馈的 bug）：
    - “2025/2/1日到2025年6月16日平均深睡时长...” 这种“slash 日期 + 日 + 到 + 中文年月日”的混用口径，
      不应被误拆成单日期（例如输出出现 “类型：无， 日到: ...”）。
    期望：
    - 能解析为周期（起止日期应为 2025/2/1~2025/6/16）
    - 健康类型应为“睡眠”
    - 包含 平均/最高/最低 深睡时长 三个指标
    """
    raw = "2025/2/1日到2025年6月16日平均深睡时长2小时10分钟, 最高深睡时长2小时40分钟, 最低深睡时长1小时25分钟"
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)

    if "类型：无， 日到:" in out or "日到:" in out:
        raise AssertionError(f"[self-test][slash_day_to] 仍出现错误拆分（日到 变成指标名）：\nraw={raw!r}\nout={out!r}")
    if "2025/2/1~2025/6/16 健康类型：睡眠" not in out:
        raise AssertionError(
            f"[self-test][slash_day_to] 未解析为期望的周期睡眠行：\n"
            f"expect_substr={'2025/2/1~2025/6/16 健康类型：睡眠'!r}\n"
            f"out={out!r}"
        )
    for k in ("平均深睡时长:", "最高深睡时长:", "最低深睡时长:"):
        if k not in out:
            raise AssertionError(f"[self-test][slash_day_to] 未看到期望字段 {k!r}：\nout={out!r}")


def _self_test_stats_composite_range_avg_should_keep_daily_avg() -> None:
    """
    回归用例：
    - stats-composite 明细列表里出现“日期 + 范围 + 平均”的脏数据时，不应把“平均xx”丢掉。
      例如：压力：[2月18日22-26,平均24,...]

    期望：
    - 每日行能输出“压力: 平均24/平均23...”等片段（避免信息遗漏）
    """
    raw = (
        "压力：[2月18日22-26,平均24,2月19日22-26,平均24,2月20日21-25,平均23,2月22日19-23,平均21,2月23日20-24,平均22], "
        "平均压力22.8正常，最低压力19正常，最高压力26正常"
    )
    patterns = explode_newlines_and_route_to_dataclasses(raw)
    out = aggregate_patterns_to_dataline_text(patterns)
    for expect in (
        "2/18 健康类型：情绪健康， 压力: 22-26（平均24）",
        "2/19 健康类型：情绪健康， 压力: 22-26（平均24）",
        "2/20 健康类型：情绪健康， 压力: 21-25（平均23）",
        "2/22 健康类型：情绪健康， 压力: 19-23（平均21）",
        "2/23 健康类型：情绪健康， 压力: 20-24（平均22）",
    ):
        if expect not in out:
            raise AssertionError(
                f"[self-test][stats_range_avg] 未看到期望输出片段：\nexpect_substr={expect!r}\nout={out!r}"
            )


if __name__ == "__main__":
    try:
        from ._personal_data_class_test_data import (  # type: ignore
            test_SingleMetricDetailRecord,
            test_PeriodValueSingleSummaryRecord,
            test_PeriodValuemMultiSummaryRecord,
            test_PeriodTextSummaryRecord,
            test_SingleMetricStatsRecord,
            test_SingleDateValueMultiSummaryRecord,
            test_SingleDateTextSummaryRecord,
            test_PeriodValueCompareRecord,
            test_SingleDateValueSingleSummaryRecord,
            test_NoTimestampTextSummaryRecord,
            test_NoDateValueSummaryRecord,
            test_UnparsedRawPersonalData,
        )
    except Exception:
        from src.data_clean._personal_data_class_test_data import (  # type: ignore
            test_SingleMetricDetailRecord,
            test_PeriodValueSingleSummaryRecord,
            test_PeriodValuemMultiSummaryRecord,
            test_PeriodTextSummaryRecord,
            test_SingleMetricStatsRecord,
            test_SingleDateValueMultiSummaryRecord,
            test_SingleDateTextSummaryRecord,
            test_PeriodValueCompareRecord,
            test_SingleDateValueSingleSummaryRecord,
            test_NoTimestampTextSummaryRecord,
            test_NoDateValueSummaryRecord,
            test_UnparsedRawPersonalData,
        )

    # 端到端（覆盖多类型，但每次只取少量样本避免日志过大）
    _self_test_aggregate_patterns_to_dataline_text(
        [
            "\n".join(test_SingleMetricDetailRecord[:3]),
            "\n".join(test_PeriodValueSingleSummaryRecord[:2]),
            "\n".join(test_PeriodValuemMultiSummaryRecord[:2]),
            "\n".join(test_PeriodTextSummaryRecord[:2]),
            "\n".join(test_SingleMetricStatsRecord[:2]),
            "\n".join(test_SingleDateValueMultiSummaryRecord[:2]),
            "\n".join(test_SingleDateTextSummaryRecord[:2]),
            "\n".join(test_PeriodValueCompareRecord[:1]),
            "\n".join(test_SingleDateValueSingleSummaryRecord[:1]),
            "\n".join(test_NoTimestampTextSummaryRecord[:1]),
            "\n".join(test_NoDateValueSummaryRecord[:1]),
            "\n".join(test_UnparsedRawPersonalData[:1]),
        ],
        max_cases=20,
        print_preview=False,
    )
    _self_test_sport_line_grouping()
    _self_test_jump_rope_avg_speed_and_count()
    _self_test_rowing_machine_stroke_rate()
    _self_test_activity_calories_not_in_sport()
    _self_test_steps_swim_style_and_dive_depth()
    _self_test_period_summary_sport_and_metric_split()
    _self_test_health_sleep_blood_oxygen_bmi_and_walk_summary()
    _self_test_single_date_value_single_summary_to_dataline()
    _self_test_metric_suffix_trailing_wei_should_split()
    _self_test_sort_by_real_date_within_same_type()
    _self_test_single_metric_stats_summary_should_have_date_range()
    _self_test_missing_blood_pressure_and_sugar_should_be_domain()
    _self_test_period_metric_suffix_then_agg_tail_should_split()
    _self_test_running_advanced_metrics_should_split_prefix()
    _self_test_total_consume_calories_should_not_become_type()
    _self_test_period_range_with_slash_date_and_trailing_day_should_parse()
    _self_test_stats_composite_range_avg_should_keep_daily_avg()
    _self_test_dirty_cn_full_datetime_without_de_should_parse_as_detail()
    print("[self-test] test_aggregate_dataline.py 全部通过")

