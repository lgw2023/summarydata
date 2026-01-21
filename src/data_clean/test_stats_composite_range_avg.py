from __future__ import annotations

from src.data_clean import explode_newlines_and_route_to_dataclasses, aggregate_patterns_to_dataline_text


def test_stats_composite_range_avg_keeps_daily_avg() -> None:
    """
    回归：
    stats-composite 明细列表出现“日期 + 范围 + 平均”的脏数据时，
    dataline 不应遗漏“平均xx”信息（应输出到每日行）。
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
        assert expect in out

