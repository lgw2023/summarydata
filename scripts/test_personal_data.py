from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_clean import test_aggregate_format as taf
from src.data_clean.test_aggregate_time import (
    _self_test_aggregate_patterns_to_time_jsonl,
    _self_test_time_bucket_merge_effect,
)
from src.data_clean.test_aggregate_dataframes import (
    _self_test_aggregate_patterns_to_dataframes,
)


def test_aggregate_formatted_text():
    print("测试aggregate_formatted_text")
    # 测试固定的数据样例
    from src.data_clean._personal_data_class_test_data import (
        test_NoDateValueSummaryRecord,
        test_NoTimestampTextSummaryRecord,
        test_PeriodTextSummaryRecord,
        test_PeriodValueCompareRecord,
        test_PeriodValueSingleSummaryRecord,
        test_PeriodValuemMultiSummaryRecord,
        test_SingleDateTextSummaryRecord,
        test_SingleDateValueMultiSummaryRecord,
        test_SingleDateValueSingleSummaryRecord,
        test_SingleMetricDetailRecord,
        test_SingleMetricStatsRecord,
        test_UnparsedRawPersonalData,
    )
    # from src.data_clean._personal_data_class_test_data_data_diff_sample import ...
    # 统一改用 aggregate_patterns_to_formatted_text() 的格式化聚合输出（Markdown 表格），
    # 替代原先逐类 self-test 的 format_print() 打印逻辑。
    _fixed_cases = [
        *list(test_SingleMetricDetailRecord),
        *list(test_PeriodValueSingleSummaryRecord),
        *list(test_PeriodTextSummaryRecord),
        *list(test_PeriodValueCompareRecord),
        *list(test_PeriodValuemMultiSummaryRecord),
        *list(test_SingleDateValueSingleSummaryRecord),
        *list(test_SingleDateTextSummaryRecord),
        *list(test_NoTimestampTextSummaryRecord),
        *list(test_NoDateValueSummaryRecord),
        *list(test_SingleMetricStatsRecord),
        *list(test_SingleDateValueMultiSummaryRecord),
        *list(test_UnparsedRawPersonalData),
    ]
    taf._self_test_aggregate_patterns_to_formatted_text(
        _fixed_cases,
        max_cases=None,
        print_preview=True,
    )
    taf._self_test_route_raw_personal_data_to_dataclass(
        test_SingleMetricDetailRecord=test_SingleMetricDetailRecord,
        test_PeriodValueSingleSummaryRecord=test_PeriodValueSingleSummaryRecord,
        test_PeriodValuemMultiSummaryRecord=test_PeriodValuemMultiSummaryRecord,
        test_PeriodTextSummaryRecord=test_PeriodTextSummaryRecord,
        test_PeriodValueCompareRecord=test_PeriodValueCompareRecord,
        test_SingleDateValueSingleSummaryRecord=test_SingleDateValueSingleSummaryRecord,
        test_SingleDateValueMultiSummaryRecord=test_SingleDateValueMultiSummaryRecord,
        test_SingleDateTextSummaryRecord=test_SingleDateTextSummaryRecord,
        test_NoDateValueSummaryRecord=test_NoDateValueSummaryRecord,
        test_NoTimestampTextSummaryRecord=test_NoTimestampTextSummaryRecord,
        test_SingleMetricStatsRecord=test_SingleMetricStatsRecord,
        test_UnparsedRawPersonalData=test_UnparsedRawPersonalData,
    )
    taf._self_test_recover_to_raw_data_roundtrip(
        test_SingleMetricDetailRecord=test_SingleMetricDetailRecord,
        test_PeriodValueSingleSummaryRecord=test_PeriodValueSingleSummaryRecord,
        test_PeriodValuemMultiSummaryRecord=test_PeriodValuemMultiSummaryRecord,
        test_PeriodTextSummaryRecord=test_PeriodTextSummaryRecord,
        test_PeriodValueCompareRecord=test_PeriodValueCompareRecord,
        test_SingleDateValueSingleSummaryRecord=test_SingleDateValueSingleSummaryRecord,
        test_SingleDateValueMultiSummaryRecord=test_SingleDateValueMultiSummaryRecord,
        test_SingleDateTextSummaryRecord=test_SingleDateTextSummaryRecord,
        test_NoDateValueSummaryRecord=test_NoDateValueSummaryRecord,
        test_NoTimestampTextSummaryRecord=test_NoTimestampTextSummaryRecord,
        test_SingleMetricStatsRecord=test_SingleMetricStatsRecord,
        test_UnparsedRawPersonalData=test_UnparsedRawPersonalData,
    )

    
    # # 测试excel数据文件
    # 从 Excel 读取 data 列；每行对应一个样本，最终得到“等长”的字符串列表
    _xlsx_path = PROJECT_ROOT / "summary_eval_diff.xlsx"
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "读取 Excel 需要依赖 pandas + openpyxl。请先安装：pip install pandas openpyxl"
        ) from exc

    _df = pd.read_excel(_xlsx_path, dtype=object)  # 默认第一个 sheet
    if "data" not in set(_df.columns.astype(str).tolist()):
        raise KeyError(
            f"Excel 中未找到 'data' 列：{_xlsx_path}；现有列={list(_df.columns)}"
        )

    _series = _df["data"]
    test_aggregate_patterns_to_formatted_text = [
        "" if pd.isna(v) else str(v) for v in _series.tolist()
    ]
    taf._self_test_aggregate_patterns_to_formatted_text(
        test_aggregate_patterns_to_formatted_text,
        max_cases=None,
        print_preview=True,
    )


def test_aggregate_by_time():
    print("测试aggregate_by_time（按日期/时间桶聚合）")

    # 测试固定的数据样例（所有样例都覆盖）
    from src.data_clean._personal_data_class_test_data import (
        test_NoDateValueSummaryRecord,
        test_NoTimestampTextSummaryRecord,
        test_PeriodTextSummaryRecord,
        test_PeriodValueCompareRecord,
        test_PeriodValueSingleSummaryRecord,
        test_PeriodValuemMultiSummaryRecord,
        test_SingleDateTextSummaryRecord,
        test_SingleDateValueMultiSummaryRecord,
        test_SingleDateValueSingleSummaryRecord,
        test_SingleMetricDetailRecord,
        test_SingleMetricStatsRecord,
        test_UnparsedRawPersonalData,
    )

    _self_test_aggregate_patterns_to_time_jsonl(test_SingleMetricDetailRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_PeriodValueSingleSummaryRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_PeriodTextSummaryRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_PeriodValueCompareRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_PeriodValuemMultiSummaryRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_SingleDateValueSingleSummaryRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_SingleDateTextSummaryRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_NoTimestampTextSummaryRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_NoDateValueSummaryRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_SingleMetricStatsRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_SingleDateValueMultiSummaryRecord, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_time_jsonl(test_UnparsedRawPersonalData, max_cases=None, print_preview=True)
    _self_test_aggregate_patterns_to_dataframes(
        ["\n".join([
            "\n".join(test_SingleMetricDetailRecord[:3]),
            "\n".join(test_PeriodValueSingleSummaryRecord[:3]),
            "\n".join(test_PeriodValuemMultiSummaryRecord[:3]),
            "\n".join(test_PeriodTextSummaryRecord[:3]),
            "\n".join(test_SingleDateValueSingleSummaryRecord[:3]),
            "\n".join(test_SingleDateValueMultiSummaryRecord[:3]),
            "\n".join(test_SingleDateTextSummaryRecord[:3]),
            "\n".join(test_NoTimestampTextSummaryRecord[:3]),
            "\n".join(test_NoDateValueSummaryRecord[:3]),
            "\n".join(test_SingleMetricStatsRecord[:3]),
            "\n".join(test_UnparsedRawPersonalData[:3]),
        ])],
        max_cases=None, print_preview=True)

    # 核心效果检查：确实发生“同一时间桶合并”
    _self_test_time_bucket_merge_effect(
        lines=[
            *list(test_SingleMetricDetailRecord),
            *list(test_PeriodValueSingleSummaryRecord),
            *list(test_PeriodValuemMultiSummaryRecord),
            *list(test_PeriodTextSummaryRecord),
            *list(test_SingleDateValueSingleSummaryRecord),
            *list(test_SingleDateValueMultiSummaryRecord),
            *list(test_SingleDateTextSummaryRecord),
            *list(test_SingleMetricStatsRecord),
        ]
    )

    # 测试 excel 数据文件（所有行都覆盖）
    _xlsx_path = PROJECT_ROOT / "summary_eval_diff.xlsx"
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "读取 Excel 需要依赖 pandas + openpyxl。请先安装：pip install pandas openpyxl"
        ) from exc

    _df = pd.read_excel(_xlsx_path, dtype=object)  # 默认第一个 sheet
    if "data" not in set(_df.columns.astype(str).tolist()):
        raise KeyError(
            f"Excel 中未找到 'data' 列：{_xlsx_path}；现有列={list(_df.columns)}"
        )

    _series = _df["data"]
    test_aggregate_patterns_to_time_jsonl = [
        "" if pd.isna(v) else str(v) for v in _series.tolist()
    ]
    _self_test_aggregate_patterns_to_time_jsonl(
        test_aggregate_patterns_to_time_jsonl,
        max_cases=None,
        print_preview=True,
    )


def test_aggregate_by_dataframes():
    print("测试aggregate_by_dataframes（聚合为 pandas DataFrame 列表）")

    # 测试固定的数据样例（所有样例都覆盖）
    from src.data_clean._personal_data_class_test_data import (
        test_NoDateValueSummaryRecord,
        test_NoTimestampTextSummaryRecord,
        test_PeriodTextSummaryRecord,
        test_PeriodValueCompareRecord,
        test_PeriodValueSingleSummaryRecord,
        test_PeriodValuemMultiSummaryRecord,
        test_SingleDateTextSummaryRecord,
        test_SingleDateValueMultiSummaryRecord,
        test_SingleDateValueSingleSummaryRecord,
        test_SingleMetricDetailRecord,
        test_SingleMetricStatsRecord,
        test_UnparsedRawPersonalData,
    )

    # _self_test_aggregate_patterns_to_dataframes(test_SingleMetricDetailRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_PeriodValueSingleSummaryRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_PeriodTextSummaryRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_PeriodValueCompareRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_PeriodValuemMultiSummaryRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_SingleDateValueSingleSummaryRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_SingleDateTextSummaryRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_NoTimestampTextSummaryRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_NoDateValueSummaryRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_SingleMetricStatsRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_SingleDateValueMultiSummaryRecord, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(test_UnparsedRawPersonalData, max_cases=None, print_preview=True)
    # _self_test_aggregate_patterns_to_dataframes(
    #     ["\n".join([
    #         "\n".join(test_SingleMetricDetailRecord[:3]),
    #         "\n".join(test_PeriodValueSingleSummaryRecord[:3]),
    #         "\n".join(test_PeriodValuemMultiSummaryRecord[:3]),
    #         "\n".join(test_PeriodTextSummaryRecord[:3]),
    #         "\n".join(test_SingleDateValueSingleSummaryRecord[:3]),
    #         "\n".join(test_SingleDateValueMultiSummaryRecord[:3]),
    #         "\n".join(test_SingleDateTextSummaryRecord[:3]),
    #         "\n".join(test_NoTimestampTextSummaryRecord[:3]),
    #         "\n".join(test_NoDateValueSummaryRecord[:3]),
    #         "\n".join(test_SingleMetricStatsRecord[:3]),
    #         "\n".join(test_UnparsedRawPersonalData[:3]),
    #     ])],
    #     max_cases=None, print_preview=True)

    # 测试 excel 数据文件（所有行都覆盖）
    _xlsx_path = PROJECT_ROOT / "summary_eval_diff.xlsx"
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "读取 Excel 需要依赖 pandas + openpyxl。请先安装：pip install pandas openpyxl"
        ) from exc

    _df = pd.read_excel(_xlsx_path, dtype=object)  # 默认第一个 sheet
    if "data" not in set(_df.columns.astype(str).tolist()):
        raise KeyError(
            f"Excel 中未找到 'data' 列：{_xlsx_path}；现有列={list(_df.columns)}"
        )

    _series = _df["data"]
    test_aggregate_patterns_to_dataframes = [
        "" if pd.isna(v) else str(v) for v in _series.tolist()
    ]
    _self_test_aggregate_patterns_to_dataframes(
        test_aggregate_patterns_to_dataframes,
        max_cases=None,
        print_preview=True,
    )


def main() -> None:
    # test_aggregate_formatted_text()
    # test_aggregate_by_time()
    test_aggregate_by_dataframes()

if __name__ == "__main__":
    main()
