from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# ==== 项目根目录 / sys.path ====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import load_env  # noqa: E402

try:  # noqa: E402
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


system_prompt = """
你是一个运动健康领域的数据分析助手，请根据提供的【个人数据】，分析数据条目、数据结构和样式，并输出JSON化的数据样式表示。
在该任务中：
1. 你不需要在意数据的缺失或者异常，仅需要关注数据条目、数据结构和样式。
2. 数据样式表示不需要起英文变量名，保留【个人数据】中原始的条目名称即可。
3. 数据样式表示的输出格式为JSON，不要输出其他内容。
4. 对于【个人数据】中的同一指标名称的多个记录，你不需要写成多条记录，只需要按示例所示写成一条记录即可，因为我们当前只是在所数据样式的总结。
5. "解析逻辑"这项只是作为样例给你看的，实际在最终输出时不需要包含这一项。
6. 对于“日期”或“开始日期”或“结束日期”或“日期范围”，你不需要提取真实的数字，只需要写明其格式即可，例如："日期": "Date (格式: YYYY/MM/DD)"。
7. 仅能严格从示例中的实体类型进行选取，如果碰到无法严格准确匹配符合已有示例中的实体类型的数据，则输出json体：[{"实体类型": "未定义"}]


以下是一些示例：
【个人数据】
2025/1/25 16:59的泳池游泳泳姿为：蝶泳,
[数据样式]
[
    {
        "实体类型": "单指标的明细记录",
        "解析逻辑": "{日期} {时间}的{指标名称}为：{数值}{单位}...（用中文或英文逗号分隔多条记录）",
        "核心字段": 
        {
            "日期": "Date (格式: YYYY/MM/DD)",
            "时间": "Timestamp",
            "指标名称": "泳池游泳泳姿",
            "数值类型": "String",
            "单位": "无"
        }
    }
]

【个人数据】
2025/2/1 06:07的户外跑步距离为：5.11千米,...,2025/2/8 06:06的户外跑步距离为：4.37千米,
[数据样式]
[
    {
        "实体类型": "单指标的明细记录",
        "解析逻辑": "{日期} {时间}的{指标名称}为：{数值}{单位}...（用中文或英文逗号分隔多条记录）",
        "核心字段": 
        {
            "日期": "Date (格式: YYYY/MM/DD)",
            "时间": "Timestamp",
            "指标名称": "户外跑步距离",
            "数值类型": "Float",
            "单位": "千米"
        }
    }
]

【个人数据】
2025/2/1 06:07的户外跑步用时为：0.47小时,...,2025/2/11 07:12的户外跑步用时为：0.68小时,
[数据样式]
[
    {
        "实体类型": "单指标的明细记录",
        "解析逻辑": "{日期} {时间}的{指标名称}为：{数值}{单位}...（用中文或英文逗号分隔多条记录）",
        "核心字段": 
        {
            "日期": "Date (格式: YYYY/MM/DD)",
            "时间": "Timestamp",
            "指标名称": "户外跑步用时",
            "数值类型": "Duration",
            "单位": "小时"
        }
    }
]

【个人数据】
2025/2/1 06:07的户外跑步热量为：311.00千卡,...,2025/2/18 07:03的户外跑步热量为：333.00千卡,
[数据样式]
[
    {
        "实体类型": "单指标的明细记录",
        "解析逻辑": "{日期} {时间}的{指标名称}为：{数值}{单位}...（用中文或英文逗号分隔多条记录）",
        "核心字段": 
        {
            "日期": "Date (格式: YYYY/MM/DD)",
            "时间": "Timestamp",
            "指标名称": "户外跑步热量",
            "数值类型": "Float",
            "单位": "千卡"
        }
    }
]

【个人数据】
2025/2/1 06:07的户外跑步步幅为：0.00,...,2025/2/18 07:03的户外跑步步幅为：0.00,
[数据样式]
[
    {
        "实体类型": "单指标的明细记录",
        "解析逻辑": "{日期} {时间}的{指标名称}为：{数值}{单位}...（用中文或英文逗号分隔多条记录）",
        "核心字段": 
        {
            "日期": "Date (格式: YYYY/MM/DD)",
            "时间": "Timestamp",
            "指标名称": "户外跑步步幅",
            "数值类型": "Float",
            "单位": "无"
        }
    }
]

【个人数据】
2025/2/1至2025/2/22的跑步总距离为201.27千米
[数据样式]
[
    {
        "实体类型": "周期数值单项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}的{指标名称}为{数值}{单位}",
        "示例": "2025/2/1至2025/2/22的跑步总距离为201.27千米",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD)",
            "指标名称": "跑步总距离",
            "数值类型": "Float",
            "单位": "千米"
        }
    }
]

【个人数据】
2025/2/1至2025/2/22的跑步总热量为12052.00千卡
[数据样式]
[
    {
        "实体类型": "周期数值单项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}的{指标名称}为{数值}{单位}",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD)",
            "指标名称": "跑步总热量",
            "数值类型": "Float",
            "单位": "千卡"
        }
    }
]

【个人数据】
2025/2/1至2025/2/22的平均户外跑步心率为136.00次/分钟
[数据样式]
[
    {
        "实体类型": "周期数值单项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}的{指标名称}为{数值}{单位}",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD)",
            "指标名称": "平均户外跑步心率",
            "数值类型": "Float",
            "单位": "次/分钟"
        }
    }
]

【个人数据】
6/16~6/22 深睡连续性偏低
[数据样式]
[
    {
        "实体类型": "周期文本单多项总结",
        "解析逻辑": "{开始日期}~{结束日期} {洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: MM/DD)",
            "结束日期": "Date (格式: MM/DD)",
            "指标名称": "深睡连续性",
            "状态描述": "String"
        }
    }
]

【个人数据】
6/23~6/26 睡眠得分一般，睡眠质量一般
[数据样式]
[
    {
        "实体类型": "周期文本单多项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期} {洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: MM/DD)",
            "结束日期": "Date (格式: MM/DD)",
            "指标名称": "睡眠得分",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "周期文本单多项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期} {洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: MM/DD)",
            "结束日期": "Date (格式: MM/DD)",
            "指标名称": "睡眠质量",
            "状态描述": "String"
        }
    }
]

【个人数据】
6/16~6/22的平均快速眼动比例为19.0%，6/23~6/26的平均快速眼动比例为22.0%，少3.0%
[数据样式]
[
    {
        "实体类型": "周期数值对比记录",
        "解析逻辑": "{日期范围1}的{指标名称}为{数值1}{单位}，{日期范围2}的{指标名称}为{数值2}{单位}，{对比逻辑类型}{差异数值}{单位}",
        "核心字段": 
        {
            "日期范围1": "Date (格式: MM/DD~MM/DD 或 MM/DD)",
            "日期范围2": "Date (格式: MM/DD~MM/DD 或 MM/DD)",
            "指标名称": "平均快速眼动比例",
            "数值类型": "Float",
            "单位": "%",
            "对比逻辑类型": "String",
            "差异数值类型": "Float"
        }
    }
]

【个人数据】
2023年6月17日到2025年6月16日平均入睡时间23:20正常，最早入睡时间22:35正常，最晚入睡时间01:30偏晚
[数据样式]
[
    {
        "实体类型": "周期数值多项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}{洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{数值}{单位}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "指标名称": "平均入睡时间",
            "数值类型": "Timestamp",
            "单位": "无",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "周期数值多项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}{洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{数值}{单位}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "指标名称": "最早入睡时间",
            "数值类型": "Timestamp",
            "单位": "无",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "周期数值多项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}{洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{数值}{单位}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "指标名称": "最晚入睡时间",
            "数值类型": "Timestamp",
            "单位": "无",
            "状态描述": "String"
        }
    }
]

【个人数据】
2023年6月17日到2025年6月16日平均零星小睡入睡时间13:00, 最晚零星小睡入睡时间13:00, 最早零星小睡入睡时间13:00 
[数据样式]
[
    {
        "实体类型": "周期数值多项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}{洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{数值}{单位}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "指标名称": "平均零星小睡入睡时间",
            "数值类型": "Timestamp",
            "单位": "无",
            "状态描述": "无"
        }
    },
    {
        "实体类型": "周期数值多项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}{洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{数值}{单位}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "指标名称": "最晚零星小睡入睡时间",
            "数值类型": "Timestamp",
            "单位": "无",
            "状态描述": "无"
        }
    },
    {
        "实体类型": "周期数值多项总结",
        "解析逻辑": "{开始日期}[到｜至｜~]{结束日期}{洞察1}，{洞察2}，...（洞察之间用中文或英文逗号分隔；每条洞察通常可拆为{指标名称}{数值}{单位}{状态描述}）",
        "核心字段": 
        {
            "开始日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "结束日期": "Date (格式: YYYY/MM/DD 或 MM/DD)",
            "指标名称": "最早零星小睡入睡时间",
            "数值类型": "Timestamp",
            "单位": "无",
            "状态描述": "无"
        }
    }
]

【个人数据】
入睡时间欠规律,睡眠得分中等，睡眠质量良好,深睡连续性偏低
[数据样式]
[
    {
        "实体类型": "无时间日期的文本总结",
        "解析逻辑": "{指标名称}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "入睡时间",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "无时间日期的文本总结",
        "解析逻辑": "{指标名称}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "睡眠得分",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "无时间日期的文本总结",
        "解析逻辑": "{指标名称}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "睡眠质量",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "无时间日期的文本总结",
        "解析逻辑": "{指标名称}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "深睡连续性",
            "状态描述": "String"
        }
    }
]


【个人数据】
平均压力均值36分正常，最低压力均值31分正常，最高压力均值49分正常
[数据样式]
[
    {
        "实体类型": "无时间日期的数值总结",
        "解析逻辑": "{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "平均压力均值",
            "数值类型": "Float",
            "单位": "分",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "无时间日期的数值总结",
        "解析逻辑": "{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "最低压力均值",
            "数值类型": "Float",
            "单位": "分",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "无时间日期的数值总结",
        "解析逻辑": "{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "最高压力均值",
            "数值类型": "Float",
            "单位": "分",
            "状态描述": "String"
        }
    }
]

【个人数据】
平均血氧98%正常，最低血氧98%正常，最高血氧99%正常
[数据样式]
[
    {
        "实体类型": "无时间日期的数值总结",
        "解析逻辑": "{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "平均血氧",
            "数值类型": "Float",
            "单位": "%",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "无时间日期的数值总结",
        "解析逻辑": "{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "最低血氧",
            "数值类型": "Float",
            "单位": "%",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "无时间日期的数值总结",
        "解析逻辑": "{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "最高血氧",
            "数值类型": "Float",
            "单位": "%",
            "状态描述": "String"
        }
    }
]

【个人数据】
7/21锻炼时长17分钟偏低
[数据样式]
[
    {
        "实体类型": "单日期数值单项总结",
        "解析逻辑": "{日期}{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "锻炼时长",
            "日期": "Date (格式: MM月DD日)",
            "数值类型": "Float",
            "单位": "分钟",
            "状态描述": "String"
        }
    },
]

【个人数据】
4/23入睡时间01:40偏晚
[数据样式]
[
    {
        "实体类型": "单日期数值单项总结",
        "解析逻辑": "{日期}{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "入睡时间",
            "日期": "Date (格式: MM/DD)",
            "数值类型": "Timestamp",
            "单位": "无",
            "状态描述": "String"
        }
    },
]

【个人数据】
2025/3/6夜间睡眠时长偏长
[数据样式]
[
    {
        "实体类型": "单日期文本总结",
        "解析逻辑": "{日期}{指标名称}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "夜间睡眠时长",
            "时间": "Date (格式: YYYY/MM/DD)",
            "状态描述": "String"
        }
    },
]

【个人数据】
8/2 睡眠得分中等，睡眠质量良好
[数据样式]
[
    {
        "实体类型": "单日期文本总结",
        "解析逻辑": "{日期} {指标名称}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "睡眠得分",
            "时间": "Date (格式: MM/DD)",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "单日期文本总结",
        "解析逻辑": "{日期} {指标名称}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "睡眠质量",
            "时间": "Date (格式: MM/DD)",
            "状态描述": "String"
        }
    },
]

【个人数据】
血氧饱和度：[8月2日97%-97%,8月3日98%-98%,8月4日96%-96%,8月5日97%-97%,8月6日97%-97%,8月7日96%-96%] , 平均血氧饱和度97%正常, 最高血氧饱和度98%正常, 最低血氧饱和度96%正常
[数据样式]
[
    {
        "实体类型": "单指标的统计复合记录",
        "解析逻辑": "{核心指标名}：[{日期}{数值}{单位}]，{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "血氧饱和度",
            "数据列表": 
            [
                {
                    "日期": "Date (格式: MM月DD日)",
                    "数值类型": "FloatRange",
                    "单位": "%"
                }
            ],
            "统计汇总描述": 
            [
                {
                    "指标名称": "平均血氧饱和度",
                    "数值类型": "Float",
                    "单位": "%",
                    "状态描述": "String"
                },
                {
                    "指标名称": "最低血氧饱和度",
                    "数值类型": "Float",
                    "单位": "%",
                    "状态描述": "String"
                },
                {
                    "指标名称": "最高血氧饱和度",
                    "数值类型": "Float",
                    "单位": "%",
                    "状态描述": "String"
                }
            ]
        }
    }
]

【个人数据】
浅睡比例：[1月1日51%, ..., 1月30日46%, 1月31日43%]，平均浅睡比例47%正常，最低浅睡比例36%正常，最高浅睡比例55%正常
[数据样式]
[
    {
        "实体类型": "单指标的统计复合记录",
        "解析逻辑": "{核心指标名}：[{日期}{数值}{单位}]，{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "浅睡比例",
            "数据列表": 
            [
                {
                    "日期": "Date (格式: MM月DD日)",
                    "数值类型": "Float",
                    "单位": "%"
                }
            ],
            "统计汇总描述": 
            [
                {
                    "指标名称": "平均浅睡比例",
                    "数值类型": "Float",
                    "单位": "%",
                    "状态描述": "String"
                },
                {
                    "指标名称": "最低浅睡比例",
                    "数值类型": "Float",
                    "单位": "%",
                    "状态描述": "String"
                },
                {
                    "指标名称": "最高浅睡比例",
                    "数值类型": "Float",
                    "单位": "%",
                    "状态描述": "String"
                }
            ]
        }
    }
]


【个人数据】
睡眠时长：[2月17日10小时10分钟,2月18日5小时58分钟,2月19日7小时6分钟,2月20日8小时47分钟,2月21日7小时45分钟,2月22日7小时42分钟,2月23日8小时40分钟],平均睡眠时长8小时1分钟正常，最短睡眠时长5小时52分钟偏短，最长睡眠时长10小时10分钟偏长
[数据样式]
[
    {
        "实体类型": "单指标的统计复合记录",
        "解析逻辑": "{核心指标名}：[{日期}{数值}{单位}]，{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "睡眠时长",
            "数据列表": 
            [
                {
                    "日期": "Date (格式: MM月DD日)",
                    "数值类型": "Duration",
                    "单位": "小时/分钟（例如：xx小时xx分钟）"
                }
            ],
            "统计汇总描述": 
            [
                {
                    "指标名称": "平均睡眠时长",
                    "数值类型": "Duration",
                    "单位": "小时/分钟（例如：xx小时xx分钟）",
                    "状态描述": "String"
                },
                {
                    "指标名称": "最短睡眠时长",
                    "数值类型": "Duration",
                    "单位": "小时/分钟（例如：xx小时xx分钟）",
                    "状态描述": "String"
                },
                {
                    "指标名称": "最长睡眠时长",
                    "数值类型": "Duration",
                    "单位": "小时/分钟（例如：xx小时xx分钟）",
                    "状态描述": "String"
                }
            ]
        }
    }
]

【个人数据】
8月7日血氧饱和度96%-96%,平均血氧饱和度96%正常, 最高血氧饱和度96%正常, 最低血氧饱和度96%正常
[数据样式]
[
    {
        "实体类型": "单日期数值多项总结",
        "解析逻辑": "{日期}{指标名称}{数值}{单位},{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "血氧饱和度",
            "日期": "Date (格式: MM月DD日)",
            "数值类型": "FloatRange",
            "单位": "%",
            "状态描述": "无"
        }
    },
    {
        "实体类型": "单日期数值多项总结",
        "解析逻辑": "{日期}{指标名称}{数值}{单位},{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "平均血氧饱和度",
            "日期": "Date (格式: MM月DD日)",
            "数值类型": "Float",
            "单位": "%",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "单日期数值多项总结",
        "解析逻辑": "{日期}{指标名称}{数值}{单位},{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        "核心字段": 
        {
            "指标名称": "最高血氧饱和度96",
            "日期": "Date (格式: MM月DD日)",
            "数值类型": "Float",
            "单位": "%",
            "状态描述": "String"
        }
    },
    {
        "实体类型": "单日期数值多项总结",
        "解析逻辑": "{日期}{指标名称}{数值}{单位},{指标名称}{数值}{单位}{状态描述}...（用中文或英文逗号分隔多条记录，若只有一条记录，则不使用逗号）",
        {
            "指标名称": "最低血氧饱和度",
            "日期": "Date (格式: MM月DD日)",
            "数值类型": "Float",
            "单位": "%",
            "状态描述": "String"
        }
    }
]

【个人数据】
8/2~8/8占比最高的情绪是愉悦
[数据样式]
[
    {
        "实体类型": "未定义", 
    }
]

【个人数据】
锻炼时长27分钟，距离目标40分钟还差13分钟
[数据样式]
[
    {
        "实体类型": "未定义", 
    }
]
"""

user_prompt = """
【个人数据】
"""

load_env()

# 默认从 .env 读取 QwenMax 的 URL/NAME/API_KEY
# 注意：不要在 import 阶段强制校验这些变量，否则 --help / 仅分析模式也会直接报错。
QWENMAX_BASE_URL = os.getenv("LLM_MODEL_QWENMAX_URL")
QWENMAX_MODEL_NAME = os.getenv("LLM_MODEL_QWENMAX_NAME")
QWENMAX_API_KEY = os.getenv("LLM_MODEL_QWENMAX_API_KEY")
# 并行度：每个样本内部逐行并行调用大模型；保持样本/行的输出顺序不变
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "8"))

_thread_local = threading.local()


def get_client() -> Any:
    """
    为并行调用准备：给每个线程创建独立的 OpenAI client，避免共享 client 的潜在线程安全问题。
    """
    if OpenAI is None:
        raise ImportError("openai 包未安装，请先 pip install openai>=1.0")
    if not QWENMAX_BASE_URL:
        raise ValueError("缺少环境变量 LLM_MODEL_QWENMAX_URL（请在 .env 中配置）")
    if not QWENMAX_MODEL_NAME:
        raise ValueError("缺少环境变量 LLM_MODEL_QWENMAX_NAME（请在 .env 中配置）")
    if not QWENMAX_API_KEY:
        raise ValueError("缺少环境变量 LLM_MODEL_QWENMAX_API_KEY（请在 .env 中配置）")

    c = getattr(_thread_local, "client", None)
    if c is None:
        c = OpenAI(api_key=QWENMAX_API_KEY, base_url=QWENMAX_BASE_URL)
        _thread_local.client = c
    return c


@dataclass(frozen=True)
class SampleRow:
    row_idx: int
    data: str
    query: str | None
    last_query: str | None
    last_answer_phone: str | None
    raw_row: dict[str, str]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    return _FENCE_RE.sub("", s.strip())


def _try_json_loads(s: str) -> Any | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_valid_json_blob(raw: str) -> Any | None:
    """
    从模型回复中尽量抽取出一个“合法 JSON”对象（优先数组/对象），并返回解析后的 Python 对象。
    解析失败返回 None。
    """
    if not raw:
        return None
    raw = _strip_code_fences(raw)

    # 1) 直接尝试
    direct = _try_json_loads(raw)
    if direct is not None:
        return direct

    # 2) 尝试截取最外层 [...] 或 {...}
    candidates: list[str] = []
    lbr = raw.find("[")
    rbr = raw.rfind("]")
    if lbr != -1 and rbr != -1 and rbr > lbr:
        candidates.append(raw[lbr : rbr + 1])
    lcb = raw.find("{")
    rcb = raw.rfind("}")
    if lcb != -1 and rcb != -1 and rcb > lcb:
        candidates.append(raw[lcb : rcb + 1])

    for c in candidates:
        parsed = _try_json_loads(c)
        if parsed is not None:
            return parsed
    return None


# ========= “未定义实体类型”输出检测 =========
def _is_unknown_entity_type_output(obj: Any) -> bool:
    """
    判断模型输出是否为“未定义实体类型”占位：
    [{"实体类型": "未定义", "个人数据": "...原始内容..."}]
    """
    if not isinstance(obj, list) or not obj:
        return False
    for it in obj:
        if isinstance(it, dict) and str(it.get("实体类型", "")).strip() == "未定义":
            return True
    return False


def _style_contains_unknown_value_type(obj: Any) -> bool:
    """
    扫描 style JSON（通常为 list[dict]）中是否出现“数值类型=未定义/差异数值类型=未定义”等情况。
    该情况通常代表历史推理输出不符合预期，需要重新推理补全。
    """
    if obj is None:
        return False
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_str = str(k).strip()
            # 兼容：数值类型 / 差异数值类型 / xxx数值类型
            if k_str == "数值类型" or k_str.endswith("数值类型"):
                if isinstance(v, str) and v.strip() == "未定义":
                    return True
            if _style_contains_unknown_value_type(v):
                return True
        return False
    if isinstance(obj, list):
        return any(_style_contains_unknown_value_type(it) for it in obj)
    return False


def _should_reinfer_existing_style(existing_line_item: dict[str, Any]) -> bool:
    """
    用于“补全缺失结果”阶段：当旧结果的 style 命中以下任一条件时，强制重新推理该行：
    - 输出为 [{"实体类型":"未定义"}] 这类占位
    - 任意（差异）数值类型字段为 "未定义"
    """
    style = existing_line_item.get("style")
    if _is_unknown_entity_type_output(style):
        return True
    if _style_contains_unknown_value_type(style):
        return True
    return False


# ========= LLM 输出内容校验（样式 JSON） =========
_DATE_LIKE_KEYS: set[str] = {
    "日期",
    "开始日期",
    "结束日期",
    "日期范围",
    "日期范围1",
    "日期范围2",
}
# 注意：FloatRange 用于“单日期数值多项总结”（例如：血氧饱和度 96%-98%）
_ALLOWED_VALUE_TYPES: set[str] = {"Int", "Float", "String", "Duration", "Timestamp", "FloatRange"}
_ALLOWED_STATUS_VALUES: set[str] = {"String", "无"}
_HAS_DIGIT_RE = re.compile(r"\d")


# ========= 实体类型白名单（强约束）=========
_ALLOWED_ENTITY_TYPES_STRICT: set[str] = {
    "单指标的明细记录",
    "周期数值单项总结",
    "周期文本单多项总结",
    "周期数值对比记录",
    "周期数值多项总结",
    "无时间日期的文本总结",
    "无时间日期的数值总结",
    "单日期数值单项总结",
    "单日期文本总结",
    "单指标的统计复合记录",
    "单日期数值多项总结",
}


def _enforce_allowed_entity_types_or_unknown(obj: Any) -> Any:
    """
    若模型返回的结果不在允许的实体类型集合中（必须严格匹配），统一改写为固定值：
    [{"实体类型": "未定义"}]
    """
    if not isinstance(obj, list) or not obj:
        return [{"实体类型": "未定义"}]
    for it in obj:
        if not isinstance(it, dict):
            return [{"实体类型": "未定义"}]
        et = str(it.get("实体类型", "")).strip()
        if et not in _ALLOWED_ENTITY_TYPES_STRICT:
            return [{"实体类型": "未定义"}]
    return obj


def _is_date_format_placeholder(v: Any) -> bool:
    """
    日期/日期范围类字段必须是“格式占位描述”，例如：
    - "Date (格式: YYYY/MM/DD)"
    - "Date (格式: MM月DD日)"
    并且不能包含任何具体数字（0-9）。
    """
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    # 要求显式写 Date + 格式，并且禁止出现任何数字
    if "Date" not in s or "格式" not in s:
        return False
    if _HAS_DIGIT_RE.search(s):
        return False
    return True


def _is_date_format_placeholder_exact(v: Any, *, expected_format: str) -> bool:
    """
    更严格的日期占位校验：要求严格等于 "Date (格式: <expected_format>)"，并且不包含任何数字。
    例如 expected_format="MM/DD" -> "Date (格式: MM/DD)"
    """
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    if _HAS_DIGIT_RE.search(s):
        return False
    return s == f"Date (格式: {expected_format})"


def _is_time_format_placeholder(v: Any) -> bool:
    """
    时间类字段必须是“格式占位描述”，例如：
    - "Time (格式: HH:mm)"
    并且不能包含任何具体数字（0-9）。
    """
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    # 要求显式写 Time + 格式，并且禁止出现任何数字
    if "Time" not in s or "格式" not in s:
        return False
    if _HAS_DIGIT_RE.search(s):
        return False
    return True


def _is_time_format_placeholder_exact(v: Any, *, expected_format: str) -> bool:
    """
    更严格的时间占位校验：要求严格等于 "Time (格式: <expected_format>)"，并且不包含任何数字。
    例如 expected_format="HH:mm" -> "Time (格式: HH:mm)"
    """
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    if _HAS_DIGIT_RE.search(s):
        return False
    return s == f"Time (格式: {expected_format})"


def _validate_style_json(obj: Any) -> tuple[bool, list[str]]:
    """
    对模型输出的“数据样式 JSON”做约束校验：
    1) 日期类字段（日期/开始日期/结束日期/日期范围...）必须是格式占位，且不出现具体数字
    2) 状态描述 只能是 "String" 或 "无"
    3) 数值类型（数值类型/差异数值类型...）只能在白名单内

    返回：(是否通过, 错误列表)；错误列表用于构造重试提示。
    """
    errors: list[str] = []

    def walk(x: Any, path: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                k_str = str(k)
                next_path = f"{path}.{k_str}" if path else k_str

                # (1) 日期字段：必须是格式占位，且不能有具体数字
                if k_str in _DATE_LIKE_KEYS:
                    if not _is_date_format_placeholder(v):
                        errors.append(
                            f"{next_path} 必须为 Date 的格式占位（例如 'Date (格式: YYYY/MM/DD)'），且不能包含任何数字：实际={v!r}"
                        )

                # (2) 状态描述字段：只能为 String / 无
                if k_str in {"状态描述"}:
                    if not isinstance(v, str) or v.strip() not in _ALLOWED_STATUS_VALUES:
                        errors.append(
                            f"{next_path} 只能是 'String' 或 '无'：实际={v!r}"
                        )

                # (3) 数值类型字段：只能为白名单
                if k_str.endswith("数值类型") or k_str == "数值类型":
                    # 兼容“周期数值多项总结”等场景：数值类型可能为 Time 占位（如 "Time (格式: HH:mm)"）
                    if not isinstance(v, str) or (v.strip() not in _ALLOWED_VALUE_TYPES and not _is_time_format_placeholder(v)):
                        errors.append(
                            f"{next_path} 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：实际={v!r}"
                        )

                walk(v, next_path)
        elif isinstance(x, list):
            for i, it in enumerate(x):
                walk(it, f"{path}[{i}]")
        else:
            return

    walk(obj, "")
    return (len(errors) == 0), errors


def _validate_data_pattern_json(obj: Any) -> tuple[bool, list[str]]:
    """
    对模型输出的“数据样式 JSON（data pattern）”做更强约束校验。

    当前只对以下实体类型进行严格校验（其他类型暂不校验）：
    - 实体类型=“单指标的明细记录”
    - 实体类型=“周期数值单项总结”
    - 实体类型=“周期文本单多项总结”
    - 实体类型=“周期数值对比记录”
    - 实体类型=“周期数值多项总结”
    - 实体类型=“无时间日期的文本总结”
    - 实体类型=“无时间日期的数值总结”
    - 实体类型=“单日期数值单项总结”
    - 实体类型=“单日期文本总结”
    - 实体类型=“单指标的统计复合记录”
    - 实体类型=“单日期数值多项总结”

    约束：
    - 整体必须是 JSON 数组（list）
    - 当存在任意 item 的 "实体类型" == "单指标的明细记录" 时：
      - 数组长度必须为 1，且唯一元素必须为 dict
      - item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"日期","时间","指标名称","数值类型","单位"}
      - "日期" 必须为 Date 格式占位（且不含数字）
      - "时间" 必须为 Time 格式占位（且不含数字）
      - "数值类型" 必须在白名单内
    - 当存在任意 item 的 "实体类型" == "周期数值单项总结" 时：
      - 数组长度必须为 1，且唯一元素必须为 dict
      - item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"开始日期","结束日期","指标名称","数值类型","单位"}
      - "开始日期"/"结束日期" 必须为 Date 格式占位（且不含数字）
      - "数值类型" 必须在白名单内
    - 当存在任意 item 的 "实体类型" == "周期文本单多项总结" 时：
      - 数组长度必须 >= 1，且每个元素必须为 dict
      - 每个 item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"开始日期","结束日期","指标名称","状态描述"}
      - "开始日期"/"结束日期" 必须严格为 "Date (格式: MM/DD)"（且不含数字）
      - "状态描述" 必须严格为 "String"
    - 当存在任意 item 的 "实体类型" == "周期数值对比记录" 时：
      - 数组长度必须为 1，且唯一元素必须为 dict
      - item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"日期范围1","日期范围2","指标名称","数值类型","单位","对比逻辑类型","差异数值类型"}
      - "日期范围1"/"日期范围2" 必须严格为 "Date (格式: MM/DD~MM/DD 或 MM/DD)"（且不含数字）
      - "数值类型"/"差异数值类型" 必须在白名单内
      - "对比逻辑类型" 必须严格为 "String"
    - 当存在任意 item 的 "实体类型" == "周期数值多项总结" 时：
      - 数组长度必须 >= 1，且每个元素必须为 dict
      - 每个 item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"开始日期","结束日期","指标名称","数值类型","单位","状态描述"}
      - "开始日期"/"结束日期" 必须严格为 "Date (格式: YYYY/MM/DD 或 MM/DD)"（且不含数字）
      - "数值类型" 必须严格为 "Time (格式: HH:mm)"（且不含数字）
      - "状态描述" 必须严格为 "String"
    - 当存在任意 item 的 "实体类型" == "无时间日期的文本总结" 时：
      - 数组长度必须 >= 1，且每个元素必须为 dict
      - 每个 item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"指标名称","状态描述"}
      - "状态描述" 必须严格为 "String"
    - 当存在任意 item 的 "实体类型" == "无时间日期的数值总结" 时：
      - 数组长度必须 >= 1，且每个元素必须为 dict
      - 每个 item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"指标名称","数值类型","单位","状态描述"}
      - "数值类型" 必须在白名单内
      - "状态描述" 必须严格为 "String"
    - 当存在任意 item 的 "实体类型" == "单日期数值单项总结" 时：
      - 数组长度必须 >= 1，且每个元素必须为 dict
      - 每个 item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"指标名称","日期","数值类型","单位","状态描述"}
      - "日期" 必须严格为 "Date (格式: MM月DD日)"（且不含数字）
      - "数值类型" 必须在白名单内
      - "状态描述" 必须严格为 "String"
    - 当存在任意 item 的 "实体类型" == "单日期文本总结" 时：
      - 数组长度必须 >= 1，且每个元素必须为 dict
      - 每个 item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"指标名称","时间","状态描述"}
      - "时间" 必须严格为 "Date (格式: MM/DD)"（且不含数字）
      - "状态描述" 必须严格为 "String"
    - 当存在任意 item 的 "实体类型" == "单指标的统计复合记录" 时：
      - 数组长度必须为 1，且唯一元素必须为 dict
      - item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"指标名称","数据列表","统计汇总描述"}
      - "数据列表" 必须为 dict 列表，且长度必须为 1；唯一元素必须且只能包含：
        {"日期","数值类型","单位"}；其中日期必须严格为 "Date (格式: MM月DD日)"
      - "统计汇总描述" 必须为 dict 列表，且长度必须 >= 1；每个元素必须且只能包含：
        {"指标名称","数值类型","单位","状态描述"}；其中状态描述必须严格为 "String"
    - 当存在任意 item 的 "实体类型" == "单日期数值多项总结" 时（新格式）：
      - 数组长度必须 >= 1，且每个元素必须为 dict
      - 每个 item 只能包含且必须包含两个 key：{"实体类型","核心字段"}
      - "核心字段" 必须为 dict，且 key 不能多不能少，严格为：
        {"指标名称","日期","数值类型","单位","状态描述"}
      - "日期" 必须严格为 "Date (格式: MM月DD日)"（且不含数字）
      - "数值类型" 必须在白名单内（允许 FloatRange / Float / ...）
      - "状态描述" 必须严格为 "String" 或 "无"
    """
    errors: list[str] = []

    if not isinstance(obj, list):
        return False, [f"顶层必须为 JSON 数组（list）：实际类型={type(obj).__name__}"]

    # 仅当出现受支持的实体类型时触发强校验
    supported_types = {
        "单指标的明细记录",
        "周期数值单项总结",
        "周期文本单多项总结",
        "周期数值对比记录",
        "周期数值多项总结",
        "无时间日期的文本总结",
        "无时间日期的数值总结",
        "单日期数值单项总结",
        "单日期文本总结",
        "单指标的统计复合记录",
        "单日期数值多项总结",
    }
    present_supported_types: set[str] = set()
    for it in obj:
        if isinstance(it, dict):
            et = str(it.get("实体类型", "")).strip()
            if et in supported_types:
                present_supported_types.add(et)

    if not present_supported_types:
        return True, []

    if len(present_supported_types) > 1:
        return False, [f"同一次输出不支持混合多种受校验的实体类型：实际={sorted(present_supported_types)}"]

    triggered_type = next(iter(present_supported_types))

    if triggered_type in {"单指标的明细记录", "周期数值单项总结", "周期数值对比记录", "单指标的统计复合记录"}:
        if len(obj) != 1:
            return False, [f"当实体类型为“{triggered_type}”时，整体必须为长度为 1 的列表：实际长度={len(obj)}"]
        items = [obj[0]]
    else:
        # 允许多条（长度>=1）
        if len(obj) < 1:
            return False, [f"当实体类型为“{triggered_type}”时，整体列表长度必须 >= 1：实际长度={len(obj)}"]
        items = obj

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(
                f"当实体类型为“{triggered_type}”时，列表元素必须为对象（dict）：index={idx} 实际={type(item).__name__}"
            )
            continue

        allowed_item_keys = {"实体类型", "核心字段"}
        item_keys = set(map(str, item.keys()))
        if item_keys != allowed_item_keys:
            extra = sorted(item_keys - allowed_item_keys)
            missing = sorted(allowed_item_keys - item_keys)
            if extra:
                errors.append(f"{triggered_type} 顶层对象不允许包含额外字段：index={idx} extra={extra}")
            if missing:
                errors.append(f"{triggered_type} 顶层对象缺少必需字段：index={idx} missing={missing}")

        et = str(item.get("实体类型", "")).strip()
        if et != triggered_type:
            errors.append(f'实体类型必须为 "{triggered_type}"：index={idx} 实际={item.get("实体类型")!r}')

        core = item.get("核心字段")
        if not isinstance(core, dict):
            errors.append(f"核心字段 必须为对象（dict）：index={idx} 实际={core!r}")
            continue

        if triggered_type == "单指标的明细记录":
            required_core_keys = {"日期", "时间", "指标名称", "数值类型", "单位"}
        elif triggered_type == "周期数值单项总结":
            required_core_keys = {"开始日期", "结束日期", "指标名称", "数值类型", "单位"}
        elif triggered_type == "周期数值对比记录":
            required_core_keys = {
                "日期范围1",
                "日期范围2",
                "指标名称",
                "数值类型",
                "单位",
                "对比逻辑类型",
                "差异数值类型",
            }
        elif triggered_type == "周期数值多项总结":
            required_core_keys = {"开始日期", "结束日期", "指标名称", "数值类型", "单位", "状态描述"}
        elif triggered_type == "无时间日期的文本总结":
            required_core_keys = {"指标名称", "状态描述"}
        elif triggered_type == "无时间日期的数值总结":
            required_core_keys = {"指标名称", "数值类型", "单位", "状态描述"}
        elif triggered_type == "单日期数值单项总结":
            required_core_keys = {"指标名称", "日期", "数值类型", "单位", "状态描述"}
        elif triggered_type == "单日期文本总结":
            required_core_keys = {"指标名称", "时间", "状态描述"}
        elif triggered_type == "单指标的统计复合记录":
            required_core_keys = {"指标名称", "数据列表", "统计汇总描述"}
        elif triggered_type == "单日期数值多项总结":
            required_core_keys = {"指标名称", "日期", "数值类型", "单位", "状态描述"}
        else:
            required_core_keys = {"开始日期", "结束日期", "指标名称", "状态描述"}

        core_keys = set(map(str, core.keys()))
        if core_keys != required_core_keys:
            extra = sorted(core_keys - required_core_keys)
            missing = sorted(required_core_keys - core_keys)
            if extra:
                errors.append(f"核心字段 不允许包含额外字段：index={idx} extra={extra}")
            if missing:
                errors.append(f"核心字段 缺少必需字段：index={idx} missing={missing}")

        # 字段值校验（仅在字段存在时做检查，避免重复噪声）
        if triggered_type == "单指标的明细记录":
            if "日期" in core and not _is_date_format_placeholder(core.get("日期")):
                errors.append(
                    f'核心字段.日期 必须为 Date 的格式占位（例如 "Date (格式: YYYY/MM/DD)"），且不能包含任何数字：index={idx} 实际={core.get("日期")!r}'
                )
            if "时间" in core and not _is_time_format_placeholder(core.get("时间")):
                errors.append(
                    f'核心字段.时间 必须为 Time 的格式占位（例如 "Time (格式: HH:mm)"），且不能包含任何数字：index={idx} 实际={core.get("时间")!r}'
                )
            if "数值类型" in core:
                vt = core.get("数值类型")
                if not isinstance(vt, str) or vt.strip() not in _ALLOWED_VALUE_TYPES:
                    errors.append(
                        f"核心字段.数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={vt!r}"
                    )
        elif triggered_type == "周期数值单项总结":
            if "开始日期" in core and not _is_date_format_placeholder(core.get("开始日期")):
                errors.append(
                    f'核心字段.开始日期 必须为 Date 的格式占位（例如 "Date (格式: YYYY/MM/DD)"），且不能包含任何数字：index={idx} 实际={core.get("开始日期")!r}'
                )
            if "结束日期" in core and not _is_date_format_placeholder(core.get("结束日期")):
                errors.append(
                    f'核心字段.结束日期 必须为 Date 的格式占位（例如 "Date (格式: YYYY/MM/DD)"），且不能包含任何数字：index={idx} 实际={core.get("结束日期")!r}'
                )
            if "数值类型" in core:
                vt = core.get("数值类型")
                if not isinstance(vt, str) or vt.strip() not in _ALLOWED_VALUE_TYPES:
                    errors.append(
                        f"核心字段.数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={vt!r}"
                    )
        elif triggered_type == "周期数值对比记录":
            # 周期数值对比记录：日期范围必须严格为指定占位；数值类型在白名单；对比逻辑类型必须为 String
            if "日期范围1" in core and not _is_date_format_placeholder_exact(
                core.get("日期范围1"), expected_format="MM/DD~MM/DD 或 MM/DD"
            ):
                errors.append(
                    f'核心字段.日期范围1 必须严格为 "Date (格式: MM/DD~MM/DD 或 MM/DD)"，且不能包含任何数字：index={idx} 实际={core.get("日期范围1")!r}'
                )
            if "日期范围2" in core and not _is_date_format_placeholder_exact(
                core.get("日期范围2"), expected_format="MM/DD~MM/DD 或 MM/DD"
            ):
                errors.append(
                    f'核心字段.日期范围2 必须严格为 "Date (格式: MM/DD~MM/DD 或 MM/DD)"，且不能包含任何数字：index={idx} 实际={core.get("日期范围2")!r}'
                )
            if "数值类型" in core:
                vt = core.get("数值类型")
                if not isinstance(vt, str) or vt.strip() not in _ALLOWED_VALUE_TYPES:
                    errors.append(
                        f"核心字段.数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={vt!r}"
                    )
            if "差异数值类型" in core:
                dvt = core.get("差异数值类型")
                if not isinstance(dvt, str) or dvt.strip() not in _ALLOWED_VALUE_TYPES:
                    errors.append(
                        f"核心字段.差异数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={dvt!r}"
                    )
            if "对比逻辑类型" in core:
                ct = core.get("对比逻辑类型")
                if not isinstance(ct, str) or ct.strip() != "String":
                    errors.append(f'核心字段.对比逻辑类型 必须严格为 "String"：index={idx} 实际={ct!r}')
        elif triggered_type == "周期数值多项总结":
            # 周期数值多项总结：日期必须严格为指定占位；数值类型必须严格为 Time 占位；状态描述必须严格为 String
            if "开始日期" in core and not _is_date_format_placeholder_exact(
                core.get("开始日期"), expected_format="YYYY/MM/DD 或 MM/DD"
            ):
                errors.append(
                    f'核心字段.开始日期 必须严格为 "Date (格式: YYYY/MM/DD 或 MM/DD)"，且不能包含任何数字：index={idx} 实际={core.get("开始日期")!r}'
                )
            if "结束日期" in core and not _is_date_format_placeholder_exact(
                core.get("结束日期"), expected_format="YYYY/MM/DD 或 MM/DD"
            ):
                errors.append(
                    f'核心字段.结束日期 必须严格为 "Date (格式: YYYY/MM/DD 或 MM/DD)"，且不能包含任何数字：index={idx} 实际={core.get("结束日期")!r}'
                )
            if "数值类型" in core and not _is_time_format_placeholder_exact(core.get("数值类型"), expected_format="HH:mm"):
                errors.append(
                    f'核心字段.数值类型 必须严格为 "Time (格式: HH:mm)"，且不能包含任何数字：index={idx} 实际={core.get("数值类型")!r}'
                )
            if "状态描述" in core:
                sd = core.get("状态描述")
                if not isinstance(sd, str) or sd.strip() != "String":
                    errors.append(f'核心字段.状态描述 必须严格为 "String"：index={idx} 实际={sd!r}')
        elif triggered_type == "无时间日期的文本总结":
            # 无时间日期的文本总结：状态描述必须严格为 String
            if "状态描述" in core:
                sd = core.get("状态描述")
                if not isinstance(sd, str) or sd.strip() != "String":
                    errors.append(f'核心字段.状态描述 必须严格为 "String"：index={idx} 实际={sd!r}')
        elif triggered_type == "无时间日期的数值总结":
            # 无时间日期的数值总结：数值类型在白名单；状态描述必须严格为 String
            if "数值类型" in core:
                vt = core.get("数值类型")
                if not isinstance(vt, str) or vt.strip() not in _ALLOWED_VALUE_TYPES:
                    errors.append(
                        f"核心字段.数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={vt!r}"
                    )
            if "状态描述" in core:
                sd = core.get("状态描述")
                if not isinstance(sd, str) or sd.strip() != "String":
                    errors.append(f'核心字段.状态描述 必须严格为 "String"：index={idx} 实际={sd!r}')
        elif triggered_type == "单日期数值单项总结":
            # 单日期数值单项总结：日期必须严格为 MM月DD日 占位；数值类型在白名单；状态描述必须严格为 String
            if "日期" in core and not _is_date_format_placeholder_exact(core.get("日期"), expected_format="MM月DD日"):
                errors.append(
                    f'核心字段.日期 必须严格为 "Date (格式: MM月DD日)"，且不能包含任何数字：index={idx} 实际={core.get("日期")!r}'
                )
            if "数值类型" in core:
                vt = core.get("数值类型")
                if not isinstance(vt, str) or vt.strip() not in _ALLOWED_VALUE_TYPES:
                    errors.append(
                        f"核心字段.数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={vt!r}"
                    )
            if "状态描述" in core:
                sd = core.get("状态描述")
                if not isinstance(sd, str) or sd.strip() != "String":
                    errors.append(f'核心字段.状态描述 必须严格为 "String"：index={idx} 实际={sd!r}')
        elif triggered_type == "单日期文本总结":
            # 单日期文本总结：时间必须严格为 MM/DD 的 Date 占位；状态描述必须严格为 String
            if "时间" in core and not _is_date_format_placeholder_exact(core.get("时间"), expected_format="MM/DD"):
                errors.append(
                    f'核心字段.时间 必须严格为 "Date (格式: MM/DD)"，且不能包含任何数字：index={idx} 实际={core.get("时间")!r}'
                )
            if "状态描述" in core:
                sd = core.get("状态描述")
                if not isinstance(sd, str) or sd.strip() != "String":
                    errors.append(f'核心字段.状态描述 必须严格为 "String"：index={idx} 实际={sd!r}')
        elif triggered_type == "单指标的统计复合记录":
            # 单指标的统计复合记录：
            # - 数据列表：dict 列表且长度=1；元素键严格为 {日期,数值类型,单位}；日期严格 MM月DD日 占位
            # - 统计汇总描述：dict 列表且长度>=1；元素键严格为 {指标名称,数值类型,单位,状态描述}；状态描述严格 String
            data_list = core.get("数据列表")
            if not isinstance(data_list, list):
                errors.append(f"核心字段.数据列表 必须为列表（list）：index={idx} 实际={type(data_list).__name__}")
            else:
                if len(data_list) != 1:
                    errors.append(f"核心字段.数据列表 列表长度必须为 1：index={idx} 实际长度={len(data_list)}")
                if len(data_list) >= 1:
                    it0 = data_list[0]
                    if not isinstance(it0, dict):
                        errors.append(
                            f"核心字段.数据列表[0] 必须为对象（dict）：index={idx} 实际={type(it0).__name__}"
                        )
                    else:
                        required_dl_keys = {"日期", "数值类型", "单位"}
                        dl_keys = set(map(str, it0.keys()))
                        if dl_keys != required_dl_keys:
                            extra = sorted(dl_keys - required_dl_keys)
                            missing = sorted(required_dl_keys - dl_keys)
                            if extra:
                                errors.append(f"核心字段.数据列表[0] 不允许包含额外字段：index={idx} extra={extra}")
                            if missing:
                                errors.append(f"核心字段.数据列表[0] 缺少必需字段：index={idx} missing={missing}")
                        if "日期" in it0 and not _is_date_format_placeholder_exact(it0.get("日期"), expected_format="MM月DD日"):
                            errors.append(
                                f'核心字段.数据列表[0].日期 必须严格为 "Date (格式: MM月DD日)"，且不能包含任何数字：index={idx} 实际={it0.get("日期")!r}'
                            )
                        if "数值类型" in it0:
                            vt = it0.get("数值类型")
                            if not isinstance(vt, str) or vt.strip() not in _ALLOWED_VALUE_TYPES:
                                errors.append(
                                    f"核心字段.数据列表[0].数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={vt!r}"
                                )

            summaries = core.get("统计汇总描述")
            if not isinstance(summaries, list):
                errors.append(f"核心字段.统计汇总描述 必须为列表（list）：index={idx} 实际={type(summaries).__name__}")
            else:
                if len(summaries) < 1:
                    errors.append(f"核心字段.统计汇总描述 列表长度必须 >= 1：index={idx} 实际长度={len(summaries)}")
                for j, sit in enumerate(summaries):
                    if not isinstance(sit, dict):
                        errors.append(
                            f"核心字段.统计汇总描述[{j}] 必须为对象（dict）：index={idx} 实际={type(sit).__name__}"
                        )
                        continue
                    required_sum_keys = {"指标名称", "数值类型", "单位", "状态描述"}
                    sum_keys = set(map(str, sit.keys()))
                    if sum_keys != required_sum_keys:
                        extra = sorted(sum_keys - required_sum_keys)
                        missing = sorted(required_sum_keys - sum_keys)
                        if extra:
                            errors.append(f"核心字段.统计汇总描述[{j}] 不允许包含额外字段：index={idx} extra={extra}")
                        if missing:
                            errors.append(f"核心字段.统计汇总描述[{j}] 缺少必需字段：index={idx} missing={missing}")
                    if "数值类型" in sit:
                        vt = sit.get("数值类型")
                        if not isinstance(vt, str) or vt.strip() not in _ALLOWED_VALUE_TYPES:
                            errors.append(
                                f"核心字段.统计汇总描述[{j}].数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={vt!r}"
                            )
                    if "状态描述" in sit:
                        st = sit.get("状态描述")
                        if not isinstance(st, str) or st.strip() != "String":
                            errors.append(f'核心字段.统计汇总描述[{j}].状态描述 必须严格为 "String"：index={idx} 实际={st!r}')
        elif triggered_type == "单日期数值多项总结":
            # 单日期数值多项总结（新格式）：
            # - 顶层：list[dict]，长度>=1
            # - 每个 item：{"实体类型","核心字段"}（前面已校验）
            # - 核心字段键严格为 {"指标名称","日期","数值类型","单位","状态描述"}
            # - 日期严格为 MM月DD日 的 Date 占位；数值类型在白名单；状态描述只能是 String/无
            if "日期" in core and not _is_date_format_placeholder_exact(core.get("日期"), expected_format="MM月DD日"):
                errors.append(
                    f'核心字段.日期 必须严格为 "Date (格式: MM月DD日)"，且不能包含任何数字：index={idx} 实际={core.get("日期")!r}'
                )
            if "数值类型" in core:
                vt = core.get("数值类型")
                if not isinstance(vt, str) or vt.strip() not in _ALLOWED_VALUE_TYPES:
                    errors.append(
                        f"核心字段.数值类型 只能是 {sorted(_ALLOWED_VALUE_TYPES)} 之一：index={idx} 实际={vt!r}"
                    )
            if "状态描述" in core:
                sd = core.get("状态描述")
                if not isinstance(sd, str) or sd.strip() not in _ALLOWED_STATUS_VALUES:
                    errors.append(f"核心字段.状态描述 只能是 'String' 或 '无'：index={idx} 实际={sd!r}")
        else:
            # 周期文本单多项总结：日期必须严格为 MM/DD 占位；状态描述描述必须严格为 String
            if "开始日期" in core and not _is_date_format_placeholder_exact(core.get("开始日期"), expected_format="MM/DD"):
                errors.append(
                    f'核心字段.开始日期 必须严格为 "Date (格式: MM/DD)"，且不能包含任何数字：index={idx} 实际={core.get("开始日期")!r}'
                )
            if "结束日期" in core and not _is_date_format_placeholder_exact(core.get("结束日期"), expected_format="MM/DD"):
                errors.append(
                    f'核心字段.结束日期 必须严格为 "Date (格式: MM/DD)"，且不能包含任何数字：index={idx} 实际={core.get("结束日期")!r}'
                )
            if "状态描述" in core:
                sd = core.get("状态描述")
                if not isinstance(sd, str) or sd.strip() != "String":
                    errors.append(f'核心字段.状态描述 必须严格为 "String"：index={idx} 实际={sd!r}')

    return (len(errors) == 0), errors


def _to_optional_str(v: Any) -> str | None:
    """
    把 CSV 单元格值规范成“可选字符串”：
    - 缺失/空串/仅空白 -> None
    - 常见空值标记（大小写不敏感）：none/null/nan -> None
    - 否则返回 strip 后的字符串
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in {"none", "null", "nan"}:
        return None
    return s


def _split_valid_lines(text: str) -> list[str]:
    """
    将多行文本拆分为 lines，并过滤掉“无效字符串”的子项：
    - 空串/仅空白
    - 常见空值标记：none/null/nan（大小写不敏感）
    - 命中“无内容/无结果”提示语（例如：没有查询到/未查询到/没有/暂无）

    注意：仅用于过滤，不改变有效行的原始内容（保留原始空白/格式），以便与历史输出做精确匹配。
    """
    raw_lines = str(text).splitlines()
    return [ln for ln in raw_lines if _to_optional_str(ln) is not None and not _should_skip_line(ln)]


# ========= “无内容/无结果”行过滤 =========
# 需求：若个人数据样本的某行内容包含以下字眼，则跳过该内容（不参与推理、也不计入 data_lines）
_SKIP_LINE_KEYWORDS: tuple[str, ...] = (
    "没有查询",
    "未查询",
    "未查",
    "暂无",
    "没有",
    "无结果",
    "无数据",
    "无法",
)


def _should_skip_line(line: str) -> bool:
    """
    判断某一行个人数据是否应被跳过。
    - 命中任一关键字（“包含”匹配）即跳过。

    注意：该规则是强过滤；若你担心 “没有” 造成误杀，可在此处将其改为更严格的模式。
    """
    s = str(line).strip()
    if not s:
        return True
    return any(kw in s for kw in _SKIP_LINE_KEYWORDS)


def iter_samples_from_csv(csv_path: Path, column_name: str = "data") -> Iterable[SampleRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or column_name not in reader.fieldnames:
            raise ValueError(f"CSV 缺少列名 {column_name!r}，实际列名={reader.fieldnames}")
        for row_idx, row in enumerate(reader):
            # DictReader 读出来 value 可能是 None，这里统一转成 str
            normalized_row: dict[str, str] = {k: (v if v is not None else "") for k, v in row.items()}
            cell = (normalized_row.get(column_name) or "").strip()

            # 按需求：last_query/last_answer_phone “有且为有效字符串才填”，否则 None
            query = _to_optional_str(normalized_row.get("query"))
            last_query = _to_optional_str(normalized_row.get("last_query"))
            last_answer_phone = _to_optional_str(normalized_row.get("last_answer_phone"))

            yield SampleRow(
                row_idx=row_idx,
                data=cell,
                query=query,
                last_query=last_query,
                last_answer_phone=last_answer_phone,
                raw_row=normalized_row,
            )


def infer_style_json_for_line(
    data_line: str,
    *,
    max_retries: int = 10,
    retry_sleep_sec: float = 0.3,
    client: Any | None = None,
) -> Any:
    """
    对单行“个人数据”做推理并抽取合法 JSON。
    失败则最多重试 max_retries 次，最终仍失败返回空 JSON 数组 []。
    """
    base_user_prompt = f"{user_prompt}{data_line}\n"
    last_bad_output: str | None = None

    c = client or get_client()
    for attempt in range(1, max_retries + 1):
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": base_user_prompt})
        if last_bad_output:
            messages.append({"role": "assistant", "content": last_bad_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "上一次输出不符合要求。请严格只输出合法 JSON（不要解释、不要代码块），并满足以下约束：\n"
                        "1) 所有“日期/开始日期/结束日期/日期范围/日期范围1/日期范围2”等字段必须写成 Date 的格式占位，"
                        "例如：\"Date (格式: YYYY/MM/DD)\"，且不能包含任何具体数字（0-9）。\n"
                        "1.1) 所有“时间”字段（如出现）必须写成 Time 的格式占位，例如：\"Time (格式: HH:mm)\"，且不能包含任何具体数字（0-9）。\n"
                        "2) 所有“状态描述”字段只能是 \"String\" 或 \"无\"，不能写具体内容（例如“欠规律”）。\n"
                        "3) 所有“数值类型/差异数值类型”等字段只能是：Int / Float / String / Duration / Timestamp（不能是 Integer 等）。\n"
                        "4) 当实体类型为“单指标的明细记录”时：整体必须是长度为 1 的列表，且仅包含一个对象；该对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：日期、时间、指标名称、数值类型、单位。\n"
                        "5) 当实体类型为“周期数值单项总结”时：整体必须是长度为 1 的列表，且仅包含一个对象；该对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：开始日期、结束日期、指标名称、数值类型、单位。\n"
                        "6) 当实体类型为“周期文本单多项总结”时：整体必须是字典列表，长度必须 >= 1；列表中每个对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：开始日期、结束日期、指标名称、状态描述；其中开始日期/结束日期必须严格为 \"Date (格式: MM/DD)\"，状态描述必须严格为 \"String\"。\n"
                        "7) 当实体类型为“周期数值对比记录”时：整体必须是长度为 1 的列表，且仅包含一个对象；该对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：日期范围1、日期范围2、指标名称、数值类型、单位、对比逻辑类型、差异数值类型；"
                        "其中日期范围1/日期范围2必须严格为 \"Date (格式: MM/DD~MM/DD 或 MM/DD)\"，对比逻辑类型必须严格为 \"String\"，数值类型/差异数值类型必须在白名单内。\n"
                        "8) 当实体类型为“周期数值多项总结”时：整体必须是字典列表，长度必须 >= 1；列表中每个对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：开始日期、结束日期、指标名称、数值类型、单位、状态描述；"
                        "其中开始日期/结束日期必须严格为 \"Date (格式: YYYY/MM/DD 或 MM/DD)\"，数值类型必须严格为 \"Time (格式: HH:mm)\"，状态描述必须严格为 \"String\"。\n"
                        "9) 当实体类型为“无时间日期的文本总结”时：整体必须是字典列表，长度必须 >= 1；列表中每个对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：指标名称、状态描述；其中状态描述必须严格为 \"String\"。\n"
                        "10) 当实体类型为“无时间日期的数值总结”时：整体必须是字典列表，长度必须 >= 1；列表中每个对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：指标名称、数值类型、单位、状态描述；其中数值类型必须在白名单内，状态描述必须严格为 \"String\"。\n"
                        "11) 当实体类型为“单指标的统计复合记录”时：整体必须是长度为 1 的列表，且仅包含一个对象；该对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：指标名称、数据列表、统计汇总描述；其中数据列表必须是字典列表且长度必须为 1，且唯一元素必须且只能包含：日期、数值类型、单位（日期必须严格为 \"Date (格式: MM月DD日)\"）；"
                        "统计汇总描述必须是字典列表且长度必须 >= 1，且每个元素必须且只能包含：指标名称、数值类型、单位、状态描述（状态描述必须严格为 \"String\"）。\n"
                        "12) 当实体类型为“单日期数值单项总结”时：整体必须是字典列表，长度必须 >= 1；列表中每个对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：指标名称、日期、数值类型、单位、状态描述；其中日期必须严格为 \"Date (格式: MM月DD日)\"，数值类型必须在白名单内，状态描述必须严格为 \"String\"。\n"
                        "13) 当实体类型为“单日期文本总结”时：整体必须是字典列表，长度必须 >= 1；列表中每个对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：指标名称、时间、状态描述；其中时间必须严格为 \"Date (格式: MM/DD)\"，状态描述必须严格为 \"String\"。\n"
                        "14) 当实体类型为“单日期数值多项总结”时：整体必须是字典列表，长度必须 >= 1；列表中每个对象只能有“实体类型/核心字段”两项；"
                        "核心字段必须且只能包含：指标名称、日期、数值类型、单位、状态描述；其中日期必须严格为 \"Date (格式: MM月DD日)\"，数值类型必须在白名单内，状态描述只能是 \"String\" 或 \"无\"。\n"
                    ),
                }
            )

        resp = c.chat.completions.create(model=QWENMAX_MODEL_NAME, messages=messages)
        content = (resp.choices[0].message.content or "").strip()
        parsed = extract_valid_json_blob(content)
        if parsed is not None:
            # 实体类型强约束：若不在白名单中，直接返回固定未定义结果（不重试）。
            parsed = _enforce_allowed_entity_types_or_unknown(parsed)
            if _is_unknown_entity_type_output(parsed):
                return parsed

            ok, errs = _validate_style_json(parsed)
            ok2, errs2 = True, []
            if ok:
                ok2, errs2 = _validate_data_pattern_json(parsed)
                if ok2:
                    # 当实体类型为“未定义”时，打印该行个人数据，便于人工排查/补充规则
                    if _is_unknown_entity_type_output(parsed):
                        print(data_line, file=sys.stderr)
                    return parsed
            # JSON 合法但不满足字段约束：触发重试
            last_bad_output = content
            # 给下一轮重试一个更明确的失败原因（截断避免太长）
            merged_errs: list[str] = []
            if errs:
                merged_errs.extend(errs)
            if errs2:
                merged_errs.extend(errs2)
            if merged_errs:
                brief = "\n".join(merged_errs[:8])
                if len(merged_errs) > 8:
                    brief += f"\n...（共 {len(merged_errs)} 条）"
                last_bad_output = f"{content}\n\n[校验失败原因]\n{brief}"
            if retry_sleep_sec > 0:
                time.sleep(retry_sleep_sec)
            continue

        last_bad_output = content
        if retry_sleep_sec > 0:
            time.sleep(retry_sleep_sec)

    return []


def _is_int_like(v: Any) -> bool:
    try:
        int(v)
        return True
    except Exception:
        return False


def load_existing_results(out_path: Path) -> dict[int, dict[str, Any]]:
    """
    读取已存在的 JSONL 输出文件，按 row_idx 建索引（同 row_idx 多次出现时保留“最后一次”）。
    读取失败/脏行将被忽略（例如上次中断导致的半行）。
    """
    existing: dict[int, dict[str, Any]] = {}
    if not out_path.exists():
        return existing
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            row_idx = obj.get("row_idx")
            if not _is_int_like(row_idx):
                continue
            existing[int(row_idx)] = obj
    return existing


def _normalize_existing_patterns(
    existing_obj: dict[str, Any] | None,
    lines: list[str],
) -> dict[int, dict[str, Any]]:
    """
    将已有结果的 data_patterns 规范为 {line_idx -> {line_idx, data, style}} 映射。
    只保留：
    - line_idx 合法且在范围内
    - data 与当前输入该行完全一致（避免复用“输入变化”的旧行结果）

    注意：style 可以是任意 JSON（包括 []）；只要这一行存在，就视为“被处理过”。
    """
    if not existing_obj:
        return {}
    raw = existing_obj.get("data_patterns")
    if not isinstance(raw, list):
        return {}
    n = len(lines)
    out: dict[int, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        li = item.get("line_idx")
        if not _is_int_like(li):
            continue
        li_int = int(li)
        if li_int < 0 or li_int >= n:
            continue
        # 兼容“LLM 异常时 data 置为 None”的记录：用 data_raw 做匹配，保证断点续跑可复用
        match_data = item.get("data")
        if match_data is None:
            match_data = item.get("data_raw")
        if str(match_data or "") != lines[li_int]:
            continue
        out_item: dict[str, Any] = {"line_idx": li_int, "data": item.get("data", lines[li_int]), "style": item.get("style", [])}
        if "data_raw" in item:
            out_item["data_raw"] = item.get("data_raw")
        out[li_int] = out_item
    return out


def _is_valid_data_value(v: Any) -> bool:
    """
    用于判断 data_patterns[*].data 是否“有效且不为空”。
    - None / 空串 / 仅空白 / 常见空值标记（none/null/nan，大小写不敏感） -> False
    - 其他 -> True
    """
    return _to_optional_str(v) is not None


def _normalize_existing_patterns_by_data_lines(
    existing_obj: dict[str, Any] | None,
    expected_lines: int | None = None,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """
    将已有结果的 data_patterns 规范为 {line_idx -> item} 映射，并返回 (data_lines, map)。

    与旧逻辑不同：不再依赖“当前输入 lines 与旧结果 data 完全一致”来决定复用，
    而是以 JSONL 结果本身的 data_lines 为准（可选用 expected_lines 做额外一致性校验）。
    """
    if not existing_obj:
        return 0, {}
    data_lines = existing_obj.get("data_lines")
    if not _is_int_like(data_lines):
        return 0, {}
    n = int(data_lines)
    if expected_lines is not None and expected_lines != n:
        # 输入行数发生变化时，不复用该样本的旧结果
        return n, {}
    raw = existing_obj.get("data_patterns")
    if not isinstance(raw, list):
        return n, {}

    out: dict[int, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        li = item.get("line_idx")
        if not _is_int_like(li):
            continue
        li_int = int(li)
        if li_int < 0 or li_int >= n:
            continue

        # 规范化字段：缺失时给默认值，保留 data_raw 便于断点补全
        normalized: dict[str, Any] = {
            "line_idx": li_int,
            "data": item.get("data", None),
            "style": item.get("style", []),
        }
        if "data_raw" in item:
            normalized["data_raw"] = item.get("data_raw")
        out[li_int] = normalized

    return n, out


def is_sample_complete(existing_obj: dict[str, Any] | None, expected_lines: int | None = None) -> bool:
    """
    完整性检查（按新规则）：
    - 以 JSONL 里的 data_lines 为准（可选 expected_lines 用来校验行数一致）
    - 需要存在 0..N-1 每个 line_idx
    - 且每个条目的 data 必须是“有效且不为空”（data=None/空串/空白/none/null/nan 都视为不完整）
    - 且每个条目的 style 不能出现“未定义不稳定输出”（例如：实体类型=未定义、或（差异）数值类型=未定义）
    """
    # 全新输入/没有旧结果时，不应被判定为“完整可复用”
    # 例外：当期望行数本身为 0（空样本）时，视为完整（无需推理）
    if expected_lines == 0:
        return True
    if existing_obj is None:
        return False

    n, m = _normalize_existing_patterns_by_data_lines(existing_obj, expected_lines=expected_lines)
    # 若旧结果宣称 data_lines=0，但当前样本并非空样本，则视为不完整
    if n == 0:
        return False
    if len(m) != n or any(i not in m for i in range(n)):
        return False
    # 既要 data 有效，也要避免复用“未定义/不稳定”的历史推理结果
    return all(_is_valid_data_value(m[i].get("data")) and (not _should_reinfer_existing_style(m[i])) for i in range(n))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 CSV 读取个人数据列（默认列名 data），对每个样本进行样式抽取并输出为 JSONL。"
        )
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        help="输入 CSV 文件路径（例如：data_diff_sample.csv）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=(
            "输出 JSONL 文件路径。默认按规则写入："
            "data/<csv_stem>/perlsonal_datapatterns.jsonl"
        ),
    )
    parser.add_argument(
        "--column",
        type=str,
        default="data",
        help="CSV 中包含个人数据文本的列名（默认：data）",
    )
    parser.add_argument(
        "--analyze-patterns",
        action="store_true",
        help=(
            "仅解析输出 JSONL（--out 或按 --raw-data 推导默认输出路径），"
            "统计所有“实体类型-指标名称”的组合并输出；不会调用大模型。"
        ),
    )
    parser.add_argument(
        "--extract-metric-only",
        type=str,
        default="",
        help=(
            "仅解析输出 JSONL（--out 或按 --raw-data 推导默认输出路径），提取并打印："
            "所有“该行(line_idx)内只出现该指标名称（不混入其他指标名称）”的行；"
            "打印内容为命中行（row_idx + line_idx + data）。不会调用大模型。"
            "（如需旧版“整条样本(row_idx)仅包含该指标”的严格过滤，请使用 --extract-metric-only-sample）"
        ),
    )
    parser.add_argument(
        "--extract-entity-type",
        type=str,
        default="",
        help=(
            "仅解析输出 JSONL（--out 或按 --raw-data 推导默认输出路径），遍历所有样本的所有行的 style 条目；"
            "只要条目的“实体类型”严格匹配该值，就把该条目打印出来（包含 row_idx/line_idx/data/style_item）。"
            "不会调用大模型。"
        ),
    )
    parser.add_argument(
        "--extract-metric-only-sample",
        type=str,
        default="",
        help=(
            "仅解析输出 JSONL（--out 或按 --raw-data 推导默认输出路径），提取并打印："
            "整条样本(row_idx)内只出现该“指标名称”（不允许混入其他指标名称）的样本；"
            "打印内容为命中该指标的行（row_idx + line_idx + data）。不会调用大模型。"
        ),
    )
    args = parser.parse_args()

    def _resolve_out_path() -> Path:
        """
        统一解析输出 JSONL 路径：
        - 若显式传入 --out：直接使用（相对路径按项目根目录拼接）
        - 否则若传入 --raw-data：按规则推导默认输出路径 data/<csv_stem>/perlsonal_datapatterns.jsonl
        - 否则报错（仅分析/提取模式也需要知道要解析哪个 JSONL）
        """
        if args.out:
            p = Path(args.out)
            return p if p.is_absolute() else (PROJECT_ROOT / p)
        if args.raw_data:
            csv_path_for_out = Path(args.raw_data)
            if not csv_path_for_out.is_absolute():
                csv_path_for_out = PROJECT_ROOT / csv_path_for_out
            # 命名规则：data_diff_sample.csv -> data/data_diff_sample/perlsonal_datapatterns.jsonl
            return PROJECT_ROOT / "data" / csv_path_for_out.stem / "perlsonal_datapatterns.jsonl"
        raise ValueError("缺少 --out 或 --raw-data：无法定位要解析/写入的输出 JSONL 路径")

    out_path = _resolve_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 仅统计模式：不跑推理、不读 CSV，只解析现有输出 JSONL
    if args.analyze_patterns:
        analyze_output_patterns(out_path)
        return

    # 仅提取模式：不跑推理、不读 CSV，只解析现有输出 JSONL
    if args.extract_metric_only:
        extract_lines_by_metric_only(out_path, metric_name=args.extract_metric_only)
        return

    # 仅提取模式：按实体类型，不跑推理、不读 CSV，只解析现有输出 JSONL
    if args.extract_entity_type:
        extract_style_items_by_entity_type(out_path, entity_type=args.extract_entity_type)
        return

    # 仅提取模式（旧版严格：按 row_idx）：不跑推理、不读 CSV，只解析现有输出 JSONL
    if args.extract_metric_only_sample:
        extract_samples_by_metric_only(out_path, metric_name=args.extract_metric_only_sample)
        return

    if not args.raw_data:
        raise ValueError("缺少 --raw-data（例如：--raw-data data_diff_sample.csv）")

    csv_path = Path(args.raw_data)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV：{csv_path}")

    # 自动加载已有输出，并检查可复用/需补全的样本规模
    existing = load_existing_results(out_path)
    total = 0
    reuse_complete = 0
    reuse_partial = 0
    for sample in iter_samples_from_csv(csv_path, column_name=args.column):
        total += 1
        lines = _split_valid_lines(sample.data)
        ex_obj = existing.get(sample.row_idx)
        if ex_obj is None:
            continue
        if is_sample_complete(ex_obj, expected_lines=len(lines)):
            reuse_complete += 1
        else:
            reuse_partial += 1
    if out_path.exists():
        print(
            f"检测到已有输出：{out_path}；"
            f"可直接复用完整样本 {reuse_complete}/{total}，"
            f"需补全/重算样本 {reuse_partial}/{total}。"
        )

    # 逐样本实时写入（JSONL）：每条 CSV 样本对应输出一行；样本内部逐行推理并一一对应
    written = 0
    reused = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sample in iter_samples_from_csv(csv_path, column_name=args.column):
            lines = _split_valid_lines(sample.data)
            ex_obj = existing.get(sample.row_idx)
            _, ex_map = _normalize_existing_patterns_by_data_lines(ex_obj, expected_lines=len(lines))

            if is_sample_complete(ex_obj, expected_lines=len(lines)):
                # 完整样本：复用已有结果（外层元信息用当前 CSV 的，以保证同步）
                line_results = [ex_map[i] for i in range(len(lines))] if lines else []
                reused += 1
            else:
                # 不完整样本：只补齐缺失/无效的行（data=None 或空值均视为不完整）
                n = len(lines)
                ordered: list[dict[str, Any] | None] = [None] * n
                missing: list[tuple[int, str]] = []
                for i, text in enumerate(lines):
                    if i in ex_map and _is_valid_data_value(ex_map[i].get("data")):
                        # data 有效：通常可保留旧结果；
                        # 但按新需求：在“补全缺失结果”时，若旧结果的（差异）数值类型为“未定义”
                        # 或 style 为“实体类型=未定义”占位，也要重新推理解析。
                        if not _should_reinfer_existing_style(ex_map[i]):
                            ordered[i] = ex_map[i]
                            continue

                    # data 无效或缺失：优先从旧结果的 data_raw 取原始个人数据来补全
                    raw_text: str | None = None
                    if i in ex_map:
                        raw_text = _to_optional_str(ex_map[i].get("data_raw"))
                    if raw_text is None:
                        raw_text = text  # 兜底：使用当前 CSV 的该行文本
                    missing.append((i, raw_text))

                if missing:
                    if LLM_MAX_WORKERS <= 1 or len(missing) == 1:
                        for line_idx, text in missing:
                            try:
                                style = infer_style_json_for_line(text, max_retries=10)
                            except Exception:
                                # LLM 调用出现意外异常：标记 data=None，方便后续排查
                                style = []
                                ordered[line_idx] = {
                                    "line_idx": line_idx,
                                    "data": None,
                                    "data_raw": text,
                                    "style": style,
                                }
                            else:
                                ordered[line_idx] = {"line_idx": line_idx, "data": text, "style": style}
                    else:
                        max_workers = min(LLM_MAX_WORKERS, len(missing))
                        with ThreadPoolExecutor(max_workers=max_workers) as ex:
                            future_map = {
                                ex.submit(
                                    infer_style_json_for_line,
                                    text,
                                    max_retries=10,
                                    client=None,
                                ): (line_idx, text)
                                for (line_idx, text) in missing
                            }
                            for fut in as_completed(future_map):
                                line_idx, text = future_map[fut]
                                try:
                                    style = fut.result()
                                except Exception:
                                    style = []
                                    ordered[line_idx] = {
                                        "line_idx": line_idx,
                                        "data": None,
                                        "data_raw": text,
                                        "style": style,
                                    }
                                else:
                                    ordered[line_idx] = {"line_idx": line_idx, "data": text, "style": style}

                # 理论上不会有 None；保守兜底（None 代表未生成，这里补空 style 也算“被处理过”）
                line_results = [
                    x if x is not None else {"line_idx": i, "data": lines[i], "style": []}
                    for i, x in enumerate(ordered)
                ]

            out = {
                "row_idx": sample.row_idx,
                "query": sample.query,
                "last_query": sample.last_query,
                "last_answer_phone": sample.last_answer_phone,
                "data_lines": len(lines),
                "data_patterns": line_results,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()  # 关键：每处理完一个样本就写入并落盘，便于中断续跑
            written += 1

    print(f"已写入 {written} 行到：{out_path}（其中直接复用完整样本 {reused} 行）")


def analyze_output_patterns(out_path: Path, *, top_k: int = 50) -> None:
    """
    解析输出结果 JSONL，以“指标名称”为主统计其出现过哪些“实体类型”及次数。

    说明：
    - 输出文件每行对应一个 CSV 样本（row_idx），其中 data_patterns[*].style 是一个 JSON 数组；
    - 每个 style item 通常为 {"实体类型": "...", "核心字段": {...}}，其中核心字段里一般包含 "指标名称"。
    """
    if not out_path.exists():
        raise FileNotFoundError(f"找不到输出 JSONL：{out_path}")

    def _norm_str(v: Any) -> str:
        s = str(v).strip()
        return s if s else "<空>"

    # 指标名称 -> {实体类型 -> count}
    metric_to_entity: dict[str, dict[str, int]] = {}
    # 指标名称 -> total count（便于排序）
    metric_total: dict[str, int] = {}

    total_rows = 0
    total_lines = 0
    total_style_items = 0
    bad_json_lines = 0

    with out_path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_json_lines += 1
                continue
            if not isinstance(obj, dict):
                continue

            total_rows += 1
            dl = obj.get("data_lines")
            if _is_int_like(dl):
                total_lines += int(dl)

            patterns = obj.get("data_patterns")
            if not isinstance(patterns, list):
                continue
            for p in patterns:
                if not isinstance(p, dict):
                    continue
                style = p.get("style")
                if not isinstance(style, list):
                    continue
                for it in style:
                    if not isinstance(it, dict):
                        continue
                    total_style_items += 1
                    metric = "<缺失指标名称>"
                    core = it.get("核心字段")
                    if isinstance(core, dict) and core.get("指标名称") is not None:
                        metric = _norm_str(core.get("指标名称"))
                    entity_type = _norm_str(it.get("实体类型", "未定义"))

                    metric_total[metric] = metric_total.get(metric, 0) + 1
                    metric_to_entity.setdefault(metric, {})
                    metric_to_entity[metric][entity_type] = metric_to_entity[metric].get(entity_type, 0) + 1

    # 输出：先按 count 降序
    sorted_metrics = sorted(metric_total.items(), key=lambda kv: (-kv[1], kv[0]))

    print(f"输出文件：{out_path}")
    print(f"样本行数（JSONL 行数）：{total_rows}")
    print(f"data_lines 总和：{total_lines}")
    print(f"style item 总数：{total_style_items}")
    if bad_json_lines:
        print(f"警告：存在无法解析的脏行：{bad_json_lines} 行（已跳过）")
    print(f"指标名称总数：{len(metric_total)}")
    print("")

    print(f"Top {min(top_k, len(sorted_metrics))} 指标名称（按出现次数降序；展示该指标的实体类型分布）：")
    for metric, total in sorted_metrics[:top_k]:
        et_map = metric_to_entity.get(metric, {})
        et_sorted = sorted(et_map.items(), key=lambda kv: (-kv[1], kv[0]))
        et_str = ", ".join([f"{et}={cnt}" for et, cnt in et_sorted])
        print(f"- {metric}  ->  total={total}, 实体类型数={len(et_map)}; {et_str}")


def extract_samples_by_metric_only(
    out_path: Path,
    *,
    metric_name: str,
    max_print: int | None = None,
) -> None:
    """
    从输出 JSONL 中提取并打印样本：
    - 该样本（同一 row_idx）内，所有出现过的“指标名称”去重集合必须严格等于 {metric_name}
      （即：不允许混入任何其他指标名称；缺失指标名称的 style item 不计入集合）
    - 且样本内至少出现过一次 metric_name

    打印内容：仅打印命中 metric_name 的行（line_idx + data）。
    """
    if not out_path.exists():
        raise FileNotFoundError(f"找不到输出 JSONL：{out_path}")

    target = str(metric_name).strip()
    if not target:
        raise ValueError("metric_name 不能为空")

    def _norm_str(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    printed = 0
    total_rows = 0
    bad_json_lines = 0

    with out_path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_json_lines += 1
                continue
            if not isinstance(obj, dict):
                continue

            total_rows += 1
            row_idx = obj.get("row_idx")
            patterns = obj.get("data_patterns")
            if not isinstance(patterns, list):
                continue

            metric_set: set[str] = set()
            hit_lines: list[tuple[int, str]] = []
            has_target = False
            has_other = False

            for p in patterns:
                if not isinstance(p, dict):
                    continue
                li = p.get("line_idx")
                data = p.get("data", "")
                data_s = str(data) if data is not None else ""

                style = p.get("style")
                if not isinstance(style, list):
                    continue

                # 每一行可能包含多个 style item；逐个提取核心字段.指标名称
                line_has_target = False
                for it in style:
                    if not isinstance(it, dict):
                        continue
                    core = it.get("核心字段")
                    if not isinstance(core, dict):
                        continue
                    mn = _norm_str(core.get("指标名称"))
                    if not mn:
                        continue
                    metric_set.add(mn)
                    if mn == target:
                        has_target = True
                        line_has_target = True
                    else:
                        has_other = True

                if line_has_target and _is_int_like(li):
                    hit_lines.append((int(li), data_s))

                # 早停：一旦确认混入其他指标，则该样本必定不符合
                if has_other:
                    break

            if not has_target or has_other:
                continue
            if metric_set != {target}:
                # 理论上 has_other=False 时 metric_set 应该只能是 {target} 或空；这里保守兜底
                continue
            if not hit_lines:
                continue

            # 打印该样本
            print(f"\n[row_idx={row_idx}] 指标名称={target} 命中行数={len(hit_lines)}")
            for li, txt in sorted(hit_lines, key=lambda x: x[0]):
                print(f"- line_idx={li}: {txt}")

            printed += 1
            if max_print is not None and printed >= max_print:
                break

    if bad_json_lines:
        print(f"\n[提示] 存在无法解析的脏行：{bad_json_lines} 行（已跳过）")
    print(f"\n完成：共扫描 {total_rows} 个样本，匹配并打印 {printed} 个样本。")


def extract_lines_by_metric_only(
    out_path: Path,
    *,
    metric_name: str,
    max_print: int | None = None,
) -> None:
    """
    从输出 JSONL 中提取并打印“行级别”的命中结果：
    - 对每个 row_idx 的每个 line_idx，统计该行 style 内出现过的“指标名称”去重集合 metric_set
      （缺失指标名称的 style item 不计入集合）
    - 当且仅当 metric_set == {metric_name} 时，认为该行命中

    打印内容：row_idx + line_idx + data。
    """
    if not out_path.exists():
        raise FileNotFoundError(f"找不到输出 JSONL：{out_path}")

    target = str(metric_name).strip()
    if not target:
        raise ValueError("metric_name 不能为空")

    def _norm_str(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    printed = 0
    total_rows = 0
    total_lines = 0
    bad_json_lines = 0

    with out_path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_json_lines += 1
                continue
            if not isinstance(obj, dict):
                continue

            total_rows += 1
            row_idx = obj.get("row_idx")
            patterns = obj.get("data_patterns")
            if not isinstance(patterns, list):
                continue

            for p in patterns:
                if not isinstance(p, dict):
                    continue
                li = p.get("line_idx")
                if not _is_int_like(li):
                    continue
                li_int = int(li)
                data = p.get("data", "")
                data_s = str(data) if data is not None else ""

                style = p.get("style")
                if not isinstance(style, list):
                    continue

                metric_set: set[str] = set()
                entity_types: set[str] = set()
                for it in style:
                    if not isinstance(it, dict):
                        continue
                    et = _norm_str(it.get("实体类型"))
                    if et:
                        entity_types.add(et)
                    core = it.get("核心字段")
                    if not isinstance(core, dict):
                        continue
                    mn = _norm_str(core.get("指标名称"))
                    if not mn:
                        continue
                    metric_set.add(mn)

                total_lines += 1
                if metric_set == {target}:
                    et_str = "|".join(sorted(entity_types)) if entity_types else "<缺失实体类型>"
                    print(f"[row_idx={row_idx}] line_idx={li_int} 【{et_str}】 【个人数据】{data_s}")
                    printed += 1
                    if max_print is not None and printed >= max_print:
                        break

            if max_print is not None and printed >= max_print:
                break

    if bad_json_lines:
        print(f"\n[提示] 存在无法解析的脏行：{bad_json_lines} 行（已跳过）")
    print(f"\n完成：共扫描 {total_rows} 个样本，扫描行数 {total_lines}，匹配并打印 {printed} 行。")


def extract_style_items_by_entity_type(
    out_path: Path,
    *,
    entity_type: str,
    max_print: int | None = None,
) -> None:
    """
    从输出 JSONL 中提取并打印“style item 级别”的命中结果：
    - 遍历每个 row_idx 的每个 line_idx 的每个 style item
    - 当且仅当 style item 的 "实体类型" == entity_type（strip 后严格相等）时，认为命中

    打印内容：row_idx + line_idx + data + style_item（单条 style item）。
    """
    if not out_path.exists():
        raise FileNotFoundError(f"找不到输出 JSONL：{out_path}")

    target = str(entity_type).strip()
    if not target:
        raise ValueError("entity_type 不能为空")

    def _norm_str(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    printed = 0
    total_rows = 0
    total_lines = 0
    total_style_items = 0
    bad_json_lines = 0

    with out_path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_json_lines += 1
                continue
            if not isinstance(obj, dict):
                continue

            total_rows += 1
            row_idx = obj.get("row_idx")
            patterns = obj.get("data_patterns")
            if not isinstance(patterns, list):
                continue

            for p in patterns:
                if not isinstance(p, dict):
                    continue
                li = p.get("line_idx")
                if not _is_int_like(li):
                    continue
                li_int = int(li)

                data = p.get("data", "")
                data_s = str(data) if data is not None else ""

                style = p.get("style")
                if not isinstance(style, list):
                    continue

                total_lines += 1
                for it in style:
                    if not isinstance(it, dict):
                        continue
                    total_style_items += 1
                    et = _norm_str(it.get("实体类型"))
                    if et != target:
                        continue

                    # 尝试补充指标名称（如果存在）
                    metric = "<缺失指标名称>"
                    core = it.get("核心字段")
                    if isinstance(core, dict) and core.get("指标名称") is not None:
                        mn = _norm_str(core.get("指标名称"))
                        if mn:
                            metric = mn

                    style_item_s = json.dumps(it, ensure_ascii=False)
                    print(f"[row_idx={row_idx}] line_idx={li_int} 实体类型={target} 指标名称={metric} 【个人数据】\n{data_s}")
                    print(f"  [style_item] {style_item_s}")

                    printed += 1
                    if max_print is not None and printed >= max_print:
                        break

                if max_print is not None and printed >= max_print:
                    break

            if max_print is not None and printed >= max_print:
                break

    if bad_json_lines:
        print(f"\n[提示] 存在无法解析的脏行：{bad_json_lines} 行（已跳过）")
    print(
        f"\n完成：共扫描 {total_rows} 个样本，扫描行数 {total_lines}，style item 总数 {total_style_items}，"
        f"匹配并打印 {printed} 条（实体类型={target}）。"
    )


if __name__ == "__main__":
    main()
"""
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --analyze-patterns
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "单日期数值多项总结"
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-metric-only "血氧饱和度"

python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "无时间日期的数值总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "无时间日期的文本总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "单日期文本总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "单日期数值单项总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "单日期数值多项总结" | grep -v "个人数据" | grep -v "style_item"  | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "周期文本总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "周期数值单项总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "周期数值多项总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "周期数值对比记录" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "单指标的明细记录" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "单指标的统计复合记录" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data summary_train_v3.csv --extract-entity-type "未定义" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'


python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "无时间日期的数值总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "无时间日期的文本总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "单日期文本总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "单日期数值单项总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "单日期数值多项总结" | grep -v "个人数据" | grep -v "style_item"  | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "周期文本总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "周期数值单项总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "周期数值多项总结" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "周期数值对比记录" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "单指标的明细记录" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "单指标的统计复合记录" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'
python scripts/predict_personal_data.py --raw-data data_diff_sample.csv --extract-entity-type "未定义" | grep -v "个人数据" | grep -v "style_item" | grep -v "共扫描" | perl -lane 'print "\"$_\"," if $_;'


"""