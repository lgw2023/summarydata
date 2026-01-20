from __future__ import annotations

"""
data_clean：个人数据样式解析/归一化/聚合渲染。

说明：
- 该包是从历史单文件 `scripts/classify_personal_data.py` 拆分而来；
- 对外导出尽量保持兼容（`from src.data_clean import *` 与旧版 `from classify_personal_data import *`）。
"""

from .models import *  # noqa: F403
from .parse import *  # noqa: F403
from .aggregate_format import *  # noqa: F403
from .aggregate_time import *  # noqa: F403
from .aggregate_dataframe import *  # noqa: F403
from .aggregate_dataline import *  # noqa: F403

# 兼容旧版：沿用原脚本的导出列表（在拆分后由本包统一汇总）
from .models import __all__ as _MODELS_ALL
from .parse import __all__ as _PARSE_ALL
from .aggregate_format import __all__ as _AGG_FORMAT_ALL
from .aggregate_time import __all__ as _AGG_TIME_ALL
from .aggregate_dataframe import __all__ as _AGG_DF_ALL
from .aggregate_dataline import __all__ as _AGG_DATALINE_ALL

__all__ = list(
    dict.fromkeys(
        [
            *_MODELS_ALL,
            *_PARSE_ALL,
            *_AGG_FORMAT_ALL,
            *_AGG_TIME_ALL,
            *_AGG_DF_ALL,
            *_AGG_DATALINE_ALL,
        ]
    )
)

