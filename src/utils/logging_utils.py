from __future__ import annotations

import logging
from typing import Optional


def init_logger(level: int = logging.INFO) -> logging.Logger:
    """
    初始化一个简单的全局 logger。
    - 统一格式：时间 | 等级 | 模块 | 信息
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # 屏蔽第三方 httpx 默认的 INFO 级别请求日志，只保留 WARNING 及以上
    # 例如：
    # 2025-12-04 20:39:59,693 | INFO | httpx | HTTP Request: POST ...
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    return logging.getLogger("summarydata")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "summarydata")


