from __future__ import annotations

"""
简单的 .env 加载工具。

README 中推荐使用 python-dotenv 管理各类模型与代理配置，
这里提供一个轻量封装，在脚本入口处调用一次即可。
"""

from typing import Final

try:  # 可选依赖：未安装时保持静默，不影响本地纯启发式调试
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


_LOADED: Final[bool] = False


def load_env() -> None:
    """
    从当前工作目录自动加载 .env 文件（如果存在）。

    - 若未安装 python-dotenv，则什么都不做，避免强制依赖。
    - 多次调用也不会重复加载。
    """
    global _LOADED  # type: ignore[global-variable-not-assigned]
    if _LOADED or load_dotenv is None:
        return
    load_dotenv()
    _LOADED = True



