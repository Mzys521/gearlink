"""GearLink 全局日志开关。

框架内部统一使用标准库 `logging`，模块级日志器命名以 `gearlink` 为根
（如 `gearlink.core.agent` / `gearlink.core.memory`）。本模块提供一键开关：

- `enable_logging()`：为 `gearlink` 命名空间配置 stderr 输出并设置级别，
  之后框架内部日志（ReAct 工具调用、长期记忆沉淀/检索、技能发现等）实时可见；
- `disable_logging()`：移除输出 handler 并把命名空间级别调至高于 CRITICAL，
  恢复静默。

默认（不调用任何开关）框架不输出日志，遵循库的静默原则；应用可按需开关。
"""

import logging
import sys

__all__ = ["enable_logging", "disable_logging"]

#: GearLink 日志命名空间根：所有内部模块日志器的共同祖先
_ROOT_LOGGER_NAME = "gearlink"

#: 默认输出级别
DEFAULT_LEVEL = logging.INFO

#: 输出格式：时间 + 级别 + 模块名 + 消息
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

#: 本开关创建的输出 handler（模块私有，仅移除自身添加的 handler）
_handler: logging.Handler | None = None


def _root_logger() -> logging.Logger:
    """返回 GearLink 日志命名空间根日志器。"""
    return logging.getLogger(_ROOT_LOGGER_NAME)


def enable_logging(level: int = DEFAULT_LEVEL) -> None:
    """一键开启 GearLink 日志输出（输出到 stderr，级别可配置）。

    重复调用幂等：不会重复添加 handler。开启后 `gearlink` 命名空间的日志
    统一经本开关输出，不再向应用根日志器传播（避免与应用自行配置的 root
    handler 重复输出）。

    Args:
        level: 日志级别（如 logging.DEBUG / logging.INFO / logging.WARNING），
            默认 INFO。
    """
    global _handler
    logger = _root_logger()
    logger.setLevel(level)
    logger.propagate = False
    if _handler is None:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(_handler)


def disable_logging() -> None:
    """一键关闭 GearLink 日志输出。

    移除本开关添加的 handler，并把命名空间级别调至高于 CRITICAL 以屏蔽全部
    内部日志。重复调用幂等；不影响应用为其他命名空间配置的日志。
    """
    global _handler
    logger = _root_logger()
    logger.setLevel(logging.CRITICAL + 1)
    logger.propagate = False
    if _handler is not None:
        logger.removeHandler(_handler)
        _handler = None
