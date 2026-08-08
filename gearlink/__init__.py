"""GearLink：轻量级 Agent 框架。

只有从此处显式导出的名称才是公共 API。
"""

from gearlink.core.agent import ReactAgent
from gearlink.core.memory import (
    LongTermMemory,
    Memory,
    MemoryEntry,
    MemoryManager,
    ShortTermMemory,
)
from gearlink.core.tool import TOOL_REGISTRY, TOOL_SCHEMAS, call_tool, register_tool
from gearlink.providers.base import ModelProvider, ModelResponse, ToolCall
from gearlink.providers.openai_provider import OpenAIProvider

# 导入工具包以触发内置工具注册（显式导入，非运行时扫描）
import gearlink.tools.builtin  # noqa: F401

__all__ = [
    "ReactAgent",
    "Memory",
    "MemoryEntry",
    "MemoryManager",
    "ShortTermMemory",
    "LongTermMemory",
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "call_tool",
    "register_tool",
    "ModelProvider",
    "ModelResponse",
    "ToolCall",
    "OpenAIProvider",
]
