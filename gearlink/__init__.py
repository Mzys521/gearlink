"""GearLink：轻量级 Agent 框架。

只有从此处显式导出的名称才是公共 API（接口设计规范 §1.5，架构设计 §2）：

- Agent：`ReactAgent`；
- 记忆：`Memory` / `MemoryEntry` / `ShortTermMemory` / `LongTermMemory` / `MemoryManager`；
- 工具：`TOOL_REGISTRY` / `TOOL_SCHEMAS` / `register_tool` / `call_tool` / `set_skill_registry`；
- 模型：`ModelProvider` / `ModelResponse` / `StreamChunk` / `ToolCall` / `OpenAIProvider`；
- 技能：`Skill` / `SkillRegistry` / `SkillLoader`；
- 异常：`GearLinkError` 体系（接口设计规范 §5）；
- 工具函数：`estimate_tokens` / `count_message_tokens`。
"""

from gearlink.core.agent import ReactAgent
from gearlink.core.memory import (
    LongTermMemory,
    Memory,
    MemoryEntry,
    MemoryManager,
    ShortTermMemory,
)
from gearlink.core.tool import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    call_tool,
    register_tool,
    set_skill_registry,
)
from gearlink.exceptions import (
    GearLinkError,
    MemoryError,
    ProviderError,
    SkillError,
    SkillExecutionError,
    SkillLoadError,
    SkillNotFoundError,
    SkillValidationError,
    ToolError,
    ToolNotFoundError,
)
from gearlink.providers.base import ModelProvider, ModelResponse, StreamChunk, ToolCall
from gearlink.providers.openai_provider import OpenAIProvider
from gearlink.skills import Skill, SkillLoader, SkillRegistry
from gearlink.utils.token_count import count_message_tokens, estimate_tokens

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
    "set_skill_registry",
    "ModelProvider",
    "ModelResponse",
    "StreamChunk",
    "ToolCall",
    "OpenAIProvider",
    "Skill",
    "SkillRegistry",
    "SkillLoader",
    "GearLinkError",
    "ProviderError",
    "ToolError",
    "ToolNotFoundError",
    "MemoryError",
    "SkillError",
    "SkillNotFoundError",
    "SkillLoadError",
    "SkillValidationError",
    "SkillExecutionError",
    "estimate_tokens",
    "count_message_tokens",
]
