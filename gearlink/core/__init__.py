"""核心抽象：Agent、Memory、Tool 引擎。

core 只面向抽象编程，不得依赖 providers/skills/tools 中的具体实现。
"""

from gearlink.core.agent import MAX_ITERATIONS, MAX_TOOL_RESULT_TOKENS, SYSTEM_PROMPT, ReactAgent
from gearlink.core.events import AgentHandoffEvent, SubtaskEndEvent, TeamPlanGeneratedEvent
from gearlink.core.memory import (
    ChromaVectorStore,
    ContextBuilder,
    LongTermMemory,
    Memory,
    MemoryEntry,
    MemoryManager,
    ProfileHookFn,
    Session,
    ShortTermMemory,
    VectorStore,
)
from gearlink.core.orchestrator import DependentOrchestrator, Orchestrator
from gearlink.core.tool import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    ToolRegistry,
    call_tool,
    register_tool,
    set_skill_registry,
)

__all__ = [
    "ReactAgent",
    "SYSTEM_PROMPT",
    "MAX_ITERATIONS",
    "MAX_TOOL_RESULT_TOKENS",
    "Memory",
    "MemoryEntry",
    "MemoryManager",
    "ProfileHookFn",
    "Session",
    "ShortTermMemory",
    "LongTermMemory",
    "VectorStore",
    "ChromaVectorStore",
    "ContextBuilder",
    "Orchestrator",
    "DependentOrchestrator",
    "AgentHandoffEvent",
    "SubtaskEndEvent",
    "TeamPlanGeneratedEvent",
    "ToolRegistry",
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "call_tool",
    "register_tool",
    "set_skill_registry",
]
