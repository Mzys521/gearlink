"""GearLink：轻量级 Agent 框架。

只有从此处显式导出的名称才是公共 API（开发规范 §5.1，架构设计 §2）：

- Agent：`Agent`（编排策略抽象）/ `ReactAgent`（ReAct）/ `PlanExecuteAgent`（规划-执行）/
  `Orchestrator`（多 Agent 协作主管-工人编排）/
  `DependentOrchestrator`（依赖编排，工人间流水线协作）/
  `AutonomousOrchestrator`（模型自主 DAG 编排，串并行混合）；
  `run_events` 事件流含 `AgentEvent` 体系与 `HookFn` 回调；
- 记忆：`Memory` / `MemoryEntry` / `Session` / `ShortTermMemory` /
  `LongTermMemory` / `MemoryManager` / `VectorStore` / `ChromaVectorStore`；
- 工具：`TOOL_REGISTRY` / `TOOL_SCHEMAS` / `register_tool` / `call_tool` / `set_skill_registry`；
- 模型：`ModelProvider` / `ModelResponse` / `StreamChunk` / `ToolCall` /
  `OpenAIProvider` / `OllamaProvider` / `AnthropicProvider`；
- 技能：`Skill` / `SkillRegistry` / `SkillLoader`；
- MCP：`McpClient`（外部 MCP 服务器工具接入，另见 `gearlink.mcp`）；
- 异常：`GearLinkError` 体系（开发规范 §7）；
- 日志：`enable_logging` / `disable_logging`（全局日志开关）；
- 可观测性：`TokenUsage` / `UsageTracker`（用量统计）/ `JsonlEventSink` / `jsonl_hook` /
  `load_jsonl_events`（事件落盘与回放）；
- 工具函数：`estimate_tokens` / `count_message_tokens`。
"""

from gearlink.core.agent import Agent, PlanExecuteAgent, ReactAgent
from gearlink.core.events import (
    AgentEvent,
    AgentHandoffEvent,
    FinalAnswerEvent,
    HookFn,
    JsonlEventSink,
    LoopAbortEvent,
    ModelMessageEvent,
    PlanGeneratedEvent,
    PlanStepEndEvent,
    PlanStepStartEvent,
    StepStartEvent,
    SubtaskEndEvent,
    TeamPlanGeneratedEvent,
    TextDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    jsonl_hook,
    load_jsonl_events,
)
from gearlink.core.orchestrator import AutonomousOrchestrator, DependentOrchestrator, Orchestrator
from gearlink.core.memory import (
    ChromaVectorStore,
    ContextBuilder,
    EmbeddingFn,
    LongTermMemory,
    Memory,
    MemoryEntry,
    MemoryManager,
    ProfileHookFn,
    Session,
    ShortTermMemory,
    VectorStore,
)
from gearlink.core.tool import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    ToolRegistry,
    build_tool_schema,
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
from gearlink.mcp import McpClient
from gearlink.providers.anthropic_provider import AnthropicProvider
from gearlink.providers.base import (
    ModelProvider,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from gearlink.providers.ollama_provider import OllamaProvider
from gearlink.providers.openai_provider import OpenAIProvider
from gearlink.skills import Skill, SkillLoader, SkillRegistry
from gearlink.utils.logging import disable_logging, enable_logging
from gearlink.utils.token_count import count_message_tokens, estimate_tokens
from gearlink.utils.usage import UsageRecord, UsageTracker

# 导入工具包以触发内置工具注册（显式导入，非运行时扫描）
import gearlink.tools.builtin  # noqa: F401

__all__ = [
    "Agent",
    "ReactAgent",
    "PlanExecuteAgent",
    "AgentEvent",
    "StepStartEvent",
    "TextDeltaEvent",
    "ModelMessageEvent",
    "ToolCallStartEvent",
    "ToolCallEndEvent",
    "FinalAnswerEvent",
    "LoopAbortEvent",
    "PlanGeneratedEvent",
    "PlanStepStartEvent",
    "PlanStepEndEvent",
    "Orchestrator",
    "DependentOrchestrator",
    "AutonomousOrchestrator",
    "TeamPlanGeneratedEvent",
    "AgentHandoffEvent",
    "SubtaskEndEvent",
    "HookFn",
    "JsonlEventSink",
    "jsonl_hook",
    "load_jsonl_events",
    "Memory",
    "MemoryEntry",
    "MemoryManager",
    "Session",
    "ShortTermMemory",
    "LongTermMemory",
    "EmbeddingFn",
    "ProfileHookFn",
    "VectorStore",
    "ChromaVectorStore",
    "ContextBuilder",
    "ToolRegistry",
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "call_tool",
    "register_tool",
    "build_tool_schema",
    "set_skill_registry",
    "ModelProvider",
    "ModelResponse",
    "StreamChunk",
    "TokenUsage",
    "ToolCall",
    "OpenAIProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "McpClient",
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
    "enable_logging",
    "disable_logging",
    "estimate_tokens",
    "count_message_tokens",
    "UsageTracker",
    "UsageRecord",
]
