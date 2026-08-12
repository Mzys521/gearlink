# GearLink

轻量级 Agent 框架：多策略 Agent 编排（ReAct / 规划-执行 / 多 Agent 协作）+ 三个可插拔维度（模型、记忆、工具）+ 技能扩展，不引入重型框架。

## 特性

- **多策略 Agent**：`Agent` 抽象统一 `run` / `run_stream` / `run_events` 契约；`ReactAgent` 编排 Reason → Act → Observe，工具失败作为可恢复信号写回模型处理；`PlanExecuteAgent` 先规划后执行（规划 → 步骤子循环 → 整合），任务分解更可控；`Orchestrator` 主管-工人多 Agent 协作（主管分派子任务，各工人独立执行后汇总）；
- **事件流 + 回调**：`run_events` 逐步产出 `AgentEvent` 体系事件，`hooks` / `add_hook` 注入回调实现 on_step 观察与干预；
- **可观测性**：`TokenUsage` 用量透传（`ModelMessageEvent.usage`）、`JsonlEventSink` 事件落盘与离线回放、`UsageTracker` 按标签聚合用量与成本估算；
- **模型可插拔**：面向 `ModelProvider` 抽象编程，内置 OpenAI 兼容实现（OpenAI / DeepSeek / 国产兼容接口）、本地 `OllamaProvider`（无需密钥）与 `AnthropicProvider`；支持可重试错误的指数退避重试（`max_retries`）与结构化输出（`response_format`）；
- **记忆可插拔**：短期滑窗 `ShortTermMemory`、向量长期记忆 `LongTermMemory`（可注入自定义 `embedding_function`，存储后端经 `VectorStore` 协议抽象，支持检索阈值过滤与 MMR 去冗余）、以及组合两者的 `MemoryManager`（上下文预算、会话摘要沉淀与动态压缩、用户画像钩子、检索注入）；`snapshot()` / `restore()` 支持会话断线恢复；
- **工具注册表**：「注册表 + JSON Schema + 调度器」三件套，`register_tool` 显式登记，`build_tool_schema` 从函数签名自动生成 schema；支持工具白名单与并行执行；
- **MCP 接入**：`McpClient` 把外部 MCP 服务器的工具映射进注册表（`mcp_<server>_<tool>`），`core/` 无感知复用；
- **技能扩展**：渐进式披露的 `SKILL.md` 知识包，`ReactAgent(skill_registry=...)` 注入，模型经 `load_skill` 工具按需加载完整指令。

## 安装

```bash
pip install -e .          # 开发安装
pip install -r requirements.txt
```

在项目根目录创建 `.env`（参考 `.env.example`）：

```
DEEPSEEK_API_KEY=你的密钥
```

## 快速开始

```python
from dotenv import load_dotenv

from gearlink import OpenAIProvider, ReactAgent

load_dotenv()

agent = ReactAgent(provider=OpenAIProvider())
print(agent.run("现在几点了？"))  # 自动调用 get_current_time 工具
```

### 流式输出

`run_stream` 逐片段产出模型文本，工具调用阶段自动执行并继续循环；
拼接全部片段即为完整答案：

```python
from gearlink import OpenAIProvider, ReactAgent

agent = ReactAgent(provider=OpenAIProvider())
for delta in agent.run_stream("用一段话介绍 GearLink"):
    print(delta, end="", flush=True)
```

若自定义 Provider 未覆写 `chat_stream`，将自动回退到非流式 `chat()`（一次性产出全文）。

### 注册自定义工具

工具实现与其 JSON Schema 同源定义，通过 `register_tool` 显式登记：

```python
from gearlink import ReactAgent, OpenAIProvider, register_tool


def add(a: float, b: float) -> float:
    """两数相加"""
    return a + b


register_tool(
    "add",
    add,
    {
        "description": "计算两个数的和",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "加数 a"},
                "b": {"type": "number", "description": "加数 b"},
            },
            "required": ["a", "b"],
        },
    },
)

agent = ReactAgent(provider=OpenAIProvider())
print(agent.run("帮我算一下 12.3 加 45.6"))
```

### 长期记忆与会话摘要

```python
import chromadb

from gearlink import LongTermMemory, MemoryManager, OpenAIProvider, ReactAgent, ShortTermMemory

memory = MemoryManager(
    short_term=ShortTermMemory(max_message=20),
    long_term=LongTermMemory(
        vector_db=chromadb.PersistentClient(path=".chroma"),
        collection_name="chat_history",
    ),
    max_context_tokens=4000,
    summarizer=my_summarizer,  # Callable[[str], str]，可选；会话结束时沉淀摘要
)
agent = ReactAgent(provider=OpenAIProvider(), memory=memory, retrieve_every_iteration=True)
print(agent.run("我之前说过喜欢什么？"))  # 自动检索长期记忆注入上下文
memory.end_session()  # 会话结束：沉淀摘要并补沉淀剩余上下文
```

### 技能注入

每个技能是一个含 `SKILL.md`（YAML frontmatter 提供 name / description）的目录；
模型按需经 `load_skill` 工具加载完整指令（渐进式披露）：

```python
from pathlib import Path

import gearlink.tools.load_skill  # 显式导入，触发 load_skill 工具注册
from gearlink import OpenAIProvider, ReactAgent, SkillLoader, SkillRegistry

registry = SkillRegistry()
for skill in SkillLoader.discover_from_directory(Path("examples/skill_demo")):
    registry.register(skill)

# 注入注册表后，默认系统提示会自动列出可用技能
agent = ReactAgent(provider=OpenAIProvider(), skill_registry=registry)
print(agent.run("帮我做一次代码审查"))
```

### 规划-执行 Agent

任务较复杂时改用 `PlanExecuteAgent`：先规划（分解步骤），再逐步执行（复用
`ReactAgent` 子循环，支持工具），最后整合为最终答案：

```python
from gearlink import OpenAIProvider, PlanExecuteAgent

agent = PlanExecuteAgent(provider=OpenAIProvider(), max_steps=5)
print(agent.run("对比 Python 与 Go 的适用场景并给出选型建议"))
```

规划器输出无法解析时自动退化为单步骤直接执行；规划步骤数超过 `max_steps`
上限时自动截断。事件流新增 `PlanGeneratedEvent` / `PlanStepStartEvent` /
`PlanStepEndEvent`，可经 `hooks` 回调观察规划与逐步执行过程。

### 多 Agent 协作（Orchestrator）

多角色分工用 `Orchestrator`：主管 Agent 把任务拆分派给登记的工人（各配独立
工具/记忆），各工人独立完成后汇总为最终答案（`parallel=True` 可并行执行）：

```python
from gearlink import Orchestrator, OpenAIProvider, ReactAgent

orchestrator = Orchestrator(
    supervisor=ReactAgent(provider=OpenAIProvider()),
    workers={
        "researcher": ReactAgent(provider=OpenAIProvider(), tools=["search"]),
        "writer": ReactAgent(provider=OpenAIProvider(), tools=[]),
    },
)
print(orchestrator.run("调研并撰写一份产品介绍"))
```

协作过程经事件流暴露：`TeamPlanGeneratedEvent`（分派清单）→ `AgentHandoffEvent`
（派单）→ 工人事件 → `SubtaskEndEvent`；与 `PlanExecuteAgent`（单模型串行步骤）
互补。分派输出无法解析时退化为把原任务派给全部工人。

### 更多模型提供者

Provider 可插拔：换提供者不改变其余任何用法。

```python
from gearlink import AnthropicProvider, OllamaProvider, ReactAgent

# 本地 Ollama：无需密钥，需先 ollama pull qwen2.5:7b 并 ollama serve
agent = ReactAgent(provider=OllamaProvider())

# Anthropic Claude：pip install gearlink[anthropic]，配 ANTHROPIC_API_KEY
agent = ReactAgent(provider=AnthropicProvider())

# 国产模型（Qwen / Kimi / GLM 等）多兼容 OpenAI 接口，直接复用 OpenAIProvider：
# OpenAIProvider(api_key=..., base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
```

网络抖动等瞬时故障可交给框架自动重试（指数退避，鉴权错误不重试）：

```python
agent = ReactAgent(provider=OllamaProvider(), max_retries=3)
```

### 会话持久化与恢复

`snapshot()` 导出会话快照（可 JSON 落盘），重启后 `restore()` 无缝继续：

```python
snapshot = memory.snapshot(model="deepseek-chat", metadata={"user_id": "u-1"})
# ... 落盘 / 进程重启 ...
session = memory.restore(snapshot)  # 短期记忆恢复，继续对话
```

### MCP 外部工具

`McpClient` 把外部 MCP 服务器的工具映射进本地注册表（需 `pip install gearlink[mcp]`）：

```python
from gearlink import McpClient

client = McpClient("external", session)  # session: 已 initialize 的 mcp.ClientSession
client.register_tools()  # 登记为 mcp_external_*，Agent 即可直接调用
```

### 可观测性

模型调用的 token 用量经 `ModelMessageEvent.usage`（`TokenUsage`）透传；事件流可经
`JsonlEventSink` 落盘、离线回放，`UsageTracker` 按标签聚合用量并估算成本：

```python
from gearlink import JsonlEventSink, UsageTracker, jsonl_hook, load_jsonl_events

with JsonlEventSink("events.jsonl") as sink:
    agent.add_hook(jsonl_hook(sink))  # 每个事件逐条落盘
    agent.run("你好")

for event in load_jsonl_events("events.jsonl"):  # 离线回放
    print(event["seq"], event["type"])
```

### 记忆深化（压缩 / 画像 / 检索质量）

长对话与检索质量的进阶能力（均默认关闭，向后兼容）：

```python
from gearlink import LongTermMemory, MemoryManager, ShortTermMemory

memory = MemoryManager(
    short_term=ShortTermMemory(max_message=50),
    max_context_tokens=8000,
    summarizer=my_summarizer,
    compress_context=True,  # 超阈值时把最旧一段压缩为 [上下文摘要]
    profile_hook=my_profile_hook,  # end_session 时沉淀用户画像，优先注入
)

long_term = LongTermMemory(
    vector_db=client,
    collection_name="chat_history",
    relevance_threshold=1.2,  # 过滤低相关检索结果（distance 阈值）
    mmr_lambda=0.7,  # MMR 去冗余，避免 top-k 全是重复语义
)
```

存储后端经 `VectorStore` 协议抽象：默认 `ChromaVectorStore`（行为不变），
也可 `LongTermMemory(store=自定义后端)` 注入内存/文件/远程向量库等实现。

### 日志开关

框架默认静默；调用 `enable_logging` 一键开启内部日志（ReAct 工具调用、长期记忆
沉淀/检索、技能发现等），`disable_logging` 一键关闭：

```python
import logging

from gearlink import disable_logging, enable_logging

enable_logging()  # 开启：输出到 stderr，级别 INFO
enable_logging(logging.DEBUG)  # 或指定更细级别
disable_logging()  # 关闭：恢复静默
```

两个函数均幂等可重复调用；开启后框架日志统一经开关输出，不再向应用根日志器传播
（避免与应用自行配置的 `basicConfig` 重复输出）。

## 公共 API

所有公共名称均从顶层包 `gearlink` 显式导出（见 `gearlink/__init__.py` 的 `__all__`）：

| 分类 | 导出名称 |
|---|---|
| Agent | `Agent`、`ReactAgent`、`PlanExecuteAgent`、`Orchestrator` |
| 事件 | `AgentEvent`、`StepStartEvent`、`TextDeltaEvent`、`ModelMessageEvent`、`ToolCallStartEvent`、`ToolCallEndEvent`、`FinalAnswerEvent`、`LoopAbortEvent`、`PlanGeneratedEvent`、`PlanStepStartEvent`、`PlanStepEndEvent`、`TeamPlanGeneratedEvent`、`AgentHandoffEvent`、`SubtaskEndEvent`、`HookFn` |
| 记忆 | `Memory`、`MemoryEntry`、`Session`、`ShortTermMemory`、`LongTermMemory`、`MemoryManager`、`EmbeddingFn`、`ProfileHookFn`、`VectorStore`、`ChromaVectorStore` |
| 工具 | `TOOL_REGISTRY`、`TOOL_SCHEMAS`、`register_tool`、`call_tool`、`build_tool_schema`、`set_skill_registry` |
| 模型 | `ModelProvider`、`ModelResponse`、`StreamChunk`、`ToolCall`、`OpenAIProvider`、`OllamaProvider`、`AnthropicProvider` |
| MCP | `McpClient` |
| 技能 | `Skill`、`SkillRegistry`、`SkillLoader` |
| 可观测性 | `TokenUsage`、`UsageTracker`、`UsageRecord`、`JsonlEventSink`、`jsonl_hook`、`load_jsonl_events` |
| 异常 | `GearLinkError`、`ProviderError`、`ToolError`、`ToolNotFoundError`、`MemoryError`、`SkillError` 及其子类 |
| 日志 | `enable_logging`、`disable_logging` |
| 工具函数 | `estimate_tokens`、`count_message_tokens` |

## 示例

标注「免密钥」的示例内置了无网络 Provider，无需配置 API key 即可直接运行，适合初次体验：

- [examples/custom_provider_demo.py](examples/custom_provider_demo.py)：自定义 `ModelProvider`（免密钥）；
- [examples/custom_memory_demo.py](examples/custom_memory_demo.py)：自定义 `Memory` 实现并注入 Agent（免密钥）；
- [examples/event_hooks_demo.py](examples/event_hooks_demo.py)：`run_events` 事件流消费 + `add_hook` 观察/替换回调（免密钥）；
- [examples/custom_tool_demo.py](examples/custom_tool_demo.py)：`register_tool` 注册自定义工具 + `call_tool` 调度与错误兜底（免密钥；配 key 后联动 Agent）；
- [examples/session_restore_demo.py](examples/session_restore_demo.py)：会话 `snapshot` / `restore` 断线恢复（免密钥）；
- [examples/observability_demo.py](examples/observability_demo.py)：token 用量透传 + JSONL 事件落盘回放 + `UsageTracker` 成本估算（免密钥）；
- [examples/memory_advanced_demo.py](examples/memory_advanced_demo.py)：上下文摘要压缩 + 用户画像 + 检索阈值/MMR + 自定义 `VectorStore`（免密钥，无需 chromadb）；
- [examples/orchestrator_demo.py](examples/orchestrator_demo.py)：`Orchestrator` 主管-工人多 Agent 协作（免密钥）；
- [examples/ollama_local_demo.py](examples/ollama_local_demo.py)：`OllamaProvider` 本地模型（无需 API key，需本地 Ollama 服务）；
- [examples/anthropic_demo.py](examples/anthropic_demo.py)：`AnthropicProvider` 接入 Claude（需 `gearlink[anthropic]` 与密钥）；
- [examples/mcp_client_demo.py](examples/mcp_client_demo.py)：`McpClient` 消费外部 MCP 服务器工具（需 `gearlink[mcp]`）；
- [examples/memory_chatbot.py](examples/memory_chatbot.py)：短期 + 长期记忆 + 会话摘要的对话助手；
- [examples/streaming_demo.py](examples/streaming_demo.py)：`run_stream` 流式输出（含工具调用阶段）；
- [examples/plan_execute_demo.py](examples/plan_execute_demo.py)：`PlanExecuteAgent` 规划-执行（含事件回调观察）；
- [examples/skill_demo/](examples/skill_demo/)：技能目录结构示例（每个技能一个含 `SKILL.md` 的子目录）。

## 文档

- [使用教程](docs/使用教程.md)：从安装到进阶的完整新手教程
- [架构设计](docs/架构设计.md)：分层架构、核心组件、运行时数据流与四类扩展点契约
- [接口设计规范](docs/接口设计规范.md)：公共 API 设计原则
- [开发规范](docs/开发规范.md)：开发流程与代码规范

## 开发

```bash
ruff format .
ruff check .
pytest
```

提交前须全部通过。贡献请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。
