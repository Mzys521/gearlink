# GearLink

轻量级 Agent 框架：多策略 Agent 编排（ReAct / 规划-执行）+ 三个可插拔维度（模型、记忆、工具）+ 技能扩展，不引入重型框架。

## 特性

- **多策略 Agent**：`Agent` 抽象统一 `run` / `run_stream` / `run_events` 契约；`ReactAgent` 编排 Reason → Act → Observe，工具失败作为可恢复信号写回模型处理；`PlanExecuteAgent` 先规划后执行（规划 → 步骤子循环 → 整合），任务分解更可控；
- **事件流 + 回调**：`run_events` 逐步产出 `AgentEvent` 体系事件，`hooks` / `add_hook` 注入回调实现 on_step 观察与干预；
- **模型可插拔**：面向 `ModelProvider` 抽象编程，内置 OpenAI 兼容实现（OpenAI / DeepSeek）；
- **记忆可插拔**：短期滑窗 `ShortTermMemory`、chromadb 向量长期记忆 `LongTermMemory`、以及组合两者的 `MemoryManager`（上下文预算、会话摘要沉淀、检索注入）；
- **工具注册表**：「注册表 + JSON Schema + 调度器」三件套，`register_tool` 显式登记；
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

### 日志开关

框架默认静默；调用 `enable_logging` 一键开启内部日志（ReAct 工具调用、长期记忆
沉淀/检索、技能发现等），`disable_logging` 一键关闭：

```python
import logging

from gearlink import disable_logging, enable_logging

enable_logging()               # 开启：输出到 stderr，级别 INFO
enable_logging(logging.DEBUG)  # 或指定更细级别
disable_logging()              # 关闭：恢复静默
```

两个函数均幂等可重复调用；开启后框架日志统一经开关输出，不再向应用根日志器传播
（避免与应用自行配置的 `basicConfig` 重复输出）。

## 公共 API

所有公共名称均从顶层包 `gearlink` 显式导出（见 `gearlink/__init__.py` 的 `__all__`）：

| 分类 | 导出名称 |
|---|---|
| Agent | `Agent`、`ReactAgent`、`PlanExecuteAgent` |
| 事件 | `AgentEvent`、`StepStartEvent`、`TextDeltaEvent`、`ModelMessageEvent`、`ToolCallStartEvent`、`ToolCallEndEvent`、`FinalAnswerEvent`、`LoopAbortEvent`、`PlanGeneratedEvent`、`PlanStepStartEvent`、`PlanStepEndEvent`、`HookFn` |
| 记忆 | `Memory`、`MemoryEntry`、`ShortTermMemory`、`LongTermMemory`、`MemoryManager` |
| 工具 | `TOOL_REGISTRY`、`TOOL_SCHEMAS`、`register_tool`、`call_tool`、`set_skill_registry` |
| 模型 | `ModelProvider`、`ModelResponse`、`StreamChunk`、`ToolCall`、`OpenAIProvider` |
| 技能 | `Skill`、`SkillRegistry`、`SkillLoader` |
| 异常 | `GearLinkError`、`ProviderError`、`ToolError`、`ToolNotFoundError`、`MemoryError`、`SkillError` 及其子类 |
| 日志 | `enable_logging`、`disable_logging` |
| 工具函数 | `estimate_tokens`、`count_message_tokens` |

## 示例

- [examples/memory_chatbot.py](examples/memory_chatbot.py)：短期 + 长期记忆 + 会话摘要的对话助手；
- [examples/streaming_demo.py](examples/streaming_demo.py)：`run_stream` 流式输出（含工具调用阶段）；
- [examples/plan_execute_demo.py](examples/plan_execute_demo.py)：`PlanExecuteAgent` 规划-执行（含事件回调观察）；
- [examples/skill_demo/](examples/skill_demo/)：技能目录结构示例（每个技能一个含 `SKILL.md` 的子目录）。

## 文档

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
