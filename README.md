# GearLink

轻量级 Agent 框架：一个 ReAct 循环编排器 + 三个可插拔维度（模型、记忆、工具）+ 技能扩展，不引入重型框架。

## 特性

- **ReAct 循环**：`ReactAgent` 编排 Reason → Act → Observe，工具失败作为可恢复信号写回模型处理，`run_stream` 支持流式输出；
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

## 公共 API

所有公共名称均从顶层包 `gearlink` 显式导出（见 `gearlink/__init__.py` 的 `__all__`）：

| 分类 | 导出名称 |
|---|---|
| Agent | `ReactAgent` |
| 记忆 | `Memory`、`MemoryEntry`、`ShortTermMemory`、`LongTermMemory`、`MemoryManager` |
| 工具 | `TOOL_REGISTRY`、`TOOL_SCHEMAS`、`register_tool`、`call_tool`、`set_skill_registry` |
| 模型 | `ModelProvider`、`ModelResponse`、`StreamChunk`、`ToolCall`、`OpenAIProvider` |
| 技能 | `Skill`、`SkillRegistry`、`SkillLoader` |
| 异常 | `GearLinkError`、`ProviderError`、`ToolError`、`ToolNotFoundError`、`MemoryError`、`SkillError` 及其子类 |
| 工具函数 | `estimate_tokens`、`count_message_tokens` |

## 示例

- [examples/memory_chatbot.py](examples/memory_chatbot.py)：短期 + 长期记忆 + 会话摘要的对话助手；
- [examples/streaming_demo.py](examples/streaming_demo.py)：`run_stream` 流式输出（含工具调用阶段）；
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
