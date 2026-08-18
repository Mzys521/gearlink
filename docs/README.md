# GearLink 文档中心

本文档中心按“项目概览 → 安装与运行 → 架构 → 数据模型 → API → 开发与治理”组织，
是 GearLink 技术文档的统一入口。README 只保留产品定位与最短上手路径；详细设计、
契约和操作说明以本目录文档为准。

## 1. 项目概览

GearLink 是面向 Python 3.10 及以上版本的轻量级 Agent 框架。框架提供 ReAct、
规划-执行和主管-工人多 Agent 编排，并通过稳定抽象支持模型、记忆、工具、技能与
MCP 扩展。公共 API 由 `gearlink/__init__.py` 集中导出。

核心能力边界：

- `Agent` 定义 `run`、`run_stream`、`run_events` 三种统一调用契约；
- `ReactAgent` 与 `PlanExecuteAgent` 实现单 Agent 执行策略；
- `Orchestrator`、`DependentOrchestrator`、`AutonomousOrchestrator` 实现多 Agent 调度；
- `ModelProvider`、`Memory`、`VectorStore` 与 `ToolRegistry` 提供可替换扩展点；
- `AgentEvent` 是运行时观察、回放和用量统计的统一事件模型。

## 2. 安装与设置

### 2.1 运行环境

- Python 3.10+
- 默认模型依赖：OpenAI 兼容 API
- 可选能力：`anthropic`、`mcp`

### 2.2 安装

```bash
pip install gearlink

# 可选扩展
pip install "gearlink[anthropic]"
pip install "gearlink[mcp]"

# 从源码开发
pip install -e ".[dev]"
```

复制 `.env.example` 为 `.env`，按所选 Provider 配置密钥。密钥只应通过环境变量或
外部密钥服务注入，不得提交到仓库。完整的首个 Agent 示例和 Provider 配置见
[使用教程](使用教程.md)。

### 2.3 验证开发环境

```bash
ruff format --check .
ruff check .
pytest
```

## 3. 架构

[架构设计](架构设计.md)是架构事实来源，包含：

- 分层依赖和模块数据流；
- Agent、Provider、Memory、Tool、Skill 与事件模型的职责；
- ReAct、规划-执行及三类多 Agent 编排的运行时流程；
- 容错、序列化、可观测性和扩展点约束。

多 Agent 编排采用 Template Method：基类执行内核统一处理派单、串并行执行、事件
重编号、未收敛兜底和最终汇总；子类只定义计划生成、拓扑分层与上游结果注入。

## 4. 数据模型

| 模型 | 模块 | 用途 |
|---|---|---|
| `ToolCall` | `providers/base.py` | 标准化工具调用标识、名称和参数 |
| `TokenUsage` | `providers/base.py` | 输入、输出和总 token 用量 |
| `ModelResponse` | `providers/base.py` | Provider 非流式响应边界模型 |
| `StreamChunk` | `providers/base.py` | Provider 流式响应边界模型 |
| `MemoryEntry` | `core/memory.py` | 长期记忆内容、元数据和检索距离 |
| `Session` | `core/memory.py` | 可快照和恢复的会话状态 |
| `AgentEvent` 子类 | `core/events.py` | 执行步骤、模型消息、工具调用、编排和终态事件 |
| `UsageRecord` | `utils/usage.py` | 按标签聚合的调用次数与 token 用量 |

持久化模型使用 `to_dict()` / `from_dict()` 保证 JSON 边界清晰；字段兼容规则和异常
行为见[架构设计：序列化约定](架构设计.md#11-序列化约定)。

## 5. API 参考

[接口文档](接口文档.md)按以下领域列出公共 API、参数、返回值和异常：

1. Agent 与多 Agent 编排
2. 事件流与回调
3. 短期、长期及组合记忆
4. 工具注册、Schema 与调度
5. Provider 与标准响应模型
6. MCP、技能、可观测性、日志和异常

顶层 `gearlink.__all__` 是公共导出的最终依据；未导出的内部函数不承诺兼容性。

## 6. 使用与示例

- [使用教程](使用教程.md)：从快速开始到工具、记忆、技能、MCP 和多 Agent 编排；
- `examples/`：按能力拆分的可执行示例；
- 根目录 `README.md`：项目定位、安装和最短上手路径。

## 7. 开发与治理

- [开发规范](开发规范.md)：目录边界、代码风格、扩展契约、测试与发布规则；
- [贡献指南](../CONTRIBUTING.md)：提交步骤和质量门禁；
- [安全策略](../SECURITY.md)：漏洞报告、密钥安全和受支持版本；
- [更新日志](../CHANGELOG.md)：已发布版本的用户可见变更；
- [开发方向](开发方向.md)：尚未承诺发布日期的演进方向。

文档维护遵循以下规则：已发布事实写入 README、教程、架构或 API 文档；计划项只写入
开发方向；用户可见变更写入更新日志，避免在多个文档中维护相互冲突的状态描述。
