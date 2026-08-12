# 更新日志

本项目所有显著变更均记录于此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

首个发布版本（0.1.0）整理中，包含以下内容。

### Added

- `ReactAgent`：ReAct 循环编排器，支持注入 `Memory` / `MemoryManager`、`retrieve_every_iteration` 每轮长期检索注入、工具结果按 `MAX_TOOL_RESULT_TOKENS` 截断；
- `ReactAgent` 新增可选构造参数 `skill_registry`：注入技能注册表并把可用技能列表拼入默认系统提示，联动 `load_skill` 工具；
- 记忆体系：`ShortTermMemory`（滑窗 + token 预算）、`LongTermMemory`（chromadb 向量检索、时间衰减、容量淘汰、写入去重）、`MemoryManager`（上下文预算、会话摘要沉淀、检索注入）；
- 工具三件套：`TOOL_REGISTRY` / `TOOL_SCHEMAS` / `call_tool` 与显式 `register_tool`；内置 `get_current_time` 工具；
- 技能扩展：`Skill` / `SkillRegistry` / `SkillLoader`（`gearlink/skills/`）与 `load_skill` 工具（`gearlink/tools/load_skill.py`）；
- `OpenAIProvider`：OpenAI 兼容接口适配，密钥从环境变量 `DEEPSEEK_API_KEY` 读取；
- 统一异常体系 `gearlink/exceptions.py`（`GearLinkError` 及其子类）；
- 流式输出：`ModelProvider.chat_stream`（默认回退非流式）与 `ReactAgent.run_stream`，新增统一流式事件 `StreamChunk`；`OpenAIProvider` 支持真流式（工具调用增量按 index 累积）；
- 事件流：`ReactAgent.run_events` 逐步产出 `AgentEvent` 体系事件（`StepStartEvent` / `ToolCallStartEvent` / `ToolCallEndEvent` / `FinalAnswerEvent` 等），`run` / `run_stream` 重构为事件流消费者；新增 `hooks` 构造参数与 `add_hook` 回调（on_step 语义）；
- Agent 抽象：`Agent`（ABC）统一编排契约——`run_events` 为循环唯一实现（子类必须提供），`run` / `run_stream` / `add_hook` 由基类实现；`ReactAgent` 继承之并移除重复实现，公共接口不变；
- `PlanExecuteAgent`：规划-执行策略（规划器分解步骤 → 内部 `ReactAgent` 执行器子循环 → 整合器汇总答案），支持 `max_steps` 截断与规划解析失败自动退化单步骤，新增 `PlanGeneratedEvent` / `PlanStepStartEvent` / `PlanStepEndEvent` 事件；
- 全局日志开关：`enable_logging` / `disable_logging` 一键开关 `gearlink` 命名空间日志（输出到 stderr，级别可配置，幂等）；
- 根目录 `examples/`（记忆对话示例与技能目录示例）；
- `examples/` 补齐公共 API 分类示例：`custom_provider_demo.py` / `custom_memory_demo.py` / `event_hooks_demo.py` / `custom_tool_demo.py`（均内置无网络 Provider，免密钥可运行）；
- 序列化：`ToolCall` / `ModelResponse` 新增 `to_dict()` / `from_dict()` 往返方法（含嵌套 tool_calls，接口设计规范 §3）；
- 新手教程 `docs/使用教程.md`：从安装到进阶的完整教程；
- 工程化：GitHub Actions CI（`ruff format --check` + `ruff check` + `pytest --cov`，Python 3.10/3.12 矩阵）+ `gearlink/core` 覆盖率 ≥ 80% 门禁（`pytest-cov`）+ gitleaks 密钥扫描 job；
- 开源合规：根目录新增 MIT `LICENSE`；`requirements.txt` 补齐示例依赖 `python-dotenv`；
- Embedding 抽象：`EmbeddingFn` 类型契约 + `LongTermMemory` 新增可选 `embedding_function` 参数（自定义向量化由应用层注入，默认行为不变，开发方向 §4.1）；
- Provider 重试与结构化输出：`ProviderError.retryable` 可重试标记；`ReactAgent` 新增 `max_retries`（指数退避，默认 0 = 现状）；`ModelProvider.chat` / `chat_stream` 新增可选 `response_format`，`OpenAIProvider` 透传（开发方向 §4.3）；
- 工具编排增强：`ReactAgent` 新增 `tools` 白名单与 `parallel_tool_calls` 并行执行（默认关闭）；新增 `build_tool_schema` 从函数签名 + docstring 推导 JSON Schema（开发方向 §4.4）；
- 会话持久化：新增 `Session` dataclass 与 `MemoryManager.snapshot()` / `restore()`，支持断线恢复（开发方向 §4.5）；配套免密钥示例 `examples/session_restore_demo.py`；
- Provider 生态：`OllamaProvider`（本地模型，无需密钥）与 `AnthropicProvider`（Messages API 双向归一化，依赖可选 `gearlink[anthropic]`）；配套示例 `examples/ollama_local_demo.py` / `examples/anthropic_demo.py`（开发方向 §4.2）；
- MCP 接入：新增 `gearlink/mcp/` 与 `McpClient`，远端 MCP 工具以 `mcp_<server>_<tool>` 命名映射进工具注册表（依赖可选 `gearlink[mcp]`）；配套示例 `examples/mcp_client_demo.py`（开发方向 §4.6）；
- 可观测性（开发方向 §5.1）：`TokenUsage` 计数类型与 `ModelResponse.usage` 字段（`OpenAIProvider` 透传）；`ModelMessageEvent.usage` 事件透传；`JsonlEventSink` / `jsonl_hook` / `load_jsonl_events` 事件落盘与离线回放；`UsageTracker` / `UsageRecord` 按标签聚合用量与成本估算；配套免密钥示例 `examples/observability_demo.py`；
- 记忆深化（开发方向 §5.2）：`MemoryManager` 新增 `compress_context`（上下文摘要动态压缩，复用 `summarizer` 注入点）与 `profile_hook`（用户画像沉淀，`build_context` 优先注入）；`LongTermMemory` 新增 `relevance_threshold`（相关性阈值过滤）与 `mmr_lambda`（MMR 去冗余重排）；存储后端收敛为 `VectorStore` 协议（默认 `ChromaVectorStore`，行为不变，可用 `store=` 注入自定义后端）；配套免密钥示例 `examples/memory_advanced_demo.py`；
- 多 Agent 协作（开发方向 §5.3）：新增编排层 `Orchestrator`（主管-工人模式：主管分派子任务 → 各工人独立执行（可 `parallel=True` 并行）→ 汇总答案；分派解析失败退化为全员兜底），新增 `TeamPlanGeneratedEvent` / `AgentHandoffEvent` / `SubtaskEndEvent` 事件；`Agent` 契约不变（纯新增）；配套免密钥示例 `examples/orchestrator_demo.py`；
- `tests/` 基础测试套件（外部服务全部 mock）。

### Fixed

- 修复 `ReactAgent` 引用未定义的 `SYSTEM_PROMPT` 常量导致的 `NameError` 与测试收集失败；
- 移除 `.gitignore` 对 `tests/` 的误排除。

### Changed

- `tools/builtin.py` 移除冗余的 FastMCP 包装；依赖清单以显式 `pyyaml` 替代 `fastmcp`；
- `skills/base.py` 规范化：现代类型语法、完整类型标注、Google 风格 docstring、标准日志器；
- 公共 API 出口补齐：技能三件套、异常体系、`set_skill_registry` 与 token 工具函数均从顶层包 `gearlink` 显式导出（见 `__all__`）；
- 移除 `core/agent.py` 的 `__main__` 演示入口，可运行示例统一收敛至根目录 `examples/`；
- 去重重构：`core/memory.py` 三处 token 预算裁剪逻辑统一为 `_keep_within_budget` 辅助函数，检索去重合并为单次遍历；`core/agent.py` 事件产出样板代码统一为基类 `_emit_event`，`provider` / `_hooks` 初始化上提至 `Agent` 基类（`PlanExecuteAgent` 与执行器共享同一回调列表）；`exceptions.py` 移除冗余 `pass`。公共 API 行为不变。
