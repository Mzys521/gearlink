# 更新日志

本项目所有显著变更均记录于此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [0.4.0] - 2026-08-15

### Added

- 模型自主编排：
  - `AutonomousOrchestrator`：`Orchestrator` 纯新增子类，主管模型每次请求自主产出 DAG 编排计划（`{"nodes": [{"id", "worker", "task"}], "edges": [{"from", "to"}]}`，节点为子任务、边为数据依赖），自主决定拆分、指派与串/并行策略；计划编译为节点级 Kahn 拓扑分层混合串并行执行（`parallel=True` 默认：层内并行、层间串行），同一工人多节点按声明顺序串行；下游任务执行前全部直接上游结果按依赖边汇总、按上游工人名分组以 `[上游结果]` 报告段落注入（层屏障保证消息同步）；计划解析失败/引用未登记工人/缺失依赖/成环时记录日志并降级为全员兜底分派；兼容主管输出旧分派格式（纯数组按序解释为串行链）；
  - 事件新增可选字段（均默认 `None`，向后兼容）：`TeamPlanGeneratedEvent.graph`（DAG 计划）/ `.parallel_groups`（拓扑分层），`AgentHandoffEvent.layer` / `.upstream`（直接上游子任务下标），`SubtaskEndEvent.layer`；兜底降级时不携带这些字段；
  - 新增 `examples/autonomous_orchestrator_demo.py`（免密钥）与 `tests/test_autonomous_orchestrator.py`（18 用例）；`docs/架构设计.md` / `docs/使用教程.md` / `docs/接口文档.md` / `docs/开发方向.md` / `README.md` 同步更新。

### 兼容性说明

- 纯新增能力：`Orchestrator` / `DependentOrchestrator` 的外部行为、事件序列与既有测试语义均不变，无需迁移；
- 事件新增字段序列化后为 `null`，旧版 JSONL 回放代码按字段缺失处理即可；
- `AutonomousOrchestrator` 默认 `parallel=True`（层内并行），要求工人自身及其工具线程安全；如需全串行传 `parallel=False`。

## [0.3.0] - 2026-08-14

### Added

- 依赖编排（开发方向 §6.8）：
  - `DependentOrchestrator`：`Orchestrator` 纯新增子类，`dependencies` 编程式声明工人依赖（worker 名 → 上游 worker 名列表），执行按 Kahn 拓扑分层（层内可并行、层间串行），上游结果以 `[上游结果]` 报告段落自动注入下游任务，实现流水线式多 Agent 协作；构造时校验未登记引用与依赖环（抛 `GearLinkError`）；`dependencies=None` 时行为等价 `Orchestrator`；
  - `TeamPlanGeneratedEvent` 新增可选 `dependencies` 字段（默认 None，向后兼容）；
  - 新增 `examples/dependent_orchestrator_demo.py` 与 `tests/test_dependent_orchestrator.py`（16 用例）；`docs/接口文档.md` / `docs/架构设计.md` / `docs/开发方向.md` / `docs/使用教程.md` / `README.md` 同步更新。

## [0.2.1] - 2026-08-14

### Added

- 新增 GitHub Issue 模板（bug_report / feature_request / config）
- 新增 GitHub PR 模板（变更类型 + 检查清单）
- 新增 `SECURITY.md` 安全策略
- 安装文档更新为 `pip install gearlink`（从 PyPI 安装）

## [0.2.0] - 2026-08-13

- 打包元数据：`pyproject.toml` 补齐 PEP 639 的 `license` / `license-files`、`authors`、`keywords`、`classifiers` 与 `[project.urls]`，`build-system` 升级为 `setuptools>=77`；`python -m build` 可产出 sdist + wheel，为发布 PyPI 就绪。

### Changed

- 日志覆盖补齐：`gearlink/` 全部有副作用的逻辑模块接入标准库 `logging` 模块日志器——本次新增 `core/tool.py`、`core/events.py`、`providers/openai_provider.py`、`providers/anthropic_provider.py`、`providers/ollama_provider.py`、`mcp/client.py`、`tools/builtin.py`、`tools/load_skill.py`，并补全 `core/agent.py` / `core/orchestrator.py` 的编排日志；统一「里程碑 info / 过程 debug / 异常兜底 warning」分级，默认静默、经 `enable_logging()` 输出 stderr；不改变公共 API 与行为。

## [0.2.0] - 2026-08-13

### Added

- 架构治理（开发方向 §6.4）：
  - `ToolRegistry` 类：工具注册表从进程级全局单例升级为可实例化组件，经 `ReactAgent(tool_registry=...)` 注入实现工具集/技能集隔离；模块级 `TOOL_REGISTRY` / `TOOL_SCHEMAS` / `register_tool` / `call_tool` / `set_skill_registry` 保留为默认实例兼容委托；
  - `contextvars` 上下文解析：`ToolRegistry.call_tool` 执行期间经 `get_current_tool_registry()` 暴露当前注册表，`load_skill` 工具据此解析所属 Agent 的技能注册表，不再依赖全局状态；
  - `ContextBuilder` 协议：统一记忆/上下文构建契约，`Memory` 基类新增 `build_context` 默认实现（返回 `get_messages()`），`ReactAgent._build_messages` 移除 `isinstance(MemoryManager)` 特判；
  - `ReactAgent` / `PlanExecuteAgent` 新增 `tool_registry` 可选参数；`McpClient` 新增 `tool_registry` 可选参数。
- 鲁棒性与确定性（开发方向 §6.5）：
  - `OpenAIProvider` / `OllamaProvider` 新增 `timeout` 参数；OpenAI SDK 内置重试关闭（`max_retries=0`），重试由框架 `max_retries` 统一管理，避免双重重试；
  - `PlanExecuteAgent._plan` / `Orchestrator._dispatch` 强制 `response_format={"type":"json_object"}`，新增 `_extract_json` 辅助函数从含 markdown 围栏或说明文字的文本中提取 JSON，加固解析容错。
- 新增 `docs/接口文档.md`：逐一说明每个公共 API 的作用与用法。
- 重写 `docs/架构设计.md`：对齐当前代码实现，补充 PlanExecuteAgent / Orchestrator 时序图与文字说明。

### Changed

- 文档重组：合并 `docs/接口设计规范.md` 到 `docs/开发规范.md`（统一为 13 节结构），删除已完成的 `docs/记忆升级方案.md`；更新全部交叉引用。

### Fixed

- 文档与代码一致性修正：
  - `docs/接口设计规范.md` §2 抽象接口示例补齐 `response_format` 参数，与 `ModelProvider.chat` / `chat_stream` 实际签名对齐；§5 异常层次补全 `SkillError` 子树（`SkillNotFoundError` / `SkillLoadError` / `SkillValidationError` / `SkillExecutionError`），与 `gearlink/exceptions.py` 实际继承关系对齐；
  - `docs/架构设计.md` §2 分层图、§3 类图、§3.1 组件表补全 `Orchestrator`（`core/orchestrator.py`，`Agent` 子类）；§7 目录结构由「目标态」改为「实际结构」并补 `tools/load_skill.py`；§9 异常层次补全 `SkillError` 子树；
  - `docs/开发规范.md` §1 项目结构补全 `mcp/`、`core/` 子文件、根级文档（`CHANGELOG.md` / `CONTRIBUTING.md` / `TODO.md` / `.env.example`）与 `.github/`；§9 开源合规更新为现状；移除已全部完成的「附：待整改清单」（迁移至 `TODO.md`）；
  - `docs/开发方向.md` 现状盘点更新（15 测试文件 / 201 用例 / `gearlink/core` 覆盖率 95.82%），P0~P2（M0/M1/M2）标记为已完成，仅保留 P3 为未来方向；
  - `README.md` 公共 API 表「异常」行显式列出全部 `SkillError` 子类；文档索引补 `docs/开发方向.md` 链接；
  - `examples/` 中 5 个自定义 `ModelProvider` 实现的 `chat()` 签名补齐 `response_format` 可选参数（接口设计规范 §2 严格合规，向后兼容）；
  - `gearlink/providers/openai_provider.py` 恢复被误删的 `DEFAULT_BASE_URL` / `DEFAULT_MODEL` 模块常量（缺失会导致 `ruff check` F821 失败及 `OpenAIProvider()` 实例化时 `NameError`）。

## [0.1.0] - 2026-08-12

首个语义化版本（提交 `a809e21`）：在早期版本（见文末「早期提交」）基础上重构并补齐，
形成多策略 Agent 编排（ReAct / 规划-执行 / 多 Agent 协作）+ 三可插拔维度
（模型 / 记忆 / 工具）+ 技能扩展的完整框架。

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

## 早期提交（语义化版本采用前）

以下提交未采用 Keep a Changelog 格式，仅按 git 历史补记（均为 0.1.0 的前身）：

- `238c4a1`（v1.0.2，2026-08-12）：`core/agent.py` 与 `core/memory.py` 重构，清理 `exceptions.py` 冗余；
- `385034a`（agent编排策略实现，2026-08-12）：编排策略落地（后续演进为 `PlanExecuteAgent`）；
- `f6c36e6`（正式版 v1.0.1，2026-08-10）：README / CONTRIBUTING / CHANGELOG 初版，`examples/` 示例目录（memory_chatbot / streaming_demo / skill_demo），`core/agent.py` 大幅重构，异常体系与公共导出补齐；
- `b9256b2` / `74438d4`（1.1 / demo 测试，2026-08-10）：demo 完善与测试；
- `7075444`（skills v1.1，2026-08-08）：技能体系初版；
- `2cd7e53` / `c248ad4` / `956c841`（上下文记忆 v1.2 / V1.1 / 记忆管理 v1，2026-08-08）：记忆体系初版与迭代；
- `bb27ad6` / `98e7577`（规范化 / 开发协议规范，2026-08-08）：开发规范与接口设计规范文档；
- `9c3f022`（最初ReAct循环，2026-08-08）：项目起点。
