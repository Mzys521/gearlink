# GearLink 待办清单（开源发布前）

> 依据 `docs/开发规范.md` 与 `docs/接口设计规范.md` 整理。已完成项不再列出。

## 已完成（本次整改）

- [x] 为 `gearlink/` 及各子目录添加 `__init__.py`，建立 `pyproject.toml`，移除所有 `sys.path` hack，改为包导入
- [x] `openai_provider.py` 硬编码 API key 改为环境变量 `DEEPSEEK_API_KEY` 读取，缺失时抛出明确错误
- [x] `memory.py` 抽象方法 `get_message` 统一为 `get_messages`，补齐 Google 风格 docstring
- [x] `core/agent.py` 扁平导入改为 `from gearlink.core.tool import ...` 等完整包路径
- [x] 建立统一异常层次 `gearlink/exceptions.py`（`GearLinkError` 体系），provider/tool 调用均包装第三方异常
- [x] 补充 `tests/` 基础测试用例（memory / tool / agent / provider，外部服务全部 mock）
- [x] `ShortTermMemory` 实现按 `max_tokens` 的截断逻辑（`utils/` 启发式 token 计数）
- [x] `MemoryManager` 支持 `max_context_tokens` 分层预算分配（系统消息优先、长期检索占 `RETRIEVAL_BUDGET_RATIO`、短期对话从最新保留）
- [x] `MemoryManager` 检索结果按 role 带说话人标签结构化注入（`[用户]` / `[助手]`，未知角色回退原始 role）
- [x] `MemoryManager` 检索结果过滤与短期窗口内容重复的条目（去重，消除即时沉淀导致的上下文冗余）
- [x] `ReactAgent` 工具结果写入记忆前按 `MAX_TOOL_RESULT_TOKENS` 截断
- [x] `MemoryManager`：`end_session` 支持注入 `summarizer` 沉淀会话摘要（`[会话摘要]` 前缀，失败不中断会话结束）
- [x] `MemoryManager`：新增 `system_budget_ratio` 非检索 system 消息预算上限（检索注入消息受保护）
- [x] `LongTermMemory`：新增 `recency_weight` 时间衰减排序与 `max_entries` 容量上限淘汰
- [x] `LongTermMemory`：新增 `dedupe` 写入去重（sha256 `content_hash`）与检索结果去重
- [x] `ReactAgent`：新增 `retrieve_every_iteration` 每轮长期记忆检索注入开关
- [x] `ReactAgent`：新增 `skill_registry` 可选参数，注入技能注册表并联动 `load_skill` 工具（默认系统提示拼入可用技能列表）
- [x] `skills/base.py` 规范化：现代类型语法、完整类型标注、Google 风格 docstring、模块级日志器
- [x] `tools/builtin.py` 移除冗余 FastMCP 包装；依赖清单以显式 `pyyaml` 替代 `fastmcp`
- [x] 示例代码迁入根目录 `examples/`（`memory_chatbot.py` 自 `demo/`、`skill_demo` 自 `gearlink/examples/`），并移除 `sys.path` hack
- [x] `.gitignore` 移除对 `tests/` 的误排除
- [x] 补齐 `tests/test_skill.py` 与 Agent 技能注入测试用例

## 已完成（M1 可插拔深化，开发方向 §4.1–4.6）

- [x] §4.1 Embedding 抽象：`EmbeddingFn` 类型契约 + `LongTermMemory(embedding_function=...)`
- [x] §4.2 Provider 生态：`OllamaProvider`（无需密钥）/ `AnthropicProvider`（Messages API 归一化，依赖可选），含测试与示例
- [x] §4.3 Provider 重试与结构化输出：`ProviderError.retryable` + `ReactAgent(max_retries=...)` 指数退避 + `response_format`
- [x] §4.4 工具编排增强：`tools` 白名单 / `parallel_tool_calls` / `build_tool_schema` schema 自动生成
- [x] §4.5 会话持久化：`Session` + `MemoryManager.snapshot()` / `restore()`，含断线恢复示例
- [x] §4.6 MCP 接入：`gearlink/mcp/` 与 `McpClient`（`mcp_<server>_<tool>` 命名映射，依赖可选），含测试与示例

## 已完成（M2 生产可用，开发方向 §5.1–5.3）

- [x] §5.1 可观测性：`TokenUsage` + `ModelResponse.usage` 透传；`JsonlEventSink` / `jsonl_hook` / `load_jsonl_events` 事件落盘回放；`UsageTracker` 用量聚合与成本估算，含测试与示例
- [x] §5.2 记忆深化：`MemoryManager(compress_context=...)` 上下文摘要动态压缩 + `profile_hook` 用户画像沉淀注入；`LongTermMemory(relevance_threshold=... / mmr_lambda=...)` 阈值过滤与 MMR 去冗余；存储后端收敛为 `VectorStore` 协议（默认 `ChromaVectorStore`），含测试与示例
- [x] §5.3 多 Agent 协作：编排层 `Orchestrator`（主管-工人，可并行）+ `TeamPlanGeneratedEvent` / `AgentHandoffEvent` / `SubtaskEndEvent` 事件（`Agent` 契约不变），含测试与示例

## 待办

### 代码完善

- [x] `LongTermMemory`：实现 `add_message` / `get_messages` / `clear`（基于 chromadb 向量检索）
- [x] `MemoryManager`：实现短期 + 长期记忆的组合管理（`add_message` / `build_context` / `end_session` / `clear`），并支持通过 `ReactAgent(memory=...)` 注入
- [x] `ReactAgent`：`print` 日志替换为标准库 `logging`
- [x] `skills/`：确定技能扩展契约（渐进式披露设计），并同步 `接口设计规范.md` / `架构设计.md`
- [x] `tools/`：将内置工具（如 `get_current_time`）从 `core/tool.py` 迁移到 `tools/`，core 仅保留注册表与调度器（含 `register_tool` 显式注册函数）

### 数据结构与序列化

- [x] 需要持久化的 dataclass（如写入记忆的结构）补充 `to_dict()` / `from_dict()` 往返方法（`MemoryEntry` / `ToolCall` / `ModelResponse`，含往返一致性测试）

### 工程化

- [x] 建立 `examples/` 目录（含记忆对话示例与技能目录示例）
- [x] 为其余公共 API 补齐可直接运行的示例（接口设计规范 §8；新增自定义 Provider / Memory / 工具 / 事件回调四个免密钥示例）
- [x] CI：GitHub Actions 流水线（`ruff format --check` + `ruff check` + `pytest`）
- [x] `core/` 测试覆盖率 ≥ 80%（`pytest-cov` 接入，`fail_under = 80` 门禁，当前约 96%）

### 开源合规（开发规范 §9）

- [x] 确定许可证类型并添加 `LICENSE`（MIT）
- [x] 编写 `README.md`（项目简介、安装、快速开始、示例链接）
- [x] 编写 `CONTRIBUTING.md` 指向 `docs/开发规范.md`
- [x] 创建 `CHANGELOG.md`（Keep a Changelog 格式）
- [x] 提供 `.env.example` 示例环境变量文件
- [ ] 发布前用 `gitleaks` 审查历史提交，确认无泄露密钥（CI 已接入 gitleaks 扫描 job；旧提交中曾硬编码 API key，发布前须轮换该密钥，必要时重写历史）

## 待办（架构债务与 DX 优化，本次审查新增）

> 依据 `docs/开发方向.md` §1.3 与 §6.4–6.7 整理，均为 P3 未排期方向，按优先级排列。
> 遵循「纯新增、默认值等价现状、公共 API 只增不改」原则，落地后同步更新 `CHANGELOG.md` 与 `docs/开发方向.md`。

### 架构治理与依赖注入（§6.4）

- [x] 工具/技能注册表去全局单例：`TOOL_REGISTRY` / `TOOL_SCHEMAS` / `_skill_registry` 改为可实例化的 `ToolRegistry` 注入 `ReactAgent`（保留模块级默认实例兼容），`load_skill` 经 `contextvars` 上下文解析，消除 `register_tool` 无锁写全局
- [x] 统一记忆/上下文契约：抽象 `ContextBuilder` 协议，`Memory` 基类新增 `build_context` 默认实现，移除 `ReactAgent._build_messages` 的 `isinstance(MemoryManager)` 特判

### 鲁棒性与确定性（§6.5）

- [x] Provider 超时：`OpenAIProvider` / `OllamaProvider` 增加 `timeout`，关闭 OpenAI SDK 默认重试（`max_retries=0`）避免双重重试
- [x] 结构化输出强制 + JSON 加固：规划/分派传 `response_format={"type":"json_object"}`，`_extract_json` 辅助函数从含围栏或说明文字的文本中提取 JSON
- [ ] token 计数可注入：定义 `TokenCounter` 协议（默认启发式，可选 tiktoken），修正工具结果按 `MAX_TOOL_RESULT_TOKENS * 4` 字符反推的截断误差
- [ ] `build_tool_schema` 完善：`Optional[X]` 生成 nullable、支持参数级描述（`typing.Annotated` 或描述字典），消除 `_nullable` 死代码

### 可观测性与异常深化（§6.6）

- [ ] `ProviderError` 子类化：派生限流/鉴权/服务端子类（向后兼容 `retryable`）
- [ ] 事件 trace 结构：`AgentEvent` 增加 `trace_id` / `parent_seq`，`Orchestrator` / `PlanExecuteAgent` 转发时保留父子关系；远期 OTLP 导出
- [ ] 版本与类型治理：提供 `__version__`、`py.typed`、mypy/pyright CI 门禁

### 异步、并发与开发者体验（§6.7）

- [ ] 异步 API：`arun` / `arun_stream` / `arun_events` + `ModelProvider.chat_async`，`Orchestrator` / `parallel_tool_calls` asyncio 变体 + 取消令牌
- [ ] 统一配置对象 `GearLinkConfig` 集中管理 env 读取，去除 Provider 内硬编码 DeepSeek 语义泄漏
- [ ] 声明式装配：YAML/JSON 描述 agent 团队 + `Agent.from_config(...)`，配合 CLI 降低多 Agent 编排门槛
