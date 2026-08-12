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
- `tests/` 基础测试套件（外部服务全部 mock）。

### Fixed

- 修复 `ReactAgent` 引用未定义的 `SYSTEM_PROMPT` 常量导致的 `NameError` 与测试收集失败；
- 移除 `.gitignore` 对 `tests/` 的误排除。

### Changed

- `tools/builtin.py` 移除冗余的 FastMCP 包装；依赖清单以显式 `pyyaml` 替代 `fastmcp`；
- `skills/base.py` 规范化：现代类型语法、完整类型标注、Google 风格 docstring、标准日志器；
- 公共 API 出口补齐：技能三件套、异常体系、`set_skill_registry` 与 token 工具函数均从顶层包 `gearlink` 显式导出（见 `__all__`）；
- 移除 `core/agent.py` 的 `__main__` 演示入口，可运行示例统一收敛至根目录 `examples/`。
