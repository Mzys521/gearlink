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

## 待办

### 代码完善

- [x] `LongTermMemory`：实现 `add_message` / `get_messages` / `clear`（基于 chromadb 向量检索）
- [x] `MemoryManager`：实现短期 + 长期记忆的组合管理（`add_message` / `build_context` / `end_session` / `clear`），并支持通过 `ReactAgent(memory=...)` 注入
- [x] `ReactAgent`：`print` 日志替换为标准库 `logging`
- [ ] `skills/`：确定技能扩展契约（接口设计规范 §4 标注「待定」），并在文档中补齐
- [x] `tools/`：将内置工具（如 `get_current_time`）从 `core/tool.py` 迁移到 `tools/`，core 仅保留注册表与调度器（含 `register_tool` 显式注册函数）

### 数据结构与序列化

- [ ] 需要持久化的 dataclass（如写入记忆的结构）补充 `to_dict()` / `from_dict()` 往返方法

### 工程化

- [ ] 建立 `examples/` 目录，为每个公共 API 提供可直接运行的示例（接口设计规范 §8）
- [ ] CI：GitHub Actions 流水线（`ruff format --check` + `ruff check` + `pytest`）
- [ ] `core/` 测试覆盖率 ≥ 80%（当前已有基础用例，需接入 coverage 检查）

### 开源合规（开发规范 §9）

- [ ] 确定许可证类型并添加 `LICENSE`
- [ ] 编写 `README.md`（项目简介、安装、快速开始、示例链接）
- [ ] 编写 `CONTRIBUTING.md` 指向 `docs/开发规范.md`
- [ ] 创建 `CHANGELOG.md`（Keep a Changelog 格式）
- [x] 提供 `.env.example` 示例环境变量文件
- [ ] 发布前用 `gitleaks` 审查历史提交，确认无泄露密钥（注意：旧提交中曾硬编码 API key，需处理历史记录或轮换密钥）
