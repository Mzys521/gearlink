# GearLink 待办清单

本文档只记录尚未完成的工作。已交付能力与版本历史见 [CHANGELOG.md](CHANGELOG.md)，
长期方向和优先级说明见 [开发方向](docs/开发方向.md)。

## 安全与发布治理

- [ ] 审核历史提交中的密钥泄露风险：持续运行 gitleaks，确认旧密钥已轮换；如历史中
  仍含有效凭据，制定并执行历史清理方案。
- [ ] 增加提示注入防护、工具权限审批和敏感信息脱敏策略。

## 鲁棒性与类型治理

- [ ] 定义可注入 `TokenCounter` 协议，保留启发式默认实现并支持精确 tokenizer；统一
  上下文预算和工具结果截断逻辑。
- [ ] 完善 `build_tool_schema`：正确表达 `Optional` / nullable，并支持参数级描述；移除
  无效的 `_nullable` 中间状态。
- [ ] 细化 `ProviderError`：增加限流、鉴权和服务端错误子类，同时兼容现有
  `retryable` 属性。
- [ ] 发布 `__version__` 与 `py.typed`，在 CI 增加 mypy 或 pyright 门禁。

## 可观测性

- [ ] 为 `AgentEvent` 增加 `trace_id` / `parent_seq`，使 Orchestrator 与
  PlanExecuteAgent 转发的子循环保留父子关系；评估 OTLP 导出。

## 异步与开发者体验

- [ ] 增加 `arun` / `arun_stream` / `arun_events` 和 Provider 异步契约，并为多 Agent
  与并行工具执行提供取消语义。
- [ ] 引入 `GearLinkConfig` 统一环境变量和 Provider 配置，消除默认实现中的厂商语义
  泄漏。
- [ ] 支持 YAML/JSON 声明式团队装配与 `Agent.from_config(...)`，并评估 CLI / 服务化
  入口。
- [ ] 扩充技能发现、版本约束、签名校验和分发治理。

## 完成标准

每个事项合入前必须同步测试、用户文档和 `CHANGELOG.md`；默认配置须保持现有外部
行为，破坏性变更必须经过弃用周期。
