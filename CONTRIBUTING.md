# 贡献指南

感谢你对 GearLink 的关注！完整的开发流程、代码规范与 API 设计原则见 [docs/开发规范.md](docs/开发规范.md)。此处仅列出关键流程。

## 提交流程

1. 从 `main` 切出分支：`feat/<描述>`、`fix/<描述>` 或 `docs/<描述>`；
2. `main` 只接受 PR 合入，一个 PR 只做一件事；
3. 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)，如 `feat(providers): 新增 AnthropicProvider 实现`。

## 提交前检查

```bash
ruff format .
ruff check .
pytest
```

三者必须全部通过；涉及外部服务（模型 API、网络）的测试一律 mock。

## 扩展贡献

四类扩展点（Provider / Tool / Skill / Memory）只允许**新增文件并注册**，不得修改 `core/`，契约见 [docs/架构设计.md §6](docs/架构设计.md)。涉及公共 API 变更的 PR 须同步更新文档与 CHANGELOG。

## 安全

严禁在代码或提交历史中硬编码 API key 等敏感信息；密钥一律经环境变量读取。
