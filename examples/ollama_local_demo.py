"""GearLink Ollama 本地模型示例

演示 `OllamaProvider`（开发方向 §4.2）：无需任何 API 密钥，
用本地 Ollama 服务驱动 ReAct Agent。

前置条件（一次性）：
    ollama pull qwen2.5:7b
    ollama serve

运行方式：
    python examples/ollama_local_demo.py

本地服务未启动时，本示例会给出友好提示而不是抛出原始堆栈。
"""

from gearlink import OllamaProvider, ProviderError, ReactAgent

# 1) 本地提供者：无需密钥，默认连接 http://localhost:11434/v1
provider = OllamaProvider()  # 也可 OllamaProvider(model="llama3.2:3b")

# 2) 照常用法：Provider 可插拔，其余 API 与 OpenAIProvider 完全一致
agent = ReactAgent(provider=provider)

try:
    answer = agent.run("用一句话介绍你自己，并告诉我现在几点了")
    print("助手：", answer)
except ProviderError as e:
    print("无法连接 Ollama 服务：", e)
    print("请先执行 `ollama serve` 并确认已 `ollama pull qwen2.5:7b`")
