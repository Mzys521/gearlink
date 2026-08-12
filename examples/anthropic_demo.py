"""GearLink Anthropic Claude 示例

演示 `AnthropicProvider`（开发方向 §4.2）：适配 Anthropic Messages API，
自动完成 OpenAI 消息格式与工具调用的双向归一化。

前置条件：
    pip install gearlink[anthropic]（或 pip install "anthropic>=0.34"）
    根目录 .env 中配置 ANTHROPIC_API_KEY（或设置同名环境变量）

运行方式：
    python examples/anthropic_demo.py

未配置密钥或未安装依赖时，本示例会给出友好提示而不是抛出原始堆栈。
"""

from pathlib import Path

from dotenv import load_dotenv

from gearlink import ProviderError, ReactAgent

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # 加载根目录 .env 中的 ANTHROPIC_API_KEY

try:
    # 延迟导入：构造时才检查依赖与密钥，给出针对性提示
    from gearlink import AnthropicProvider

    provider = AnthropicProvider()  # 也可 AnthropicProvider(model="claude-haiku")
except ValueError as e:
    print("Anthropic 提供者不可用：", e)
    raise SystemExit(0)

# Provider 可插拔：换提供者不改变其余任何用法
agent = ReactAgent(provider=provider)

try:
    answer = agent.run("现在几点了？请先调用工具查询，再用一句话总结")
    print("助手：", answer)
except ProviderError as e:
    print("Anthropic 服务调用失败：", e)
