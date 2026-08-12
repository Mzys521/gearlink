"""OllamaProvider 测试：本地模型提供者，无需密钥（mock 外部服务）。"""

from unittest.mock import MagicMock

from gearlink.providers.ollama_provider import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaProvider


def test_provider_works_without_any_api_key(monkeypatch):
    # 验收（开发方向 §4.2）：环境变量缺失时仍可构造，无需密钥
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    provider = OllamaProvider()

    assert provider.model == DEFAULT_MODEL
    assert str(provider.client.base_url).rstrip("/").endswith(DEFAULT_BASE_URL.rstrip("/"))


def test_provider_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.10:11434/v1")

    provider = OllamaProvider()

    assert provider.model == "llama3.2:3b"
    assert "192.168.1.10" in str(provider.client.base_url)


def test_explicit_args_take_precedence(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")

    provider = OllamaProvider(model="qwen2.5:14b", base_url="http://localhost:11434/v1")

    assert provider.model == "qwen2.5:14b"


def test_chat_reuses_openai_compliant_path():
    provider = OllamaProvider()
    fake_message = MagicMock(content="本地模型回复", tool_calls=[])
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_message)]
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = fake_response

    response = provider.chat(messages=[{"role": "user", "content": "你好"}])

    assert response.content == "本地模型回复"
    # 走的是 OpenAI 兼容 chat.completions 通道
    kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
