"""Ollama 本地模型提供者：复用 OpenAI 兼容接口，无需 API 密钥。

Ollama 在 ``/v1`` 路径暴露 OpenAI 兼容服务，因此 :class:`OllamaProvider`
直接继承 :class:`OpenAIProvider`，仅替换服务地址与默认模型（开发方向 §4.2）。
"""

import os

from gearlink.providers.openai_provider import OpenAIProvider

#: Ollama 本地服务默认地址（OpenAI 兼容端点）
DEFAULT_BASE_URL = "http://localhost:11434/v1"

#: 默认模型；可按本地实际拉取的模型替换
DEFAULT_MODEL = "qwen2.5:7b"


class OllamaProvider(OpenAIProvider):
    """Ollama 本地模型提供者：无需密钥，利于本地演示与离线开发。

    使用前需先启动 Ollama 服务并拉取模型，例如：

    .. code-block:: bash

        ollama pull qwen2.5:7b
        ollama serve
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """初始化 Ollama 提供者。

        各参数未传入时回退到环境变量，再到内置默认值：
        model <- OLLAMA_MODEL；base_url <- OLLAMA_BASE_URL。

        Args:
            model: Ollama 中已拉取的模型名称，默认 ``qwen2.5:7b``。
            base_url: Ollama 服务地址，默认指向本机 ``http://localhost:11434/v1``。
            timeout: 请求超时秒数（开发方向 §6.5）；None 时使用 SDK 默认值。
        """
        # Ollama 不校验密钥，传入占位值即可；显式指定 model/base_url，
        # 避免落入父类的 DEEPSEEK_* 环境变量回退路径
        super().__init__(
            model=model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL,
            api_key="ollama",
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL,
            timeout=timeout,
        )
