"""可注入的 token 计数与预算内文本截断工具。

默认实现沿用无第三方依赖的启发式规则：CJK 表意字符约 1 token，其余字符约
4 字符/token。需要模型对应的精确 BPE 计数时，可安装 ``gearlink[tokenizers]``
并注入 :class:`TiktokenTokenCounter`，或按 :class:`TokenCounter` 协议提供自定义实现。
"""

import importlib
import json
import re
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

# 常见表意字符范围：CJK 统一表意文字、日文假名、谚文、全角符号
_CJK_CHARS = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af\uff00-\uffef]")


@runtime_checkable
class TokenCounter(Protocol):
    """文本、消息计数及预算内截断的可注入契约。"""

    def count_text(self, text: str) -> int:
        """返回文本占用的 token 数。"""
        ...

    def count_message(self, message: dict[str, Any]) -> int:
        """返回一条 OpenAI 格式消息的内容 token 数。"""
        ...

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """返回不超过 ``max_tokens`` 的有效文本前缀。"""
        ...


def _count_message(message: dict[str, Any], count_text: Callable[[str], int]) -> int:
    """使用给定文本计数函数统计消息内容与工具调用参数。"""
    content = message.get("content")
    if content is None:
        content_text = ""
    elif isinstance(content, str):
        content_text = content
    else:
        content_text = json.dumps(content, ensure_ascii=False)

    tokens = count_text(content_text)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        tokens += count_text(json.dumps(tool_calls, ensure_ascii=False))
    return tokens


class HeuristicTokenCounter:
    """无依赖的确定性启发式 token 计数器。"""

    def count_text(self, text: str) -> int:
        """按 CJK 每字 1 token、其余每 4 字符 1 token 估算。"""
        if not text:
            return 0
        cjk_count = len(_CJK_CHARS.findall(text))
        other_count = len(text) - cjk_count
        return cjk_count + (other_count + 3) // 4

    def count_message(self, message: dict[str, Any]) -> int:
        """统计消息 ``content`` 与 ``tool_calls`` 的估算 token 数。"""
        return _count_message(message, self.count_text)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """按启发式计数返回预算内最长字符前缀。"""
        if max_tokens < 0:
            raise ValueError("max_tokens 不能为负数")
        if self.count_text(text) <= max_tokens:
            return text

        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count_text(text[:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        return text[:low]


class TiktokenTokenCounter:
    """基于可选 ``tiktoken`` 依赖的 BPE token 计数器。

    ``model`` 与 ``encoding_name`` 只能指定一个。对非 OpenAI 或无法自动识别的
    模型，应显式传入与目标模型兼容的 ``encoding_name``。
    """

    def __init__(self, *, model: str | None = None, encoding_name: str | None = None) -> None:
        if model is not None and encoding_name is not None:
            raise ValueError("model 与 encoding_name 不能同时指定")
        try:
            tiktoken = importlib.import_module("tiktoken")
        except ImportError as exc:
            raise ImportError(
                "TiktokenTokenCounter 需要可选依赖：pip install 'gearlink[tokenizers]'"
            ) from exc

        if model is not None:
            self.encoding = tiktoken.encoding_for_model(model)
        else:
            self.encoding = tiktoken.get_encoding(encoding_name or "cl100k_base")

    def _encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, disallowed_special=())

    def count_text(self, text: str) -> int:
        """使用配置的 BPE encoding 精确统计文本 token 数。"""
        return len(self._encode(text))

    def count_message(self, message: dict[str, Any]) -> int:
        """统计消息 ``content`` 与 ``tool_calls`` 的 BPE token 数。"""
        return _count_message(message, self.count_text)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """按 token 边界截断，并丢弃不完整 UTF-8 尾字节。"""
        if max_tokens < 0:
            raise ValueError("max_tokens 不能为负数")
        tokens = self._encode(text)
        if len(tokens) <= max_tokens:
            return text
        raw = b"".join(
            self.encoding.decode_single_token_bytes(token) for token in tokens[:max_tokens]
        )
        return raw.decode("utf-8", errors="ignore")


DEFAULT_TOKEN_COUNTER: TokenCounter = HeuristicTokenCounter()


def estimate_tokens(text: str) -> int:
    """使用默认启发式计数器估算文本 token 数（向后兼容函数）。"""
    return DEFAULT_TOKEN_COUNTER.count_text(text)


def count_message_tokens(message: dict[str, Any]) -> int:
    """使用默认启发式计数器估算单条消息 token 数（向后兼容函数）。"""
    return DEFAULT_TOKEN_COUNTER.count_message(message)


def truncate_text(
    text: str,
    max_tokens: int,
    *,
    token_counter: TokenCounter | None = None,
    suffix: str = "",
) -> str:
    """按计数器将文本与可选后缀共同限制在 token 预算内。

    未超预算时原样返回且不追加后缀；发生截断时，后缀本身也计入预算。若后缀
    独占预算仍过长，则先把后缀截到预算内。
    """
    if max_tokens < 0:
        raise ValueError("max_tokens 不能为负数")
    counter = token_counter if token_counter is not None else DEFAULT_TOKEN_COUNTER
    if counter.count_text(text) <= max_tokens:
        return text

    fitted_suffix = counter.truncate_text(suffix, max_tokens)
    suffix_tokens = counter.count_text(fitted_suffix)
    content_budget = max(0, max_tokens - suffix_tokens)
    prefix = counter.truncate_text(text, content_budget)
    candidate = prefix + fitted_suffix

    # BPE 在拼接边界可能重新分词；逐 token 收紧前缀，确保最终硬上限成立。
    while content_budget > 0 and counter.count_text(candidate) > max_tokens:
        content_budget -= 1
        prefix = counter.truncate_text(text, content_budget)
        candidate = prefix + fitted_suffix
    if counter.count_text(candidate) > max_tokens:
        return counter.truncate_text(candidate, max_tokens)
    return candidate
