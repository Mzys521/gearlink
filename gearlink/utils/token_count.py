"""Token 计数工具：为上下文预算分配提供无依赖的启发式估算。

估算规则：CJK 表意字符按每字约 1 token 计，其余字符按 4 字符/token 计。
结果确定、无网络与第三方依赖，适合作为上下文预算的近似分配依据；
需要精确计数时可在调用方替换为 tiktoken 等专用实现。
"""

import json
import re
from typing import Any

# 常见表意字符范围：CJK 统一表意文字、日文假名、谚文、全角符号
_CJK_CHARS = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（启发式）。

    Args:
        text: 待估算的原始文本。

    Returns:
        int: 估算的 token 数；空文本返回 0。
    """
    if not text:
        return 0
    cjk_count = len(_CJK_CHARS.findall(text))
    ascii_count = len(text) - cjk_count
    return cjk_count + (ascii_count + 3) // 4


def count_message_tokens(message: dict[str, Any]) -> int:
    """估算一条 OpenAI 格式消息的 token 数。

    content 文本与 tool_calls 参数部分均计入；缺失字段按 0 处理。

    Args:
        message: OpenAI 消息格式字典。

    Returns:
        int: 估算的 token 数。
    """
    tokens = estimate_tokens(message.get("content") or "")
    tool_calls = message.get("tool_calls")
    if tool_calls:
        tokens += estimate_tokens(json.dumps(tool_calls, ensure_ascii=False))
    return tokens
