"""GearLink 通用工具函数。

只有从此处显式导出的名称才是 utils 的公共 API。
"""

from gearlink.utils.token_count import count_message_tokens, estimate_tokens

__all__ = ["estimate_tokens", "count_message_tokens"]
