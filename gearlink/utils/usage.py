"""Token 用量聚合统计（可观测性，开发方向 §5.1）。

按标签（如 provider / 模型名 / 会话 ID）聚合 :class:`TokenUsage`，
并支持按单价表估算成本。纯内存实现，应用层按需持有。
"""

from dataclasses import dataclass, field
from typing import Any

from gearlink.providers.base import TokenUsage


@dataclass
class UsageRecord:
    """单个标签下的累计用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，与 from_dict 保证往返一致。"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageRecord":
        """从字典反序列化。

        Args:
            data: to_dict 产出的字典（兼容缺少 calls 字段的数据）。

        Returns:
            UsageRecord 实例。
        """
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            calls=data.get("calls", 0),
        )


@dataclass
class UsageTracker:
    """Token 用量聚合器：按标签累计多次调用的用量，支持成本估算。

    典型用法（经 hooks 从事件流采集）::

        tracker = UsageTracker()

        def on_event(event):
            if getattr(event, "usage", None) is not None:
                tracker.add(event.usage, label="deepseek-chat")

        agent.add_hook(on_event)
    """

    records: dict[str, UsageRecord] = field(default_factory=dict)

    def add(self, usage: TokenUsage | None, label: str = "default") -> None:
        """累计一次调用的用量。

        Args:
            usage: 本次调用用量；None 时忽略（提供者未上报）。
            label: 聚合标签（如模型名 / 会话 ID），默认 ``"default"``。
        """
        if usage is None:
            return
        record = self.records.setdefault(label, UsageRecord())
        record.input_tokens += usage.input_tokens
        record.output_tokens += usage.output_tokens
        record.calls += 1

    def total(self) -> TokenUsage:
        """全部标签的用量总和。"""
        result = TokenUsage()
        for record in self.records.values():
            result = result + TokenUsage(record.input_tokens, record.output_tokens)
        return result

    def estimate_cost(self, prices: dict[str, tuple[float, float]]) -> float:
        """按单价表估算累计成本。

        Args:
            prices: 标签 → (输入单价, 输出单价) 的映射，单价以「每千 token」计；
                未收录的标签不计入成本。

        Returns:
            估算成本（与单价表同币种/单位）。
        """
        cost = 0.0
        for label, record in self.records.items():
            if label not in prices:
                continue
            input_price, output_price = prices[label]
            cost += record.input_tokens / 1000 * input_price
            cost += record.output_tokens / 1000 * output_price
        return cost

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，与 from_dict 保证往返一致。"""
        return {label: record.to_dict() for label, record in self.records.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageTracker":
        """从字典反序列化。

        Args:
            data: to_dict 产出的字典。

        Returns:
            UsageTracker 实例。
        """
        return cls(records={label: UsageRecord.from_dict(v) for label, v in data.items()})
