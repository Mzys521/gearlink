from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """模型返回的一次工具调用"""
    id: str
    name: str
    arguments: str  # JSON 字符串


@dataclass
class ModelResponse:
    """模型响应的统一表示"""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


class ModelProvider(ABC):
    """模型提供者抽象接口：屏蔽具体模型服务的调用细节"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """根据消息列表发起一次对话，返回统一的 ModelResponse"""
        pass
