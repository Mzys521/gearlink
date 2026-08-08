import chromadb
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from chromadb.config import Settings


class Memory():

    @abstractmethod
    def add_message(self, message: Dict[str, Any]) -> None:
        """添加一条记忆"""
        pass

    @abstractmethod
    def get_message(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取记忆中的消息列表，可限制条数"""
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class ShortTermMemory(Memory):
    """短期记忆：存储最近N条对话"""

    def __init__(self, max_tokens: Optional[int] = None, max_message: Optional[int] = 20) -> None:
        """
        :param max_tokens : 保留最大 token 数
        :param max_message : 保留最大消息数
        """
        self.max_messages = max_message
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)

        # 移除最早信息
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        # token 计数截断逻辑

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit is not None:
            return self.messages[-limit:]
        return self.messages

    def clear(self) -> None:
        self.messages = []


class LongTermMemory(Memory):
    """长期记忆：基于向量检索的语义信息"""
    pass


class MemoryManager:
    """记忆管理器：结合短期喝长期记忆"""
    pass
