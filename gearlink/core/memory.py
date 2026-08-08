import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gearlink.exceptions import MemoryError

if TYPE_CHECKING:
    import chromadb


class Memory(ABC):
    """记忆抽象接口：屏蔽短期/长期记忆的存储细节"""

    @abstractmethod
    def add_message(self, message: dict[str, Any]) -> None:
        """添加一条消息到记忆。

        Args:
            message: OpenAI 消息格式字典，须包含 role 字段。
        """

    @abstractmethod
    def get_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        """获取记忆中的消息列表。

        Args:
            limit: 限制返回的最大条数；None 表示返回全部。

        Returns:
            按时间顺序排列的消息列表。
        """

    @abstractmethod
    def clear(self) -> None:
        """清空记忆中的所有消息。"""


class ShortTermMemory(Memory):
    """短期记忆：存储最近 N 条对话"""

    def __init__(self, max_tokens: int | None = None, max_message: int | None = 20) -> None:
        """初始化短期记忆。

        Args:
            max_tokens: 保留的最大 token 数（暂未实现截断）。
            max_message: 保留的最大消息数，超出时移除最早的消息。
        """
        self.max_messages = max_message
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []

    def add_message(self, message: dict[str, Any]) -> None:
        """添加一条消息到短期记忆。

        Args:
            message: OpenAI 消息格式字典。
        """
        self.messages.append(message)

        # 超出上限时移除最早的消息
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

        # TODO: token 计数截断逻辑

    def get_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        """获取最近的对话消息。

        Args:
            limit: 限制返回的最大条数；None 表示返回全部。

        Returns:
            按时间顺序排列的消息列表。
        """
        if limit is not None:
            return self.messages[-limit:]
        return self.messages

    def clear(self) -> None:
        """清空所有对话消息。"""
        self.messages = []


class MemoryManager:
    """记忆管理器：结合短期和长期记忆（待实现）"""


@dataclass(frozen=True)
class MemoryEntry:
    """长期记忆中的一条记录，可序列化存储"""

    id: str
    role: str
    content: str
    timestamp: float
    distance: float | None = None  # 与查询的 L2 距离，越小越相关；非检索场景为 None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，与 from_dict 保证往返一致。"""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "distance": self.distance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """从字典反序列化。

        Args:
            data: to_dict 产出的字典（兼容缺少 distance 字段的旧数据）。

        Returns:
            MemoryEntry 实例。
        """
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            distance=data.get("distance"),
        )


class LongTermMemory(Memory):
    """长期记忆：基于向量检索的语义信息。

    写入的消息以 content 作为向量化文本（由 chromadb 默认嵌入函数处理），
    metadata 中存 role 与写入时间戳，支持按语义相似度检索。
    """

    def __init__(self, vector_db: "chromadb.Client", collection_name: str) -> None:
        """初始化长期记忆。

        Args:
            vector_db: chromadb.Client 实例。
            collection_name: 向量库集合名称，不存在时自动创建。

        Raises:
            MemoryError: 集合创建/获取失败时抛出。
        """
        self.vector_db = vector_db
        self.collection_name = collection_name
        try:
            self.collection = vector_db.get_or_create_collection(name=collection_name)
        except Exception as e:
            raise MemoryError(f"创建向量集合 {collection_name} 失败: {e}") from e

    def add_message(self, message: dict[str, Any]) -> None:
        """添加一条消息到长期记忆。

        Args:
            message: OpenAI 消息格式字典，须包含 role 与非空 content 字段。

        Raises:
            ValueError: message 缺少 role 或 content 字段时抛出。
            MemoryError: 向量库写入失败时抛出。
        """
        role = message.get("role")
        content = message.get("content")
        if not role or not content:
            raise ValueError("message 须包含非空的 role 与 content 字段")
        try:
            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[content],
                metadatas=[{"role": role, "timestamp": time.time()}],
            )
        except Exception as e:
            raise MemoryError(f"长期记忆写入失败: {e}") from e

    def get_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        """获取长期记忆中的全部消息，按写入时间升序。

        Args:
            limit: 限制返回的最大条数（取最近 N 条）；None 表示返回全部。

        Returns:
            消息字典列表，含 role / content / timestamp 字段。

        Raises:
            MemoryError: 向量库读取失败时抛出。
        """
        try:
            result = self.collection.get(include=["metadatas", "documents"])
        except Exception as e:
            raise MemoryError(f"长期记忆读取失败: {e}") from e
        messages = sorted(
            (
                {"role": meta["role"], "content": doc, "timestamp": meta["timestamp"]}
                for meta, doc in zip(result["metadatas"], result["documents"])
            ),
            key=lambda item: item["timestamp"],
        )
        if limit is not None:
            messages = messages[-limit:]
        return messages

    def get_relevant_messages(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """按语义相似度检索与查询最相关的记忆。

        Args:
            query: 自然语言查询文本。
            limit: 返回的最大条数，默认 5。

        Returns:
            MemoryEntry 列表，按相关度从高到低（distance 升序）排列。

        Raises:
            MemoryError: 向量库检索失败时抛出。
        """
        try:
            result = self.collection.query(query_texts=[query], n_results=limit)
        except Exception as e:
            raise MemoryError(f"长期记忆检索失败: {e}") from e
        ids, documents = result["ids"][0], result["documents"][0]
        metadatas, distances = result["metadatas"][0], result["distances"][0]
        return [
            MemoryEntry(
                id=entry_id,
                role=meta["role"],
                content=doc,
                timestamp=meta["timestamp"],
                distance=distance,
            )
            for entry_id, doc, meta, distance in zip(ids, documents, metadatas, distances)
        ]

    def clear(self) -> None:
        """清空长期记忆中的所有记录。

        Raises:
            MemoryError: 向量库删除失败时抛出。
        """
        try:
            ids = self.collection.get(include=[])["ids"]
            if ids:  # 部分 chromadb 版本对空 ids 列表报错
                self.collection.delete(ids=ids)
        except Exception as e:
            raise MemoryError(f"长期记忆清空失败: {e}") from e
