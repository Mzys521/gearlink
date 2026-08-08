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
    """记忆管理器：组合短期与长期记忆，统一为 Agent 提供上下文。

    职责划分：

    - 短期记忆管「当下对话」：所有消息先写入短期记忆，每轮全量注入模型请求。
    - 长期记忆管「沉淀知识」：user/assistant 的文本消息在写入时即选择性沉淀；
      会话结束时可调用 :meth:`end_session` 将短期记忆中剩余的消息补沉淀后清空短期。
    - :meth:`build_context` 将长期记忆的 top-k 语义检索结果以 system 消息注入
      短期消息头部，组装出本轮请求的完整消息列表。

    注意：本类是组合器而非 ``Memory`` 子类——它不对齐 ``Memory`` 抽象的三方法签名，
    而是提供更贴合「短期 + 长期」场景的专用接口（`build_context` / `end_session`）。
    """

    #: 写入时即沉淀到长期记忆的消息角色（system/tool 消息不沉淀）
    _SEDIMENT_ROLES = ("user", "assistant")

    def __init__(
        self,
        short_term: Memory,
        long_term: "LongTermMemory | None" = None,
        relevant_limit: int = 5,
    ) -> None:
        """初始化记忆管理器。

        Args:
            short_term: 短期记忆实例，承载当下对话消息。
            long_term: 长期记忆实例；None 表示不启用语义检索与沉淀。
            relevant_limit: build_context 检索长期记忆时的默认 top-k 条数。
        """
        self.short_term = short_term
        self.long_term = long_term
        self.relevant_limit = relevant_limit

    def add_message(self, message: dict[str, Any]) -> None:
        """写入一条消息：必进短期记忆，按策略选择性沉淀到长期记忆。

        沉淀策略：role 为 user/assistant 且含非空 content 的文本消息立即沉淀；
        system/tool 消息与工具调用中间态不沉淀（避免噪声）。

        Args:
            message: OpenAI 消息格式字典，须包含 role 字段。

        Raises:
            MemoryError: 长期记忆写入失败时抛出（短期记忆已写入成功）。
        """
        self.short_term.add_message(message)

        if (
            self.long_term is not None
            and message.get("role") in self._SEDIMENT_ROLES
            and message.get("content")
        ):
            self.long_term.add_message({"role": message["role"], "content": message["content"]})

    def end_session(self) -> None:
        """结束会话：将短期记忆中尚未沉淀的消息补写入长期记忆，然后清空短期记忆。

        与 :meth:`add_message` 的即时沉淀配合使用：user/assistant 文本消息已沉淀，
        此处主要补齐 system/tool 等会话过程消息，保证长期记忆不丢失本轮上下文。
        未配置长期记忆时仅清空短期记忆。

        Raises:
            MemoryError: 长期记忆写入失败时抛出。
        """
        if self.long_term is not None:
            for message in self.short_term.get_messages():
                content = message.get("content")
                if content:
                    self.long_term.add_message(
                        {"role": message.get("role", "user"), "content": content}
                    )
        self.short_term.clear()

    def build_context(self, query: str, relevant_limit: int | None = None) -> list[dict[str, Any]]:
        """组装本轮请求的消息列表：长期记忆检索结果注入 + 短期记忆全量。

        检索到的相关历史以 system 消息形式插入短期消息头部；未配置长期记忆
        或 query 为空时仅返回短期记忆全量消息。

        Args:
            query: 语义检索查询文本（通常为本轮用户输入）。
            relevant_limit: 检索 top-k 条数；None 表示使用构造时的默认值。

        Returns:
            可直接传给 ModelProvider.chat 的消息列表。

        Raises:
            MemoryError: 长期记忆检索失败时抛出。
        """
        messages: list[dict[str, Any]] = []

        if self.long_term is not None and query:
            limit = self.relevant_limit if relevant_limit is None else relevant_limit
            entries = self.long_term.get_relevant_messages(query, limit=limit)
            if entries:
                messages.append(
                    {
                        "role": "system",
                        "content": "以下是与当前问题相关的历史记忆：\n"
                        + "\n".join(f"- {entry.content}" for entry in entries),
                    }
                )

        messages.extend(self.short_term.get_messages())
        return messages

    def clear(self) -> None:
        """清空短期与长期记忆中的所有内容。

        Raises:
            MemoryError: 长期记忆清空失败时抛出。
        """
        self.short_term.clear()
        if self.long_term is not None:
            self.long_term.clear()


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
