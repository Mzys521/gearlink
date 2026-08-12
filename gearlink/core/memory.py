import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gearlink.exceptions import MemoryError
from gearlink.utils.token_count import count_message_tokens, estimate_tokens

if TYPE_CHECKING:
    import chromadb

#: 模块级日志器：记录短期记忆截断、长期记忆沉淀/检索与上下文组装过程
logger = logging.getLogger(__name__)

#: 日志中展示消息内容的预览长度，避免超长内容刷屏
_LOG_PREVIEW_LENGTH = 60


def _preview(text: str, length: int = _LOG_PREVIEW_LENGTH) -> str:
    """截断文本用于日志展示。

    Args:
        text: 原始文本。
        length: 预览长度上限。

    Returns:
        截断后的预览文本，超出部分以省略号结尾。
    """
    if len(text) <= length:
        return text
    return text[:length] + "..."


#: 长期记忆检索结果在上下文总预算中的最大占比，超出部分按相关度裁剪
RETRIEVAL_BUDGET_RATIO = 0.25

#: 检索注入时各角色对应的说话人显示名（未知角色回退为原始 role 值）
_ROLE_LABELS = {"user": "用户", "assistant": "助手"}

#: 时间衰减归一化周期：30 天，用于把条目年龄折算到 [0,1] 区间
_RECENCY_NORM_SECONDS = 30 * 24 * 60 * 60

#: 会话摘要沉淀时写入长期记忆的内容前缀，用于区分摘要与原始消息
_SESSION_SUMMARY_PREFIX = "[会话摘要] "


def _keep_within_budget(
    items: Sequence[Any],
    quota: int,
    tokens_of: Callable[[Any], int],
    *,
    always_keep_newest: bool = False,
) -> list[int]:
    """从最新一项向前贪心填充，返回不超出 token 预算的项索引（升序）。

    各处上下文裁剪（检索预算、短期消息、非检索 system 消息）共用的统一策略：
    按最新优先保留，累计 token 超出预算即停止。

    Args:
        items: 按最旧到最新排列的待裁剪项。
        quota: token 预算上限。
        tokens_of: 计算单项 token 数的函数。
        always_keep_newest: 最新一项即使单独超出预算也保留（软约束，
            避免裁剪出空结果）。

    Returns:
        保留项在 items 中的索引列表，按升序排列。
    """
    kept: list[int] = []
    total = 0
    for index in range(len(items) - 1, -1, -1):
        tokens = tokens_of(items[index])
        if total + tokens > quota and (kept or not always_keep_newest):
            break
        kept.append(index)
        total += tokens
    kept.reverse()
    return kept


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
            max_tokens: 保留的最大 token 数；超出时从最旧消息开始移除，
                至少保留最新一条消息。None 表示不按 token 数裁剪。
            max_message: 保留的最大消息数，超出时移除最早的消息。
        """
        self.max_messages = max_message
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []

    def add_message(self, message: dict[str, Any]) -> None:
        """添加一条消息到短期记忆。

        超出 max_message 条数或 max_tokens 预算时，从最旧消息开始移除；
        单条消息自身超出预算时仍保留最新一条，避免清空记忆。

        Args:
            message: OpenAI 消息格式字典。
        """
        self.messages.append(message)

        # 超出条数上限时移除最早的消息
        if self.max_messages and len(self.messages) > self.max_messages:
            removed = len(self.messages) - self.max_messages
            self.messages = self.messages[-self.max_messages :]
            logger.debug(
                "短期记忆按条数上限截断: 移除 %d 条最旧消息（上限 %d 条）",
                removed,
                self.max_messages,
            )

        # 超出 token 预算时按总量递减，从最旧消息开始移除
        if self.max_tokens:
            total = sum(count_message_tokens(m) for m in self.messages)
            removed = 0
            while total > self.max_tokens and len(self.messages) > 1:
                removed_message = self.messages.pop(0)
                total -= count_message_tokens(removed_message)
                removed += 1
            if removed:
                logger.debug(
                    "短期记忆按 token 预算截断: 移除 %d 条最旧消息（预算 %d token）",
                    removed,
                    self.max_tokens,
                )

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
    - 配置 ``max_context_tokens`` 后，组装结果按「系统消息优先、长期检索固定
      占比、短期对话从最新保留」的顺序裁剪，确保请求不超出上下文预算。

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
        max_context_tokens: int | None = None,
        summarizer: Callable[[str], str] | None = None,
        system_budget_ratio: float = 1.0,
    ) -> None:
        """初始化记忆管理器。

        Args:
            short_term: 短期记忆实例，承载当下对话消息。
            long_term: 长期记忆实例；None 表示不启用语义检索与沉淀。
            relevant_limit: build_context 检索长期记忆时的默认 top-k 条数。
            max_context_tokens: 组装上下文的 token 预算上限；None 表示不设限。
                预算分配顺序为系统消息优先保留、长期检索占固定比例
                （RETRIEVAL_BUDGET_RATIO）、短期对话从最新消息向前填充。
            summarizer: 会话摘要生成器 `Callable[[str], str]`；注入时
                :meth:`end_session` 会先把会话文本拼成 transcript 交由它生成摘要
                沉淀进长期记忆。None 表示不启用摘要沉淀（保持现状）。
                摘要依赖模型生成，具体实现由应用层注入（core 不直接依赖 providers）。
            system_budget_ratio: 非检索 system 消息占上下文预算的比例上限，
                取值 (0, 1.0]；1.0 表示不裁剪（现状）。检索注入的 system 消息
                已由 RETRIEVAL_BUDGET_RATIO 独立保护，不参与本配额裁剪。
        """
        self.short_term = short_term
        self.long_term = long_term
        self.relevant_limit = relevant_limit
        self.max_context_tokens = max_context_tokens
        self.summarizer = summarizer
        self.system_budget_ratio = system_budget_ratio

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
            logger.debug(
                "沉淀到长期记忆: role=%s, content=%s",
                message["role"],
                _preview(message["content"]),
            )
            self.long_term.add_message({"role": message["role"], "content": message["content"]})
        elif self.long_term is not None:
            logger.debug("跳过沉淀: role=%r（非 user/assistant 或空内容）", message.get("role"))

    def end_session(self) -> None:
        """结束会话：沉淀会话摘要（可选），补沉淀剩余消息，然后清空短期记忆。

        与 :meth:`add_message` 的即时沉淀配合使用：user/assistant 文本消息已沉淀，
        此处主要补齐 system/tool 等会话过程消息，保证长期记忆不丢失本轮上下文。
        配置了 ``summarizer`` 时，会先把短期记忆中的 user/assistant 文本拼成
        transcript 交由它生成会话摘要并沉淀进长期记忆（失败不中断会话结束）。
        未配置长期记忆时仅清空短期记忆。

        Raises:
            MemoryError: 长期记忆写入失败时抛出。
        """
        if self.long_term is not None and self.summarizer is not None:
            transcript = self._build_transcript()
            if transcript:
                try:
                    summary = self.summarizer(transcript)
                except Exception:
                    logger.warning("会话摘要生成失败，跳过摘要沉淀", exc_info=True)
                else:
                    self.long_term.add_message(
                        {
                            "role": "assistant",
                            "content": f"{_SESSION_SUMMARY_PREFIX}{summary}",
                        }
                    )
                    logger.info("会话结束: 沉淀会话摘要到长期记忆（%d 字）", len(summary))

        if self.long_term is not None:
            sedimented = 0
            for message in self.short_term.get_messages():
                content = message.get("content")
                if content:
                    self.long_term.add_message(
                        {"role": message.get("role", "user"), "content": content}
                    )
                    sedimented += 1
            if sedimented:
                logger.info("会话结束: 补沉淀 %d 条短期记忆到长期记忆", sedimented)
        self.short_term.clear()

    def _build_transcript(self) -> str:
        """将短期记忆中的 user/assistant 文本消息拼成会话文本。

        格式为逐行 ``用户: ...`` / ``助手: ...``，仅包含有非空内容的文本消息；
        无任何可拼接消息时返回空字符串。

        Returns:
            会话 transcript 文本。
        """
        lines = []
        for message in self.short_term.get_messages():
            role = message.get("role")
            content = message.get("content")
            if role in self._SEDIMENT_ROLES and content:
                label = _ROLE_LABELS.get(role, role)
                lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def build_context(self, query: str, relevant_limit: int | None = None) -> list[dict[str, Any]]:
        """组装本轮请求的消息列表：长期记忆检索结果注入 + 短期记忆全量。

        检索到的相关历史以 system 消息形式插入短期消息头部，每条带说话人标签
        （如 [用户] / [助手]），且过滤掉与短期窗口内容重复的条目；未配置长期
        记忆或 query 为空时仅返回短期记忆全量消息。配置了 max_context_tokens
        时，按「系统消息优先保留、长期检索占固定比例、短期消息从最新保留」
        的顺序裁剪，确保请求不超出上下文预算；配置了 system_budget_ratio 时，
        检索注入之外的 system 消息还会额外按比例配额裁剪（检索注入消息受保护）。

        Args:
            query: 语义检索查询文本（通常为本轮用户输入）。
            relevant_limit: 检索 top-k 条数；None 表示使用构造时的默认值。

        Returns:
            可直接传给 ModelProvider.chat 的消息列表。

        Raises:
            MemoryError: 长期记忆检索失败时抛出。
        """
        messages: list[dict[str, Any]] = []
        protected_system: dict[str, Any] | None = None

        if self.long_term is not None and query:
            limit = self.relevant_limit if relevant_limit is None else relevant_limit
            entries = self.long_term.get_relevant_messages(query, limit=limit)
            logger.debug("长期记忆检索: query=%r, top-k=%d, 命中 %d 条", query, limit, len(entries))
            entries = self._dedupe_against_short_term(entries)
            entries = self._fit_retrieval_budget(entries)
            if entries:
                retrieval_content = "以下是与当前问题相关的历史记忆：\n" + "\n".join(
                    self._format_retrieved_entry(entry) for entry in entries
                )
                protected_system = {"role": "system", "content": retrieval_content}
                messages.append(protected_system)
                logger.info(
                    "注入长期记忆 %d 条（约 %d token）: %s",
                    len(entries),
                    estimate_tokens(retrieval_content),
                    " | ".join(_preview(entry.content) for entry in entries),
                )

        messages.extend(self.short_term.get_messages())

        if self.max_context_tokens is not None:
            messages = self._trim_to_budget(messages, self.max_context_tokens)
            if self.system_budget_ratio < 1.0:
                messages = self._trim_system_to_ratio(messages, protected_system)
        return messages

    def _dedupe_against_short_term(self, entries: "list[MemoryEntry]") -> "list[MemoryEntry]":
        """过滤掉与短期窗口内容重复的检索条目，避免同一内容注入两次。

        短期与长期记忆共享同一写入内容（即时沉淀），此处做精确内容匹配去重，
        消除 top-k 检索与滑窗之间天然重叠的上下文冗余。

        Args:
            entries: 检索结果列表。

        Returns:
            去除与短期记忆内容重复后的条目列表。
        """
        short_term_contents = {
            message.get("content")
            for message in self.short_term.get_messages()
            if message.get("content")
        }
        kept: list[MemoryEntry] = []
        for entry in entries:
            if entry.content in short_term_contents:
                logger.debug(
                    "检索去重: 过滤与短期窗口重复的记忆 role=%s content=%s",
                    entry.role,
                    _preview(entry.content),
                )
                continue
            kept.append(entry)
        if len(kept) != len(entries):
            logger.info(
                "检索去重: 过滤 %d 条与短期窗口重复的记忆，保留 %d 条",
                len(entries) - len(kept),
                len(kept),
            )
        return kept

    def _format_retrieved_entry(self, entry: "MemoryEntry") -> str:
        """将一条检索到的长期记忆格式化为带说话人标签的文本。

        Args:
            entry: 检索结果条目。

        Returns:
            形如 ``- [用户]：内容`` 的注入文本；未知角色回退为原始 role 值。
        """
        label = _ROLE_LABELS.get(entry.role, entry.role)
        return f"- [{label}]：{entry.content}"

    def _fit_retrieval_budget(self, entries: "list[MemoryEntry]") -> "list[MemoryEntry]":
        """按检索预算裁剪长期记忆条目，超出部分从最不相关的一端裁剪。

        Args:
            entries: 按相关度从高到低排列的检索结果。

        Returns:
            预算内保留的条目；未配置 max_context_tokens 时原样返回。
        """
        if self.max_context_tokens is None:
            return entries
        quota = int(self.max_context_tokens * RETRIEVAL_BUDGET_RATIO)
        # 首条即使超预算也保留（软约束），其余超出即停止
        kept_indices = _keep_within_budget(
            entries, quota, lambda entry: estimate_tokens(entry.content), always_keep_newest=True
        )
        kept = [entries[index] for index in kept_indices]
        logger.debug(
            "检索预算裁剪: 配额 %d token（总预算 %d），保留 %d/%d 条",
            quota,
            self.max_context_tokens,
            len(kept),
            len(entries),
        )
        return kept

    def _trim_to_budget(self, messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        """在 token 预算内裁剪消息：system 消息始终保留，其余从最旧开始丢弃。

        兜底策略：若裁剪后没有任何消息（无 system 消息且其余全部超出预算），
        保留最新一条，避免向模型发出空请求。

        Args:
            messages: 组装完成的消息列表。
            budget: 上下文总预算。

        Returns:
            裁剪后不超出预算的消息列表。
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        others = [m for m in messages if m.get("role") != "system"]
        before_tokens = sum(count_message_tokens(m) for m in messages)
        remaining = budget - sum(count_message_tokens(m) for m in system_msgs)
        kept_indices = _keep_within_budget(others, remaining, count_message_tokens)
        # 兜底：整体为空（无 system 且其余全超预算）时保留最新一条，避免空请求
        if not kept_indices and not system_msgs and others:
            kept_indices = [len(others) - 1]

        messages = system_msgs + [others[index] for index in kept_indices]
        logger.info(
            "上下文预算裁剪: 预算 %d token，裁剪前 %d 条（约 %d token），"
            "裁剪后 %d 条（约 %d token）",
            budget,
            len(system_msgs) + len(others),
            before_tokens,
            len(messages),
            sum(count_message_tokens(m) for m in messages),
        )
        return messages

    def _trim_system_to_ratio(
        self,
        messages: list[dict[str, Any]],
        protected: "dict[str, Any] | None" = None,
    ) -> list[dict[str, Any]]:
        """按 system_budget_ratio 配额裁剪「非检索」system 消息。

        检索注入的 system 消息（受保护）已由 RETRIEVAL_BUDGET_RATIO 独立管控，
        不参与本配额裁剪；对其余 system 消息从最旧开始丢弃，直至总 token 不超过
        ``max_context_tokens * system_budget_ratio``。兜底保留最新一条，避免清空提示。

        Args:
            messages: 已按总预算裁剪过的消息列表。
            protected: 受保护的检索注入 system 消息对象；None 表示本轮无检索注入。

        Returns:
            裁剪非检索 system 后的消息列表。
        """
        quota = int(self.max_context_tokens * self.system_budget_ratio)
        trimmable = [m for m in messages if m.get("role") == "system" and m is not protected]
        if not trimmable:
            return messages
        # 从最旧开始丢弃直至配额内，兜底保留最新一条（软约束）
        kept_indices = _keep_within_budget(
            trimmable, quota, count_message_tokens, always_keep_newest=True
        )
        if len(kept_indices) == len(trimmable):
            return messages
        kept_ids = {id(trimmable[index]) for index in kept_indices}
        drop_ids = {id(m) for m in trimmable} - kept_ids

        result = [m for m in messages if id(m) not in drop_ids]
        logger.info(
            "system 预算裁剪: 配额 %d token（ratio=%s，总预算 %d），"
            "丢弃 %d 条非检索 system，保留 %d 条",
            quota,
            self.system_budget_ratio,
            self.max_context_tokens,
            len(trimmable) - len(kept_indices),
            len(kept_indices),
        )
        return result

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
    metadata 中存 role、写入时间戳与 content 的 sha256 哈希，支持按语义相似度检索。

    可选能力（均默认关闭，等价于纯向量检索现状）：

    - ``recency_weight``：检索排序引入时间衰减，让新近记忆适度排前；
    - ``max_entries``：容量上限，超出时按写入时间淘汰最旧条目；
    - ``dedupe``：写入前按内容哈希去重，避免同内容反复堆积。
    """

    def __init__(
        self,
        vector_db: "chromadb.Client",
        collection_name: str,
        recency_weight: float = 0.0,
        max_entries: int | None = None,
        dedupe: bool = False,
    ) -> None:
        """初始化长期记忆。

        Args:
            vector_db: chromadb.Client 实例。
            collection_name: 向量库集合名称，不存在时自动创建。
            recency_weight: 检索排序中新旧衰减权重；0.0 表示纯相关度排序。
                大于 0 时按 ``score = distance + recency_weight * age_ratio`` 重排，
                age_ratio 为条目年龄相对 30 天归一化的 [0,1] 值。
            max_entries: 长期库容量上限；写入超出时淘汰时间戳最旧的条目。
                None 表示不设限。
            dedupe: 写入前去重开关；True 时先按 content 的 sha256 哈希查询，
                命中则跳过本次写入。False 表示原样写入。

        Raises:
            MemoryError: 集合创建/获取失败时抛出。
        """
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.recency_weight = recency_weight
        self.max_entries = max_entries
        self.dedupe = dedupe
        try:
            self.collection = vector_db.get_or_create_collection(name=collection_name)
        except Exception as e:
            raise MemoryError(f"创建向量集合 {collection_name} 失败: {e}") from e

    def add_message(self, message: dict[str, Any]) -> None:
        """添加一条消息到长期记忆。

        写入的 metadata 恒包含 content 的 sha256 哈希（content_hash），供去重查询使用。
        配置了 ``dedupe`` 时，写入前先按 content_hash 查询，命中则跳过本次写入。
        写入后若配置了 ``max_entries`` 且条目数超出上限，按时间戳淘汰最旧的
        超出部分，保证库容量不超过上限。

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
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if self.dedupe:
            try:
                hits = self.collection.get(where={"content_hash": content_hash})["ids"]
            except Exception as e:
                raise MemoryError(f"长期记忆去重查询失败: {e}") from e
            if hits:
                logger.debug("长期记忆去重: 内容已存在（hash=%s），跳过写入", content_hash[:8])
                return

        try:
            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[content],
                metadatas=[
                    {
                        "role": role,
                        "timestamp": time.time(),
                        "content_hash": content_hash,
                    }
                ],
            )
        except Exception as e:
            raise MemoryError(f"长期记忆写入失败: {e}") from e

        if self.max_entries is not None:
            self._evict_over_max_entries()

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

        返回前会按 content 精确去重（保留相关度首个），避免重复内容注入上下文。
        配置了 ``recency_weight`` 时，去重后结果会按
        ``score = distance + recency_weight * age_ratio`` 重排，新近记忆适度排前。

        Args:
            query: 自然语言查询文本。
            limit: 返回的最大条数，默认 5。

        Returns:
            MemoryEntry 列表，按相关度从高到低（distance 升序，开启时间衰减后
            按综合得分升序）排列。

        Raises:
            MemoryError: 向量库检索失败时抛出。
        """
        try:
            result = self.collection.query(query_texts=[query], n_results=limit)
        except Exception as e:
            raise MemoryError(f"长期记忆检索失败: {e}") from e
        ids, documents = result["ids"][0], result["documents"][0]
        metadatas, distances = result["metadatas"][0], result["distances"][0]
        entries = [
            MemoryEntry(
                id=entry_id,
                role=meta["role"],
                content=doc,
                timestamp=meta["timestamp"],
                distance=distance,
            )
            for entry_id, doc, meta, distance in zip(ids, documents, metadatas, distances)
        ]
        entries = self._dedupe_by_content(entries)
        if self.recency_weight > 0:
            entries = self._sort_by_recency(entries)
        return entries

    def _dedupe_by_content(self, entries: "list[MemoryEntry]") -> "list[MemoryEntry]":
        """按 content 精确去重检索结果，保留相关度首个。

        向量库中同一文本可能被写入多次（如 dedupe 未开启时的历史堆积），
        检索去重避免同一内容重复注入上下文。

        Args:
            entries: 按相关度从高到低排列的检索结果。

        Returns:
            去除重复 content 后的条目列表，顺序不变。
        """
        seen: set[str] = set()
        kept: list[MemoryEntry] = []
        for entry in entries:
            if entry.content in seen:
                logger.debug("检索去重: 过滤重复内容 %s", _preview(entry.content))
                continue
            seen.add(entry.content)
            kept.append(entry)
        if len(kept) != len(entries):
            logger.info(
                "检索去重: 过滤 %d 条重复记忆，保留 %d 条",
                len(entries) - len(kept),
                len(kept),
            )
        return kept

    def _sort_by_recency(self, entries: "list[MemoryEntry]") -> "list[MemoryEntry]":
        """按时间衰减综合得分对检索结果重新排序。

        得分 ``score = distance + recency_weight * age_ratio``，age_ratio 为条目
        年龄相对 30 天归一化的 [0,1] 值；得分越低越靠前。

        Args:
            entries: 按相关度排列的检索结果。

        Returns:
            按综合得分升序重排后的条目列表。
        """
        now = time.time()
        scored = []
        for entry in entries:
            age_seconds = max(now - entry.timestamp, 0.0)
            age_ratio = min(age_seconds / _RECENCY_NORM_SECONDS, 1.0)
            distance = entry.distance if entry.distance is not None else 0.0
            scored.append((distance + self.recency_weight * age_ratio, entry))
        scored.sort(key=lambda item: item[0])
        return [entry for _, entry in scored]

    def _evict_over_max_entries(self) -> None:
        """超出容量上限时，按时间戳淘汰最旧的条目。

        仅在配置了 ``max_entries`` 且当前条目数超过上限时执行淘汰。

        Raises:
            MemoryError: 向量库读取/删除失败时抛出。
        """
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as e:
            raise MemoryError(f"长期记忆读取失败: {e}") from e
        ids, metadatas = result["ids"], result["metadatas"]
        if len(ids) <= self.max_entries:
            return
        excess = len(ids) - self.max_entries
        oldest = sorted(zip(ids, metadatas), key=lambda item: item[1]["timestamp"])[:excess]
        try:
            self.collection.delete(ids=[entry_id for entry_id, _ in oldest])
        except Exception as e:
            raise MemoryError(f"长期记忆淘汰失败: {e}") from e
        logger.info(
            "长期记忆容量上限: 淘汰 %d 条最旧记录（上限 %d 条）",
            excess,
            self.max_entries,
        )

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
