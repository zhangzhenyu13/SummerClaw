"""GroupChat 协调空间 —— 所有 Agent 的公共对话通道。

完全去中心化设计：不设中央调度器，Agent 自由在频道中发言。
提供消息历史管理和轮次截断 + 摘要压缩策略以控制 token 爆炸。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """GroupChat 消息类型。"""
    SYSTEM = "system"              # 系统消息（任务广播、状态通知）
    NOMINATION = "nomination"      # Agent 提名发言
    REVIEW = "review"              # Agent 互评发言
    VOTE = "vote"                  # 投票消息
    RESULT = "result"              # 选举结果
    MEMORY_REPORT = "memory_report"  # 记忆报告
    SUMMARY = "summary"            # 整合摘要
    MAINTENANCE = "maintenance"    # 维护辩论
    INFO = "info"                  # 一般信息


@dataclass
class ChatMessage:
    """GroupChat 中的一条消息。"""
    id: str
    sender_id: str                # 发送者 ID（"system" 表示系统消息）
    sender_name: str              # 发送者名称
    msg_type: MessageType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    round_number: int = 0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "msg_type": self.msg_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "round_number": self.round_number,
        }

    def to_prompt_text(self) -> str:
        """转为 LLM 可读的文本格式。"""
        header = f"[{self.sender_name}]"
        if self.msg_type == MessageType.SYSTEM:
            return f"📢 **系统广播**：{self.content}"
        elif self.msg_type == MessageType.NOMINATION:
            return f"🗳️ {header} 提名发言：\n{self.content}"
        elif self.msg_type == MessageType.REVIEW:
            return f"💬 {header} 评审意见：\n{self.content}"
        elif self.msg_type == MessageType.MEMORY_REPORT:
            return f"📋 {header} 记忆报告：\n{self.content}"
        elif self.msg_type == MessageType.RESULT:
            return f"✅ **选举结果**：{self.content}"
        elif self.msg_type == MessageType.SUMMARY:
            return f"📊 **整合摘要**：\n{self.content}"
        else:
            return f"{header}：{self.content}"


class GroupChat:
    """去中心化的 GroupChat 协调空间。

    Agent 在此频道中自由发言、互评、投票。
    提供轮次截断和摘要压缩功能以控制上下文长度。

    Parameters:
        max_history_rounds: 保留的最近轮次数（超出部分会被摘要压缩）
        max_context_chars: 上下文最大字符数（约等于 token 数的 1/4）
    """

    def __init__(
        self,
        max_history_rounds: int = 20,
        max_context_chars: int = 32000,
    ) -> None:
        self.max_history_rounds = max_history_rounds
        self.max_context_chars = max_context_chars
        self._messages: list[ChatMessage] = []
        self._round_count = 0
        self._compressed_summary: str = ""
        self._compressed_rounds: int = 0

    # ── 消息管理 ─────────────────────────────────────────────────────

    def broadcast(
        self,
        sender_id: str,
        sender_name: str,
        content: str,
        msg_type: MessageType = MessageType.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """向 GroupChat 广播一条消息。

        Args:
            sender_id: 发送者 ID
            sender_name: 发送者名称
            content: 消息内容
            msg_type: 消息类型
            metadata: 附加元数据

        Returns:
            创建的 ChatMessage 对象
        """
        import uuid

        msg = ChatMessage(
            id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            sender_name=sender_name,
            msg_type=msg_type,
            content=content,
            metadata=metadata or {},
            round_number=self._round_count,
        )
        self._messages.append(msg)
        self._maybe_compress()
        return msg

    def system_broadcast(self, content: str) -> ChatMessage:
        """发送系统广播消息。"""
        self._round_count += 1
        return self.broadcast(
            sender_id="system",
            sender_name="系统",
            content=content,
            msg_type=MessageType.SYSTEM,
        )

    def advance_round(self) -> int:
        """进入下一轮对话。"""
        self._round_count += 1
        return self._round_count

    # ── 历史查询 ─────────────────────────────────────────────────────

    def get_messages(
        self,
        msg_type: MessageType | None = None,
        last_n_rounds: int | None = None,
        include_compressed: bool = False,
    ) -> list[ChatMessage]:
        """获取消息历史。

        Args:
            msg_type: 按类型过滤（None 表示不过滤）
            last_n_rounds: 仅返回最后 N 轮的消息
            include_compressed: 是否包含已被压缩的消息

        Returns:
            消息列表
        """
        messages = self._messages

        if msg_type is not None:
            messages = [m for m in messages if m.msg_type == msg_type]

        if last_n_rounds is not None:
            min_round = self._round_count - last_n_rounds
            messages = [m for m in messages if m.round_number >= max(min_round, 0)]

        return messages

    def get_recent_history(self, n_rounds: int = 5) -> str:
        """获取最近 N 轮的格式化历史文本。"""
        if self._compressed_summary:
            parts = [f"## 历史摘要（第 1-{self._compressed_rounds} 轮）\n{self._compressed_summary}"]
        else:
            parts = []

        recent = self.get_messages(last_n_rounds=n_rounds)
        if recent:
            parts.append(f"## 最近 {n_rounds} 轮对话")
            current_round = -1
            for m in recent:
                if m.round_number != current_round:
                    current_round = m.round_number
                    parts.append(f"\n### 第 {current_round} 轮")
                parts.append(m.to_prompt_text())

        return "\n".join(parts)

    def get_nomination_messages(self, round_number: int | None = None) -> list[ChatMessage]:
        """获取提名消息。"""
        msgs = self.get_messages(msg_type=MessageType.NOMINATION)
        if round_number is not None:
            msgs = [m for m in msgs if m.round_number == round_number]
        return msgs

    # ── 压缩策略 ─────────────────────────────────────────────────────

    def _maybe_compress(self) -> None:
        """检查是否需要压缩历史消息。"""
        total_chars = sum(len(m.content) for m in self._messages)
        if total_chars > self.max_context_chars:
            self._compress_old_rounds()

    def _compress_old_rounds(self) -> None:
        """压缩最旧的轮次为摘要，替换原始消息。"""
        if len(self._messages) <= self.max_history_rounds * 2:
            return

        # 找出需要压缩的轮次范围
        rounds = sorted(set(m.round_number for m in self._messages))
        if len(rounds) <= self.max_history_rounds:
            return

        # 保留最近的 max_history_rounds 轮
        keep_rounds = set(rounds[-self.max_history_rounds:])
        to_compress = [m for m in self._messages if m.round_number not in keep_rounds]

        if not to_compress:
            return

        # 生成摘要
        summary_parts = []
        compressed_first_round = min(m.round_number for m in to_compress)
        compressed_last_round = max(m.round_number for m in to_compress)

        for m in to_compress:
            if m.msg_type == MessageType.NOMINATION:
                summary_parts.append(f"- {m.sender_name} 提名参与（关联度见元数据）")
            elif m.msg_type == MessageType.MEMORY_REPORT:
                summary_parts.append(f"- {m.sender_name} 提交了记忆报告")
            elif m.msg_type == MessageType.RESULT:
                summary_parts.append(f"- 选举结果：{m.content[:100]}")

        self._compressed_summary = (
            f"第 {compressed_first_round}-{compressed_last_round} 轮摘要：\n"
            + "\n".join(summary_parts[:30])
        )
        self._compressed_rounds = compressed_last_round

        # 移除被压缩的消息
        self._messages = [m for m in self._messages if m.round_number in keep_rounds]

    # ── LLM 上下文构建 ──────────────────────────────────────────────

    def to_llm_context(self, max_rounds: int = 10) -> str:
        """构建完整的 LLM 可读上下文，用于驱动协商和决策。

        将 GroupChat 中的全部消息（含压缩摘要）格式化为 LLM prompt 片段，
        让 LLM 能够全面感知对话历史和各 Agent 的发言。

        Args:
            max_rounds: 包含的最近轮次数

        Returns:
            格式化的 LLM 上下文文本
        """
        parts: list[str] = []

        if self._compressed_summary:
            parts.append(f"## 历史对话摘要（第 1-{self._compressed_rounds} 轮）\n{self._compressed_summary}")

        recent = self.get_messages(last_n_rounds=max_rounds)
        if recent:
            parts.append(f"## 最近 {min(max_rounds, len(set(m.round_number for m in recent)))} 轮对话")
            current_round = -1
            for m in recent:
                if m.round_number != current_round:
                    current_round = m.round_number
                    parts.append(f"\n### 第 {current_round} 轮")
                parts.append(m.to_prompt_text())

        if not parts:
            return "(GroupChat 中暂无对话记录)"

        return "\n".join(parts)

    def get_candidates_context(self) -> str:
        """提取所有提名发言的上下文，供 LLM 评审和排序使用。"""
        nominations = self.get_messages(msg_type=MessageType.NOMINATION)
        reviews = self.get_messages(msg_type=MessageType.REVIEW)

        parts: list[str] = []
        if nominations:
            parts.append("## 各 Agent 提名发言")
            for m in nominations:
                score = m.metadata.get("relevance_score", "?")
                category = m.metadata.get("category", "")
                mem_role = m.metadata.get("memory_role", "")
                parts.append(
                    f"### {m.sender_name}（{mem_role}，类别：{category}）\n"
                    f"关联度评分：{score}\n"
                    f"{m.content}"
                )

        if reviews:
            parts.append("\n## 评审意见")
            for m in reviews:
                parts.append(f"[{m.sender_name}]: {m.content}")

        if not parts:
            return "(暂无提名信息)"

        return "\n".join(parts)

    def clear(self) -> None:
        """清空聊天历史。"""
        self._messages.clear()
        self._round_count = 0
        self._compressed_summary = ""
        self._compressed_rounds = 0

    # ── 统计 ─────────────────────────────────────────────────────────

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def current_round(self) -> int:
        return self._round_count

    def stats(self) -> dict[str, Any]:
        """返回聊天统计。"""
        type_counts: dict[str, int] = {}
        for m in self._messages:
            key = m.msg_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "total_messages": len(self._messages),
            "current_round": self._round_count,
            "compressed_rounds": self._compressed_rounds,
            "messages_by_type": type_counts,
        }