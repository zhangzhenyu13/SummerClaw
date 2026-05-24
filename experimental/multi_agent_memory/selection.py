"""动态选择协议 —— 去中心化的 Borda 计票法选择 ≤K 个记忆 Agent。

实现 GroupChat 自主提名 + 投票机制：
1. 自我评估与提名阶段（N 个 Agent 并行发言）
2. 群体评审与收敛阶段（最多 2 轮对话）
3. Borda 计票法产生最终参与列表

核心特点：选择过程本身利用各 Agent 已有的记忆来评估任务关联度，
形成基于视角互补的选组，而非随机或轮询。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from experimental.multi_agent_memory.group_chat import GroupChat, MessageType
from experimental.multi_agent_memory.role_agent import RoleAgent, NominationSpeech

if TYPE_CHECKING:
    pass


@dataclass
class SelectionResult:
    """选择结果。"""
    selected_agents: list[str]        # 按优先级排序的选中 Agent ID 列表
    all_nominations: list[NominationSpeech]  # 所有提名发言
    vote_summary: dict[str, float]    # agent_id → Borda 得分
    selection_rounds: int             # 选择消耗的轮次数


class SelectionProtocol:
    """去中心化的 Agent 选择协议。

    通过 Borda 计票法从 N 个 Agent 中选出最多 K 个参与记忆读写。
    选择过程包括自我评估提名、群体评审和投票收敛三个阶段。

    Parameters:
        k: 最多选择的 Agent 数量（3~5 推荐）
        max_review_rounds: 评审收敛的最大对话轮次
    """

    def __init__(self, k: int = 3, max_review_rounds: int = 2) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.max_review_rounds = max_review_rounds

    # ── 主流程 ───────────────────────────────────────────────────────

    async def select(
        self,
        agents: list[RoleAgent],
        task_description: str,
        group_chat: GroupChat,
    ) -> SelectionResult:
        """执行完整的动态选择流程。

        Args:
            agents: 所有可用 Agent 列表
            task_description: 任务描述
            group_chat: GroupChat 协调空间

        Returns:
            SelectionResult 包含选中的 Agent 列表和投票详情
        """
        if len(agents) <= self.k:
            logger.info(f"Agent 数量 ({len(agents)}) ≤ K ({self.k})，全部选中")
            return SelectionResult(
                selected_agents=[a.agent_id for a in agents],
                all_nominations=[],
                vote_summary={a.agent_id: 1.0 for a in agents},
                selection_rounds=0,
            )

        # 阶段 1：广播任务描述
        group_chat.system_broadcast(
            f"📋 新任务到达，需要从 {len(agents)} 个 Agent 中选出最多 {self.k} 个参与记忆读写。\n\n"
            f"任务描述：{task_description}"
        )

        # 阶段 2：自我评估与提名阶段（N 个 Agent 并行发言）
        logger.info(f"阶段 1/3：{len(agents)} 个 Agent 并行自我评估与提名")
        all_nominations = await self._nomination_phase(agents, task_description, group_chat)
        if not all_nominations:
            logger.warning("没有 Agent 提名，无法继续选择")
            return SelectionResult(
                selected_agents=[],
                all_nominations=[],
                vote_summary={},
                selection_rounds=group_chat.current_round,
            )

        # 阶段 3：群体评审与收敛（最多 max_review_rounds 轮）
        logger.info(f"阶段 2/3：群体评审与收敛（最多 {self.max_review_rounds} 轮）")
        await self._review_phase(agents, all_nominations, task_description, group_chat)

        # 阶段 4：Borda 计票
        logger.info("阶段 3/3：Borda 计票法投票")
        vote_result = await self._borda_vote(agents, all_nominations, task_description, group_chat)

        # 广播结果
        selected_names = [
            a.role_name for a in agents if a.agent_id in vote_result["selected"]
        ]
        result_msg = (
            f"从 {len(agents)} 个 Agent 中选出 {len(vote_result['selected'])} 个：\n"
            + "\n".join(f"  {i+1}. {name} (Borda: {vote_result['scores'].get(aid, 0):.1f})"
                        for i, (name, aid) in enumerate(
                            zip(selected_names, vote_result["selected"])
                        ))
        )
        group_chat.broadcast(
            sender_id="system",
            sender_name="选举系统",
            content=result_msg,
            msg_type=MessageType.RESULT,
        )

        # 打印 Borda 计票详情
        logger.info(f"Borda 计票结果（前 {self.k} 名）：")
        sorted_scores = sorted(
            vote_result["scores"].items(), key=lambda x: x[1], reverse=True
        )
        for rank, (aid, score) in enumerate(sorted_scores, 1):
            name = next((a.role_name for a in agents if a.agent_id == aid), "?")
            status = "✅ 入选" if aid in vote_result["selected"] else "  —"
            logger.info(f"  {rank}. {name} | Borda: {score:.0f} {status}")
        # 打印每张选票
        ballots = vote_result.get("ballots", {})
        if ballots:
            logger.info(f"  各 Agent 投票详情（共 {len(ballots)} 张选票）：")
            for voter_id, ranking in ballots.items():
                voter_name = next((a.role_name for a in agents if a.agent_id == voter_id), voter_id)
                ranked_names = [
                    next((a.role_name for a in agents if a.agent_id == aid), aid)
                    for aid in ranking
                ]
                logger.info(f"    {voter_name} → {' > '.join(ranked_names)}")

        return SelectionResult(
            selected_agents=vote_result["selected"],
            all_nominations=all_nominations,
            vote_summary=vote_result["scores"],
            selection_rounds=group_chat.current_round,
        )

    # ── 提名阶段 ─────────────────────────────────────────────────────

    async def _nomination_phase(
        self,
        agents: list[RoleAgent],
        task_description: str,
        group_chat: GroupChat,
    ) -> list[NominationSpeech]:
        """所有 Agent 并行自我评估并发表提名发言。"""
        group_chat.advance_round()

        async def _agent_nominate(agent: "RoleAgent") -> NominationSpeech | None:
            try:
                speech = await agent.evaluate_task_relevance(task_description)
                # 广播提名发言到 GroupChat
                content = (
                    f"**关联度评分**：{speech.relevance_score:.2f}\n"
                    f"**理由**：{speech.reasoning}\n"
                    f"**记忆读写计划**：{speech.memory_plan}\n"
                    f"**协作期望**：{speech.collaboration_expectation}"
                )
                group_chat.broadcast(
                    sender_id=agent.agent_id,
                    sender_name=agent.role_name,
                    content=content,
                    msg_type=MessageType.NOMINATION,
                    metadata={
                        "relevance_score": speech.relevance_score,
                        "category": agent.category,
                        "memory_role": agent.memory_role,
                    },
                )
                return speech
            except Exception as e:
                logger.error(f"[{agent.agent_id}] 提名发言失败: {e}")
                return None

        # 并行执行所有 Agent 的提名
        tasks = [_agent_nominate(a) for a in agents]
        results = await asyncio.gather(*tasks)

        nominations = [r for r in results if r is not None]
        logger.info(
            f"提名阶段完成：{len(nominations)}/{len(agents)} 个 Agent 发表了提名发言"
        )
        # 打印所有提名详情
        for n in sorted(nominations, key=lambda x: x.relevance_score, reverse=True):
            logger.info(
                f"  🗳️ {n.role_name} | 关联度 {n.relevance_score:.2f} | "
                f"理由: {n.reasoning[:120]}..."
            )
        return nominations

    # ── 评审阶段 ─────────────────────────────────────────────────────

    async def _review_phase(
        self,
        agents: list[RoleAgent],
        nominations: list[NominationSpeech],
        task_description: str,
        group_chat: GroupChat,
    ) -> None:
        """LLM 驱动的 Agent 互审阶段。

        每个 Agent 使用 LLM 综合分析所有提名发言，发表有见地的评审意见。
        规则兜底：基于关联度区间和类别差异的简单启发式。

        最多 max_review_rounds 轮对话。
        """
        nom_summary = self._build_nomination_summary(nominations)

        reviews: list[ChatMessage] = []
        for review_round in range(self.max_review_rounds):
            group_chat.advance_round()
            group_chat.system_broadcast(
                f"评审轮次 {review_round + 1}/{self.max_review_rounds}：\n"
                f"请各 Agent 阅读以上提名发言。如果你对其他 Agent 的提名有补充或质疑，"
                f"请发表评审意见。也可以选择沉默。\n\n{nom_summary}"
            )

            review_tasks = [
                self._agent_review(agent, nominations, task_description, group_chat)
                for agent in agents
            ]
            await asyncio.gather(*review_tasks)

        # 打印评审意见汇总
        review_msgs = group_chat.get_messages(msg_type=MessageType.REVIEW)
        if review_msgs:
            logger.info(f"评审阶段完成，共 {len(review_msgs)} 条评审意见：")
            for m in review_msgs:
                logger.info(f"  💬 [{m.sender_name}]: {m.content[:150]}...")
        else:
            logger.info("评审阶段完成，无 Agent 发表评审意见")

        group_chat.system_broadcast(
            "评审阶段结束。进入投票阶段，各 Agent 请准备好意向票。"
        )

    async def _agent_review(
        self,
        agent: RoleAgent,
        nominations: list[NominationSpeech],
        task_description: str,
        group_chat: GroupChat,
    ) -> None:
        """单个 Agent 的纯 LLM 驱动评审发言。

        使用 LLM 综合分析所有提名，发现互补价值或潜在问题。
        """
        other_nominations = [
            n for n in nominations if n.agent_id != agent.agent_id
        ]
        if not other_nominations:
            return

        # LLM 驱动评审
        if agent.provider is None:
            raise RuntimeError("纯 LLM 模式需要 provider，请初始化时传入 provider 和 model")

        try:
            review_content = await self._llm_review(
                agent, other_nominations, task_description, group_chat
            )
            if review_content:
                group_chat.broadcast(
                    sender_id=agent.agent_id,
                    sender_name=agent.role_name,
                    content=review_content,
                    msg_type=MessageType.REVIEW,
                )
        except Exception as e:
            logger.warning(f"[{agent.agent_id}] LLM review failed: {e}")
            # 如果 LLM 评审失败，不发表评审意见（纯 LLM 模式下）

    async def _llm_review(
        self,
        agent: RoleAgent,
        other_nominations: list[NominationSpeech],
        task_description: str,
        group_chat: GroupChat,
    ) -> str | None:
        """使用 LLM 驱动单个 Agent 的评审发言。"""
        nominees_text = "\n".join([
            f"- {n.role_name}（关联度 {n.relevance_score:.2f}）：{n.reasoning[:150]}"
            for n in other_nominations
        ])

        system_prompt = f"""你是 {agent.role_name}（{agent.memory_role}），正在参与多角色协作系统的 Agent 选择评审。

你的角色档案中与决策相关的内容：
{agent._nomination_instruction[:500]}

请阅读其他 Agent 的提名发言，从你的专业视角发表评审意见：
1. 是否有被低估的候选者？其独特视角如何互补？
2. 是否有被高估的候选者？哪些维度被忽略？
3. 整体组合是否足够多样化？

如果所有提名都合理，可以简单表示认可。评审意见应简洁（150字以内）。"""

        user_prompt = f"""## 当前任务
{task_description}

## 其他 Agent 的提名
{nominees_text}

请发表你的评审意见："""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await agent._call_llm(messages)
        if response:
            return response.strip()[:500]
        return None

    def _build_nomination_summary(self, nominations: list[NominationSpeech]) -> str:
        """构建提名摘要。"""
        lines = ["## 提名摘要", ""]
        for i, n in enumerate(sorted(nominations, key=lambda x: x.relevance_score, reverse=True)):
            lines.append(
                f"{i+1}. **{n.role_name}** — 关联度 {n.relevance_score:.2f} | "
                f"{n.reasoning[:80]}..."
            )
        return "\n".join(lines)

    # ── Borda 计票 ───────────────────────────────────────────────────

    async def _borda_vote(
        self,
        agents: list[RoleAgent],
        nominations: list[NominationSpeech],
        task_description: str,
        group_chat: GroupChat,
    ) -> dict:
        """执行 Borda 计票法投票（LLM 驱动排序，并行投票）。

        每个 Agent 并行使用 LLM 对候选者进行深度排序，Borda 计分汇总。

        Returns:
            {"selected": [...], "scores": {...}, "ballots": {...}}
        """
        # 构建候选池（排除关联度极低的 Agent）
        min_relevance = 0.05
        nominee_map: dict[str, NominationSpeech] = {}
        for n in nominations:
            if n.relevance_score >= min_relevance:
                nominee_map[n.agent_id] = n

        if len(nominee_map) <= self.k:
            return {
                "selected": list(nominee_map.keys()),
                "scores": {aid: float(n.relevance_score) for aid, n in nominee_map.items()},
                "ballots": {},
            }

        candidate_ids = list(nominee_map.keys())
        ballots: dict[str, list[str]] = {}

        # 并行收集选票：所有 Agent 同时投票，不再串行等待
        async def _agent_vote(agent: RoleAgent) -> tuple[str, list[str]] | None:
            candidates_for_agent = [
                {
                    "agent_id": n.agent_id,
                    "role_name": n.role_name,
                    "relevance_score": n.relevance_score,
                    "category": getattr(
                        next((a for a in agents if a.agent_id == n.agent_id), None),
                        "category", "",
                    ),
                }
                for n in nominations
                if n.agent_id in candidate_ids and n.agent_id != agent.agent_id
            ]
            if not candidates_for_agent:
                return None
            try:
                ranking = await agent.rank_candidates(
                    candidates_for_agent, task_description, group_chat
                )
                return agent.agent_id, ranking
            except Exception as e:
                logger.warning(f"[{agent.agent_id}] 投票失败: {e}")
                return None

        vote_tasks = [_agent_vote(a) for a in agents]
        vote_results = await asyncio.gather(*vote_tasks)

        for result in vote_results:
            if result is not None:
                agent_id, ranking = result
                ballots[agent_id] = ranking

        # Borda 计分
        n_candidates = len(candidate_ids)
        scores: dict[str, float] = {cid: 0.0 for cid in candidate_ids}

        for voter_id, ranking in ballots.items():
            for rank, candidate_id in enumerate(ranking):
                if candidate_id in scores:
                    points = n_candidates - rank - 1
                    scores[candidate_id] += points

        # 取前 K 名
        sorted_candidates = sorted(
            scores.items(),
            key=lambda x: (x[1], nominee_map[x[0]].relevance_score if x[0] in nominee_map else 0),
            reverse=True,
        )
        selected = [cid for cid, _ in sorted_candidates[:self.k]]

        logger.info(
            f"Borda 计票完成：{len(ballots)} 张选票，选出 {len(selected)} 个 Agent"
        )

        return {
            "selected": selected,
            "scores": scores,
            "ballots": {k: v for k, v in ballots.items()},
        }


# ── 模块级便捷函数 ──────────────────────────────────────────────────

async def select_agents(
    agents: list[RoleAgent],
    task: str,
    group_chat: GroupChat,
    k: int = 3,
) -> SelectionResult:
    """快捷函数：执行动态选择并返回结果。"""
    protocol = SelectionProtocol(k=k)
    return await protocol.select(agents, task, group_chat)