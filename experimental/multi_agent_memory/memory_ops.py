"""记忆操作协调器 —— 视角分化的并行读写与结果聚合。

实现文档中描述的"视角分化的并行操作"：
- 读取（Recall）：每个选中的 Agent 独立使用角色滤镜改写查询后检索记忆
- 写入（Store）：根据任务和他人报告，独立决定存储策略
- 结果汇总：广播角色记忆报告，生成整合摘要
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from loguru import logger

from experimental.multi_agent_memory.group_chat import MessageType

if TYPE_CHECKING:
    from experimental.multi_agent_memory.role_agent import RoleAgent, MemoryReport
    from experimental.multi_agent_memory.group_chat import GroupChat


class MemoryOperations:
    """视角分化的并行记忆读写协调器。

    被选中的 K 个 Agent 并行执行记忆读写，各自遵循其角色策略，
    产生差异化的记忆操作。完全去中心化，无中央调度器。

    Parameters:
        agent_read_top_k: 每个 Agent 检索的记忆条目数上限
        enable_llm_integration: 是否启用 LLM 整合摘要
    """

    def __init__(
        self,
        agent_read_top_k: int = 10,
        enable_llm_integration: bool = True,
    ) -> None:
        self.agent_read_top_k = agent_read_top_k
        self.enable_llm_integration = enable_llm_integration

    # ── 并行读取 ──────────────────────────────────────────────────────

    async def parallel_read(
        self,
        selected_agents: list["RoleAgent"],
        task_description: str,
        group_chat: "GroupChat",
    ) -> list["MemoryReport"]:
        """每个选中的 Agent 独立检索记忆，生成角色记忆报告。

        Args:
            selected_agents: 选中的 Agent 列表
            task_description: 任务描述
            group_chat: GroupChat 协调空间

        Returns:
            所有 Agent 的记忆报告列表
        """
        group_chat.advance_round()
        group_chat.system_broadcast(
            f"📖 记忆读取阶段开始 —— {len(selected_agents)} 个 Agent 并行检索记忆"
        )

        async def _agent_read(agent: "RoleAgent") -> "MemoryReport | None":
            try:
                # Step 1: 角色化查询改写 + 检索
                query_results = agent.query(
                    query_text=task_description,
                    top_k=self.agent_read_top_k,
                )
                logger.debug(
                    f"[{agent.agent_id}] 检索到 {len(query_results)} 条记忆"
                )

                # Step 2: 生成角色记忆报告
                report = await agent.generate_memory_report(
                    task_description=task_description,
                    query_results=query_results,
                )

                # Step 3: 广播报告到 GroupChat
                report_content = self._format_report_for_chat(report)
                group_chat.broadcast(
                    sender_id=agent.agent_id,
                    sender_name=agent.role_name,
                    content=report_content,
                    msg_type=MessageType.MEMORY_REPORT,
                    metadata={
                        "num_retrieved": len(query_results),
                        "memory_role": agent.memory_role,
                    },
                )
                return report
            except Exception as e:
                logger.error(f"[{agent.agent_id}] 记忆检索失败: {e}")
                # 返回一个空报告
                from experimental.multi_agent_memory.role_agent import MemoryReport
                return MemoryReport(
                    agent_id=agent.agent_id,
                    role_name=agent.role_name,
                    memory_role=agent.memory_role,
                    query_rewritten=task_description,
                    reasoning=f"记忆检索失败：{e}",
                    uncertainties=["检索过程中发生错误"],
                )

        # 并行执行
        tasks = [_agent_read(a) for a in selected_agents]
        results = await asyncio.gather(*tasks)
        reports = [r for r in results if r is not None]

        logger.info(f"并行读取完成：{len(reports)}/{len(selected_agents)} 份报告生成")
        return reports

    # ── 并行写入 ──────────────────────────────────────────────────────

    async def parallel_write(
        self,
        selected_agents: list["RoleAgent"],
        task_description: str,
        reports: list["MemoryReport"],
        group_chat: "GroupChat",
    ) -> dict[str, int]:
        """每个选中的 Agent 独立决定存储策略并写入新记忆。

        差异化的写入行为：
        - 不同角色存入不同类型的信息（事实、推理、矛盾、隐喻等）
        - Agent 可以选择不写入（如果没有值得记录的信息）
        - 写入时自动触发角色化的主动链接和标签添加

        Args:
            selected_agents: 选中的 Agent 列表
            task_description: 任务描述
            reports: 所有 Agent 的记忆报告（用于交叉参考）
            group_chat: GroupChat 协调空间

        Returns:
            agent_id → 写入条数的映射
        """
        group_chat.advance_round()
        group_chat.system_broadcast(
            f"✏️ 记忆写入阶段开始 —— {len(selected_agents)} 个 Agent 独立决定存储策略"
        )

        write_stats: dict[str, int] = {}

        async def _agent_write(agent: "RoleAgent") -> tuple[str, int]:
            count = 0
            try:
                # 获取该 Agent 的报告（如果有）
                own_report = next(
                    (r for r in reports if r.agent_id == agent.agent_id), None
                )
                # 获取其他 Agent 的报告
                other_reports = [r for r in reports if r.agent_id != agent.agent_id]

                # 根据角色策略决定写入内容
                storage_decisions = self._get_storage_decisions(
                    agent, own_report, other_reports, task_description
                )

                for decision in storage_decisions:
                    text = decision.get("text", "")
                    if not text:
                        continue
                    metadata = decision.get("metadata", {})
                    metadata["task"] = task_description[:200]
                    metadata["source"] = "multi_agent_collaboration"

                    # 注入角色标签
                    metadata.setdefault("memory_type", decision.get("type", "fact"))
                    metadata.setdefault("role", agent.role_name)
                    metadata.setdefault("memory_role", agent.memory_role)

                    result_id = agent.store(text, metadata)
                    if result_id:
                        count += 1
                        logger.debug(
                            f"[{agent.agent_id}] 写入记忆: {text[:60]}..."
                        )

                if count > 0:
                    group_chat.broadcast(
                        sender_id=agent.agent_id,
                        sender_name=agent.role_name,
                        content=f"存储了 {count} 条新记忆",
                        msg_type=MessageType.INFO,
                    )
            except Exception as e:
                logger.error(f"[{agent.agent_id}] 记忆写入失败: {e}")

            return agent.agent_id, count

        tasks = [_agent_write(a) for a in selected_agents]
        results = await asyncio.gather(*tasks)
        for agent_id, count in results:
            write_stats[agent_id] = count

        total_written = sum(write_stats.values())
        logger.info(f"并行写入完成：共 {total_written} 条新记忆被存储")
        return write_stats

    def _get_storage_decisions(
        self,
        agent: "RoleAgent",
        own_report: "MemoryReport | None",
        other_reports: list["MemoryReport"],
        task_description: str,
    ) -> list[dict]:
        """根据角色策略决定写入哪些记忆条目。

        纯 LLM 驱动：优先使用 Agent 报告中 LLM 生成的 storage_decisions。
        """
        decisions: list[dict] = []

        # 优先使用 Agent 自己报告中 LLM 生成的存储决策
        if own_report and own_report.storage_decisions:
            for sd_text in own_report.storage_decisions:
                memory_type = self._classify_memory_type(agent, sd_text)
                decisions.append({
                    "text": sd_text,
                    "type": memory_type,
                    "metadata": {"auto_generated": True},
                })

        # 记录其他 Agent 的不确定性（交叉引用）
        if own_report and other_reports:
            for other in other_reports:
                if other.uncertainties:
                    contradiction_text = (
                        f"[交叉参考] {other.role_name} 在任务中标记了关键不确定性："
                        f"{'；'.join(other.uncertainties[:2])}"
                    )
                    decisions.append({
                        "text": contradiction_text,
                        "type": "cross_reference",
                        "metadata": {"related_agent": other.agent_id},
                    })

        return decisions

    def _classify_memory_type(self, agent: "RoleAgent", text: str) -> str:
        """根据角色类别和文本内容分类记忆类型。"""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["矛盾", "不一致", "冲突", "contradict"]):
            return "contradiction"
        if any(kw in text_lower for kw in ["推断", "推测", "假设", "可能", "推理"]):
            return "inference"
        if any(kw in text_lower for kw in ["方法", "流程", "步骤", "策略", "框架"]):
            return "methodology"
        if any(kw in text_lower for kw in ["情感", "叙事", "隐喻", "故事"]):
            return "narrative"
        return "fact"

    def _build_heuristic_storage(
        self,
        agent: "RoleAgent",
        report: "MemoryReport",
        task_description: str,
    ) -> list[dict]:
        """基于角色类型的启发式存储决策。"""
        decisions = []

        # 根据角色类别决定存储内容
        category = agent.category

        # 记录任务参与事实
        decisions.append({
            "text": f"[任务记录] 作为 {agent.role_name} 参与了关于 '{task_description[:100]}' 的记忆协作。检索到 {len(report.retrieved_items)} 条相关记忆。",
            "type": "fact",
            "metadata": {"is_static": False},
        })

        # 如果有推理结果，记录推理
        if report.reasoning and len(report.reasoning) > 20:
            decisions.append({
                "text": f"[角色推理] {report.reasoning[:300]}",
                "type": "inference",
                "metadata": {"is_static": False},
            })

        # 记录不确定性
        for uncertainty in report.uncertainties[:3]:
            decisions.append({
                "text": f"[不确定性] {uncertainty}",
                "type": "uncertainty",
                "metadata": {"is_static": False},
            })

        return decisions

    # ── 结果汇总 ──────────────────────────────────────────────────────

    async def aggregate_results(
        self,
        reports: list["MemoryReport"],
        group_chat: "GroupChat",
        provider: Any = None,
        model: str = "",
    ) -> str:
        """纯 LLM 驱动的整合摘要，指出共识与分歧。

        使用 LLM 综合分析所有报告，生成高质量整合摘要。
        保留认知多样性，不强求统一答案。

        Args:
            reports: 所有 Agent 的记忆报告
            group_chat: GroupChat 协调空间
            provider: LLM provider（可选）
            model: 模型名称

        Returns:
            整合摘要文本
        """
        group_chat.advance_round()

        if not reports:
            msg = "没有生成任何记忆报告，无法整合。"
            group_chat.system_broadcast(msg)
            return msg

        # 纯 LLM 主导整合
        if provider is None:
            raise RuntimeError("纯 LLM 模式需要 provider，请初始化时传入 provider 和 model")

        try:
            llm_summary = await self._llm_integrate(
                reports, provider, model
            )
            if llm_summary:
                group_chat.broadcast(
                    sender_id="system",
                    sender_name="整合系统",
                    content=llm_summary,
                    msg_type=MessageType.SUMMARY,
                )
                return llm_summary
        except Exception as e:
            logger.warning(f"LLM 整合摘要失败: {e}")

        # 如果 LLM 整合失败，返回错误信息
        error_msg = "LLM 整合摘要失败，无法生成整合结果。"
        group_chat.system_broadcast(error_msg)
        return error_msg

    async def _extract_consensus_and_divergences_llm(
        self,
        reports: list["MemoryReport"],
        provider: Any,
        model: str,
    ) -> tuple[list[str], list[str]] | None:
        """LLM 驱动的共识与分歧提取。

        Returns:
            (consensus, divergences) 或 None（LLM 不可用时）
        """
        if provider is None:
            return None

        reports_text = "\n\n".join([
            f"[{r.role_name}] 推理：{r.reasoning[:200]}\n"
            f"不确定性：{'；'.join(r.uncertainties[:3]) if r.uncertainties else '无'}"
            for r in reports
        ])

        messages = [
            {"role": "system", "content": (
                "你是多角色协作分析助手。请从以下角色报告中提取共识和分歧。"
                "返回 JSON 格式。"
            )},
            {"role": "user", "content": (
                f"## 各角色报告\n{reports_text}\n\n"
                f"请提取共识点和分歧点，返回 JSON：\n"
                f'{{"consensus": ["共识1", "共识2"], "divergences": ["分歧1", "分歧2"]}}'
            )},
        ]

        try:
            response = await asyncio.wait_for(
                provider.chat_with_retry(
                    messages=messages, model=model, retry_mode="standard"
                ),
                timeout=60,
            )
            if hasattr(response, "content") and response.content:
                json_match = __import__("re").search(r'\{[\s\S]*\}', str(response.content))
                if json_match:
                    data = __import__("json").loads(json_match.group(0))
                    return (
                        data.get("consensus", []),
                        data.get("divergences", []),
                    )
        except Exception as e:
            logger.warning(f"LLM 共识提取失败: {e}")

        return None

    def _extract_consensus_and_divergences(
        self,
        reports: list["MemoryReport"],
    ) -> tuple[list[str], list[str]]:
        """从多份报告中提取共识和分歧（规则兜底版）。"""
        consensus: list[str] = []
        divergences: list[str] = []

        # 所有 Agent 的推理收集
        reasonings = [(r.role_name, r.reasoning) for r in reports if r.reasoning]

        if len(reasonings) >= 2:
            # 简单共识检测：看是否有多个 Agent 提到了相似的关键词
            all_keywords: list[set[str]] = []
            import re
            for _, reasoning_text in reasonings:
                words = set(re.findall(r'[\u4e00-\u9fff]{2,}', reasoning_text))
                all_keywords.append(words)

            if len(all_keywords) >= 2:
                # 找出在至少 2/3 报告中出现的关键词
                common_words = all_keywords[0].copy()
                for kw_set in all_keywords[1:]:
                    common_words &= kw_set

                if common_words:
                    consensus.append(f"多角色共识关键词：{'、'.join(list(common_words)[:5])}")
                else:
                    divergences.append("不同角色的分析角度差异较大，未形成明显共识")

        # 收集所有不确定性
        all_uncertainties = []
        for r in reports:
            for u in r.uncertainties:
                all_uncertainties.append(f"[{r.role_name}] {u}")

        if all_uncertainties:
            divergences.append(
                f"各角色标记的不确定性（共 {len(all_uncertainties)} 项）：\n"
                + "\n".join(f"  - {u}" for u in all_uncertainties[:10])
            )

        return consensus, divergences

    def _build_structured_summary(
        self,
        reports: list["MemoryReport"],
        consensus: list[str],
        divergences: list[str],
    ) -> str:
        """构建结构化的整合摘要。"""
        lines = [
            f"# 多角色记忆协作整合摘要",
            f"",
            f"**参与角色数**：{len(reports)}",
            f"**总检索记忆条目**：{sum(len(r.retrieved_items) for r in reports)}",
            f"",
        ]

        # 各角色视角摘要
        lines.append("## 各角色视角")
        for i, r in enumerate(reports, 1):
            items_count = len(r.retrieved_items)
            lines.append(f"\n### {i}. {r.role_name}（{r.memory_role}）")
            lines.append(f"- 检索条目数：{items_count}")
            if r.reasoning:
                lines.append(f"- 核心判断：{r.reasoning[:200]}")
            if r.uncertainties:
                lines.append(f"- 不确定性：{'；'.join(r.uncertainties[:3])}")

        # 共识
        if consensus:
            lines.append("\n## 🤝 共识发现")
            for c in consensus:
                lines.append(f"- {c}")

        # 分歧
        if divergences:
            lines.append("\n## 🔀 视角分歧")
            for d in divergences:
                lines.append(f"- {d}")

        lines.append(f"\n---\n*记忆系统的多视角协作完成。各角色保留了独立的记忆视角，"
                      f"分歧被保留为系统的认知多样性。*")

        return "\n".join(lines)

    def _format_report_for_chat(self, report: "MemoryReport") -> str:
        """格式化记忆报告为 GroupChat 可读文本。"""
        lines = [
            f"### {report.role_name} 记忆报告",
            f"**记忆子角色**：{report.memory_role}",
            f"**检索条目数**：{len(report.retrieved_items)}",
        ]

        # 展示 top-3 检索结果摘要
        if report.retrieved_items:
            lines.append("\n**关键记忆条目**：")
            for item in report.retrieved_items[:3]:
                text = item.get("text", "")[:120]
                score = item.get("score", 0)
                lines.append(f"  - [{score:.2f}] {text}")

        if report.reasoning:
            lines.append(f"\n**角色推理**：{report.reasoning[:300]}")

        if report.uncertainties:
            lines.append(f"\n**不确定性**：{'；'.join(report.uncertainties[:5])}")

        return "\n".join(lines)

    async def _llm_integrate(
        self,
        reports: list["MemoryReport"],
        provider: Any,
        model: str,
    ) -> str | None:
        """使用 LLM 综合分析所有报告，生成高质量整合摘要。

        LLM 自行发现共识和分歧，无需依赖规则预提取。
        """
        reports_text = "\n\n".join([
            f"### {r.role_name}（{r.memory_role}）\n"
            f"查询改写：{r.query_rewritten[:100]}\n"
            f"检索条目：{len(r.retrieved_items)} 条\n"
            f"核心推理：{r.reasoning[:300]}\n"
            f"不确定性：{'；'.join(r.uncertainties[:5]) if r.uncertainties else '无'}\n"
            f"存储建议：{'；'.join(r.storage_decisions[:3]) if r.storage_decisions else '无'}"
            for r in reports
        ])

        # 提取所有角色的快速概览
        roles_overview = "、".join([f"{r.role_name}({r.memory_role})" for r in reports])

        messages = [
            {
                "role": "system",
                "content": (
                    "你是多角色记忆协作的整合者。参与协作的角色包括："
                    f"{roles_overview}。\n\n"
                    "请基于各角色的记忆报告，生成一份综合分析摘要（不超过 800 字），"
                    "必须包含以下四个部分：\n\n"
                    "## 1. 各角色核心发现\n"
                    "概括每个角色从自身视角得出的关键洞察。\n\n"
                    "## 2. 共识发现\n"
                    "识别多个角色共同关注或一致认同的观点。\n\n"
                    "## 3. 视角分歧\n"
                    "指出不同角色间的重要分歧、认知空白和互补视角。\n\n"
                    "## 4. 综合建议\n"
                    "基于多视角分析，给出后续行动建议。\n\n"
                    "保留认知多样性，不强求统一答案。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## 各角色记忆报告\n\n{reports_text}\n\n"
                    f"请生成整合摘要（按上述四部分格式）。"
                ),
            },
        ]

        try:
            response = await asyncio.wait_for(
                provider.chat_with_retry(
                    messages=messages,
                    model=model,
                    retry_mode="standard",
                ),
                timeout=120,
            )
            if hasattr(response, "content"):
                return str(response.content or "")
        except Exception as e:
            logger.warning(f"LLM 整合失败: {e}")

        return None


# ── 模块级便捷函数 ──────────────────────────────────────────────────

async def process_memory_collaboration(
    selected_agents: list["RoleAgent"],
    task: str,
    group_chat: "GroupChat",
    provider: Any = None,
    model: str = "",
) -> tuple[list["MemoryReport"], dict[str, int], str]:
    """快捷函数：执行完整的记忆协作流程。

    Returns:
        (report_list, write_stats, integrated_summary)
    """
    ops = MemoryOperations()

    # 并行读取
    reports = await ops.parallel_read(selected_agents, task, group_chat)

    # 并行写入
    write_stats = await ops.parallel_write(selected_agents, task, reports, group_chat)

    # 结果汇总
    summary = await ops.aggregate_results(reports, group_chat, provider, model)

    return reports, write_stats, summary