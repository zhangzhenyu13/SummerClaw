"""多角色记忆系统 —— 主入口。

根据 resources/roles 下的角色定义，构建 N 个角色驱动的记忆 Agent，
通过 GroupChat 自主协商选出 ≤K 个 Agent 参与记忆读写，
实现视角分化的记忆管理。

纯 LLM 驱动设计：Agent 选择、评审、排序、查询改写、报告生成、
共识提取等核心决策均由 LLM 驱动，无规则兜底逻辑。
完全去中心化设计，符合 Avilization 群体演进智能思想。
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from experimental.multi_agent_memory.role_agent import RoleAgent, DEFAULT_ROLE_PROFILE
from experimental.multi_agent_memory.group_chat import GroupChat
from experimental.multi_agent_memory.selection import SelectionProtocol, SelectionResult
from experimental.multi_agent_memory.memory_ops import MemoryOperations

# ── 常量 ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCES_ROLES = PROJECT_ROOT / "resources" / "roles"
DEFAULT_K = 3   # 默认每轮任务选出的 Agent 数量
DEFAULT_N = 10  # 默认初始化的 Agent 总数


class MultiAgentMemorySystem:
    """多角色记忆系统 —— 纯 LLM 驱动、去中心化、自组织的群体记忆管理。

    三层架构：
    - 角色实例层：N 个 RoleAgent，每个拥有独立 Supermemory 实例
    - 协商协作层：GroupChat + Borda 计票动态选择（纯 LLM 驱动排序和评审）
    - 记忆存储层：视角分化的并行读写操作（纯 LLM 驱动查询改写和报告生成）

    所有核心决策均由 LLM 驱动，无规则兜底逻辑。

    Usage::

        system = MultiAgentMemorySystem(k=3)
        await system.initialize(max_agents=10)

        result = await system.process_task("分析最近的AI发展趋势")
        print(result.summary)
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        roles_dir: Path | None = None,
        memory_base_dir: Path | None = None,
        config: Any = None,
        provider: Any = None,
        model: str = "",
    ) -> None:
        """初始化多角色记忆系统。

        纯 LLM 驱动模式必须传入 provider 和 model；无 provider 时
        系统将无法运行。

        Args:
            k: 每轮任务最多选择的 Agent 数量
            roles_dir: 角色文件目录（默认 resources/roles）
            memory_base_dir: 记忆存储根目录（默认 ./multi_agent_memory_data）
            config: summerclaw Config 对象（可选）
            provider: LLM provider（必须传入，纯 LLM 驱动模式）
            model: 模型名称
        """
        self.k = k
        self.roles_dir = roles_dir or RESOURCES_ROLES
        self.memory_base_dir = Path(memory_base_dir) if memory_base_dir else Path("./multi_agent_memory_data")
        self.config = config
        self.provider = provider
        self.model = model

        # 核心组件（延迟初始化）
        self._agents: list[RoleAgent] = []
        self._agent_map: dict[str, RoleAgent] = {}
        self._group_chat: GroupChat | None = None
        self._selection: SelectionProtocol = SelectionProtocol(k=k)
        self._memory_ops: MemoryOperations = MemoryOperations()

        # 状态
        self._initialized = False
        self._available_roles: list[dict[str, str]] = []

    # ── 初始化 ───────────────────────────────────────────────────────

    async def initialize(
        self,
        max_agents: int = DEFAULT_N,
        role_filter: list[str] | None = None,
        category_filter: list[str] | None = None,
    ) -> int:
        """扫描角色文件并初始化 Agent 实例。

        Args:
            max_agents: 最多初始化的 Agent 数量
            role_filter: 仅加载指定名称的角色（为空则加载全部）
            category_filter: 仅加载指定类别的角色（为空则加载全部）

        Returns:
            成功初始化的 Agent 数量
        """
        logger.info(f"初始化多角色记忆系统 (K={self.k}, max_agents={max_agents})")

        # 扫描可用角色
        self._available_roles = self._scan_roles(role_filter, category_filter)
        if not self._available_roles:
            logger.error(f"未找到可用角色文件于 {self.roles_dir}")
            return 0

        # 限制数量
        roles_to_load = self._available_roles[:max_agents]
        logger.info(
            f"从 {len(self._available_roles)} 个角色中选择 {len(roles_to_load)} 个加载"
        )

        # 并行加载角色档案并创建 Agent
        self._agents = []
        self._agent_map = {}

        for i, role_info in enumerate(roles_to_load):
            try:
                profile = self._load_role_profile(role_info["file_path"])
                category = role_info.get("category_dir", "")

                agent = RoleAgent(
                    agent_id=f"agent_{i:03d}",
                    role_name=role_info["role_name"],
                    role_profile=profile,
                    category=category,
                    memory_dir=self.memory_base_dir,
                    provider=self.provider,
                    model=self.model,
                )
                self._agents.append(agent)
                self._agent_map[agent.agent_id] = agent
                logger.debug(
                    f"[{i+1}/{len(roles_to_load)}] 加载角色: {agent.role_name} "
                    f"({agent.memory_role})"
                )
            except Exception as e:
                logger.warning(f"加载角色 {role_info.get('role_name', '?')} 失败: {e}")

        # 初始化 GroupChat
        self._group_chat = GroupChat()

        self._initialized = True
        logger.info(
            f"多角色记忆系统初始化完成：{len(self._agents)} 个 Agent 就绪"
        )
        return len(self._agents)

    def _scan_roles(
        self,
        role_filter: list[str] | None = None,
        category_filter: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """扫描角色目录，返回角色信息列表。"""
        roles = []
        if not self.roles_dir.exists():
            logger.warning(f"角色目录不存在: {self.roles_dir}")
            return roles

        role_name_set = set(role_filter) if role_filter else None
        category_set = set(category_filter) if category_filter else None

        for category_dir in sorted(self.roles_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            if category_dir.name == "__pycache__":
                continue

            category_name = category_dir.name
            if category_set and category_name not in category_set:
                continue

            for role_file in sorted(category_dir.glob("*.md")):
                role_name = role_file.stem
                if role_name_set and role_name not in role_name_set:
                    continue
                roles.append({
                    "category_dir": category_name,
                    "category": category_name.replace("_", " ").title(),
                    "role_name": role_name,
                    "file_path": str(role_file),
                })

        return roles

    @staticmethod
    def _load_role_profile(file_path: str) -> dict[str, Any]:
        """从 Markdown 角色文件加载结构化档案。

        解析 YAML front matter 中的角色定义段，提取每个
        编号小节（如 "1. 身份与背景"）的 内容 字段。
        """
        try:
            content = Path(file_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"无法读取角色文件 {file_path}: {e}")
            return DEFAULT_ROLE_PROFILE

        profile: dict[str, Any] = {}

        # 提取 YAML 块
        yaml_match = re.search(r'```yaml\s*\n(.*?)\n```', content, re.DOTALL)
        if not yaml_match:
            # 回退：尝试解析纯 Markdown 格式的角色文件
            return MultiAgentMemorySystem._load_markdown_profile(content, file_path)

        yaml_text = yaml_match.group(1)

        # 按编号小节分割（匹配 "  N. Name:" 模式）
        sections = re.split(r'\n(\s+\d+\.\s+.+?:)\s*\n', yaml_text)
        # sections: [preamble, header1, body1, header2, body2, ...]
        # 跳过第一个 preamble（"角色定义:"）

        current_name: str | None = None
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            body = sections[i + 1] if i + 1 < len(sections) else ""

            # 提取小节名称（去掉编号和末尾冒号）
            name_match = re.match(r'^\d+\.\s+(.+?):$', header)
            if not name_match:
                continue
            name = name_match.group(1).strip()

            # 从 body 中提取 内容 字段（支持 "内容: |" 多行格式）
            content_value = MultiAgentMemorySystem._extract_yaml_content(body)
            if content_value:
                profile[name] = {
                    "描述": "",
                    "内容": content_value,
                }

        if len(profile) < 2:
            logger.warning(f"角色文件 {file_path} 解析内容不足 ({len(profile)} 段)，使用默认档案")
            return DEFAULT_ROLE_PROFILE

        return profile

    @staticmethod
    def _extract_yaml_content(body: str) -> str:
        """从小节 body 中提取 内容 字段的值。

        支持两种格式：
        - 内容: |\n  缩进的多行文本
        - 内容: 单行文本
        """
        # 匹配 "内容: |" 后的多行缩进文本
        content_block = re.search(r'内容:\s*\|\s*\n(.*?)(?=\n\s*\d+\.|\n\s*经典方法论|\Z)', body, re.DOTALL)
        if content_block:
            # 提取内容，去除公共缩进
            raw = content_block.group(1)
            lines = raw.split('\n')
            # 检测最小缩进
            min_indent = None
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    if min_indent is None or indent < min_indent:
                        min_indent = indent
            if min_indent and min_indent > 0:
                lines = [line[min_indent:] if line.strip() else line for line in lines]
            return '\n'.join(lines).strip()

        # 匹配 "内容: 单行文本"
        content_line = re.search(r'内容:\s*"?(.+?)"?\s*$', body, re.MULTILINE)
        if content_line:
            val = content_line.group(1).strip()
            if val and val != '|':
                return val

        return ""

    @staticmethod
    def _load_markdown_profile(content: str, file_path: str) -> dict[str, Any]:
        """解析纯 Markdown 格式的角色文件（回退方案）。

        格式: ### N. 名称\n**描述**: ...\n**内容**: \n...content...
        """
        profile: dict[str, Any] = {}

        # 按 "### N. " 分割小节
        sections = re.split(r'\n###\s+\d+\.\s+', content)
        # sections[0] 是标题行，后续每段是一个小节

        for section in sections[1:] if len(sections) > 1 else []:
            # 提取小节名称（第一行）
            lines = section.split('\n')
            name_line = lines[0].strip() if lines else ""
            # 名称不一定以冒号结尾
            name = name_line.rstrip('：:')

            if not name:
                continue

            # 提取 **内容**: 后的内容
            content_match = re.search(r'\*\*内容\*\*\s*:\s*\n?(.*)', section, re.DOTALL)
            if content_match:
                content_value = content_match.group(1).strip()
                if content_value:
                    profile[name] = {
                        "描述": "",
                        "内容": content_value,
                    }

        if len(profile) < 2:
            logger.warning(f"角色文件 {file_path} Markdown 解析内容不足，使用默认档案")
            return DEFAULT_ROLE_PROFILE

        return profile

    # ── 主流程：处理任务 ─────────────────────────────────────────────

    async def process_task(self, task_description: str) -> dict[str, Any]:
        """处理一个任务 —— 纯 LLM 驱动的完整去中心化协作流程。

        流程：
        1. 广播任务到 GroupChat
        2. LLM 驱动选择 ≤K 个 Agent（自我评估 + LLM 评审 + LLM 排序投票）
        3. LLM 驱动并行记忆读取（角色化查询改写 + 语义检索）
        4. LLM 驱动并行记忆写入（智能存储策略 + 交叉引用）
        5. LLM 驱动结果汇总（综合分析 + 共识分歧识别）

        所有步骤均由 LLM 驱动，无规则兜底。

        Args:
            task_description: 任务描述

        Returns:
            包含完整协作结果的字典：
            - summary: 整合摘要
            - selected_agents: 选中的 Agent 信息
            - reports: 各 Agent 的记忆报告
            - write_stats: 写入统计
            - chat_stats: GroupChat 统计
            - timestamp: 时间戳
        """
        if not self._initialized:
            raise RuntimeError("系统未初始化，请先调用 initialize()")
        if not self._agents:
            return self._empty_result("没有可用的 Agent")

        logger.info(f"开始处理任务: {task_description[:100]}...")
        start_time = datetime.now()

        # ── 阶段 1/5：广播任务 ────────────────────────────────────────
        logger.info(f"[阶段 1/5] 广播任务到 GroupChat（{len(self._agents)} 个 Agent 在线）")
        self._group_chat.clear()
        self._group_chat.system_broadcast(
            f"🚀 新任务到达\n\n**任务描述**：{task_description}"
        )

        # ── 阶段 2/5：动态选择 Agent ──────────────────────────────────
        logger.info(f"[阶段 2/5] LLM 驱动选择 ≤{self.k} 个 Agent（提名+评审+投票）")
        t2 = datetime.now()
        selection_result = await self._selection.select(
            agents=self._agents,
            task_description=task_description,
            group_chat=self._group_chat,
        )

        if not selection_result.selected_agents:
            logger.warning("没有 Agent 被选中参与此任务")
            return self._empty_result("没有 Agent 被选中（关联度过低）")

        selected = [
            self._agent_map[aid]
            for aid in selection_result.selected_agents
            if aid in self._agent_map
        ]
        if not selected:
            return self._empty_result("选中的 Agent 不在可用列表中")

        logger.info(
            f"[阶段 2/5] 完成 ({(datetime.now() - t2).total_seconds():.1f}s)："
            f"选出 {len(selected)} 个 Agent — {[a.role_name for a in selected]}"
        )

        # ── 阶段 3/5：并行记忆读取 ────────────────────────────────────
        logger.info(f"[阶段 3/5] LLM 驱动并行记忆读取（{len(selected)} 个 Agent）")
        t3 = datetime.now()
        reports = await self._memory_ops.parallel_read(
            selected_agents=selected,
            task_description=task_description,
            group_chat=self._group_chat,
        )
        total_retrieved = sum(len(r.retrieved_items) for r in reports)
        logger.info(
            f"[阶段 3/5] 完成 ({(datetime.now() - t3).total_seconds():.1f}s)："
            f"生成 {len(reports)} 份报告，共检索 {total_retrieved} 条记忆"
        )

        # ── 阶段 4/5：并行记忆写入 ────────────────────────────────────
        logger.info(f"[阶段 4/5] LLM 驱动并行记忆写入（{len(selected)} 个 Agent）")
        t4 = datetime.now()
        write_stats = await self._memory_ops.parallel_write(
            selected_agents=selected,
            task_description=task_description,
            reports=reports,
            group_chat=self._group_chat,
        )
        total_written = sum(write_stats.values())
        logger.info(
            f"[阶段 4/5] 完成 ({(datetime.now() - t4).total_seconds():.1f}s)："
            f"共写入 {total_written} 条新记忆"
        )

        # ── 阶段 5/5：结果汇总 ────────────────────────────────────────
        logger.info(f"[阶段 5/5] LLM 驱动结果汇总")
        t5 = datetime.now()
        summary = await self._memory_ops.aggregate_results(
            reports=reports,
            group_chat=self._group_chat,
            provider=self.provider,
            model=self.model,
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"[阶段 5/5] 完成 ({(datetime.now() - t5).total_seconds():.1f}s)\n"
            f"任务处理完成，总耗时 {elapsed:.1f}s\n"
            f"  Agent选择: {len(selected)} 个 | 记忆检索: {total_retrieved} 条 | 新写入: {total_written} 条"
        )

        return {
            "summary": summary,
            "selected_agents": [
                {
                    "agent_id": aid,
                    "role_name": self._agent_map[aid].role_name,
                    "memory_role": self._agent_map[aid].memory_role,
                }
                for aid in selection_result.selected_agents
                if aid in self._agent_map
            ],
            "reports": [
                {
                    "agent_id": r.agent_id,
                    "role_name": r.role_name,
                    "memory_role": r.memory_role,
                    "reasoning": r.reasoning,
                    "uncertainties": r.uncertainties,
                    "storage_decisions": r.storage_decisions,
                    "retrieved_count": len(r.retrieved_items),
                }
                for r in reports
            ],
            "write_stats": write_stats,
            "chat_stats": self._group_chat.stats(),
            "vote_summary": selection_result.vote_summary,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    def _empty_result(self, reason: str) -> dict[str, Any]:
        """返回空结果。"""
        return {
            "summary": reason,
            "selected_agents": [],
            "reports": [],
            "write_stats": {},
            "chat_stats": {},
            "vote_summary": {},
            "elapsed_seconds": 0,
            "timestamp": datetime.now().isoformat(),
        }

    # ── 查询历史 ─────────────────────────────────────────────────────

    def get_chat_history(self, last_n_rounds: int = 10) -> str:
        """获取 GroupChat 最近的历史文本。"""
        if self._group_chat is None:
            return "(GroupChat 未初始化)"
        return self._group_chat.get_recent_history(last_n_rounds)

    # ── 维护操作 ─────────────────────────────────────────────────────

    async def maintenance_cycle(self) -> dict[str, Any]:
        """执行一次定期维护循环。

        每个 Agent 根据自身角色档案中的 maintenance_routine 执行后台任务。
        """
        if not self._initialized or not self._agents:
            return {"status": "not_initialized"}

        results = {}
        for agent in self._agents:
            try:
                store = agent._ensure_store()
                # 自动清理过期记忆
                if hasattr(store, "auto_forget_expired"):
                    count = store.auto_forget_expired()
                    if count > 0:
                        results[agent.agent_id] = {
                            "role": agent.role_name,
                            "forgotten": count,
                        }
            except Exception as e:
                logger.warning(f"[{agent.agent_id}] 维护失败: {e}")
                results[agent.agent_id] = {"error": str(e)}

        logger.info(f"维护循环完成: {len(results)} 个 Agent 执行了维护")
        return {"status": "completed", "details": results}

    def get_system_stats(self) -> dict[str, Any]:
        """获取系统整体统计信息。"""
        stats: dict[str, Any] = {
            "total_agents": len(self._agents),
            "k": self.k,
            "initialized": self._initialized,
            "agent_details": [],
        }

        for agent in self._agents:
            try:
                mem_stats = agent.stats()
            except Exception:
                mem_stats = {}
            stats["agent_details"].append({
                "agent_id": agent.agent_id,
                "role_name": agent.role_name,
                "memory_role": agent.memory_role,
                "category": agent.category,
                "memory_stats": mem_stats,
            })

        if self._group_chat:
            stats["chat_stats"] = self._group_chat.stats()

        return stats

    # ── 角色信息 ─────────────────────────────────────────────────────

    def list_available_roles(self) -> list[dict[str, str]]:
        """列出所有可用角色（不初始化 Agent）。"""
        if not self._available_roles:
            self._available_roles = self._scan_roles()
        return self._available_roles

    def list_loaded_agents(self) -> list[dict[str, str]]:
        """列出已加载的 Agent 信息。"""
        return [
            {
                "agent_id": a.agent_id,
                "role_name": a.role_name,
                "memory_role": a.memory_role,
                "category": a.category,
            }
            for a in self._agents
        ]


# ── 同步便捷函数 ────────────────────────────────────────────────────


def process_task_sync(
    system: MultiAgentMemorySystem,
    task_description: str,
) -> dict[str, Any]:
    """同步版本的任务处理入口。"""
    return asyncio.run(system.process_task(task_description))


# ── CLI 演示入口 ────────────────────────────────────────────────────

async def _demo():
    """命令行演示入口。"""
    print("=" * 60)
    print("  多角色记忆系统 (Multi-Agent Memory System)")
    print("=" * 60)

    system = MultiAgentMemorySystem(k=3)

    # 初始化（加载少量 Agent 做演示）
    count = await system.initialize(max_agents=8)
    print(f"\n✓ 已加载 {count} 个 Agent")
    for a in system._agents:
        print(f"  - [{a.agent_id}] {a.role_name} ({a.memory_role})")

    # 处理示例任务
    print("\n" + "=" * 60)
    print("  处理任务演示")
    print("=" * 60)

    result = await system.process_task(
        "请分析最近AI大模型的发展趋势，并评估其对软件工程行业的影响"
    )

    print(f"\n📊 **整合摘要**：\n{result['summary'][:500]}")
    print(f"\n⏱ 耗时：{result['elapsed_seconds']:.1f}s")
    print(f"👥 参与Agent：{len(result['selected_agents'])} 个")
    print(f"💾 写入统计：{result['write_stats']}")

    # 系统统计
    stats = system.get_system_stats()
    print(f"\n📈 系统统计：{stats['total_agents']} 个 Agent 就绪")

    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(_demo())