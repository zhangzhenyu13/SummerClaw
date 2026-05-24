"""Role Agent —— 单个角色驱动的记忆 Agent，拥有独立 Supermemory 实例和角色化行为指令。

每个 RoleAgent 封装了：
- 角色档案（从 resources/roles/*.md 加载的结构化 profile）
- 独立的 SupermemoryStore（物理/逻辑命名空间隔离）
- 三类运行时行为指令：记忆读写、协商决策、维护循环
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ── 常量 ────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 120  # LLM 请求超时秒数
MAX_RETRIES = 3

# 默认角色档案（当无法加载角色文件时使用）
DEFAULT_ROLE_PROFILE: dict[str, Any] = {
    "角色定义": {
        "身份与背景": "通用知识工作者，具备广泛的知识和灵活的思维模式。",
        "思维模式": "理性分析与创造性思维并重，善于从多角度审视问题。",
        "核心知识体系": "通识教育背景，具备跨学科知识整合能力。",
    },
    "记忆策略": {
        "编码偏好": "结构化记录，关注事实、逻辑关系和上下文脉络。",
        "检索偏好": "优先检索与当前任务主题相关度高的记忆，兼顾时间新近度。",
        "维护策略": "定期整理记忆，标记过时信息，强化高频使用的重要记忆。",
    },
}

# 角色类型到记忆 Agent 子角色的映射 —— 基于 role_selector 的角色分类
CATEGORY_MEMORY_ROLE: dict[str, str] = {
    "scientific_research": "科学史官",
    "data_analysis": "数据分析师",
    "literary_creation": "叙事诗人",
    "engineering_technology": "技术档案员",
    "ai_computer_science": "知识架构师",
    "business_management": "战略记录员",
    "design": "视觉档案员",
    "finance_accounting": "财务审计员",
    "healthcare": "临床观察员",
    "education": "教育编年者",
    "legal": "法律文书官",
    "media_communication": "传播记录者",
    "agriculture_food": "农业日志员",
    "sports_fitness": "运动记录员",
    "environment_sustainability": "生态观察员",
    "transport_logistics": "物流追踪员",
    "real_estate_construction": "工程档案员",
    "entertainment_arts": "文艺档案员",
    "public_service_social_work": "公共服务记录员",
    "manufacturing_production": "生产记录员",
    "emerging_professions": "未来观察员",
    "other_professional": "专业记录员",
}


@dataclass
class NominationSpeech:
    """Agent 在 GroupChat 中发表的提名发言。"""
    agent_id: str
    role_name: str
    relevance_score: float       # 0-1 关联度评分
    reasoning: str               # 评分理由
    memory_plan: str             # 计划如何读写记忆
    collaboration_expectation: str  # 对协作的期望


@dataclass
class MemoryReport:
    """角色记忆报告 —— Agent 完成记忆检索后的输出。"""
    agent_id: str
    role_name: str
    memory_role: str             # 记忆子角色名
    query_rewritten: str         # 角色化改写后的查询
    retrieved_items: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""          # 基于记忆的推理/判断
    uncertainties: list[str] = field(default_factory=list)  # 不确定性或缺失标记
    storage_decisions: list[str] = field(default_factory=list)  # 决定存储的新记忆


class RoleAgent:
    """基于角色 Profile 的记忆 Agent。

    每个 Agent 拥有独立的 SupermemoryStore 实例，通过命名空间隔离。
    角色档案被"翻译"为三类运行时行为指令：
    - 记忆读写指令
    - 协商决策指令
    - 维护循环指令
    """

    def __init__(
        self,
        agent_id: str,
        role_name: str,
        role_profile: dict[str, Any],
        category: str = "",
        memory_dir: Path | None = None,
        provider: Any = None,
        model: str = "",
        embedding_provider: Any = None,
        embedding_model: str = "",
    ) -> None:
        """初始化角色 Agent。

        Args:
            agent_id: Agent 唯一标识
            role_name: 角色名称（如"AI研究员"）
            role_profile: 从角色文件解析出的结构化档案
            category: 角色所属类别目录名
            memory_dir: 记忆存储目录
            provider: LLM provider（来自 summerclaw.providers）
            model: 模型名称
            embedding_provider: Embedding provider（可选）
            embedding_model: Embedding 模型名称
        """
        self.agent_id = agent_id
        self.role_name = role_name
        self.role_profile = role_profile
        self.category = category
        self.memory_role = self._derive_memory_role(category, role_name)
        self.provider = provider
        self.model = model
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

        # 独立的记忆存储目录
        if memory_dir is None:
            memory_dir = Path("./memory")
        self.memory_dir = memory_dir / f"agent_{agent_id}"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 延迟初始化 SupermemoryStore（避免循环依赖）
        self._store: Any = None

        # 构建行为指令
        self._read_instruction = self._build_read_instruction()
        self._write_instruction = self._build_write_instruction()
        self._nomination_instruction = self._build_nomination_instruction()
        self._maintenance_instruction = self._build_maintenance_instruction()

    # ── 角色档案解析 ────────────────────────────────────────────────

    @staticmethod
    def _derive_memory_role(category: str, role_name: str) -> str:
        """根据角色类别推导记忆子角色名称。"""
        if category in CATEGORY_MEMORY_ROLE:
            return CATEGORY_MEMORY_ROLE[category]
        # 直接映射常见记忆角色名
        memory_roles = {
            "侦探": "探案记录官",
            "史官": "历史编年者",
            "诗人": "情感叙事者",
            "图书管理员": "知识管理员",
            "档案管理员": "档案记录官",
        }
        return memory_roles.get(role_name, f"记忆{role_name}")

    def _extract_profile_section(self, section_name: str) -> str:
        """从角色档案中提取指定字段的内容。"""
        for key, value in self.role_profile.items():
            if section_name in str(key):
                if isinstance(value, dict) and "内容" in value:
                    content = value["内容"]
                    if isinstance(content, str):
                        return content
                    return str(content)
                if isinstance(value, str):
                    return value
        return ""

    def _extract_yaml_field(self, field_path: list[str]) -> str:
        """按路径提取角色 profile 中的字段内容。"""
        data: Any = self.role_profile
        for key in field_path:
            if isinstance(data, dict):
                found = None
                for k, v in data.items():
                    if key in str(k):
                        found = v
                        break
                data = found
            else:
                return ""
        if isinstance(data, dict) and "内容" in data:
            content = data["内容"]
            return content if isinstance(content, str) else str(content)
        if isinstance(data, str):
            return data
        return ""

    # ── 行为指令构建 ────────────────────────────────────────────────

    def _build_read_instruction(self) -> str:
        """构建记忆读取指令 —— 描述 Agent 应如何查询和过滤记忆。"""
        identity = self._extract_yaml_field(["身份与背景"])
        thinking = self._extract_yaml_field(["思维模式"])
        knowledge = self._extract_yaml_field(["核心知识体系"])

        parts = [
            f"## 记忆读取指令 —— {self.role_name}（{self.memory_role}）",
            "",
            f"**身份视角**：{identity[:300] if identity else '通用知识工作者'}",
            "",
            "### 检索策略",
            f"- 从 {self.role_name} 的专业视角审视查询，识别关键词汇与潜在隐含需求",
        ]

        if thinking:
            parts.append(f"- 运用以下思维模式解读查询意图：{thinking[:200]}")

        parts.extend([
            "",
            "### 查询改写规则",
            "1. 将原始查询中的通用术语替换为本领域专业术语",
            "2. 追加领域相关的补充搜索维度（如从'代码质量'追加'架构合理性、性能瓶颈'）",
            "3. 在查询中注入角色特有的关注点（如风险偏好、美学标准、伦理考量）",
            "",
            "### 结果过滤规则",
            "1. 优先返回与角色核心知识体系相关的记忆条目",
            "2. 对检索结果按角色价值标准进行排序和加权",
            "3. 标记信息缺失、不确定性和矛盾之处",
        ])

        return "\n".join(parts)

    def _build_write_instruction(self) -> str:
        """构建记忆写入指令 —— 描述 Agent 应如何存储新记忆。"""
        identity = self._extract_yaml_field(["身份与背景"])
        boundaries = self._extract_yaml_field(["边界与局限"])
        methodology = self._extract_yaml_field(["工作流与方法论"])

        parts = [
            f"## 记忆写入指令 —— {self.role_name}（{self.memory_role}）",
            "",
            "### 存储决策原则",
            "1. 仅记录从角色视角看具有长期价值的信息",
            "2. 优先记录：事实记录、矛盾发现、推理线索、方法论洞察",
            "3. 可跳过：琐碎闲聊、已知常识、与角色无关的细节",
        ]

        if identity:
            parts.insert(1, f"**身份视角**：{identity[:200]}")

        if boundaries:
            parts.append(f"\n### 职责边界\n{boundaries[:300]}")

        parts.extend([
            "",
            "### 存储格式",
            "- 使用结构化标签标注记忆类型（事实/推理/矛盾/方法论）",
            "- 标注信息源、时间上下文和可信度",
            "- 若存储时发现与已有记忆的矛盾，主动标记并建立关联",
        ])

        return "\n".join(parts)

    def _build_nomination_instruction(self) -> str:
        """构建协商决策指令 —— 在 GroupChat 中如何提名和评审。"""
        identity = self._extract_yaml_field(["身份与背景"])
        thinking = self._extract_yaml_field(["思维模式"])
        decision = self._extract_yaml_field(["决策与判断框架"])

        parts = [
            f"## 协商决策指令 —— {self.role_name}",
            "",
            "### 自我评估规则",
            "分析任务后判断关联度时，考虑以下维度：",
            "1. 主题关联：任务主题是否与你的专业知识域高度重叠？",
            "2. 记忆储备：你的记忆中是否已有相关案例、知识或模式？",
            "3. 视角独特性：你能否提供其他角色难以提供的独特视角？",
            "4. 互补价值：你的参与是否能补充其他角色可能忽略的维度？",
        ]

        if decision:
            parts.append(f"\n### 决策框架\n{decision[:400]}")

        parts.extend([
            "",
            "### 提名发言格式",
            "1. 关联度评分（0-1的浮点数，说明理由）",
            "2. 计划如何读写记忆（具体操作描述）",
            "3. 对协作的期望（需要其他角色补充什么信息）",
        ])

        return "\n".join(parts)

    def _build_maintenance_instruction(self) -> str:
        """构建维护循环指令 —— 后台定期执行的记忆整理任务。"""
        maintenance_tasks = [
            "定期扫描记忆库，标记过时或低质量记忆",
            "交叉验证记忆间的逻辑一致性，标记矛盾",
            "整合分散的记忆条目为更凝练的知识单元",
            "更新记忆的重要性权重，清理低价值信息",
        ]
        parts = [
            f"## 维护循环指令 —— {self.role_name}",
            "",
            "### 定时维护任务",
        ]
        for i, task in enumerate(maintenance_tasks, 1):
            parts.append(f"{i}. {task}")
        return "\n".join(parts)

    # ── 记忆存储初始化 ──────────────────────────────────────────────

    def _ensure_store(self) -> Any:
        """延迟初始化 SupermemoryStore。"""
        if self._store is not None:
            return self._store

        try:
            from summerclaw.memory.supermemory_memory.store import SupermemoryStore
            self._store = SupermemoryStore(
                workspace=self.memory_dir,
                algo_name=f"agent_{self.agent_id}",
            )
            logger.debug(f"[{self.agent_id}] SupermemoryStore initialized at {self.memory_dir}")
        except Exception as e:
            logger.warning(f"[{self.agent_id}] Cannot initialize SupermemoryStore: {e}, using in-memory fallback")
            self._store = _InMemoryStore()

        return self._store

    # ── 记忆读写 API ─────────────────────────────────────────────────

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str | None:
        """存储原始或角色加工后的记忆条目。

        Args:
            text: 记忆文本
            metadata: 附加元数据（标签、来源、时间等）

        Returns:
            存储的节点 ID，或 None（若选择不存储）
        """
        store = self._ensure_store()
        metadata = metadata or {}

        # 注入角色标签
        metadata.setdefault("role", self.role_name)
        metadata.setdefault("memory_role", self.memory_role)
        metadata.setdefault("category", self.category)
        metadata.setdefault("timestamp", datetime.now().isoformat())

        try:
            from summerclaw.memory.supermemory_memory.store import MemoryNode
            node = MemoryNode(
                id=str(uuid.uuid4()),
                memory=text,
                content=metadata.get("content", text),
                document_date=datetime.now().strftime("%Y-%m-%d"),
                event_date=metadata.get("event_date"),
                is_static=metadata.get("is_static", False),
            )
            store.add_node(node)
            logger.debug(f"[{self.agent_id}] Stored: {text[:80]}...")
            return node.id
        except Exception:
            # fallback for in-memory store
            return store.store(text, metadata)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """基于语义检索返回记忆（角色化改写查询后执行）。

        Args:
            query_text: 原始查询文本
            top_k: 返回结果数量上限
            filter_tags: 可选的标签过滤列表

        Returns:
            记忆条目列表，每个包含 text、metadata、score
        """
        store = self._ensure_store()

        # 角色化查询改写（简化版——完整版应由 LLM 驱动）
        rewritten_query = self._rewrite_query(query_text)

        try:
            # 尝试语义搜索
            results = store.search_memories_by_keyword(rewritten_query, limit=top_k)
            if results:
                return [
                    {
                        "id": n.id,
                        "text": n.memory,
                        "metadata": {
                            "role": self.role_name,
                            "memory_role": self.memory_role,
                            "created_at": n.created_at,
                            "is_static": n.is_static,
                        },
                        "score": 1.0,
                    }
                    for n in results
                ]
            return []
        except Exception as e:
            logger.debug(f"[{self.agent_id}] search_memories_by_keyword failed: {e}")
            return []

    def _rewrite_query(self, query_text: str) -> str:
        """角色化查询改写 —— 纯 LLM 驱动从角色视角深度改写查询。

        此为同步回退方案，实际应由 generate_memory_report() 中的异步路径处理。
        """
        # 纯 LLM 模式下，此同步方法不应被调用，应由异步方法处理
        return query_text

    async def _llm_rewrite_query(self, query_text: str) -> str | None:
        """使用 LLM 从角色视角深度改写查询（异步）。"""
        if self.provider is None:
            return None

        identity = self._extract_yaml_field(["身份与背景"])
        knowledge = self._extract_yaml_field(["核心知识体系"])
        thinking = self._extract_yaml_field(["思维模式"])

        system_prompt = f"""你是 {self.role_name}（{self.memory_role}）。

你的身份与背景：{identity[:300] if identity else '通用知识工作者'}
你的核心知识体系：{knowledge[:300] if knowledge else '通识教育背景'}
你的思维模式：{thinking[:200] if thinking else '理性分析与创造性思维并重'}

你正在准备从自己的记忆库中检索信息。请将用户的原始查询改写为更适合从你的角色视角检索的版本：
1. 将通用术语替换为本领域的专业术语
2. 追加与你专业领域相关的补充搜索维度
3. 注入你角色特有的关注点和判断标准

请直接返回改写后的查询文本（中文，不超过 200 字），不要添加任何解释。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"原始查询：{query_text}\n\n请改写为从{self.role_name}视角的检索查询："},
        ]

        response = await self._call_llm(messages)
        if response:
            return response.strip()
        return None

    # ── 协商阶段行为 ─────────────────────────────────────────────────

    async def evaluate_task_relevance(self, task_description: str) -> NominationSpeech:
        """自我评估与提名 —— 纯 LLM 驱动分析任务关联度，生成提名发言。

        使用 LLM 深度分析任务与角色档案的语义关联。

        Args:
            task_description: 任务描述

        Returns:
            NominationSpeech 对象
        """
        if self.provider is None:
            raise RuntimeError("纯 LLM 模式需要 provider，请初始化时传入 provider 和 model")

        speech = await self._llm_evaluate_task(task_description, 0.0)
        if speech is not None:
            return speech

        # 如果 LLM 评估失败，抛出异常
        raise RuntimeError(f"[{self.agent_id}] LLM 评估任务关联度失败，无法生成提名发言")

    def _compute_base_relevance(self, task_description: str) -> float:
        """基于关键词匹配计算基础关联度。"""
        knowledge = self._extract_yaml_field(["核心知识体系"])
        identity = self._extract_yaml_field(["身份与背景"])

        # 合并角色描述文本
        combined = (knowledge + " " + identity + " " + self.role_name).lower()
        task_lower = task_description.lower()

        # 提取词语（中文按2-4字滑窗，英文按单词）
        def extract_keywords(text: str) -> set[str]:
            words = set()
            # 提取英文/数字词
            words.update(re.findall(r'[a-z0-9]+', text))
            # 提取中文2-4字词组
            for length in [2, 3, 4]:
                for i in range(len(text) - length + 1):
                    chunk = text[i:i+length]
                    # 只保留纯中文或中英混合的有意义词组
                    if any('\u4e00' <= c <= '\u9fff' for c in chunk):
                        # 过滤纯标点/数字
                        if re.search(r'[\u4e00-\u9fff]', chunk):
                            words.add(chunk)
            return words

        role_keywords = extract_keywords(combined)
        task_keywords = extract_keywords(task_lower)

        if not role_keywords:
            return 0.1

        overlap = len(role_keywords & task_keywords)
        # 使用更宽容的分母
        score = min(1.0, overlap / max(min(len(role_keywords), len(task_keywords)) * 0.15, 1))
        return max(0.05, round(score, 2))

    async def _llm_evaluate_task(
        self,
        task_description: str,
        base_relevance: float,
    ) -> NominationSpeech | None:
        """使用 LLM 深度评估任务关联度并生成提名发言。"""
        if self.provider is None:
            return None

        system_prompt = self._build_nomination_system_prompt()
        user_prompt = f"""## 当前任务描述

{task_description}

## 你的角色

你是 **{self.role_name}**（记忆子角色：{self.memory_role}）。

## 要求

请评估你与此任务的关联度，并以 JSON 格式返回：

```json
{{
    "relevance_score": 0.0-1.0,
    "reasoning": "关联度评估的详细理由",
    "memory_plan": "计划如何读写记忆",
    "collaboration_expectation": "对协作的期望"
}}
```
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self._call_llm(messages)
        if response is None:
            return None

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                return NominationSpeech(
                    agent_id=self.agent_id,
                    role_name=self.role_name,
                    relevance_score=float(data.get("relevance_score", base_relevance)),
                    reasoning=str(data.get("reasoning", "")),
                    memory_plan=str(data.get("memory_plan", "")),
                    collaboration_expectation=str(data.get("collaboration_expectation", "")),
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"[{self.agent_id}] Failed to parse LLM evaluation: {e}")

        return None

    def _build_nomination_system_prompt(self) -> str:
        """构建协商评估的系统提示词。"""
        return f"""{self._nomination_instruction}

{self._read_instruction}

当前你的角色是 {self.role_name}（{self.memory_role}）。

在评估任务时，请严格从你的角色视角出发，诚实评估你的关联度。
如果任务与你的专业领域无关，请给出较低的关联度评分并说明原因。
"""

    # ── 投票阶段行为 ─────────────────────────────────────────────────

    async def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        task_description: str,
        group_chat: Any = None,
    ) -> list[str]:
        """纯 LLM 驱动的候选 Agent 排序。

        使用 LLM 综合评估每个候选者的关联度、视角互补性和协作价值，
        给出排序。

        Args:
            candidates: 候选列表，每项含 agent_id、role_name、relevance_score、category 等
            task_description: 任务描述
            group_chat: GroupChat 协调空间（用于获取提名和评审上下文）

        Returns:
            按偏好排序的 agent_id 列表（第一位是最偏好）
        """
        if self.provider is None:
            raise RuntimeError("纯 LLM 模式需要 provider，请初始化时传入 provider 和 model")

        llm_ranking = await self._llm_rank_candidates(
            candidates, task_description, group_chat
        )
        
        if llm_ranking and len(llm_ranking) >= len(candidates) * 0.5:
            # 验证返回的 ID 都在候选列表中
            valid_ids = {c["agent_id"] for c in candidates}
            filtered = [aid for aid in llm_ranking if aid in valid_ids]
            if len(filtered) >= 1:
                # 追加 LLM 遗漏的候选者到末尾
                for c in candidates:
                    if c["agent_id"] not in filtered:
                        filtered.append(c["agent_id"])
                return filtered
        
        # 如果 LLM 排序失败，抛出异常
        raise RuntimeError(f"[{self.agent_id}] LLM 候选排序失败，无法完成投票")

    async def _llm_rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        task_description: str,
        group_chat: Any = None,
    ) -> list[str] | None:
        """使用 LLM 深度分析并排序候选 Agent。"""
        if self.provider is None:
            return None

        # 构建候选信息
        candidates_text = "\n".join([
            f"- ID: {c['agent_id']} | 角色: {c.get('role_name', '?')} | "
            f"关联度: {c.get('relevance_score', 0):.2f} | 类别: {c.get('category', '?')}"
            for c in candidates
        ])

        # 获取 GroupChat 中的评审上下文
        chat_context = ""
        if group_chat is not None:
            try:
                chat_context = group_chat.get_candidates_context()
            except Exception:
                pass

        system_prompt = f"""你是 {self.role_name}（{self.memory_role}），正在参与多角色协作系统的Agent选择。

你需要对候选 Agent 进行排序，选出最适合参与当前任务的同伴。

排序时请综合考虑：
1. **任务关联度**：候选者的专业知识与任务的匹配程度
2. **视角互补性**：候选者是否能提供你角色视角可能遗漏的维度
3. **协作价值**：候选者的参与是否能提升整体协作质量
4. **多样性**：避免选择观点过于相似的 Agent

请返回 JSON 格式的排序结果。"""

        user_prompt = f"""## 当前任务
{task_description}

## 候选 Agent 列表
{candidates_text}

## 对话上下文
{chat_context or '(无额外上下文)'}

请从你的角色视角，对以上候选 Agent 按偏好排序，返回 JSON：

```json
{{
    "ranking": ["agent_id_1", "agent_id_2", ...],
    "reasoning": "排序理由（200字以内）"
}}
```

ranking 数组应包含所有候选者的 agent_id，按最偏好到最不偏好排列。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self._call_llm(messages)
        if response is None:
            return None

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                ranking = data.get("ranking", [])
                if isinstance(ranking, list) and len(ranking) > 0:
                    return ranking
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"[{self.agent_id}] Failed to parse LLM ranking: {e}")

        return None

    # ── 记忆报告生成 ─────────────────────────────────────────────────

    async def generate_memory_report(
        self,
        task_description: str,
        query_results: list[dict[str, Any]],
        other_reports: list[dict[str, Any]] | None = None,
    ) -> MemoryReport:
        """生成角色记忆报告 —— 纯 LLM 驱动的查询改写、推理和不确定性标记。

        使用 LLM 从角色视角深度改写查询、分析检索结果、生成推理。

        Args:
            task_description: 原始任务描述
            query_results: 从自身记忆库检索到的结果
            other_reports: 其他 Agent 的报告（用于交叉参考）

        Returns:
            MemoryReport 对象
        """
        if self.provider is None:
            raise RuntimeError("纯 LLM 模式需要 provider，请初始化时传入 provider 和 model")

        # LLM 驱动查询改写
        rewritten_query = task_description
        try:
            llm_rewritten = await self._llm_rewrite_query(task_description)
            if llm_rewritten:
                rewritten_query = llm_rewritten
        except Exception as e:
            logger.warning(f"[{self.agent_id}] LLM 查询改写失败: {e}")

        report = MemoryReport(
            agent_id=self.agent_id,
            role_name=self.role_name,
            memory_role=self.memory_role,
            query_rewritten=rewritten_query,
            retrieved_items=query_results,
        )

        # LLM 主导报告生成
        try:
            enriched = await self._llm_enrich_report(
                task_description, query_results, other_reports
            )
            if enriched:
                report.reasoning = enriched.get("reasoning", "")
                report.uncertainties = enriched.get("uncertainties", [])
                report.storage_decisions = enriched.get("storage_decisions", [])
        except Exception as e:
            logger.warning(f"[{self.agent_id}] LLM 报告生成失败: {e}")
            # 如果 LLM 报告生成失败，使用基础推理
            if query_results:
                report.reasoning = (
                    f"从 {self.role_name} 视角检索到 {len(query_results)} 条相关记忆。"
                    f"这些记忆覆盖了任务涉及的部分领域，但完整度需要进一步验证。"
                )
            else:
                report.reasoning = f"从 {self.role_name} 视角未检索到直接相关的记忆条目。"
                report.uncertainties.append("记忆库中缺乏与当前任务直接关联的信息")

        return report

    async def _llm_enrich_report(
        self,
        task_description: str,
        query_results: list[dict[str, Any]],
        other_reports: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """使用 LLM 丰富记忆报告内容。"""
        if self.provider is None:
            return None

        memory_summary = "\n".join([
            f"- [{r.get('id', '?')[:8]}] {r.get('text', '')[:200]}"
            for r in query_results[:10]
        ])

        other_summary = ""
        if other_reports:
            other_summary = "\n".join([
                f"- [{r.get('role_name', '?')}]: {r.get('reasoning', '')[:200]}"
                for r in other_reports[:5]
            ])

        system_prompt = self._build_report_system_prompt()
        user_prompt = f"""## 任务描述
{task_description}

## 从自身记忆库检索到的记忆条目
{memory_summary or '(无相关记忆)'}

## 其他角色的记忆报告（供交叉参考）
{other_summary or '(暂无其他报告)'}

请以 JSON 格式返回你的记忆分析报告：
```json
{{
    "reasoning": "基于记忆的推理/判断（200字以内）",
    "uncertainties": ["不确定性1", "不确定性2"],
    "storage_decisions": ["计划存储的新记忆描述1", "计划存储的新记忆描述2"]
}}
```
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self._call_llm(messages)
        if response is None:
            return None

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                return {
                    "reasoning": str(data.get("reasoning", "")),
                    "uncertainties": data.get("uncertainties", []),
                    "storage_decisions": data.get("storage_decisions", []),
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[{self.agent_id}] Failed to parse LLM report: {e}")

        return None

    def _build_report_system_prompt(self) -> str:
        """构建记忆报告的系统提示词。"""
        return f"""{self._read_instruction}

{self._write_instruction}

当前你的角色是 {self.role_name}（{self.memory_role}）。

请基于检索到的记忆条目，从你的角色视角进行分析：
1. 推断当前任务的关键信息
2. 标记记忆中的不确定性和空白
3. 决定是否需要存储新的记忆条目
"""

    # ── LLM 调用辅助 ─────────────────────────────────────────────────

    async def _call_llm(self, messages: list[dict[str, str]]) -> str | None:
        """统一的 LLM 调用封装。"""
        if self.provider is None:
            return None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                from summerclaw.providers.base import LLMResponse

                response: LLMResponse = await asyncio.wait_for(
                    self.provider.chat_with_retry(
                        messages=messages,
                        model=self.model,
                        retry_mode="standard",
                    ),
                    timeout=REQUEST_TIMEOUT,
                )

                if response.finish_reason == "error":
                    logger.warning(
                        f"[{self.agent_id}] LLM error (attempt {attempt}): {response.content[:200]}"
                    )
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 * attempt)
                        continue
                    return None

                return response.content or ""

            except asyncio.TimeoutError:
                logger.warning(f"[{self.agent_id}] LLM timeout (attempt {attempt})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)
                    continue
                return None
            except Exception as e:
                logger.warning(f"[{self.agent_id}] LLM exception: {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)
                    continue
                return None

        return None

    # ── 统计与调试 ───────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """返回 Agent 记忆统计。"""
        store = self._ensure_store()
        try:
            return store.stats()
        except Exception:
            return {"agent_id": self.agent_id, "role_name": self.role_name}

    def __repr__(self) -> str:
        return f"RoleAgent(id={self.agent_id}, role={self.role_name}, mem_role={self.memory_role})"


# ── 内置简易内存存储（回退方案） ────────────────────────────────────


class _InMemoryStore:
    """当无法加载 SupermemoryStore 时使用的简易内存回退方案。"""

    def __init__(self) -> None:
        self._memories: list[dict[str, Any]] = []

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        mem_id = str(uuid.uuid4())
        self._memories.append({
            "id": mem_id,
            "text": text,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        })
        return mem_id

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """简单的关键词匹配检索。"""
        query_lower = query_text.lower()
        scored = []
        for mem in self._memories:
            text_lower = mem["text"].lower()
            score = 0
            if query_lower in text_lower:
                score += 10
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            score += len(query_words & text_words) * 3
            if score > 0:
                scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"id": m["id"], "text": m["text"], "metadata": m["metadata"], "score": float(s)}
                for s, m in scored[:top_k]]

    def stats(self) -> dict[str, Any]:
        return {"total_memories": len(self._memories)}