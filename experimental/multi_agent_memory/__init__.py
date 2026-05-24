"""多角色记忆系统 (Multi-Agent Memory System)

基于 resources/roles 角色定义，构建 N 个角色驱动的记忆 Agent，
通过 GroupChat 自主协商选出 ≤K 个 Agent 参与记忆读写，
实现视角分化的记忆管理。

LLM 主导设计：Agent 选择、评审、排序、查询改写、报告生成、
共识提取等核心决策均由 LLM 驱动，规则仅在 LLM 不可用时兜底。

核心组件：
- MultiAgentMemorySystem:  主系统入口
- RoleAgent:               单角色记忆 Agent（独立 Supermemory 实例）
- GroupChat:               去中心化协商空间（含 LLM 上下文构建）
- SelectionProtocol:       LLM 驱动的 Borda 计票动态选择协议
- MemoryOperations:        LLM 驱动的视角分化并行读写协调器

Usage::

    from experimental.multi_agent_memory import MultiAgentMemorySystem
    from summerclaw.providers import OpenAICompatProvider
    from summerclaw.config import load_config
    import asyncio

    async def main():
        config = load_config()
        provider = OpenAICompatProvider(
            api_key=config.providers.dashscope.api_key,
            api_base=config.providers.dashscope.api_base,
        )
        system = MultiAgentMemorySystem(k=3, provider=provider, model="qwen-plus")
        await system.initialize(max_agents=10)
        result = await system.process_task("分析AI发展趋势")
        print(result["summary"])

    asyncio.run(main())
"""

from experimental.multi_agent_memory.role_agent import (
    RoleAgent,
    NominationSpeech,
    MemoryReport,
    DEFAULT_ROLE_PROFILE,
)
from experimental.multi_agent_memory.group_chat import (
    GroupChat,
    ChatMessage,
    MessageType,
)
from experimental.multi_agent_memory.selection import (
    SelectionProtocol,
    SelectionResult,
    select_agents,
)
from experimental.multi_agent_memory.memory_ops import (
    MemoryOperations,
    process_memory_collaboration,
)
from experimental.multi_agent_memory.system import (
    MultiAgentMemorySystem,
    process_task_sync,
)

__all__ = [
    # 主入口
    "MultiAgentMemorySystem",
    "process_task_sync",
    # 核心组件
    "RoleAgent",
    "GroupChat",
    "SelectionProtocol",
    "MemoryOperations",
    # 数据类
    "NominationSpeech",
    "MemoryReport",
    "SelectionResult",
    "ChatMessage",
    "MessageType",
    # 常量
    "DEFAULT_ROLE_PROFILE",
    # 便捷函数
    "select_agents",
    "process_memory_collaboration",
]