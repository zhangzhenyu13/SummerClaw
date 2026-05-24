import asyncio
from experimental.multi_agent_memory import MultiAgentMemorySystem
from summerclaw.providers import OpenAICompatProvider
from summerclaw.config import Config, load_config
import json

async def main():
    # 初始化系统：每轮最多选 3 个 Agent
    config : Config = load_config()
    provider = OpenAICompatProvider(api_key=config.providers.dashscope.api_key, api_base=config.providers.dashscope.api_base)
    system = MultiAgentMemorySystem(k=3, provider=provider, model="qwen3.6-plus")

    # 加载 AI 与数据分析相关角色的 Agent（最多 10 个）
    count = await system.initialize(
        max_agents=10,
        category_filter=["ai_computer_science", "data_analysis"]
    )
    print(f"已加载 {count} 个 Agent")

    # 处理任务
    result = await system.process_task("分析最新AI大模型发展趋势")

    print(f"选中 Agent: {len(result['selected_agents'])} 个")
    print(f"整合摘要: {result['summary'][:300]}")
    print(f"耗时: {result['elapsed_seconds']:.1f}s")
    print(f"写入统计: {result['write_stats']}")

    # 查看系统状态
    stats = system.get_system_stats()
    print(f"系统统计: {stats['total_agents']} 个 Agent 就绪")

    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

asyncio.run(main())