# 多角色记忆系统 (Multi-Agent Memory System)

基于 `resources/roles` 下 1000+ 个结构化角色定义，构建 N 个角色驱动的记忆 Agent，通过 GroupChat 自主协商动态选出 ≤K 个 Agent 参与记忆读写，实现**视角分化、去中心化、自组织**的群体记忆管理。

---

## 1. 设计理念

传统的记忆系统是"一个大脑记一切"——单一视角存储和检索，无法捕捉同一信息的多元理解。本系统将 **Avilization 群体演进智能**思想落地到记忆层：

- **多视角共存**：同一事件由不同角色（如侦探、诗人、史官）独立编码，保留认知多样性
- **去中心化协商**：没有中央调度器，Agent 通过 GroupChat 自主提名、互评、投票
- **记忆即社会**：记忆在多角色的碰撞中不断丰富、验证和演进，而非被简单存储

---

## 2. 三层架构

```
用户 / 任务输入
        │
        ▼
┌─────────────────────────────────┐
│      GroupChat 协调空间          │  ← 协商协作层
│  (Agent 自由讨论、提名、投票)     │
└────────────┬────────────────────┘
             │ Borda 计票选出 ≤K 个 Agent
      ┌──────┴──────┐
      │  被选中的    │
      │ 记忆Agent组  │               ← 角色实例层
      │  (K 个 RoleAgent)            │
      └──────┬──────┘
             │ 每个 Agent 独立调用
      ┌──────┴──────┐
      │ Supermemory │               ← 记忆存储层
      │ (独立实例/命名空间隔离)       │
      └─────────────┘
```

| 层 | 组件 | 职责 |
|----|------|------|
| **角色实例层** | `RoleAgent` | N 个 Agent，每个拥有独立 Supermemory 实例 + 角色化行为指令 |
| **协商协作层** | `GroupChat` + `SelectionProtocol` | 去中心化的提名、互评、Borda 计票 |
| **记忆存储层** | `MemoryOperations` | 视角分化的并行读写 + 结果聚合 |

---

## 3. 核心组件

### 3.1 RoleAgent —— 单角色记忆 Agent

每个 Agent 封装了三要素：

- **角色档案**：从 `resources/roles/*.md` 加载的 8-10 节结构化 YAML/Markdown 档案，包含身份背景、思维模式、核心知识体系、决策框架等
- **独立记忆库**：每个 Agent 拥有独立的 `SupermemoryStore` 实例（通过不同目录/命名空间物理隔离），底层算法完全共享，角色化逻辑全部上移到 Agent 层
- **三类行为指令**（从角色档案自动"翻译"生成）：

| 行为指令 | 构建方法 | 内容 |
|----------|----------|------|
| `_read_instruction` | 读取指令 | 如何改写查询、过滤结果、加权排序——"从本角色视角审视信息" |
| `_write_instruction` | 写入指令 | 存储决策原则、职责边界、格式要求——"仅记录有长期价值的角色相关信息" |
| `_nomination_instruction` | 协商指令 | 自我评估规则（主题关联/记忆储备/视角独特性/互补价值）、提名发言格式 |

**记忆子角色映射**：根据角色类别（如 `ai_computer_science`）自动推导记忆子角色名（如"知识架构师"），22 个类别均有对应映射。

**纯 LLM 驱动**：所有核心逻辑均依赖 LLM 实现，无 LLM 时系统无法运行。必须传入 LLM provider 以获得完整的语义理解和决策能力。

### 3.2 GroupChat —— 去中心化协商空间

完全不设中央调度器，Agent 自由在频道中发言。9 种消息类型覆盖完整协作流程：

| 消息类型 | 用途 |
|----------|------|
| `SYSTEM` | 系统广播（任务通知、阶段切换） |
| `NOMINATION` | Agent 提名发言（含关联度评分和理由） |
| `REVIEW` | Agent 互评发言（补充或质疑） |
| `VOTE` | 投票消息 |
| `RESULT` | 选举结果广播 |
| `MEMORY_REPORT` | 角色记忆报告 |
| `SUMMARY` | 整合摘要 |
| `MAINTENANCE` | 维护辩论 |
| `INFO` | 一般信息 |

**上下文管理**：内置轮次截断（默认保留 20 轮）+ 摘要压缩（超出时自动压缩旧轮次为摘要），防止 token 爆炸。

### 3.3 SelectionProtocol —— 纯 LLM 驱动的 Borda 计票动态选择

从 N 个 Agent 中选出最多 K 个参与记忆读写的三阶段协议：

**阶段 1 —— 自我评估与提名**（N 个 Agent 并行发言）
- 纯 LLM 驱动：每个 Agent 使用 LLM 深度评估对当前任务的语义关联度
- 生成 `NominationSpeech`：关联度评分(0-1)、理由、读写计划、协作期望

**阶段 2 —— LLM 驱动群体评审**（最多 2 轮对话）
- 纯 LLM 驱动：每个 Agent 使用 LLM 分析所有提名，发表有见地的评审
- 发现被低估/高估的候选者，评估组合多样性

**阶段 3 —— LLM 驱动 Borda 计票**
- 纯 LLM 驱动：每个 Agent 使用 LLM 对候选者深度排序（综合关联度、互补性、多样性）
- Borda 计分汇总，取前 K 名

### 3.4 MemoryOperations —— 纯 LLM 驱动的视角分化并行读写

**并行读取（Recall）**：
1. 每个选中的 Agent 使用 LLM 从角色视角深度改写查询
2. 调用自身 `SupermemoryStore` 检索记忆
3. LLM 生成 `MemoryReport`：角色推理、不确定性标记、存储建议
4. 所有报告广播到 GroupChat

**并行写入（Store）**：
- 纯 LLM 驱动：每个 Agent 的存储决策来自 LLM 生成的 `storage_decisions`
- 差异化行为：不同角色存入不同类型信息（事实/推理/矛盾/方法论/叙事）
- Agent 可以选择不写入——"没有值得记录的信息"本身就是角色策略

**结果汇总**：
- 纯 LLM 驱动：综合分析所有报告，自行发现共识和分歧
- 生成结构化四部分摘要（核心发现/共识/分歧/建议）

### 3.5 MultiAgentMemorySystem —— 主入口

对外暴露的顶层 API，串联完整协作流程：

```
initialize(max_agents, category_filter)
    │
    ▼ 扫描角色文件 → 加载档案 → 创建 N 个 RoleAgent
    │
process_task(task_description)
    │
    ├─ 1. 广播任务到 GroupChat
    ├─ 2. SelectionProtocol.select() ── 选出 ≤K 个 Agent
    ├─ 3. MemoryOperations.parallel_read() ── 视角分化的并行检索
    ├─ 4. MemoryOperations.parallel_write() ── 差异化的并行存储
    └─ 5. MemoryOperations.aggregate_results() ── 整合摘要
```

还提供 `maintenance_cycle()`（定期后台维护）、`get_system_stats()`（系统统计）等方法。

---

## 4. 任务处理完整流程

以一个具体流程为例：用户提交任务"分析最新 AI 大模型发展趋势"：

```
N=6 个 Agent 就绪 (AI研究员, LLM大模型工程师, 诗人, 史官, 数据分析师, 图像处理工程师)
K=3

┌─ 第 1 轮：系统广播任务 ─────────────────────────────────┐
│ 📢 "新任务到达：分析最新 AI 大模型发展趋势..."          │
└─────────────────────────────────────────────────────────┘

┌─ 第 2 轮：并行提名 (6 个 Agent 同时发言) ─────────────────┐
│ 🗳️ AI研究员：关联度 0.85，"任务与我的核心知识域高度重叠"  │
│ 🗳️ LLM大模型工程师：关联度 0.92，"可直接从技术视角分析"   │
│ 🗳️ 数据分析师：关联度 0.65，"可从数据趋势维度补充"        │
│ 🗳️ 诗人：关联度 0.12，"任务与文学创作领域关联度低"        │
│ 🗳️ 图像处理工程师：关联度 0.08，"任务与图像领域无关"      │
│ 🗳️ 史官：关联度 0.35，"可从历史演进视角记录"              │
└─────────────────────────────────────────────────────────┘

┌─ 第 3-4 轮：评审收敛 ───────────────────────────────────┐
│ 💬 AI研究员："史官的关联度虽低但历史视角有互补价值"      │
│ 💬 LLM大模型工程师："数据分析师的趋势量化视角值得加入"    │
└─────────────────────────────────────────────────────────┘

┌─ Borda 计票结果 ────────────────────────────────────────┐
│ 1. LLM大模型工程师 (Borda: 12.0)                         │
│ 2. AI研究员 (Borda: 10.0)                                │
│ 3. 数据分析师 (Borda: 8.0)                               │
│ ─── 以下未入选 ───                                       │
│ 4. 史官 (Borda: 4.0)                                     │
│ 5. 诗人 (Borda: 1.0)                                     │
│ 6. 图像处理工程师 (Borda: 0.0, 关联度过低被排除)          │
└─────────────────────────────────────────────────────────┘

┌─ 第 5 轮：并行记忆读取 (3 个 Agent 同时检索) ────────────┐
│ 📋 LLM大模型工程师：检索到 5 条相关记忆，核心判断："..."  │
│ 📋 AI研究员：检索到 3 条相关记忆，核心判断："..."         │
│ 📋 数据分析师：检索到 2 条相关记忆，核心判断："..."        │
└─────────────────────────────────────────────────────────┘

┌─ 第 6 轮：并行记忆写入 (3 个 Agent 各自决定存储策略) ───┐
│ ✏️ LLM大模型工程师：存储了 3 条新记忆                     │
│ ✏️ AI研究员：存储了 2 条新记忆                            │
│ ✏️ 数据分析师：存储了 2 条新记忆                           │
└─────────────────────────────────────────────────────────┘

┌─ 第 7 轮：整合摘要 ─────────────────────────────────────┐
│ 📊 共识：大模型技术正从规模竞赛转向效率优化...            │
│ 🔀 分歧：LLM工程师关注架构创新，AI研究员关注应用落地...   │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 使用方式

```python
import asyncio
from experimental.multi_agent_memory import MultiAgentMemorySystem
from summerclaw.providers import OpenAICompatProvider
from summerclaw.config import load_config

async def main():
    # 先加载配置并创建 LLM provider（纯 LLM 驱动模式要求）
    config = load_config()
    provider = OpenAICompatProvider(
        api_key=config.providers.dashscope.api_key,
        api_base=config.providers.dashscope.api_base,
    )

    # 传入 provider 和 model 初始化系统
    system = MultiAgentMemorySystem(k=3, provider=provider, model="qwen-plus")

    # 加载 AI 与数据分析相关角色的 Agent（最多 10 个）
    count = await system.initialize(
        max_agents=10,
        category_filter=["ai_computer_science", "data_analysis"]
    )
    print(f"已加载 {count} 个 Agent")

    # 处理任务（所有决策均由 LLM 驱动）
    result = await system.process_task("分析最新AI大模型发展趋势")

    print(f"选中 Agent: {len(result['selected_agents'])} 个")
    print(f"整合摘要: {result['summary'][:300]}")
    print(f"耗时: {result['elapsed_seconds']:.1f}s")
    print(f"写入统计: {result['write_stats']}")

    # 查看系统状态
    stats = system.get_system_stats()
    print(f"系统统计: {stats['total_agents']} 个 Agent 就绪")

asyncio.run(main())
```

必选参数：
- `k`：每轮最多选择的 Agent 数（推荐 3~5）
- `provider` / `model`：**LLM provider 和模型名（必须传入，纯 LLM 驱动模式）**

可选参数：
- `max_agents`：初始化加载的 Agent 总数
- `category_filter`：按类别过滤（如 `["ai_computer_science", "healthcare"]`）
- `role_filter`：按角色名过滤

---

## 6. 设计决策

| 设计点 | 选择 | 理由 |
|--------|------|------|
| **决策范式** | 纯 LLM 驱动 | LLM 深度理解语义和上下文，无规则兜底逻辑 |
| **选择机制** | GroupChat 自主提名 + LLM 评审 + LLM 排序 Borda 计票 | LLM 综合评估关联度、互补性、多样性，避免规则盲区 |
| **Agent 排序** | LLM 驱动 `rank_candidates` | 从角色视角深度分析所有候选者的协作价值 |
| **查询改写** | LLM 驱动 `_llm_rewrite_query` | 将通用查询深度改写为角色视角的专业检索语句 |
| **评审阶段** | LLM 驱动 `_llm_review` | 综合分析提名发言，发现被低估/高估的候选者 |
| **K 值控制** | 动态，固定上限 | 防止过多 Agent 涌入导致共识困难，同时保证多样性 |
| **读写完全并行且独立** | 是 | 最大化利用多视角，避免视角污染，真实反映角色差异 |
| **结果融合** | LLM 综合分析 | LLM 自行发现共识分歧，保留认知多样性 |
| **底层 Supermemory 不改** | 是 | 完全解耦，通过上层的角色 Prompt 实现差异化 |
| **通信负担** | 轮次截断 + 摘要压缩 | 控制 token 消耗，避免上下文爆炸 |
| **LLM 依赖** | 必须 | 纯 LLM 驱动模式，无 LLM 时系统无法运行 |
| **角色文件格式** | 双格式兼容 | 同时支持 YAML front matter 和纯 Markdown 格式 |

---

## 7. 文件结构

```
experimental/multi_agent_memory/
├── INFO.md              # 本文件
├── __init__.py           # 包导出，统一 API
├── role_agent.py         # 单角色记忆 Agent（纯 LLM 驱动评估/排序/改写）
├── group_chat.py         # 去中心化 GroupChat（含 LLM 上下文构建）
├── selection.py          # 纯 LLM 驱动的 Borda 计票动态选择协议
├── memory_ops.py         # 纯 LLM 驱动的视角分化并行读写协调器
└── system.py             # MultiAgentMemorySystem 主入口
```

纯 LLM 驱动核心方法：
- `role_agent.py`: `_llm_evaluate_task`, `_llm_rank_candidates`, `_llm_rewrite_query`, `_llm_enrich_report`
- `selection.py`: `_llm_review`
- `memory_ops.py`: `_llm_integrate`, `_extract_consensus_and_divergences_llm`
- `group_chat.py`: `to_llm_context`, `get_candidates_context`

---

## 8. 与现有系统的关系

- **依赖 `summerclaw.memory.supermemory_memory`**：每个 RoleAgent 的底层记忆存储使用 SupermemoryStore
- **依赖 `summerclaw.providers`**：LLM provider 用于可选的协商增强和报告深化
- **复用 `resources/roles`**：直接从已有的 1000+ 个角色定义加载档案
- **与 `summerclaw.agent.role_selector` 互补**：role_selector 负责为单 Agent 选择角色；本系统让多角色**同时**参与并自主协商记忆读写