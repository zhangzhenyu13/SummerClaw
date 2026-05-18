# Layerga 记忆算法

> GenericAgent 风格的 L0-L4 分层记忆系统——Agent 既是执行者也是记忆管理员，信息按价值和可推断性自动分类到五层。

## 核心思想

大多数记忆系统把所有信息一视同仁地"塞进"一个文本文件或向量库，但这导致三个问题：
1. **Token 污染**：LLM 上下文被大量低价值信息占据
2. **检索低效**：要找一条配置信息，需要在所有记忆中全文搜索
3. **退化**：没有维护机制，记忆随时间膨胀退化

LayergaMemory 的核心哲学：**信息的价值分布高度不均匀，记忆系统应该根据信息的确定性、可推断性和复用频率，将信息自动分配到不同层次。**

四大核心原则：
- **"No Execution, No Memory"**（无执行，无记忆）：只有经过工具调用验证的**执行结果**才能写入记忆
- **"Minimum Sufficient Pointer"**（最小充分指针）：上层只保留最短的定位符，不复制内容
- **"Self-Evolution"**（自进化）：Agent 自主决定记什么、记在哪、怎么记
- **"LLM-driven Management"**（LLM 驱动管理）：Agent 同时是执行者和记忆图书管理员

## 架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Layerga L0-L4 分层记忆系统                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  L0 • 宪法 (Constitution)    layerga/constitution.md             │ │
│  │      元规则 — 记忆系统的"法律"：什么能存、什么不能存、存到哪层      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │ 驱动决策                               │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │  L1 • 洞察索引   memory/layer_insight.txt  (≤30 行硬约束)         ││
│  │      最小导航 — 每个 L2/L3 条目的关键词指针                        ││
│  │      Tier 1: 高频场景 key→value  (ROI 高)                        ││
│  │      Tier 2: 低频场景 keyword only  (ROI 中)                      ││
│  │      [RULES] 跨任务陷阱规则 (≤1 压缩句，ROI 最高)                  ││
│  └──────────────────────────────────────────────────────────────────┘│
│                              │ 指向                                  │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │  L2 • 事实库   memory/layer_facts.txt  (## [SECTION] 块)         ││
│  │      环境事实 — LLM 无法零样本推断的信息                           ││
│  │      API 密钥、代理端口、目录路径、配置文件位置...                  ││
│  └──────────────────────────────────────────────────────────────────┘│
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │  L3 • 任务记录   memory/sop/*.md + *.py                          ││
│  │      作战标准 — 可复用工作流 + 实用脚本                             ││
│  │      只记录：隐藏前提条件 + 典型陷阱                                ││
│  │      不记录：普通步骤、可推断路径                                   ││
│  └──────────────────────────────────────────────────────────────────┘│
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │  L4 • 会话归档   memory/archives/all_histories.txt               ││
│  │      压缩历史 — 已处理会话的摘要归档                                ││
│  │      [2024-03-15 14:30] 用户询问了代理配置...                      ││
│  └──────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│  写入路径：                                                           │
│    工具调用成功 → VerifiedFact → L0DecisionTree.classify()            │
│        │                                                             │
│        ├── L1_RULES → 追加到 insight [RULES] 下                       │
│        ├── L2 → 追加为 facts 新 ## [SECTION]                          │
│        ├── L3_SOP → 创建 sop/<name>_sop.md                           │
│        ├── L3_SCRIPT → 创建 sop/<name>.py                             │
│        └── DROP → 丢弃（常识/易失状态/冗余）                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## L0 宪法 — 记忆法是核心

```
layerga/constitution.md

## 0. Core Axioms
1. No Execution, No Memory
2. Sanctity of Verified Data
3. No Volatile State (timestamps, PIDs, session IDs...)
4. Minimum Sufficient Pointer

## 1. Layer Architecture
[...每一层的定义和写入规则...]

## 2. Decision Tree
1. 是环境特定事实（LLM 无法推断）? → L2 (+ L1 同步)
2. 是通用操作规则? → L1 [RULES]
3. 是任务特定技术? → L3 (SOP 或 script)
4. 否则 → 丢弃
```

L0 不是静态文档，而是**可编程的决策引擎**——`L0DecisionTree` 类将其翻译为 Python 规则执行。

## L0DecisionTree — 可编程分类引擎

四个判定步骤，每个步骤由 regex 模式驱动：

**Step 0: 公理检查（硬约束）**
- 未经验证 → 拒绝（Axiom 1）
- 含易失状态 → 拒绝（Axiom 3）
- 超过 500 字符 → 警告（Axiom 4）

**Step 1: 环境事实检测**
```python
_ENV_FACT_PATTERNS = [
    r"\b(?:api[_\s]?key|proxy[_\s]?port|endpoint|directory|config)\b"
]
```
命中 → L2（首次同步到 L1 Tier 1）

**Step 2: 通用规则检测**
```python
_RULE_PATTERNS = [
    r"\b(?:never|always|must|禁止|必须|绝不)\b"
    r"\b(?:warning|caution|注意|小心)\b"
]
```
命中 ≥2 个 → L1 [RULES]（压缩为 1 句）

**Step 3: 任务技术检测**
```python
_TASK_TECH_PATTERNS = [
    r"\b(?:trick|hack|workaround|特殊|隐藏)\b"
    r"\b(?:retry|重试|多次尝试|花.*时间)\b"
]
```
命中 → L3（有代码片段走 L3_SCRIPT，否则 L3_SOP）

**Step 4: 丢弃** — 常识、问候语、标准方法等

### ROI 评估（L1 维护）

L1 只有 30 行空间，如何取舍？

```python
ROI = (mistake_probability × mistake_cost_tokens) / avg_tokens_per_line
```

- `mistake_probability`：没有这条信息时犯错的概率
- `mistake_cost_tokens`：犯错后修复的 token 成本
- 结果 ≥ `min_roi` (0.5) 才保留

这实现了**经济学视角的记忆管理**——每行 L1 都是经过成本效益分析的。

## 四大组件

### LayergaStore

继承 `MemoryStore`（获得所有标准文件 I/O），额外增加 L0-L4 分层文件：

| 层 | 方法 | 特点 |
|:--:|------|------|
| L0 | `read_constitution()`, `get_constitution_summary()` | 模板初始化，公理提取 |
| L1 | `read_insight()`, `write_insight()`, `patch_insight()` | 最小补丁修改 |
| L2 | `read_facts()`, `write_facts()`, `patch_facts()` | ## [SECTION] 管理 |
| L3 | `list_sops()`, `read_sop()`, `write_sop()`, `patch_sop()` | SOP + scripts |
| L4 | `append_archive()`, `read_archive()`, `compact_archives()` | 追加+裁剪 |

关键特性：所有层的写操作都支持 `patch_*` 方法——最小化文件修改，避免全量覆写。

### LayergaConsolidator

扩展 naive 的 `Consolidator`，在 `archive()` 中增加：

1. **提取验证事实**：从被归档消息中找出 `role=tool` 且状态为成功的消息
2. **L0 分类**：每个事实经 `L0DecisionTree.classify()` 判定写入层
3. **分层写入**：根据分类结果写入 L1/L2/L3
4. **L1 同步**：写入后自动调用 `sync_l1_index()`

### LayergaDream

扩展 naive 的 `Dream`，**三阶段**处理：

**Phase 1 — 分析（分层上下文注入）**：
- 标准 MEMORY.md/SOUL.md/USER.md
- **L0 宪法摘要**（核心公理）
- **L1 洞察索引**
- **L2 事实段列表**
- **L3 SOP 列表**
- **L2 事实全文** → LLM 能看到所有环境特定信息

**Phase 2 — 执行（分层编辑规则）**：
- AgentRunner 获得分层编辑指令（L1 ≤30 行硬约束、L2 最小补丁、L3 SOP 创建规则）
- 可创建 `dreamed-*` 技能

**Phase 3 — L1 清理（可选）**：
- 统计 L1 行数
- 若超过 `l1_max_lines`（30），按 ROI 排序
- 移除 ROI 最低的行直到回到限制范围内

### LayergaAutoCompact

扩展 naive 的 `AutoCompact`：

1. 会话压缩摘要写入 **L4 会话归档**（`append_archive()`）
2. 检查归档消息是否包含值得长期记忆的模式（`_maybe_trigger_long_term_update()`）
3. 若有 → 触发 L0 分类管线

## 存储布局

```
workspace/
├── SOUL.md
├── USER.md
├── MEMORY.md
├── layerga/
│   └── constitution.md       # L0 宪法
└── memory/
    ├── history.jsonl
    ├── .cursor
    ├── .dream_cursor
    ├── .verified_facts.jsonl # 已验证事实审计日志
    ├── layer_insight.txt     # L1 洞察索引 (≤30行)
    ├── layer_facts.txt       # L2 事实库
    ├── sop/                   # L3 任务标准
    │   ├── proxy_setup_sop.md
    │   └── deploy_script.py
    └── archives/              # L4 会话归档
        └── all_histories.txt
```

## 上下文注入策略

与 naive 的"始终注入全部 MEMORY.md"不同，Layerga 采用**分层按需注入**：

```
系统提示词注入 (初次):
  L0 宪法摘要 (4 条公理，约 500 chars)

每轮对话注入 (持续):
  L1 洞察索引全文 (≤30 行，约 400-800 chars)
  L2 段名指针 (N 个 [SECTION] 名称)
  L3 SOP 名称列表

按需检索 (Agent 主动):
  read_file("memory/layer_facts.txt")   # 看 L2 全文
  read_file("memory/sop/proxy_setup_sop.md")  # 看特定 SOP
```

**关键设计**：不注入 L2/L3 全文进入系统提示词——而是让 Agent 知道"有什么"，再用 `read_file` 工具按需获取。

## 使用方式

```python
from summerclaw.memory.registry import MemoryRegistry
from summerclaw.memory.layerga_memory import LayergaMemoryAlgorithm

registry = MemoryRegistry()
registry.register(LayergaMemoryAlgorithm())

algo = registry.get("layerga_memory")
components = algo.build(
    workspace=Path("./agent_workspace"),
    provider=llm_provider,
    model="gpt-4o",
    sessions=session_manager,
    context_window_tokens=128_000,
    build_messages=build_messages_fn,
    get_tool_definitions=get_tool_definitions_fn,
    max_completion_tokens=4096,
    session_ttl_minutes=60,
    max_batch_size=20,
    max_iterations=10,
    max_tool_result_chars=16_000,
    annotate_line_ages=True,
)
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `l1_max_lines` | 30 | L1 洞察索引硬上限 |
| `confidence_threshold` | 0.5 | L0 分类置信度阈值 |
| `enable_classification` | True | Consolidator 是否启用分类 |
| `enable_l1_cleanup` | True | Dream 是否启用 Phase 3 清理 |
| `enable_auto_crystallize` | True | Dream 是否自动结晶技能 |
| `enable_l4_archive` | True | AutoCompact 是否写入 L4 |

## 设计精髓：信息经济学

LayergaMemory 的本质是**信息经济学在 Agent 记忆中的应用**：

1. **稀缺性管理**：L1 只有 30 行——这是人为制造的稀缺，迫使系统做出取舍
2. **成本核算**：每条信息都有 token 成本，ROI 决定去留
3. **最小充分原则**：不写"可以推断"的，不写"会自然过期"的
4. **可审计性**：每条 L1/L2/L3 写入都记录到 `.verified_facts.jsonl`，有源可查

## 与其他算法的对比

| 特性 | naive | layerga |
|------|:---:|:---:|
| 存储结构 | 单层 MEMORY.md | L0-L4 五层 |
| 写入时机 | 文本周期 | 工具调用成功后 |
| 信息过滤 | 无 | L0 分类引擎 + 公理检查 |
| 容量控制 | 无 | L1 ≤30 行硬约束 + ROI 机制 |
| 上下文注入 | 全量 | L1 + L2/L3 指针 + 按需读取 |
| Agent 角色 | 用户/写手 | 执行者 + 记忆管理员 |
