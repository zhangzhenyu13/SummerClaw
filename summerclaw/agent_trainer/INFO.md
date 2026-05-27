# Agent Trainer — 可插拔技能优化框架

Agent Trainer 是 SummerClaw 的**零侵入式**训练模块，通过可插拔算法（如 SkillOpt）对 Agent 的 Skill 文档进行自动优化，同时保持在线 Agent 正常运行。

---

## 核心设计原则

### 1. 数据隔离（Data Isolation）

训练使用**独立工作空间** `train-outputs/<algorithm>-<task>/`，包含：
- 独立的 Memory Store（记忆数据不污染在线 Agent）
- 独立的 Session（训练 session_key 与在线完全隔离）
- 独立的输出文件（skill 版本、patch 文件、评估报告等）

```
train-outputs/skillopt-task/
├── config.json              # 训练配置快照
├── best_skill.md            # 当前最优 skill
├── history.json             # 训练历史
├── runtime_state.json       # 断点续训状态
├── skills/
│   ├── skill_v0000.md       # 初始 skill
│   ├── skill_v0001.md       # step 1 后的 skill
│   └── ...
├── steps/
│   ├── step_0001/
│   │   ├── patches/         # Reflect 阶段产生的 patch 文件
│   │   ├── merged_patch.json
│   │   ├── selected_edits.json
│   │   ├── edit_apply_report.json
│   │   ├── candidate_skill.md
│   │   └── eval/            # 评估结果
│   └── ...
└── baseline/                # 基线评估结果
```

### 2. 运行时一致性（Runtime Consistency）

训练 Agent 与在线 Agent 保持**完全一致**的运行时环境：
- **工具集**：使用相同的 `ToolRegistry` 配置
- **记忆算法类型**：在线用 nemori，训练也用 nemori（但数据存储在隔离目录）
- **模型配置**：复用在线 Agent 的 Provider / Model / Temperature / MaxTokens 等
- **ContextBuilder**：相同的构建逻辑，仅 Memory Store 指向隔离目录

### 3. 零侵入（Zero Intrusion）

- **不修改**现有 Agent 核心文件（`runner.py`、`context.py`、`loop.py` 等）
- 通过 `SummerClawEnvAdapter` 适配器桥接 AgentRunner 到训练环境
- 训练以 background asyncio task 运行，不影响在线服务

### 4. 可插拔算法（Pluggable Algorithms）

基于 `BaseAlgorithm` + Registry 模式的插件化架构：

```python
# 注册新算法
@algorithm("my_algo")
class MyAlgorithm(BaseAlgorithm):
    async def rollout(...)  -> list[RolloutResult]
    async def reflect(...)  -> list[RawPatch]
    async def aggregate(...) -> Patch
    async def select(...)   -> Patch
    async def update(...)   -> tuple[str, list[dict]]
    async def evaluate(...) -> float
```

---

## 模块结构

```
agent_trainer/
├── __init__.py              # 模块入口与功能概述
├── base.py                  # BaseAlgorithm 抽象基类（6 阶段 pipeline）
├── types.py                 # 公共类型定义（Edit, Patch, RolloutResult, GateResult 等）
├── config.py                # 训练配置构建（从 AgentConfig 合并默认值 + 用户覆盖）
├── registry.py              # 算法注册表（name → class 映射，支持装饰器注册）
├── command.py               # Channel 命令处理器（/train, /train_status, /train_stop）
├── engine/
│   └── trainer.py           # TrainerEngine — 算法无关的训练引擎（编排 6 阶段 pipeline）
├── env/
│   └── summerclaw_env.py    # SummerClawEnvAdapter — 桥接 ReACT Agent + Memory 系统
├── datasets/
│   └── loader.py            # DataLoader — 标准 train/val/test 分目录数据加载
├── evaluation/
│   └── gate.py              # 验证门控 — accept/reject 候选 skill 的纯决策函数
├── dashboard/
│   └── app.py               # Gradio WebUI + FastAPI REST 状态端点
└── algorithms/
    └── skillopt/            # SkillOpt 算法实现
        ├── algorithm.py     # SkillOptAlgorithm 主入口
        ├── reflect.py       # Minibatch 轨迹分析引擎
        ├── aggregate.py     # 层级式 Patch 合并
        ├── select.py        # LLM 驱动的编辑排序与选择
        ├── update.py        # 编辑应用（含 SLOW_UPDATE 区域保护）
        └── slow_update.py   # Epoch 级纵向比较 + EMA 正则化
```

---

## 6 阶段 Per-Step Pipeline

每个训练 Step 执行以下 6 个阶段：

```
┌─────────────────────────────────────────────────────────┐
│  ① Rollout    使用当前 Skill 执行 episode（批量推理）     │
│  ② Reflect    分析轨迹，生成 Patch（Minibatch 并行）      │
│  ③ Aggregate  层级合并多个 Patch                         │
│  ④ Select     按重要性排序并选择 Top-L 编辑（梯度裁剪）   │
│  ⑤ Update     将选中的编辑应用到 Skill 文档（优化器步进） │
│  ⑥ Evaluate   在验证集上评估候选 Skill，Accept/Reject     │
└─────────────────────────────────────────────────────────┘
```

### 阶段详解

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| **Rollout** | skill + items | `list[RolloutResult]` | 通过 `SummerClawEnvAdapter.rollout_batch()` 执行，支持并发控制 |
| **Reflect** | results + skill | `list[RawPatch]` | 分 failure/success 两组，按 minibatch 并行调用 LLM 分析 |
| **Aggregate** | patches + skill | `Patch` | 先分别合并 failure/success patches，再做最终合并（failure 优先） |
| **Select** | patch + budget | `Patch` | LLM 对编辑按重要性排序，保留 Top-L（L = edit_budget） |
| **Update** | skill + patch | `(str, report)` | 顺序应用编辑，支持 append/insert_after/replace/delete 四种操作 |
| **Evaluate** | candidate + val | `float` | 在验证集上 rollout 并计算 hard accuracy |

---

## 训练循环

```
for epoch in 1..num_epochs:
    batches = split(train_data, batch_size)
    for batch in batches:
        run 6-stage pipeline → GateResult (accept/reject)
        save state (断点续训)
    on_epoch_end() hook
```

- **Baseline**：训练开始前先在验证集上评估初始 skill，作为基准分数
- **Gate 决策**：`evaluate_gate()` 纯函数比较 candidate 分数与 current/best，返回 accept_new_best / accept / reject
- **断点续训**：每步结束后持久化 `runtime_state.json`，重启后自动从 last_completed_step 恢复

---

## 数据格式

训练数据使用标准 split 目录结构：

```
train-data/
├── train/items.json
├── val/items.json
└── test/items.json
```

每个 item 至少包含：

```json
{
    "id": "task_001",
    "question": "用户问题文本",
    "answers": ["候选答案1", "候选答案2"],
    "context": "可选上下文"
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 唯一样本标识符 |
| `question` | string | 是 | 用户输入 / 问题 |
| `answers` | list[string] | 是 | 候选答案列表，任意一个匹配即视为正确 |
| `context` | string | 否 | 问题的附加上下文 |
| `scorer` | string | 否 | 评分方式：`exact_match`（默认）、`llm_judge`、`custom` |

**自定义评分器（custom scorer）：**

当 `scorer` 设为 `custom` 时，需在任务输出目录中放置 `custom-scorer.py` 文件，定义如下函数：

```python
def score(sample: dict, predicted: str) -> float:
    # sample: 完整数据样本 dict（包含 id, question, answers, context 等）
    # predicted: Agent 预测的答案字符串
    # 返回: 0.0 ~ 1.0 的分数
    answers = sample.get("answers", [])
    predicted_lower = predicted.strip().lower()
    for ans in answers:
        if ans.lower() in predicted_lower:
            return 1.0
    return 0.0
```

---

## Channel 命令接口

从任意 Channel 发送命令即可启动训练：

| 命令 | 功能 |
|------|------|
| `/train <algorithm>` | 启动训练（返回 Dashboard URL） |
| `/train` | 列出可用算法 |
| `/train_status` | 查看所有活跃训练会话 |
| `/train_stop <algorithm>` | 请求取消指定训练 |

训练进度会通过 Channel 自动推送通知（epoch 完成、新 best、训练完成等）。

---

## Dashboard（监控面板）

Dashboard 同时提供 **Gradio WebUI** 和 **FastAPI REST** 两套接口：

### Gradio WebUI

- 实时状态显示（Running/Idle、Best Score、Total Steps）
- 训练历史表格（Step、Epoch、Score、Action、Hash、Edits、Rejected）
- 分数趋势折线图
- 事件日志流
- 手动控制（Stop Training、Deploy Best Skill）

### REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 当前训练状态 |
| `/api/history` | GET | 完整训练历史 |
| `/api/best_skill` | GET | 最优 Skill 内容 |
| `/api/current_skill` | GET | 当前 Skill 内容 |
| `/api/cancel` | POST | 取消训练 |
| `/api/deploy` | POST | 部署最优 Skill 到指定路径 |

---

## 配置参数

通过 `config.json` 的 `trainer` 字段或自动检测进行配置：

```json
{
    "trainer": {
        "num_epochs": 3,
        "batch_size": 5,
        "edit_budget": 4,
        "seed": 42,
        "workers": 4,
        "minibatch_size": 5,
        "optimizer_model": null,
        "dashboard_port": 7860,
        "dashboard_share": true
    }
}
```

**自动检测**：
- `data_dir`：`<workspace>/train-data/`
- `skill_init`：`<workspace>/skills/SKILL.md`

---

## 核心类型

| 类型 | 说明 |
|------|------|
| `Edit` | 单个编辑操作（op: append/insert_after/replace/delete） |
| `Patch` | 编辑集合 + 推理说明 |
| `RawPatch` | Reflect 阶段输出（Patch + 来源类型 + 失败摘要） |
| `RolloutResult` | 单次 episode 结果（hard/soft 分数 + 轨迹） |
| `GateResult` | 验证门控决策（accept_new_best/accept/reject） |
| `TrainingStep` | 单步训练快照 |
| `TrainingHistory` | 训练历史（步骤列表 + best score） |

---

## 环境适配器

`SummerClawEnvAdapter` 是训练环境与在线 Agent 之间的桥梁：

- **Memory 隔离**：使用与在线相同类型的 Memory 算法（如 nemori），但数据写入训练工作空间
- **Skill 注入**：将当前 Skill 内容注入到 System Prompt 的 `# Active Skill` 区域
- **评分机制**：支持 exact_match（大小写不敏感子串匹配）和关键词重叠的 soft 评分
- **并发控制**：通过 `asyncio.Semaphore` 控制 rollout 并发数

---

## 关键文件索引

| 文件 | 职责 |
|------|------|
| `base.py` | 定义 `BaseAlgorithm` 抽象基类（6 个抽象方法 + epoch hook） |
| `types.py` | 所有共享 dataclass 定义（支持 dict 双向转换） |
| `config.py` | `build_trainer_config()` — 合并默认值 + 用户配置 + 算法覆盖 |
| `registry.py` | `@algorithm` 装饰器 + `get_algorithm()` / `list_algorithms()` |
| `command.py` | `/train` 命令处理、训练启动、进度通知转发 |
| `engine/trainer.py` | `TrainerEngine.train()` — 完整训练循环 + 断点续训 + 状态持久化 |
| `env/summerclaw_env.py` | `SummerClawEnvAdapter` — AgentRunner 封装 + Memory 隔离 + 评分 |
| `datasets/loader.py` | `DataLoader` / `DataSplit` — JSON 数据加载 + batch 迭代 |
| `evaluation/gate.py` | `evaluate_gate()` — 纯函数验证门控决策 |
| `dashboard/app.py` | `DashboardServer` — Gradio + FastAPI 后台服务 |
