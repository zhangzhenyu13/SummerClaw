# nanobot/agent 模块

## 概述

`nanobot/agent` 是 UnionClaw/nanobot 项目的**核心 Agent 引擎模块**，负责实现从消息接收、上下文构建、LLM 调用、工具执行到响应发送的完整 Agent 循环生命周期。该模块采用了 **ReAct（Reasoning + Acting）** 范式，并在此基础上扩展了 **Plan-and-Solve** 规划模式、**搜索增强规划**、**闭环评估与重规划**、**子代理（Subagent）后台执行**、**技能自动生成**以及**多模式前缀驱动执行**等高级能力。

---

## 模块架构

```
nanobot/agent/
├── __init__.py              # 模块公开接口
├── loop.py                  # AgentLoop：核心处理引擎（81.6 KB）
├── runner.py                # AgentRunner：共享 LLM 工具调用循环（41.8 KB）
├── context.py               # ContextBuilder：上下文构建器（9.4 KB）
├── hook.py                  # AgentHook：生命周期钩子系统（3.5 KB）
├── planner.py               # TaskPlanner：规划和重规划器（6.9 KB）
├── search_planner.py        # SearchEnhancedPlanner：搜索增强规划层（31.5 KB）
├── evaluator.py             # TaskEvaluator：闭环评估器（5.5 KB）
├── complexity.py            # LLMComplexityEvaluator：任务复杂度评估（8.0 KB）
├── subagent.py              # SubagentManager：后台子代理管理器（20.6 KB）
├── skill_autogen.py         # SkillAutogen：技能自动生成器（12.0 KB）
├── skills.py                # SkillsLoader：技能加载器（8.6 KB）
├── task_persistence.py      # TaskRegistry：任务持久化与崩溃恢复（5.2 KB）
└── tools/                   # 工具集子模块（20 个文件）
    ├── __init__.py
    ├── base.py              # Tool 基类与装饰器
    ├── registry.py          # ToolRegistry 工具注册表
    ├── schema.py            # JSON Schema 类型定义
    ├── filesystem.py        # 文件系统工具（read/write/edit/list）
    ├── shell.py             # ExecTool：Shell 命令执行
    ├── search.py            # Glob/Grep 代码搜索工具
    ├── web.py               # WebSearch/WebFetch 网络工具
    ├── browser.py           # BrowserSearch/BrowserFetch 浏览器工具
    ├── browser_control.py   # 浏览器控制（Playwright）
    ├── mcp.py               # MCP 协议工具集成
    ├── message.py           # 消息通道发送工具
    ├── ask_user.py          # ask_user 用户交互工具
    ├── cron.py              # 定时任务调度工具
    ├── notebook.py          # Notebook 编辑工具
    ├── self.py              # MyTool：自修改工具
    ├── spawn.py             # SpawnTool：子代理生成工具
    ├── file_state.py        # 文件状态追踪
    └── sandbox.py           # 沙箱工具
```

---

## 核心组件详解

### 1. AgentLoop（`loop.py`）

AgentLoop 是整个系统的**中央调度器**，负责协调所有子组件完成一次完整的 Agent 交互。

**核心职责：**
- 从 MessageBus 接收用户消息
- 构建上下文（系统提示 + 历史 + 记忆 + 技能）
- 解析消息前缀决定执行模式
- 调用 LLM 并执行工具调用
- 通过 MessageBus 发送响应

**四种执行模式（前缀驱动）：**

| 前缀 | 模式 | 行为 |
|------|------|------|
| `/simple` | 简单模式 | 跳过规划，直接进入 ReAct 循环 |
| `/plan` | 规划模式 | 始终使用 TaskPlanner 生成执行计划 |
| `/search-plan` | 搜索增强规划 | 始终进行 Web 搜索 + 规划 |
| `/auto` | 自动模式（默认） | 由 LLMComplexityEvaluator 判断是否需要规划 |

**关键特性：**
- 支持内存算法可插拔（通过 `memory_algorithm_name` 配置）
- 内置并发请求控制（`NANOBOT_MAX_CONCURRENT_REQUESTS` 环境变量，默认 3）
- 支持 SkillAutogen 后台技能自动生成
- 支持 Dream（记忆固化）调度
- 支持运行时 checkpoint 持久化（崩溃恢复）
- 支持 `injection_callback` 中途注入消息（用于 ask_user 等交互场景）

### 2. AgentRunner（`runner.py`）

AgentRunner 是**纯净的 LLM 工具调用执行循环**，剥离了所有产品层关注点。

**核心职责：**
- 管理 LLM 请求/响应的迭代循环
- 执行工具调用（支持并发批处理）
- 上下文治理：孤儿工具结果清理、缺失回填、微压缩、历史裁剪
- Token 用量追踪与上下文窗口管理
- 注入消息的排空与合并
- 空响应重试、长度恢复重试

**关键数据结构：**
- `AgentRunSpec`：单次执行的配置（消息、工具、模型、迭代次数等）
- `AgentRunResult`：执行结果（最终内容、工具列表、用量、停止原因等）

**上下文治理机制：**
- `_drop_orphan_tool_results`：移除没有对应 tool_call 的孤儿工具结果
- `_backfill_missing_tool_results`：为缺少结果的 tool_call 插入合成错误
- `_microcompact`：将旧的紧凑型工具结果压缩为单行摘要
- `_snip_history`：基于 token 预算裁剪历史消息

### 3. ContextBuilder（`context.py`）

负责**组装 Agent 的完整上下文**（系统提示 + 消息列表）。

**核心职责：**
- 构建系统提示（身份信息 → 引导文件 → 记忆 → 技能 → 最近历史）
- 构建运行时上下文元数据块（时间、频道、聊天 ID）
- 合并用户消息内容（支持多模态：文本 + base64 图片）
- 维护消息列表（assistant 消息追加、工具结果追加）

**引导文件加载顺序：**
1. `AGENTS.md` — Agent 配置与行为规范
2. `SOUL.md` — Agent 人格定义
3. `USER.md` — 用户偏好与信息
4. `TOOLS.md` — 工具使用说明

**记忆上下文来源：**
- `MEMORY.md`（长期记忆，若用户已自定义则包含）
- `history.jsonl` 中未处理的最近历史条目（最多 50 条）

### 4. AgentHook（`hook.py`）

提供 **Agent 执行生命周期的可扩展钩子系统**。

**生命周期钩子：**

| 钩子方法 | 触发时机 |
|----------|----------|
| `wants_streaming()` | 查询是否需要流式输出 |
| `before_iteration()` | 每次迭代开始前 |
| `on_stream(delta)` | 流式输出每个 delta |
| `on_stream_end(resuming)` | 流式输出结束时 |
| `before_execute_tools()` | 工具执行前 |
| `after_iteration()` | 每次迭代结束后 |
| `finalize_content(content)` | 最终内容后处理（管道模式） |

**CompositeHook**：支持多个钩子的组合执行，提供错误隔离（单个钩子异常不会影响其他钩子）。

### 5. TaskPlanner（`planner.py`）

实现 **Plan-and-Solve** 模式的任务规划。

**核心职责：**
- `plan()`：生成结构化执行计划（纯 LLM 调用，无工具）
- `replan()`：基于评估反馈重新生成计划，支持 `local`（局部调整）和 `global`（全局重建）两种模式

**规划流程：**
1. 接收用户任务描述
2. 单次 LLM 调用生成包含任务列表和依赖标注的计划
3. 将计划注入主 Agent 上下文

### 6. SearchEnhancedPlanner（`search_planner.py`）

**搜索增强规划层**，在 TaskPlanner 之前插入 Web 搜索步骤。

**核心职责：**
- **SearchDecider**（子模块 8.1）：LLM 决策是否需要搜索（返回 TRIGGER/SKIP）
- **SearchAgent**（子模块 8.2）：小型 agentic 搜索循环，LLM 自主决定搜索策略
- **knowledge_cutoff 系统**：磁盘缓存的模型知识截止日期，支持自动刷新（7 天过期）
- 搜索结果注入 TaskPlanner 上下文

**搜索后端选择：**
- `auto`：优先浏览器工具，回退到 HTTP API 工具
- `browser`：仅使用浏览器工具（Playwright）
- `web`：仅使用 HTTP API 工具

**优雅降级：** 任何阶段失败都不会阻塞主流，自动回退到普通规划模式。

### 7. TaskEvaluator（`evaluator.py`）

**闭环评估器**，对执行结果进行量化评分。

**三种决策：**

| 决策 | 含义 |
|------|------|
| `PASS` | 目标已达成，无需重规划 |
| `LOCAL_REPLAN` | 局部失败，仅需调整特定子任务分支 |
| `GLOBAL_REPLAN` | 整体方法有误或环境变化，需要完整重建计划 |

**保守策略：** 解析失败时默认返回 `PASS`，避免触发不必要的重规划。

### 8. LLMComplexityEvaluator（`complexity.py`）

**两阶段混合复杂度评估**，决定 auto 模式下是否触发规划。

**阶段 1 — 正则预过滤（零成本）：**
- 极短消息（< 15 字符）
- 问候语模式（hi, hello, good morning...）
- 感谢/确认模式（thanks, ok, got it...）
- 简单确认模式（yes, no, go ahead...）

**阶段 2 — LLM 分类（精确）：**
- 单轮无工具 LLM 调用
- 约 200 输入 token + 1 输出 token
- 保守回退：任何错误默认返回 COMPLEX

### 9. SubagentManager（`subagent.py`）

**后台子代理管理器**，支持异步任务委派。

**核心职责：**
- `spawn()`：生成子代理执行后台任务
- 深度控制（RDC 逻辑）：仅在 `depth < max_depth` 时注册 SpawnTool
- 实时状态追踪（`SubagentStatus`）
- 任务持久化（崩溃恢复）
- 结果通过 MessageBus 注入主 Agent

**子代理工具集：**
- 文件系统工具（read/write/edit/list/glob/grep）
- Shell 执行工具（ExecTool）
- Web 搜索工具（WebSearch/WebFetch，可选）
- 浏览器工具（可选）
- SpawnTool（递归分解，可控深度）

### 10. SkillsLoader（`skills.py`）

**技能加载器**，管理系统提示中的 Agent 技能。

**核心职责：**
- 加载 workspace 和 builtin 技能目录中的 `SKILL.md` 文件
- 解析 YAML frontmatter（元数据、依赖要求）
- 筛选可用技能（检查 CLI 依赖、环境变量）
- 区分 `always` 技能（始终加载）与按需技能（渐进式加载）
- 构建技能摘要供 Agent 按需读取

### 11. SkillAutogen（`skill_autogen.py`）

**技能自动生成器**，从对话中提取可复用工作流。

**触发机制：** 累计工具调用超过 `nudge_interval` 阈值时触发

**工作流程：**
1. 汇总最近对话中的工具调用模式
2. 调用 LLM 审查是否可提取为技能
3. 自动生成 `skills/<name>/SKILL.md` 文件
4. 技能目录名强制带 `hermes--<memory_algo_name>-` 前缀

**与 Dream 的区别：**
- Dream：定时任务，处理 `history.jsonl`，更新 MEMORY/SOUL/USER.md
- SkillAutogen：阈值触发，审查当前轮消息，创建 SKILL.md

### 12. TaskRegistry（`task_persistence.py`）

**磁盘持久化的任务注册表**，支持进程崩溃后恢复。

**核心职责：**
- 任务启动时写入 `{workspace}/task_registry/{task_id}.json`
- 正常完成或取消时删除
- 重启时自动发现中断的任务
- 支持 `subagent` 和 `main_session` 两种任务类型

---

## 公开接口（`__init__.py`）

```python
from nanobot.agent import (
    AgentHook,        # 生命周期钩子基类
    AgentHookContext, # 钩子上下文数据类
    AgentLoop,        # 核心处理引擎
    CompositeHook,    # 组合钩子
    ContextBuilder,   # 上下文构建器
    Dream,            # 记忆固化（从 nanobot.memory 重导出）
    MemoryStore,      # 记忆存储（从 nanobot.memory 重导出）
    SkillsLoader,     # 技能加载器
    SubagentManager,  # 子代理管理器
)
```

---

## 执行流程概览

```
用户消息 → MessageBus
    │
    ▼
AgentLoop.process_direct() / _dispatch()
    │
    ├── 前缀解析 → 确定执行模式 (simple/plan/search-plan/auto)
    │
    ├── [auto 模式] LLMComplexityEvaluator → 判断任务复杂度
    │
    ├── [规划模式] SearchEnhancedPlanner （可选）
    │   ├── SearchDecider → 是否需要搜索？
    │   ├── SearchAgent → Agentic 搜索
    │   └── TaskPlanner.plan() → 生成计划
    │
    ├── ContextBuilder → 构建完整上下文
    │   ├── 身份 + 引导文件
    │   ├── 记忆上下文
    │   ├── 技能系统提示
    │   └── 最近历史
    │
    ├── AgentRunner.run() → ReAct 循环
    │   ├── LLM 调用（支持流式）
    │   ├── 工具调用执行
    │   ├── ask_user 交互
    │   └── 上下文治理（清理/压缩/裁剪）
    │
    ├── [闭环模式] TaskEvaluator → 评估结果
    │   ├── PASS → 完成
    │   ├── LOCAL_REPLAN → 局部重规划
    │   └── GLOBAL_REPLAN → 全局重规划（可选重新搜索）
    │
    └── MessageBus → 响应发送
```

---

## 关键设计原则

1. **前缀驱动执行模式**：通过 `/simple`、`/plan`、`/search-plan`、`/auto` 前缀实时切换执行策略
2. **优雅降级**：SearchAgent 失败回退到普通规划，Evaluator 失败默认为 PASS，确保不阻塞流程
3. **可插拔记忆算法**：通过 `memory_algorithm_name` 支持多种记忆后端（naive/hindsight/supermemory/mem0v3/mastra_om/nemori/layerga/emem）
4. **全链路可观测性**：各组件通过 loguru 输出分级日志，支持 INFO/DEBUG/TRACE 级别
5. **崩溃恢复**：通过 runtime checkpoint 和 TaskRegistry 实现进程重启后的任务恢复
6. **并发安全**：AgentRunner 支持并发工具执行批处理，AgentLoop 通过 Semaphore 控制并发请求
7. **Hook 可扩展**：CompositeHook 支持出错隔离，自定义 Hook 异常不会影响主循环