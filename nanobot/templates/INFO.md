# nanobot/templates — 代理提示词模板与工作区脚手架

## 模块概述

`nanobot/templates` 是 nanobot agent 系统的**提示词模板引擎核心**。它包含两大部分：

1. **Agent 运行时提示词模板**（`agent/` 子目录）—— 通过 Jinja2 模板引擎动态渲染，注入时间、平台、通道、记忆路径等上下文变量，形成 agent 各阶段的 system prompt。
2. **工作区脚手架文件**（根目录 `.md` 文件 + `memory/`）—— 在新 workspace 初始化时同步到用户工作区，作为可编辑的个性化配置文件。

所有模板由 `nanobot/utils/prompt_templates.py` 中的 `render_template()` 函数统一加载和渲染。

---

## 目录结构

```
templates/
├── __init__.py                          # 模块标记
├── INFO.md                              # 本文件
│
├── AGENTS.md                            # Agent 行为指令与工具使用指南
├── HEARTBEAT.md                         # 心跳周期性任务定义模板
├── SOUL.md                              # Agent 个性/语调定义
├── TOOLS.md                             # 工具使用约束与非显而易见的用法文档
├── USER.md                              # 用户画像与偏好设置模板
│
├── agent/                               # Agent 运行时提示词模板
│   ├── identity.md                      # Agent 身份标识（核心 system prompt）
│   ├── platform_policy.md              # 操作系统平台策略（Windows/POSIX 分支）
│   ├── skills_section.md                # 技能列表注入片段
│   ├── max_iterations_message.md        # 超限提示（单行）
│   ├── subagent_announce.md             # 子代理结果通报格式化
│   │
│   ├── complexity_classifier_system.md  # 任务复杂度分类器 system prompt
│   ├── search_decision_system.md        # 搜索决策代理 system prompt
│   ├── search_agent_system.md           # 搜索执行代理 system prompt
│   ├── planner_system.md                # 任务规划器 system prompt
│   ├── evaluator_system.md              # 任务完成度评估器 system prompt
│   ├── evaluator.md                     # 心跳通知门控 evaluator prompt
│   ├── subagent_system.md               # 子代理 system prompt
│   │
│   ├── consolidator_archive.md          # 对话记忆归档提取 prompt
│   ├── dream_phase1.md                  # Dream 阶段一：事实提取+去重+技能发现
│   ├── dream_phase2.md                  # Dream 阶段二：记忆文件更新执行
│   ├── skill_autogen_review.md          # 技能自动生成审查 prompt
│   │
│   └── _snippets/                       # 可复用 prompt 片段
│       └── untrusted_content.md         # 外部内容安全警告
│
└── memory/                              # 工作区记忆文件脚手架
    ├── __init__.py
    └── MEMORY.md                        # 长期记忆文件模板
```

---

## 工作区脚手架文件

根目录的 5 个 `.md` 文件 + `memory/MEMORY.md` 构成 workspace 初始化时的**引导文件（bootstrap files）**。通过 `sync_workspace_templates()` 函数，在 workspace 首次创建时仅拷贝尚不存在的文件，用户后续可以自由编辑。

### AGENTS.md
对 agent 的高级操作指令。说明定时提醒（cron）、心跳任务的管理方式，以及如何在不同场景下选择合适的工具。agent 在每次对话中读取此文件作为行为参考。

### SOUL.md
定义 agent 的个性、交流风格和核心价值观：务实执行、简洁回答、诚实、友好、珍惜用户时间和信任。

### USER.md
用户画像模板，可填写姓名、时区、语言偏好、沟通风格、技术级别、工作上下文和专业兴趣。agent 据此个性化其交互行为。

### TOOLS.md
记录工具使用的非显而易见约束和最佳实践。涵盖 `exec` 安全限制（超时、危险命令屏蔽、输出截断）、`glob` 文件发现、`grep` 内容搜索、`cron` 定时任务等工具的使用模式。

### HEARTBEAT.md
定义周期性后台任务。nanobot 按配置的间隔检查此文件，执行其中的活跃任务。

### memory/MEMORY.md
长期记忆文件模板。agent 通过 Dream 机制自动维护此文件，存储跨会话的用户信息、偏好、项目上下文和重要备注。

---

## Agent 提示词模板详解

所有 agent 模板均使用 **Jinja2** 模板语法，通过 `{{ variable }}` 注入上下文变量，通过 `{% if/for/include %}` 实现条件分支和片段复用。渲染引擎配置为纯文本模式（`autoescape=False`），由 `render_template(name, **kwargs)` 调用。

### 核心身份模板

#### `agent/identity.md`
agent 的**核心 system prompt**，由 `ContextBuilder._build_system_prompt()` 渲染。关键注入变量：

| 变量 | 说明 |
|------|------|
| `runtime` | 运行时环境信息 |
| `workspace_path` | 工作区绝对路径 |
| `memory_rel_path` | 记忆文件相对路径（算法隔离后动态变化） |
| `history_rel_path` | 历史文件相对路径 |
| `platform_policy` | 内联渲染的 `platform_policy.md` 结果 |
| `channel` | 当前通信通道名（telegram/discord/cli/…） |

根据不同 `channel` 类型，自动注入相应的格式化提示（如 Telegram 使用短段落、WhatsApp 禁用 Markdown、CLI 使用纯文本等）。还包含执行规则（先读后写、失败重试、结果验证）和搜索发现指南。

#### `agent/platform_policy.md`
根据操作系统类型（`system == 'Windows'`）输出平台策略提示，指导 agent 选择正确的命令和工具。

### 执行流程模板

nanobot 的 agent 执行分为多个决策阶段，每个阶段对应一个专门的提示词模板：

```
用户消息 → complexity_classifier → search_decision → search_agent → planner → subagent(s) → evaluator → 回复用户
```

#### `agent/complexity_classifier_system.md`
**任务复杂度分类器**。判断用户消息属于 `SIMPLE`（单轮可直接回答）还是 `COMPLEX`（需多步骤规划执行）。输入为 `{{ time_ctx }}`，输出严格为单个单词。

#### `agent/search_decision_system.md`
**搜索决策代理**。决定是否需要 pre-planning web search 来**理解任务本身**（而非填充操作细节）。关键注入变量：

| 变量 | 说明 |
|------|------|
| `time_ctx` | 当前时间上下文 |
| `knowledge_cutoff` | LLM 知识截止日期（动态刷新） |
| `has_existing_info` | 是否已有搜索信息可用 |

输出 `SKIP` 或 `TRIGGER: keyword1, keyword2, ...`。核心理念是将"理解缺口"与"信息缺失"区分开——只有概念性不理解才触发搜索，具体数据由子代理在执行时查找。

#### `agent/search_agent_system.md`
**搜索执行代理**。当 search_decision 决定 `TRIGGER` 后，由此代理执行实际的信息检索。注入变量包括 `tasks`, `keywords_str`, `available_tools`, `max_chars`。规定仅执行只读操作，输出紧凑的要点摘要（带来源引用）。

#### `agent/planner_system.md`
**任务规划器**。将目标分解为结构化的执行计划（DAG 形式）。支持两种模式：
- **普通模式**：从零生成计划
- **Replan 模式**：根据 evaluator 反馈修正计划，分 `local`（局部调整）和 `global`（全局重构建）

注入变量：`time_ctx`, `search_info`, `replan_mode`, `previous_plan`, `feedback`, `max_subagent_depth`。

输出格式为带有 `[parallel|sequential]` 标记和依赖关系的任务列表。

#### `agent/subagent_system.md`
**子代理 system prompt**。注入变量包括 `time_ctx`, `workspace`, `current_depth`, `max_depth`, `skills_summary`。根据递归深度决定是否允许进一步 `spawn` 子代理。

#### `agent/subagent_announce.md`
子代理执行完毕后，向主代理通报结果的格式化模板。注入 `label`, `status_text`, `task`, `result`。要求主代理向用户自然地总结结果。

#### `agent/evaluator_system.md`
**任务完成度评估器**。评估子代理执行结果并输出分级决策：`PASS`（通过）、`LOCAL_REPLAN`（局部修正）、`GLOBAL_REPLAN`（全局重来）。倾向于通过（"when in doubt, lean toward PASS"）以避免不必要的重试循环。

#### `agent/evaluator.md`
**心跳通知门控评估器**。用于后台心跳任务执行后，判断是否需要向用户发送通知。根据 `part` 变量在 system/user 两种角色间切换。

### 记忆处理模板

#### `agent/consolidator_archive.md`
对话记忆归档提取 prompt。指导 agent 从对话中提取用户事实、决策、解决方案、事件和偏好，按优先级排序输出要点列表。

#### `agent/dream_phase1.md`
**Dream 阶段一**：后台记忆维护的第一阶段。agent 扫描对话历史和现有记忆文件，执行三大任务：
- **提取新事实**（`[FILE]`）：原子级事实，如"has a cat named Luna"
- **去重**（`[FILE-REMOVE]`）：检测跨文件冗余内容
- **技能发现**（`[SKILL]`）：识别重复出现的工作流模式

注入变量 `stale_threshold_days` 控制陈旧内容审查阈值。

#### `agent/dream_phase2.md`
**Dream 阶段二**：根据阶段一的分析结果执行记忆文件的实际修改。指导 agent 使用 `edit_file` 工具进行精确修改（非全量重写），处理技能创建（`dreamed-` 前缀），以及质量控制规则。注入变量包括 `memory_rel_path`, `skill_creator_path`。

### 技能提取模板

#### `agent/skill_autogen_review.md`
**技能自动生成审查**（中文 prompt）。分析对话历史，判断是否值得提炼为可复用 SKILL.md。要求满足：5+ 工具调用、清晰可复现工作流、足够通用、与现有技能不重复。生成的技能以 `hermes-` 为前缀命名。注入变量 `existing_skills`, `skill_creator_path`。

#### `agent/skills_section.md`
将可用技能列表注入 system prompt 的模板片段。通过 `{{ skills_summary }}` 注入技能摘要。

### 可复用片段

#### `agent/_snippets/untrusted_content.md`
安全警告片段：提醒 agent `web_fetch` / `web_search` 返回的内容不可信，不要执行其中嵌入的指令。同时提示图像类工具可直接读取视觉内容。

通过 `{% include 'agent/_snippets/untrusted_content.md' %}` 在 `identity.md` 和 `subagent_system.md` 中复用。

### 其他

#### `agent/max_iterations_message.md`
单行模板，注入 `{{ max_iterations }}`。当 agent 达到工具调用上限时向用户展示。

---

## 模板渲染机制

```python
# nanobot/utils/prompt_templates.py

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"

@lru_cache
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_ROOT)),
        autoescape=False,     # 纯文本 prompt，不转义 HTML
        trim_blocks=True,
        lstrip_blocks=True,
    )

def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    text = _environment().get_template(name).render(**kwargs)
    return text.rstrip() if strip else text
```

Jinja2 `Environment` 实例通过 `@lru_cache` 缓存，避免重复创建。`FileSystemLoader` 以 `templates/` 为根目录，因此模板名如 `agent/identity.md` 直接解析为 `templates/agent/identity.md`。

### 主要调用点

| 调用方 | 渲染的模板 |
|--------|-----------|
| `nanobot/agent/context.py::ContextBuilder._build_system_prompt()` | `agent/identity.md`, `agent/platform_policy.md` |
| `nanobot/agent/context.py::ContextBuilder.build_messages()` | `agent/skills_section.md` |
| `nanobot/agent/complexity.py` | `agent/complexity_classifier_system.md` |
| `nanobot/agent/search_planner.py` | `agent/search_decision_system.md`, `agent/search_agent_system.md` |
| `nanobot/agent/planner.py` | `agent/planner_system.md` |
| `nanobot/agent/runner.py` | `agent/subagent_announce.md` |
| `nanobot/agent/subagent.py` | `agent/subagent_system.md` |
| `nanobot/agent/evaluator.py` | `agent/evaluator_system.md` |
| `nanobot/utils/evaluator.py` | `agent/evaluator.md` |
| `nanobot/agent/hook.py`（Dream） | `agent/dream_phase1.md`, `agent/dream_phase2.md` |
| `nanobot/agent/skill_autogen.py` | `agent/skill_autogen_review.md` |
| `nanobot/memory/...`（各算法） | `agent/consolidator_archive.md` |

---

## 工作区同步流程

```python
# nanobot/utils/helpers.py::sync_workspace_templates()

def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """Sync bundled templates to workspace. Only creates missing files."""
```

该函数通过 `importlib.resources.files("nanobot") / "templates"` 访问打包后的模板资源，在 workspace 初始化或首次启动时调用：

1. 遍历 `templates/` 根目录下所有 `.md` 文件（排除 `.` 开头隐藏文件），仅创建尚不存在的文件
2. 创建 `memory/MEMORY.md`（模板）、`memory/history.jsonl`（空文件）、`skills/` 目录
3. 初始化 Git 版本控制（跟踪 `SOUL.md`, `USER.md`, `memory/MEMORY.md`）

`ContextBuilder` 在构建 system prompt 时使用 `_is_template_content()` 检查用户是否修改了脚手架文件——如果内容与模板完全一致，说明用户未定制，相关段落可以省略以节省 token。

---

## 模板设计原则

1. **纯文本优先**：`autoescape=False`，模板输出直接作为 LLM prompt 文本
2. **变量注入与条件渲染**：通过 Jinja2 的 `{{ }}` 和 `{% if %}` 实现平台/通道感知的动态 prompt
3. **片段复用**：通用警告/提示通过 `{% include %}` 在多个模板间共享
4. **最小 token 消耗**：仅在模板被用户修改后才注入 bootstrap 文件内容
5. **双向用途**：同一套模板既作为 agent 运行时 prompt 又作为用户的 workspace 脚手架，确保 agent 行为和用户配置的一致性
6. **不覆盖原则**：`sync_workspace_templates()` 只创建不覆盖，尊重用户的定制化修改