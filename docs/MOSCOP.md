# MOSCOPT: Mixture-of-Skill Collective Optimization

**一种全文本态、参数自由的智能体技能集体进化算法**

---

## 摘要

智能体技能的自动优化是构建自主智能系统的核心挑战。SkillOpt 通过有界文本编辑与验证门控，实现了单技能的安全自进化，但它未能利用多技能间的协同与竞争。本文提出 MOSCOPT，将优化对象从单技能扩展为**技能池与门控调度器的联合文本集合**。所有组件均为自然语言提示，无需任何参数化模型或梯度更新。门控本身是一个文本技能，在每个任务或执行步骤中，根据渐进披露的技能摘要，动态激活固定数量（K）的技能供智能体全量加载。整个系统通过有界编辑、验证门控与集体进化循环，实现技能与调度策略的协同优化。MOSCOPT 完全向后兼容 SkillOpt（K=1, N=1 时退化为单技能优化），同时为复杂、长程任务提供了更鲁棒、更灵活的技能集成方案。

---

## 1. 引言

大型语言模型（LLM）驱动的智能体在执行复杂任务时，常依赖手工编写的文本提示或技能描述。这些技能的质量直接影响智能体表现的稳定性和效率。SkillOpt 提出了一种参数自由的技能优化方法，通过分析任务轨迹，对技能文本进行可控的局部编辑，并利用验证集确保每次编辑都带来严格提升，从而避免了灾难性遗忘和不稳定行为。

然而，SkillOpt 仅能优化**单个技能**，这限制了其在以下场景中的潜力：
- 长程任务的不同阶段需要截然不同的策略；
- 同一任务存在多种有效解路径，单技能可能过早收敛到局部最优；
- 技能之间的互补性无法被利用，例如一个技能擅长规划，另一个擅长执行。

直观上，我们可以独立运行多个 SkillOpt 实例，然后挑选最优者。但这种做法忽略了技能间的**实时交互**——在任务执行过程中，智能体可能需要根据当前子任务动态切换技能，而不是始终绑定一个静态提示。此外，独立优化无法捕获技能组合带来的协同增益。

本文提出的 **MOSCOPT**（Mixture-Of-Skill Collective OPTimization）正是为解决这一问题而设计。它维护一个包含 N 个文本技能的技能池，并额外引入一个**门控文本技能 G**。G 在每个任务或执行步骤中，根据当前任务状态和技能池的渐进披露摘要，选择恰好 K 个技能进行激活（K 为固定超参数，1 ≤ K ≤ N）。被激活的技能全量加载至智能体的上下文中，指导动作生成。整个技能池和门控 G 均通过 SkillOpt 式的有界编辑和验证门控进行安全进化，同时伴随淘汰、繁殖和慢更新等集体进化机制。

MOSCOPT 的核心贡献如下：
- **混合技能集体优化**：首次将多技能协同问题形式化为完全文本态、参数自由的优化框架。
- **门控即文本技能**：门控不依赖任何数值参数，自身也是一个通过轨迹反馈优化的自然语言调度器。
- **渐进披露与固定激活**：门控基于技能摘要（而非全量）做选择，实现高效的渐进披露；每步激活固定数量技能，保障上下文长度可控。
- **联合安全进化**：所有编辑操作均通过验证门控确保单调提升，且技能与门控以三阶段交错方式更新，避免震荡。

---

## 2. 背景：SkillOpt 简述

SkillOpt 的核心是一个迭代编辑循环，在实现中体现为 6 阶段 pipeline：

1. **Rollout（轨迹收集）**：使用当前技能文本执行任务，记录成功/失败轨迹。
2. **Reflect（反思分析）**：对失败轨迹进行 minibatch 分组分析，由 LLM 生成文本级的局部修改建议（如添加例外规则、调整优先级描述），编辑幅度受**文本学习率**（最大改动字符比例）约束。
3. **Aggregate（补丁聚合）**：合并来自不同失败样本的编辑建议，去重并排序。
4. **Select（补丁选择）**：从聚合后的候选中选出最优补丁。
5. **Update（更新验证）**：在验证集上测试候选编辑，**仅当成功率严格提升**时才正式接受，否则将编辑加入**拒绝缓存**，禁止重复尝试。学习率调度器根据连续接受/拒绝情况动态调整编辑预算。
6. **Evaluate（评估）**：在独立的测试集上评估当前技能版本的质量。

此外，SkillOpt 还包含**慢更新**机制：定期从高分技能中抽取稳定规则固化至元规则库，以及**拒绝缓存**防止浪费尝试。此方法确保技能优化过程**单调且安全**，完全不需要修改 LLM 参数。MOSCOPT 完整继承了这一机制，并将其作用域从单技能拓展到技能池与门控。

---

## 3. MOSCOPT 算法框架

### 3.0 架构层级

MOSCOPT 包含三个严格区分的运行层级，避免概念混淆：

| 层级 | 名称 | 含义 | 类比 |
|------|------|------|------|
| **优化步（Optimization Step）** | opt_step | 一次完整的 6 阶段 pipeline：rollout → reflect → aggregate → select → update → evaluate | SkillOpt 的单个 step |
| **纪元（Epoch）** | epoch | 包含若干个 opt_step，epoch 结束后触发集体进化（淘汰/繁殖/合并） | 机器学习的 epoch |
| **执行步（Execution Step）** | exec_step | agent 在单条轨迹中的一次决策（工具调用、动作生成），记为 \(t = 1 \dots T\) | RL 中的 time step |

**全文约定**：
- 外循环（优化）统一使用 `opt_step`。
- 内循环（轨迹内）统一使用 `exec_step`。
- `epoch` 保持原名，是 opt_step 与集体进化的桥梁。

### 3.1 问题形式化

给定一个任务分布 \(\mathcal{T}\) 和一个基础 LLM 智能体，智能体的行为由**激活的技能子集**和这些技能的完整文本决定。系统包含：
- 技能池 \(\mathcal{S} = \{s_1, s_2, \dots, s_N\}\)，每个 \(s_i\) 是一段自然语言策略描述。
- 门控文本技能 \(G\)，负责选择激活的 K 个技能。
- 超参数 \(K\)：每次激活的技能数量（固定，1 ≤ K ≤ N）。
- 技能摘要函数 \(\Sigma(\mathcal{S})\)：为门控提供轻量级的技能描述。

在任务执行过程中，门控 G 根据当前状态和技能摘要，输出激活的技能索引集合 \(\mathcal{A}\)，满足 \(|\mathcal{A}| = K\)。被激活技能的完整文本与任务状态共同构成智能体的输入 Prompt，智能体据此生成动作。

目标是优化 \(\mathcal{S}\) 和 \(G\)，以最大化任务成功率 \(\mathbb{E}_{\mathcal{T}}[R(\tau)]\)，其中 \(\tau\) 为完成任务的轨迹。

### 3.2 技能池初始化

利用基础 LLM 从任务描述和种子提示生成 N 个多样化技能：
- 提供不同的角色指令（如"你是谨慎的规划者"、"你是高效的执行者"）和温度参数 > 0.7，生成多个候选文本。
- 通过去重（基于文本相似度）和基础质量过滤（在少量任务上成功率 > 阈值）保留 N 个技能。
- 为每个技能生成简短标签和初始摘要。

### 3.3 门控文本技能 G 的初始化

G 同样由 LLM 生成，是一个自然语言调度器。初始 Prompt 包含以下要素：

**输入规范**：G 接收一个技能摘要表，格式为：

```
| ID | 标签         | 近期得分 | 专长简述       |
|----|-------------|---------|---------------|
| 1  | 保守型规划   | 0.72    | 多步推理、验证 |
| 2  | 高效执行     | 0.85    | 代码生成、快速  |
| 3  | 数学专长     | 0.60    | 计算、公式推导  |
```

**输出规范**：G 必须输出恰好 K 个技能 ID，格式为 `ACTIVATE: id1, id2, ...`。例如：`ACTIVATE: 2, 5`。

**回退规则**：若 G 的输出无法解析为恰好 K 个有效 ID（LLM 输出格式不合规），系统回退到**基于近期 Q-score 的 top-K 规则**选择，并将此次解析失败作为编辑信号反馈给门控优化。

**初始规则示例**：
```
你是一个技能调度器。根据当前任务状态和技能摘要表，从下列技能中恰好选择 K 个最合适的技能激活。
规则：
- 初始阶段优先激活规划类技能；
- 若任务涉及计算，至少选择一个数学专长技能；
- 仅输出被激活的技能编号，格式为 "ACTIVATE: id1, id2"。
```

### 3.4 执行流程与门控粒度

门控的选择频率（粒度）是一个关键设计维度，根据任务类型不同可采用不同策略：

**Task-level gating（默认）**：每个任务开始时，门控选择一次 K 个技能，整个任务使用相同组合。适用于单轮 QA 和短轨迹任务。对于 SkillOpt 主要处理的单轮 QA 任务，门控只做一次选择。

**Step-level gating（可选）**：每个 exec_step 重新调用门控选择。适用于长程多步任务（如 SWE-bench、WebShop），轨迹内不同步骤可激活不同技能。

**自适应粒度**：门控文本中可以包含"何时切换"的规则，由门控自身通过优化学习决定切换时机。

**单次 rollout 的执行流程**（以 step-level gating 为例）：

1. 初始化状态 \(x_1\)。
2. **For** \(t = 1\) **to** \(T\)（每个 exec_step）：
   - 构建当前技能摘要表 \(\Sigma_t(\mathcal{S})\)。
   - 调用门控 \(G\)，输入 \((x_t, \text{history}, \Sigma_t(\mathcal{S}))\)，解析输出获得激活集合 \(\mathcal{A}_t\)（确保 \(|\mathcal{A}_t| = K\)，否则回退到 top-K 规则）。
   - 加载激活技能的完整文本，与 \(x_t\) 拼接为 Agent Prompt：
     ```
     [Activated Skills]
     Skill 2: <完整文本>
     Skill 5: <完整文本>
     
     [Current State]
     <环境观察和历史>
     
     Please decide the next action using the activated strategies.
     ```
   - Agent 生成动作 \(a_t\)，执行后获得新状态 \(x_{t+1}\) 和即时奖励。
   - 记录每一步的激活技能 ID、门控输出以及动作结果。
3. 轨迹结束时，获得总回报 \(R(\tau)\)。

**关键约束**：未激活技能的完整文本对 Agent **完全不可见**，仅通过摘要表间接感知其存在。这避免了上下文过载和不相关技能的干扰。

#### 3.4.1 渐进披露的三重维度

MOSCOPT 中的渐进披露体现在三个正交维度：

**1. 空间维度——按需加载**

门控 G 仅读取技能的摘要描述（标签、得分、专长），输出 K 个激活 ID 后，**仅被激活的技能才全量加载**到 Agent 上下文中。未激活技能的完整文本对 Agent 完全不可见。这使得 Agent 的上下文仅包含当前最相关的 K 份技能文本，而非全部 N 份。

**2. 时间维度——渐进式摘要丰富（Progressive Summary Enrichment）**

技能摘要表 \(\Sigma_t(\mathcal{S})\) 的信息量随 epoch 逐步丰富：

| Epoch 阶段 | 摘要表包含的信息 |
|-----------|----------------|
| 1–2（初期） | 仅 ID 和标签，防止门控过拟合到不准确的早期统计 |
| 3–4（中期） | 加入近期 Q-score（滑动窗口成功率）|
| 5+（后期） | 加入共现统计、专长描述、被激活频率 |

这种渐进式丰富让门控在学习初期不被噪声统计误导，随着技能质量稳定后再利用更丰富的信息做出更优选择。

**3. 策略维度——轨迹内阶段切换**

在 step-level gating 模式下，轨迹内不同 exec_step 可以激活不同的 K 个技能，自然实现策略的阶段性切换。例如：任务早期偏向"规划型"技能，中期切换到"执行型"技能，后期激活"验证型"技能。这种动态切换本身就是一种策略层面的渐进披露——Agent 在任务的不同阶段只看到与之最相关的策略子集。

### 3.5 集体评分与归因（Collective Credit Assignment）

为了后续优化，需要将轨迹的总回报分配给参与该轨迹的各个技能和门控。

**技能个体得分更新**：对于每个技能 \(s_i\)，统计其在最近一批轨迹中**被激活的 exec_step** 与该轨迹最终回报的加权关系。设技能 \(s_i\) 在轨迹 \(\tau\) 中被激活了 \(c_i(\tau)\) 个 exec_step，轨迹回报为 \(R(\tau)\)，则：

\[
Q_{skill}(s_i) = \text{EMA}\left(\frac{\sum_{\tau} c_i(\tau) \cdot R(\tau)}{\sum_{\tau} c_i(\tau)}\right)
\]

其中 EMA 为指数移动平均，平滑跨 epoch 的得分波动。

**技能协同得分**：若两个技能 \(s_i, s_j\) 在同一轨迹中被共同激活（不必同时步），且轨迹成功，则增加协同计数 \(C(i,j)\)。协同得分定义为成功轨迹中的共现频率，用于指导协同繁殖。

**门控选择归因**：记录门控 G 每次选择的激活组合 \(\mathcal{A}_t\) 对应的轨迹结果。G 的整体质量近似为其激活组合的平均技能得分。具体编辑时，通过 reflect 阶段分析 G 的"选择错误模式"生成更精细的归因信号。

### 3.6 技能池的有界编辑优化（继承 SkillOpt）

每个技能 \(s_i\) 的优化完全遵循 SkillOpt 框架，但验证时需保持**门控 G 和其他技能不变**。编辑采用 **Patch 模式**——通过结构化的编辑操作（append / insert_after / replace / delete）对技能文本进行局部修改，而非全量重写。

**编辑生成（Reflect 阶段）**：收集最近一批轨迹中 \(s_i\) 被激活且最终步骤失败的片段，按 minibatch 分组由 LLM 并行分析。每个 Reflect 调用的 LLM prompt 注入三类上下文：
- **step_buffer_context**（Step Buffer 累积）：本 epoch 内历次 Reflect 的分析结果摘要，帮助 LLM 了解已发现的失败模式，避免重复分析。
- **meta_skill_context**（Meta Skill 优化器记忆）：跨 epoch 蒸馏的优化器指导，帮助 LLM 在证据模糊时做出更好的编辑决策。
- **rejected_buffer_context**（拒绝缓存负反馈）：本 epoch 内被验证门控拒绝的编辑摘要，防止 LLM 再次生成类似的无效编辑。

LLM 生成的编辑为结构化 Patch，包含若干 Edit 操作：
```json
{
  "reasoning": "分析推理过程",
  "edits": [
    {"op": "append", "content": "新的规则文本"},
    {"op": "replace", "target": "原有文本片段", "content": "替换后的文本"},
    {"op": "insert_after", "target": "锚点文本", "content": "在此后插入的新内容"},
    {"op": "delete", "target": "需要删除的文本片段"}
  ]
}
```

**编辑聚合（Aggregate 阶段）**：来自不同失败样本的 Patch 通过**层次化合并**（hierarchical merge）整合。失败驱动的 Patch 优先级高于成功驱动的 Patch。合并过程同样注入 meta_skill_context 和 rejected_buffer_context，确保 LLM 在去重和冲突解决时参考历史优化经验。

**编辑预算（Select 阶段）**：从聚合后的编辑池中选出最优的 L 个编辑（L = edit_budget），由 **LR Scheduler** 动态控制：
- **constant**：固定编辑预算（如始终允许 8 个编辑）。
- **linear**：线性衰减（从 max_lr 递减至 min_lr）。
- **cosine**：余弦退火（平滑衰减）。
- **autonomous**：由 LLM 根据当前 rollout 得分自主决定编辑数量。

若编辑池超出预算，由 LLM 按重要性排序并截取 top-L。

**编辑应用（Update 阶段）**：选中的编辑通过 Patch 模式顺序应用到技能文档。每个编辑操作独立执行并报告状态（applied / skipped / error），支持以下保护机制：
- **Slow Update 保护区**：`<!-- SLOW_UPDATE_START -->` 至 `<!-- SLOW_UPDATE_END -->` 之间的文本区域由慢更新独占写入，step-level 编辑**不可修改**此区域。
- **append 操作**自动插入到 Slow Update 区域之前，保持保护区在文档末尾。

**验证门控（Evaluate 阶段）**：将编辑后的候选技能 \(s_i'\) 在验证集上运行 rollout（使用当前 G 和其他技能），比较 \(s_i'\) 替换 \(s_i\) 后的成功率。只有**严格提升**才接受 \(s_i'\)，否则记录到**拒绝缓存**。

### 3.7 优化策略体系

MOSCOPT 完整继承 SkillOpt 的四大优化策略，并将其适配到多技能池场景。每个策略在多技能环境下有独立的实例和作用域。

#### 3.7.1 Step Buffer（步内累积上下文）

**作用**：在单个 epoch 内，累积历次 Reflect 调用的分析结果摘要，作为上下文注入后续的 Reflect 和 Aggregate 调用。

**机制**：
- 每次 Reflect 完成后，将本次分析的失败模式和编辑建议摘要追加到 step_buffer。
- 后续 Reflect 调用读取 step_buffer_context，了解本 epoch 已发现的问题，避免重复分析相同失败模式。
- **Epoch-local**：每个 epoch 开始时清空，不跨 epoch 持久化。

**多技能适配**：每个技能 \(s_i\) 维护独立的 step_buffer，因为不同技能的失败模式不同。门控 G 也有独立的 step_buffer。

#### 3.7.2 Rejected Buffer（拒绝缓存负反馈）

**作用**：追踪被验证门控拒绝的编辑，将拒绝信息（编辑摘要 + 得分变化 + 失败模式）作为负反馈注入后续 Reflect/Aggregate 的 LLM prompt，避免重复尝试类似编辑。

**机制**：
- 数据结构为**环形 FIFO 缓冲区**（Circular Buffer），容量由 `rejected_buffer_max_size`（默认 10）控制。
- 每条记录包含：step 编号、拒绝前后的得分（score_before / score_after）、编辑摘要列表、关联的失败模式。
- 格式化为 prompt 时，输出标题为 `## Previously Rejected Edits (this epoch)`，明确告知 LLM 避免生成类似编辑。
- **Epoch-local**：每个 epoch 开始时清空。

**多技能适配**：每个技能 \(s_i\) 维护独立的 `skill_reject_buffers[s_i]`，门控 G 维护独立的 `gate_reject_buffer`。拒绝缓存的编辑摘要长度受 `rejected_buffer_max_summary_chars`（默认 200）限制，保持 prompt 在 token 预算内。

#### 3.7.3 Slow Update（慢更新保护区）

**作用**：在 epoch 边界，通过纵向对比（longitudinal comparison）前后两个 epoch 的 per-item 结果，由 LLM 产生指导性文本，写入技能文档的**受保护区域**。

**机制**：
- 比较前后 epoch 对同一批任务的评测结果，分类为四种模式：
  - **improved**（wrong→right）：编辑成功修复了问题。
  - **regressed**（right→wrong）：编辑引入了回归，**最高优先级**。
  - **persistent_fail**（wrong→wrong）：持续失败的问题。
  - **stable_success**（right→right）：稳定成功，无需关注。
- LLM 分析这些变化，产生指导性文本（guidance），写入 `<!-- SLOW_UPDATE_START -->` / `<!-- SLOW_UPDATE_END -->` 标记的保护区域。
- 该区域**只能由 slow update 过程写入**，step-level 的 patch 编辑（append/replace/delete）不可修改此区域。
- 新技能繁殖时，从慢更新保护区提取稳定规则作为“继承基因”。

**多技能适配**：每个技能 \(s_i\) 的文档内嵌独立的 Slow Update 保护区。门控 G 也包含自己的保护区，用于固化通用的成功选择模式（如“规划阶段优先激活规划类技能”）。

#### 3.7.4 Meta Skill（优化器记忆）

**作用**：维护一个紧凑的优化器端记忆，从相邻 epoch 的技能变化中蒸馏出指导性知识。与 Slow Update 不同，Meta Skill **不修改技能文档**，而是注入到后续所有 LLM 调用（Reflect / Aggregate / Select）的 prompt 中，改善优化器自身的决策质量。

**机制**：
- 在 epoch 结束时，比较前后两个 epoch 的技能版本和纵向对比结果。
- LLM 分析“什么类型的编辑有效、什么类型的编辑无效”，蒸馏为简洁的指导文本。
- 格式化为 `## Optimizer Meta Skill` 上下文块，注入后续所有 LLM prompt。
- 指导原则：“当证据模糊时优先参考 meta skill，但当轨迹明确矛盾时以轨迹为准”。

**多技能适配**：维护**全局** meta_skill_content（跨所有技能蒸馏），因为优化器层面的经验（如“添加边界条件检查通常有效”）对所有技能和门控都适用。这避免了为 N 个技能分别维护 meta skill 的开销。

#### 3.7.5 LR Scheduler（学习率调度器）

**作用**：动态控制每个 opt_step 的编辑预算（最大编辑数量 L），避免早期过度编辑或后期编辑不足。

**四种调度模式**：

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| constant | 固定预算 | 任务难度均匀 |
| linear | 从 max_lr 线性衰减至 min_lr | 技能趋于稳定时减少扰动 |
| cosine | 余弦退火 | 平滑衰减，避免突变 |
| autonomous | LLM 自主决定编辑数 | 任务难度波动大 |

**多技能适配**：全局共享一个 LR Scheduler 实例，因为编辑预算是 opt_step 级别的资源约束。autonomous 模式下，LLM 根据当前 rollout 的 hard/soft 得分决定本步编辑多少个编辑。

### 3.8 门控文本技能 G 的优化

门控 G 本身也是文本技能，同样通过有界编辑 + 验证门控进行优化。但编辑的生成和验证略有不同。

**编辑信号——选择错误模式**：从轨迹中提取 G 的选择错误，分为正反馈和负反馈两类：

- **负反馈编辑**（选择失误）：
  - 若某 exec_step G 选择的激活组合导致后续动作失败，且失败原因与技能选择不当有关（例如错过了一个本应激活的高成功率技能），则生成编辑建议。
  - 示例：`"当检测到子任务涉及数学计算时，不要激活 Skill 5（文本生成类），应优先激活 Skill 3（数学专长）。"`
  - 示例：`"在任务的最后 20% 步骤，优先激活具有'最终检验'标签的技能。"`

- **正反馈编辑**（成功经验固化）：
  - 若某激活组合明显提升了效率，记录该模式用于慢更新时固化规则。
  - 示例：`"在规划阶段，优先激活 Skill 2 和 Skill 7 的组合，成功率达 92%。"`

**有界编辑**：与技能编辑相同，修改幅度受文本学习率约束（字符比例模式或词数上限模式），进行外科手术式修补，不改变 G 的整体结构。

**验证门控**：生成候选 \(G'\)。在验证集上**固定技能池**，比较 \(G'\) 与原 G 的整体成功率。严格提升则接受，否则加入门控的拒绝缓存。

### 3.9 三阶段交错更新

为避免技能与门控同时变化导致不可控的相互影响，采用**三阶段交错更新**策略：

**Phase 1：技能编辑（门控固定）**

在当前 G 不变的前提下，对所有技能按失败贡献程度排序，依次尝试编辑：
1. 根据 \(Q_{skill}\) 和失败频率，确定编辑优先级。
2. 对每个候选技能执行 SkillOpt 的 6 阶段 pipeline（rollout → reflect → aggregate → select → update → evaluate），每次 Reflect/Aggregate 调用注入该技能的 step_buffer、rejected_buffer 和全局 meta_skill 上下文。
3. 验证时使用当前 G 的选择行为，确保编辑效果可归因到单一技能。
4. 仅通过验证门控的编辑才被接受；被拒绝的编辑记录到 `skill_reject_buffers[s_i]`。

**Phase 2：门控编辑（技能池固定）**

在技能池稳定后（本 opt_step 技能更新完成），尝试编辑 G：
1. 从轨迹中提取门控的选择错误模式（Section 3.8）。
2. 生成有界编辑候选 \(G'\)，Reflect 调用注入门控的 step_buffer、gate_reject_buffer 和全局 meta_skill 上下文。
3. 在验证集上**固定技能池**，比较 \(G'\) 与原 G 的整体成功率。
4. 严格提升则接受，否则加入 `gate_reject_buffer`。

**Phase 3：集体进化（每 E 个 epoch 触发一次）**

在多个 opt_step 完成后，执行集体进化操作：
1. **淘汰**：移除综合得分 \(Q_{skill}\) 最低的 M 个技能（保持 \(N \geq K+M\)）。
2. **繁殖**：从得分最高的技能中选取"亲本"，使用 LLM 生成变异版本（如修改 30% 的文本、引入新的策略词汇），形成 M 个新技能加入池中。
3. **协同繁殖**（可选）：若某对技能 \((i,j)\) 的协同得分 \(C(i,j)\) 极高，尝试用 LLM 将它们合并为一个新技能 \(s_{ij}\)，结合双方优势。
4. **慢更新（Slow Update）**：在 epoch 边界，对每个技能执行纵向对比——比较前后 epoch 的 per-item 结果，分类为 improved / regressed / persistent_fail / stable_success，由 LLM 产生指导性文本写入技能文档的 `<!-- SLOW_UPDATE_START -->` / `<!-- SLOW_UPDATE_END -->` 保护区。门控 G 同样执行慢更新，固化通用的成功选择模式。新技能繁殖时，从慢更新保护区提取稳定规则作为继承基因。
5. **Meta Skill 更新**：在 epoch 结束时，从前后 epoch 的技能变化中蒸馏优化器记忆，更新全局 meta_skill_content，注入后续所有 LLM 调用。
6. **摘要更新**：随着技能池变化，更新摘要表 \(\Sigma(\mathcal{S})\)，并根据当前 epoch 适当增加披露的信息粒度（渐进式摘要丰富）。

这种三阶段交错确保了技能、门控和池结构三个维度不会同时变化，从而避免相互干扰导致的优化震荡。

### 3.10 收敛与输出

当技能池规模稳定、门控选择分布趋于集中、验证集成功率不再提升时，停止优化。最终系统可根据需要输出不同形式：

- **完整混合系统**：\(\mathcal{S}, G, K\)，部署时继续动态激活。适用于支持运行时门控的环境。
- **最优单技能**：由门控激活频率最高的技能，或通过 LLM 将所有高分技能合并为单个文本，以兼容无门控环境（如原生 SkillOpt 部署）。
- **静态技能子集 + 路由表**：提取门控的决策逻辑，固化为一组 if-then 规则和一个精简技能集，减少推理开销。

### 3.11 计算复杂度分析

MOSCOPT 相较于 SkillOpt 引入了额外的计算开销，需要显式分析和缓解。

**每个 opt_step 的 LLM 调用次数**：

| 操作 | LLM 调用次数 | 说明 |
|------|-------------|------|
| 门控选择（rollout 阶段） | \(B \times T_{exec}\) | B 个 task，每 task 平均 \(T_{exec}\) 个 exec_step |
| Agent rollout | \(B \times T_{exec}\) | 与 SkillOpt 相同 |
| 技能 Reflect | \(\lceil B/M \rceil \times N_{edit}\) | M = minibatch_size，\(N_{edit}\) = 需编辑的技能数 |
| 技能验证 | \(N_{edit} \times |\text{val}|\) | 每个编辑候选需在验证集上完整 rollout |
| 门控验证 | \(1 \times |\text{val}|\) | 门控编辑候选的验证 |
| 集体进化（每 E epoch） | \(M \times 2 + C\) | M 次变异 + 可选的 C 次协同合并 |

**与 SkillOpt 的对比**：SkillOpt 每个 opt_step 的 LLM 调用为 \(O(B + R)\)（B 次 rollout + R 次 reflect），MOSCOPT 则增加至 \(O(B \cdot T_{exec} + N \cdot |\text{val}| + |\text{val}|)\)。当 K=1, N=1 时，门控选择退化为固定选择，复杂度回落至 SkillOpt 级别。

**缓解策略**：
- **采样验证**：验证时仅使用验证集的子集（如 50%），减少验证 rollout 次数。
- **延迟验证**：积累多个编辑候选后批量验证，共享 rollout 基础设施。
- **按需门控**：task-level gating 时，每 task 仅调用一次门控 LLM（而非每 exec_step），大幅降低门控调用次数。
- **并行化**：技能编辑阶段的各技能验证可并行执行；门控选择与 Agent rollout 可流水线化。

---

## 4. 超参数汇总

| 超参数 | 符号 | 默认值 | 说明 |
|--------|------|--------|------|
| 技能池大小 | N | 5 | 池中技能总数，淘汰/繁殖后保持恒定 |
| 激活数量 | K | 2 | 每次激活的技能数，1 ≤ K ≤ N |
| 淘汰/繁殖数 | M | 1 | 每轮集体进化替换的技能数 |
| 进化间隔 | E | 5 epochs | 集体进化的触发频率 |
| 门控粒度 | — | task | task-level（默认）或 step-level |
| 摘要丰富策略 | — | epoch-based | 渐进式摘要丰富的披露节奏 |
| 编辑预算（初始） | max_lr | 8 | 每 opt_step 最大编辑数（LR scheduler 的初始值） |
| 编辑预算（最小） | min_lr | 2 | 衰减调度器的最小编辑数 |
| LR 调度器 | — | cosine | constant / linear / cosine / autonomous |
| 拒绝缓冲大小 | R | 10 | 每技能/门控的拒绝缓存容量（FIFO） |
| 拒绝摘要字符上限 | — | 200 | 每条拒绝编辑摘要的最大字符数 |
| Minibatch 大小 | M_b | 4 | Reflect 阶段的分组大小 |
| 合并批大小 | M_g | 8 | Aggregate 层次化合并的批大小 |
| 验证集采样比例 | p_val | 1.0 | 验证时使用的验证集比例（1.0 = 全量） |
| EMA 平滑系数 | \(\beta\) | 0.3 | Q-score 指数移动平均的平滑系数 |
| 最小激活次数 | \(c_{min}\) | 5 | 技能 Q-score 生效的最低激活次数阈值 |
| 启用 Slow Update | — | true | 是否在 epoch 边界执行慢更新 |
| 启用 Meta Skill | — | true | 是否在 epoch 结束时更新优化器记忆 |
| 纵向对比策略 | — | mixed | mixed / changed / unchanged |

---

## 5. 失败模式与缓解策略

### 5.1 门控输出解析失败

**现象**：G 输出的文本无法解析为恰好 K 个有效技能 ID（如输出了 3 个 ID 当 K=2，或输出了不存在的 ID）。

**缓解**：
- **即时回退**：解析失败时，按 Q-score 排序选择 top-K 技能。
- **编辑信号**：将解析失败事件记录为门控的负反馈，在 Phase 2 中触发门控编辑（如添加规则："严格输出恰好 K 个 ID，不要多也不要少"）。
- **频率监控**：若连续多个 opt_step 的解析失败率 > 30%，触发门控重建（重新初始化 G）。

### 5.2 技能多样性崩塌

**现象**：经过多轮淘汰和繁殖，技能池中的 N 个技能文本趋于相似（高分技能的"近亲繁殖"）。

**缓解**：
- **多样性度量**：每个 epoch 计算技能池的文本相似度矩阵（如基于编辑距离或 embedding 余弦相似度）。若平均相似度超过阈值（如 > 0.85），触发强制变异。
- **强制变异**：在繁殖阶段，对新生成的技能施加更大的变异幅度（如修改 50% 的文本，或引入全新的角色指令）。
- **外来基因注入**：从种子提示重新采样 1–2 个全新技能替换最低分技能。

### 5.3 归因噪声导致错误淘汰

**现象**：由于激活组合的随机性或验证集噪声，某个实际上有价值的技能因 Q-score 暂时偏低被错误淘汰。

**缓解**：
- **滑动窗口平滑**：Q-score 使用 EMA 而非单 epoch 得分，平滑短期波动。
- **最小激活次数阈值**：仅当技能被激活次数 ≥ \(c_{min}\) 时，Q-score 才参与淘汰排序；激活次数不足的技能免于淘汰。
- **保护机制**：协同得分 \(C(i,j)\) 高的技能对受到保护，即使个体 Q-score 偏低也不轻易淘汰（因为其组合价值高）。

### 5.4 K=N 退化

**现象**：当 K 接近或等于 N 时，门控失去选择意义，退化为全量加载，上下文长度膨胀。

**缓解**：
- **约束检查**：系统启动时校验 K < N，若不满足则发出警告并建议调整。
- **自动建议**：当检测到 K/N > 0.8 时，建议用户增大 N 或减小 K。

### 5.5 验证开销爆炸

**现象**：N 个技能逐一验证 + 门控验证，每个 opt_step 的验证 rollout 次数可达 \((N+1) \times |\text{val}|\)，计算成本不可承受。

**缓解**：
- **采样验证**：仅使用验证集的子集（参数 \(p_{val}\)）。
- **优先级验证**：仅对失败频率高于阈值的技能进行编辑和验证，跳过表现稳定的技能。
- **异步验证**：将验证推迟到下一个 opt_step 的 rollout 阶段并行执行。

---

## 6. 算法伪代码

```
# =============================================================
# MOSCOPT: Mixture-of-Skill Collective Optimization
# 融合 SkillOpt 优化策略体系：Patch 模式 + Step Buffer +
# Rejected Buffer + Slow Update + Meta Skill + LR Scheduler
# =============================================================

Input: task distribution T, base LLM agent, K, N, M, E, max_lr, min_lr
Output: optimized skill pool S, gating skill G

# ----- 初始化 -----
S = {s_1, ..., s_N}                       # 通过多样采样和过滤
for s_i in S:
    s_i = inject_empty_slow_update_field(s_i)  # 初始化 Slow Update 保护区
G = generate_initial_gating_prompt()          # 文本门控，含选择规则和回退说明
G = inject_empty_slow_update_field(G)         # 门控也包含 Slow Update 保护区
Summaries = generate_summaries(S)             # 初始摘要：仅 ID + 标签

# --- 优化器状态（每个技能/门控独立 + 全局共享） ---
skill_reject_buffers = {s_i: RejectedBuffer(max_size=R, max_summary_chars=200) for s_i in S}
gate_reject_buffer = RejectedBuffer(max_size=R, max_summary_chars=200)
skill_step_buffers = {s_i: [] for s_i in S}   # epoch-local 累积
gate_step_buffer = []                          # epoch-local 累积
Q_scores = {s_i: 0.0 for s_i in S}            # EMA Q-score
cooccurrence = {}                              # 协同得分矩阵
lr_scheduler = build_scheduler(mode="cosine", max_lr=max_lr, min_lr=min_lr, total_steps=total)
meta_skill_content = ""                        # 全局优化器记忆
prev_epoch_results = {}                        # 用于 slow update 纵向对比

# =============================================================
# 外循环：Epoch
# =============================================================
for epoch in 1..MAX_EPOCHS:
    
    # ---- on_epoch_start: 清空 epoch-local 缓冲区 ----
    for buf in skill_reject_buffers.values(): buf.clear()
    gate_reject_buffer.clear()
    for sb in skill_step_buffers.values(): sb.clear()
    gate_step_buffer.clear()
    
    # =========================================================
    # 外循环：Optimization Step (opt_step)
    # 每个 opt_step 包含完整的 6 阶段 pipeline
    # =========================================================
    edit_budget = lr_scheduler.step()   # LR Scheduler 决定本步编辑预算
    
    # ---- Stage 1: Rollout (数据收集) ----
    # 内循环：Execution Step (exec_step)
    trajectories = []
    for each task in T_train:
        tau = []    # 单条轨迹
        state = task.init
        
        # === 门控粒度选择 ===
        if gating_granularity == "task-level":
            activated_ids = G(state, history=[], Summaries)
            if parse_fail(activated_ids) or len(activated_ids) != K:
                activated_ids = fallback_topK(Q_scores, K)  # 回退规则
        
        for t in 1..T_MAX:   # exec_step 内循环
            if gating_granularity == "step-level":
                activated_ids = G(state, history, Summaries)
                if parse_fail(activated_ids) or len(activated_ids) != K:
                    activated_ids = fallback_topK(Q_scores, K)
            
            # 全量加载被激活的技能（渐进披露：未激活不可见）
            skill_texts = [full_text(S[id]) for id in activated_ids]
            prompt = build_agent_prompt(skill_texts, state)
            action = Agent(prompt)
            next_state, step_reward = task.step(action)
            tau.append( (state, activated_ids, action, step_reward) )
            state = next_state
            if task.done: break
        
        total_reward = task.evaluate(tau)
        trajectories.append( (tau, total_reward) )
    
    curr_results = trajectories   # 保留用于后续 slow update
    
    # ---- 归因与评分 ----
    for each skill s_i in S:
        Q_scores[s_i] = ema_update(Q_scores[s_i], compute_skill_reward(s_i, trajectories))
    update_cooccurrence(cooccurrence, trajectories)
    
    # ---- Phase 1: 技能编辑 (门控 G 固定) ----
    # Patch 模式：reflect → aggregate → select → update → evaluate
    edit_candidates = rank_by_failure_contribution(S, trajectories)
    for s_i in edit_candidates:
        failure_ctx = get_failures_for_skill(s_i, trajectories)
        if failure_ctx is empty: continue
        
        # Stage 2: Reflect (minibatch 分组分析)
        # 注入三类上下文：step_buffer + meta_skill + rejected_buffer
        sb_ctx = format_step_buffer(skill_step_buffers[s_i])
        ms_ctx = format_meta_skill_context(meta_skill_content)
        rb_ctx = skill_reject_buffers[s_i].format_context()
        
        minibatches = split(failure_ctx, minibatch_size=M_b)
        patches = []
        for mb in minibatches:
            patch = reflect_analyze(
                mb, s_i, budget=edit_budget,
                step_buffer_context=sb_ctx,
                meta_skill_context=ms_ctx,
                rejected_buffer_context=rb_ctx,
            )
            patches.append(patch)
            # 累积到 step_buffer
            skill_step_buffers[s_i].append(summarize_reflect(patch))
        
        # Stage 3: Aggregate (层次化合并，失败优先)
        merged_patch = merge_patches(
            patches, skill_content=s_i,
            failure_priority=True,
            meta_skill_context=ms_ctx,
            rejected_buffer_context=rb_ctx,
            batch_size=M_g,
        )
        
        # Stage 4: Select (LR Scheduler 控制编辑预算)
        actual_budget = edit_budget
        if lr_mode == "autonomous":
            actual_budget = llm_decide_budget(
                skill=s_i, merged_patch=merged_patch,
                rollout_hard=curr_hard, rollout_soft=curr_soft,
                meta_skill_context=ms_ctx,
            )
        candidate = rank_and_select(
            merged_patch, skill_content=s_i,
            max_edits=actual_budget,
            meta_skill_context=ms_ctx,
        )
        if candidate is None or candidate.edits is empty: continue
        
        # Stage 5: Update (Patch 模式应用编辑)
        # 每个编辑独立执行: append/insert_after/replace/delete
        # Slow Update 保护区内的文本不可被修改
        candidate_skill = apply_patch_with_report(s_i, candidate)
        
        # Stage 6: Evaluate (验证门控)
        new_score = validate(candidate_skill, S_replace={s_i: candidate_skill}, G=G, tasks=T_val)
        old_score = evaluate_current(S, G, T_val)
        if new_score > old_score:    # 严格提升才接受
            S[s_i] = candidate_skill
        else:
            # 记录到 Rejected Buffer（包含编辑摘要 + 得分变化 + 失败模式）
            skill_reject_buffers[s_i].add(
                step=step, edits=candidate.edits,
                score_before=old_score, score_after=new_score,
                failure_patterns=extract_failure_patterns(trajectories, s_i),
            )
    
    # ---- Phase 2: 门控编辑 (技能池 S 固定) ----
    selection_failures = get_failures_for_gate(G, trajectories)
    if selection_failures not empty:
        # Reflect: 分析门控选择错误模式
        g_sb_ctx = format_step_buffer(gate_step_buffer)
        g_rb_ctx = gate_reject_buffer.format_context()
        gate_patch = reflect_gate(
            selection_failures, G, budget=edit_budget,
            step_buffer_context=g_sb_ctx,
            meta_skill_context=ms_ctx,
            rejected_buffer_context=g_rb_ctx,
        )
        gate_step_buffer.append(summarize_reflect(gate_patch))
        
        # Select + Update
        gate_candidate = rank_and_select(gate_patch, skill_content=G, max_edits=edit_budget)
        if gate_candidate is not None:
            gate_candidate_text = apply_patch_with_report(G, gate_candidate)
            new_gate_score = validate_gate(gate_candidate_text, S, T_val)
            old_gate_score = evaluate_gate_current(G, S, T_val)
            if new_gate_score > old_gate_score:
                G = gate_candidate_text
            else:
                gate_reject_buffer.add(
                    step=step, edits=gate_candidate.edits,
                    score_before=old_gate_score, score_after=new_gate_score,
                )
    
    # ---- Evaluate: 本 opt_step 整体评估 ----
    epoch_score = evaluate(S, G, T_val)
    log(epoch, epoch_score, Q_scores, len(S))
    
    # =========================================================
    # on_epoch_end: Slow Update + Meta Skill
    # =========================================================
    if prev_epoch_results and curr_results:
        # ---- Slow Update (每个技能 + 门控) ----
        for s_i in S:
            comparison_pairs = build_comparison_pairs(
                prev_epoch_results[s_i], curr_results[s_i],
                policy=longitudinal_pair_policy,  # mixed / changed / unchanged
            )
            su_guidance = run_slow_update(
                prev_skill=prev_epoch_skill[s_i],
                curr_skill=s_i,
                comparison_pairs=comparison_pairs,
            )
            if su_guidance:
                # 写入 Slow Update 保护区（step-level 编辑不可修改）
                S[s_i] = replace_slow_update_field(S[s_i], su_guidance)
        
        # 门控 Slow Update
        gate_comparison = build_comparison_pairs(prev_gate_results, curr_gate_results)
        gate_su = run_slow_update(prev_skill=prev_G, curr_skill=G, comparison_pairs=gate_comparison)
        if gate_su:
            G = replace_slow_update_field(G, gate_su)
        
        # ---- Meta Skill 更新（全局优化器记忆） ----
        meta_result = run_meta_skill(
            prev_skill=prev_epoch_aggregate_skill,
            curr_skill=current_aggregate_skill,
            comparison_pairs=all_comparison_pairs,
            prev_meta_skill_content=meta_skill_content,
        )
        if meta_result:
            meta_skill_content = meta_result["meta_skill_content"]
    
    prev_epoch_results = curr_results    # 保留用于下 epoch 的 slow update
    prev_epoch_skill = {s_i: S[s_i] for s_i in S}
    
    # ---- Phase 3: 集体进化 (每 E 个 epoch 触发) ----
    if epoch % E == 0:
        # 3a. 淘汰
        lowest = select_lowest_scored(S, Q_scores, M, min_activations=c_min)
        S = S - lowest
        
        # 3b. 繁殖（继承 Slow Update 保护区的稳定规则）
        parents = select_top(S, Q_scores, M)
        new_skills = []
        for p in parents:
            slow_rules = extract_slow_update_field(S[p])   # 从保护区提取稳定规则
            new_skills.append( mutate(p, inherit_rules=slow_rules) )
        S = S ∪ new_skills[:M]    # 保持总数 N
        for s_new in new_skills:
            s_new = inject_empty_slow_update_field(s_new)   # 初始化保护区
            skill_reject_buffers[s_new] = RejectedBuffer()
            skill_step_buffers[s_new] = []
        
        # 3c. 协同合并 (可选)
        top_pair = get_highest_cooccurrence(cooccurrence, S)
        if top_pair and C(top_pair) > threshold:
            merged = merge_skills(top_pair)
            merged = inject_empty_slow_update_field(merged)
            S = S - {lowest_remaining} ∪ {merged}
        
        # 3d. 摘要更新（渐进式摘要丰富）
        Summaries = generate_rich_summaries(S, Q_scores, cooccurrence, epoch)
        
        # 3e. 多样性检查
        if diversity(S) < diversity_threshold:
            S = inject_diversity(S, seed_prompts)

# ----- 输出 -----
return best_mixture(S, G) or distill_top_skill(S, G)
```

---

## 7. 与 SkillOpt 的对比分析

| 特性 | SkillOpt | MOSCOPT |
|------|----------|---------|
| 优化对象 | 单个文本技能 | 文本技能池 + 文本门控 |
| 探索能力 | 单一路径，易局部最优 | 多技能并行探索，门控动态组合 |
| 长程任务 | 固定策略全程 | 可根据子任务分阶段切换技能 |
| 协同利用 | 无 | 显式建模和奖励技能间协同 |
| 安全编辑 | 有界编辑 + 验证门控 | 完全继承，且三阶段交错更新防振荡 |
| 每 opt_step LLM 调用 | \(O(B + R)\) | \(O(B \cdot T + N \cdot |\text{val}| + |\text{val}|)\) |
| 上下文长度 | \(1 \times \text{skill\_len}\) | \(K \times \text{skill\_len} + \text{summary\_len}\) |
| 适用任务类型 | 单轮 QA / 短轨迹 | 多阶段 / 长轨迹 |
| 部署复杂度 | 低（单文本注入） | 中（池 + 门控运行时调度） |
| 局部最优抵抗 | 低（单一搜索路径） | 高（多技能竞争 + 淘汰繁殖） |
| 向后兼容 | — | K=1 且 N=1 时完全退化为 SkillOpt |

MOSCOPT 本质上构建了一个**技能生态**，而 SkillOpt 只是其中一支单一种群。通过引入门控选择与集体进化，MOSCOPT 大幅提升了策略的覆盖面和鲁棒性，代价是显著增加的计算开销和部署复杂度。

---

## 8. 何时使用 MOSCOPT

并非所有场景都需要 MOSCOPT。以下决策树帮助用户选择合适的算法：

```
任务是单轮 QA / 短轨迹？
├── 是 → SkillOpt（简单高效，无额外开销）
└── 否 → 任务有多个明显阶段？
    ├── 是 + 阶段间策略差异大 → MOSCOPT (task-level gating)
    └── 是 + 长程多步交互 → MOSCOPT (step-level gating)

计算预算有限？
├── 是 → SkillOpt（MOSCOPT 开销约 N 倍）
└── 否 → MOSCOPT

需要多种策略互补？
├── 是 → MOSCOPT（多技能竞争 + 协同利用）
└── 否 → SkillOpt
```

**简而言之**：SkillOpt 是默认选择；当任务具有多阶段、多策略互补需求且计算预算允许时，升级到 MOSCOPT。

---

## 9. 与 SummerClaw 的集成设计

MOSCOPT 通过实现 `BaseAlgorithm` 接口接入 TrainerEngine，无需修改引擎核心代码。

### 9.1 接口适配

MOSCOPT 实现 `BaseAlgorithm` 的 6 个抽象方法，内部维护技能池和门控状态：

- `rollout(env, skill, items, out_dir)`：解析复合技能文档，提取技能池和门控，按门控粒度执行 rollout。
- `reflect(results, skill, out_dir)`：分析轨迹，同时生成技能编辑补丁和门控编辑补丁。
- `aggregate(patches, skill)`：合并技能和门控的编辑候选。
- `select(patch, budget, skill)`：在编辑预算内选择最优候选。
- `update(skill, patch)`：应用编辑，更新技能池或门控文本。
- `evaluate(env, skill, items, out_dir)`：在验证/测试集上评估当前混合策略。

### 9.2 复合技能文档

为兼容 TrainerEngine 的 `_current_skill: str` 单文本接口，MOSCOPT 将技能池和门控序列化为一个结构化文本块：

```
<!-- MOSCOPT Pool Start -->
<!-- N=5, K=2, epoch=3 -->

## Gate
<门控 G 的完整文本>

## Skill 1: 保守型规划
<skill_1 的完整文本>

## Skill 2: 高效执行
<skill_2 的完整文本>

...

<!-- MOSCOPT Pool End -->
```

`rollout()` 方法解析此文档，提取各组件；`update()` 方法在编辑后重新序列化。

### 9.3 Dashboard 可视化

MOSCOPT 扩展 Dashboard 的可视化能力：
- **技能池 Q-score 曲线**：每个技能一条曲线，展示其 Q-score 随 epoch 的变化，淘汰/繁殖事件标记为垂直虚线。
- **门控选择分布**：热力图展示不同技能被激活的频率和共现模式。
- **技能池组成变化**：时间线展示技能池成员的变化（淘汰、繁殖、合并事件）。

### 9.4 向后兼容

当 K=1 且 N=1 时：
- 技能池退化为单个技能。
- 门控 G 固定输出 `ACTIVATE: 1`，无需 LLM 调用。
- 集体进化不触发（M=0）。
- 所有阶段等价于原生 SkillOpt 的 6 阶段 pipeline。

这确保了 MOSCOPT 可以作为 SkillOpt 的**超集**存在，用户无需维护两套代码。

---

## 10. 实验设计建议

为验证 MOSCOPT 的有效性，可设计以下实验：

1. **复杂多阶段任务**：例如 WebShop（购物需搜索、比较、下单）、科学推理链（多个子问题）。对比 SkillOpt 单技能、固定技能池随机选择、独立多 SkillOpt 取优。
2. **消融实验**：
   - 关闭门控进化（G 固定）vs 完整 MOSCOPT；
   - 不同 K 值对性能的影响；
   - 有无协同繁殖的影响；
   - task-level vs step-level gating 对比。
3. **安全性测试**：观察优化过程中技能文本是否出现有害或矛盾指令，验证编辑的单调性。
4. **可解释性分析**：可视化门控在不同任务阶段选择的技能分布，展示渐进披露带来的变化。
5. **计算效率对比**：记录不同 N、K 配置下的 opt_step 耗时和 LLM 调用次数，验证复杂度分析。
6. **向后兼容验证**：在 K=1, N=1 配置下，确认 MOSCOPT 与 SkillOpt 的输出完全一致。

预期结果：MOSCOPT 在长程、复杂任务上显著优于单技能优化，且进化过程稳定，不会出现技能退化。

---

## 11. 结论

MOSCOPT 将 SkillOpt 的文本态安全优化拓展到了多技能协同场景，首次实现了完全参数自由的技能集体进化。通过将门控设计为可优化的文本技能，并采用渐进披露的固定数量激活机制，系统在保持高效上下文利用的同时，获得了动态任务适应的能力。三阶段交错更新（技能编辑 → 门控编辑 → 集体进化）确保了各组件的独立安全演化，三重渐进披露（空间按需加载、时间摘要丰富、策略阶段切换）实现了多粒度的信息控制。通过完整继承 SkillOpt 的四大优化策略（Step Buffer 累积、Rejected Buffer 负反馈、Slow Update 保护区、Meta Skill 优化器记忆）并采用 Patch 模式执行结构化编辑，MOSCOPT 在多技能环境下实现了与 SkillOpt 同等的编辑安全性和优化稳定性。该框架完全向后兼容 SkillOpt（K=1, N=1 时退化），为构建自主进化、可解释、可部署的智能体技能系统提供了坚实基础。

---

**关键词**：智能体技能优化，文本态进化，技能混合，门控机制，渐进披露，Patch 模式，SkillOpt
