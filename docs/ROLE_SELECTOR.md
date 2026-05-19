# Role Selector — 角色选择器

根据项目需求文档，从 `resources/roles` 的 **1016 个专业角色**（22 个分类）中，调用 LLM 智能选择最合适的角色组合，存储到工作目录的 `roles/selected/`。

---

## 目录

1. [快速开始](#快速开始)
2. [配置说明](#配置说明)
3. [需求文件](#需求文件)
4. [使用案例](#使用案例)
5. [命令行参数](#命令行参数)
6. [工作原理](#工作原理)
7. [故障排除](#故障排除)
8. [编程使用](#编程使用)
9. [最佳实践](#最佳实践)

---

## 快速开始

### 1. 配置

在 `~/.summerclaw/config.json` 中添加：

```json
{
  "agents": {
    "defaults": {
      "role_selector": {
        "enabled": true,
        "requirements": "roles/requirements.md",
        "count": 5
      }
    }
  }
}
```

### 2. 编辑需求（可选）

首次运行时系统会自动创建默认需求文件。你也可以手动编辑：

```bash
nano ~/.summerclaw/workspace/roles/requirements.md
```

### 3. 执行

```bash
summerclaw roles select
```

### 4. 查看

```bash
ls ~/.summerclaw/workspace/roles/selected/
```

---

## 配置说明

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | `bool` | `false` | 是否启用 |
| `requirements` | `str` | `"roles/requirements.md"` | 需求文件路径（相对于 workspace） |
| `count` | `int` | `5` | 选择数量（1~50） |
| `model_override` | `str|null` | `null` | 可选的模型覆盖 |

### 通过命令行覆盖

```bash
summerclaw roles select -m anthropic/claude-opus-4-5   # 指定模型
summerclaw roles select -w /path/to/workspace            # 指定工作目录
summerclaw roles select -c /path/to/config.json          # 指定配置
```

---

## 需求文件

需求文件默认位于 `workspace/roles/requirements.md`。首次运行时若文件不存在，自动创建：

```markdown
# 角色选择需求文档

请根据以下需求选择合适的角色：

需要数据分析师、软件工程师、UI设计师、金融分析师、LLM大模型工程师
```

### 自定义路径

```json
{ "role_selector": { "requirements": "docs/my-project.md" } }
```

支持相对于 workspace 的路径或绝对路径。

---

## 使用案例

### 案例 1：智能客服系统

需求文件内容：

```markdown
构建一个智能客服系统，要求能够：
1. 理解用户问题并提供专业回答
2. 分析用户行为数据，优化服务流程
3. 设计友好的用户界面和交互体验
4. 管理系统项目和团队协作
5. 处理技术问题和系统集成
```

可能选中：客服专员、数据分析师、UI设计师、产品经理、系统架构师

### 案例 2：电商平台

```markdown
为电商平台开发完整运营体系：
1. 前端开发（React/Vue 移动端适配）
2. 后端开发（订单、支付、库存）
3. 数据分析（用户行为、销售决策）
4. 用户体验（购物流程优化）
5. 项目管理（跨团队协调）
6. 营销推广（品牌策略）

选择 8 个角色。
```

可能选中：前端工程师、后端工程师、数据分析师、UX设计师、项目经理、客服专员、数字营销专家、产品经理

### 案例 3：AI 产品研发

```markdown
开发基于大语言模型的智能助手产品：
1. AI算法研发（微调、Prompt Engineering、RAG）
2. 软件工程（高并发、可扩展架构）
3. 产品设计（人机交互体验）
4. 数据工程（训练数据 pipeline）
5. 运维部署（推理服务 + 监控）

选择 6 个角色。
```

可能选中：LLM大模型工程师、提示词工程师、软件工程师、AI产品经理、数据工程师、DevOps工程师

### 案例 4：新媒体内容团队

```markdown
建立科技类新媒体内容创作工作室：
1. 内容策划（选题、内容日历）
2. 文案写作（深度文章、技术解读）
3. 视觉设计（配图、信息图、封面）
4. 视频制作（脚本、拍摄、后期）
5. 社交媒体运营（多平台分发、数据分析）
6. SEO优化

选择 5 个角色。
```

可能选中：内容创作者、技术作家、平面设计师、视频编导、社交媒体经理

---

## 需求文档编写技巧

### ✅ 好的需求

| 类型 | 示例 |
|---|---|
| 具体明确 | "需要 React 前端、FastAPI 后端、PostgreSQL 数据库" |
| 包含上下文 | "金融科技创业公司，开发个人理财 App" |
| 列出优先级 | "最高优先级：AI算法；重要：后端开发；辅助：测试" |

### ❌ 避免

| 类型 | 示例 |
|---|---|
| 过于模糊 | "我需要一些角色" |
| 缺乏上下文 | "选择5个角色" |
| 不切实际 | "一个角色完成开发、设计、营销、法务全栈" |

---

## 命令行参数

```bash
summerclaw roles select [OPTIONS]

Options:
  -w, --workspace TEXT    工作目录路径
  -c, --config TEXT       配置文件路径
  -m, --model TEXT        覆盖模型名称
  -h, --help              显示帮助信息
```

---

## 工作原理

```
配置检查 → enabled? → 否: 跳过
    ↓ 是
需求文件检查 → 不存在? → 自动创建默认文件
    ↓
缓存检查 → selected/ 目录已有角色? → 提示已存在，告知删除重试
    ↓ 否
扫描资源 → 1016 个角色，22 个分类
    ↓
构建 Prompt → 需求 + 角色列表
    ↓
LLM 选择 → 返回 JSON 数组
    ↓
复制文件 → workspace/roles/selected/
    ↓
完成 ✓
```

### 缓存机制（跳过机制）

如果 `workspace/roles/selected/` 已存在且有角色文件，**自动跳过**并提示：

```
⚠ 角色已存在 (5 个)，跳过选择
  位置: ~/.summerclaw/workspace/roles/selected
  如需重新选择，请删除 ~/.summerclaw/workspace/roles/selected 后重试
```

重新选择：

```bash
rm -rf ~/.summerclaw/workspace/roles/selected
summerclaw roles select
```

---

## 故障排除

### 未启用

```
Role selector is not enabled in config.
```

**解决**：配置中设置 `"enabled": true`

### 已存在跳过

```
⚠ 角色已存在，如需重新选择请删除后重试
```

**解决**：`rm -rf ~/.summerclaw/workspace/roles/selected`

### 选择失败

常见原因：
- API 密钥未配置 → `summerclaw status` 检查
- 网络连接问题 → 检查网络
- 需求太模糊 → 补充具体信息

### 结果不符预期

- 增加需求文档细节
- 明确列出所需能力
- 角色数控制在 3~8 个

---

## 编程使用

### 同步调用

```python
from summerclaw.agent.role_selector import select_roles_sync

success = select_roles_sync()
if success:
    print("角色选择成功")
```

### 异步调用

```python
import asyncio
from summerclaw.agent.role_selector import select_roles

async def main():
    return await select_roles()

asyncio.run(main())
```

### 自定义配置

```python
from summerclaw.config.schema import Config, RoleSelectorConfig

config = Config()
config.agents.defaults.role_selector = RoleSelectorConfig(
    enabled=True,
    requirements="roles/my-requirements.md",
    count=5
)
select_roles_sync(config=config)
```

---

## 最佳实践

1. **需求具体化**：需求写清楚项目背景、技术栈、核心能力，LLM 选择更精准
2. **分阶段选择**：先选 3~5 个核心角色，后续按需补充
3. **合理数量**：建议 3~10 个，过多会稀释匹配精度
4. **推荐模型**：Claude Opus/Sonnet、GPT-4 等理解能力强的模型
5. **Git 管理**：将 `requirements.md` 纳入版本控制，方便团队协作和追踪

---

## 相关文件

| 文件 | 说明 |
|---|---|
| `summerclaw/agent/role_selector.py` | 核心实现 |
| `summerclaw/config/schema.py` | 配置 Schema |
| `summerclaw/cli/commands.py` | CLI 命令 |
| `resources/roles/` | 可用角色库（1016个） |
| `config.role_selector.example.json` | 示例配置 |