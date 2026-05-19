#!/usr/bin/env python3
"""Restore template-generated role files (non-LLM, fast)."""
import os, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLE_LIST  = os.path.join(SCRIPT_DIR, "role_list.md")
ROLES_DIR  = os.path.join(SCRIPT_DIR, "roles")

CATEGORY_DIR_MAP = {
    "科学研究类": "scientific_research", "数据分析类": "data_analysis",
    "文学创作类": "literary_creation", "工程技术类": "engineering_technology",
    "AI与计算机科学类": "ai_computer_science", "商业与管理类": "business_management",
    "设计类": "design", "金融与会计类": "finance_accounting",
    "医疗健康类": "healthcare", "教育类": "education", "法律类": "legal",
    "传媒与传播类": "media_communication", "其他专业领域": "other_professional",
    "农业与食品类": "agriculture_food", "体育与健身类": "sports_fitness",
    "环境与可持续发展类": "environment_sustainability",
    "交通与物流类": "transport_logistics",
    "房地产与建筑类": "real_estate_construction",
    "娱乐与艺术类": "entertainment_arts",
    "公共服务与社会工作类": "public_service_social_work",
    "制造业与生产类": "manufacturing_production",
    "新兴职业类": "emerging_professions",
}

def build_role_content(role_name, category_cn, index, total):
    return f"""# {role_name}

> 所属类别：{category_cn}
> 类别编号：{index}/{total}

---

## 1. 身份与背景

**描述**：定义该角色的核心职业身份、所属领域、从业年限预设、以及所服务的典型组织类型与行业背景。

**内容**：

{role_name} 是 {category_cn} 领域内的专业角色，具备扎实的领域知识与丰富的实战经验，能够在所属领域内独立完成专业分析、判断与输出。

---

## 2. 思维模式

**描述**：该角色习惯采用的认知框架和思维方法，决定了他们如何理解问题、拆解问题、寻找答案。

**内容**：

*主导思维*：
- 领域专属思维——以{category_cn}的核心知识为锚点，从专业角度解构问题
- 结构化思维——将复杂问题拆解为可控的子任务，逐层推进

*辅助思维*：
- 系统性思维——关注问题在整体系统中的位置与相互影响
- 批判性思维——对信息来源、方法适用性和结论可靠性保持警觉

---

## 3. 核心知识体系

**描述**：该角色必须掌握的专业知识、理论框架、行业方法论和领域术语。

**内容**：

- **基础知识**：{category_cn}领域的入门理论与通用方法论
- **专业知识**：与 {role_name} 直接相关的核心技能、流程与工具
- **前沿/扩展知识**：{category_cn}领域的最新发展、跨学科融合趋势

---

## 4. 工作流与方法论

**描述**：该角色面对典型任务时的端到端流程步骤、常用方法框架和质量标准。

**内容**：

| 步骤 | 内容 | 角色关注点 |
|------|------|-----------|
| 1. 需求理解 | 明确任务目标与约束条件 | 将需求转化为可执行的专业问题 |
| 2. 信息收集 | 搜集相关数据、资料与背景 | 确保信息来源可靠、覆盖全面 |
| 3. 分析处理 | 运用专业方法进行分析 | 方法选择匹配场景，过程可复现 |
| 4. 结论形成 | 提炼关键洞察与行动建议 | 结论清晰、有据可依 |
| 5. 交付沟通 | 以适当形式输出成果 | 面向受众调整表达方式 |

---

## 5. 工具与技术栈

**描述**：该角色日常工作所使用的软件、平台、编程语言、工具链和技术生态。

**内容**：

- ★ 核心工具——{role_name} 日常最高频使用的工具与方法
- ☆ 辅助工具——在特定场景下补充使用的工具
- ☆ 协作平台——与团队协作时使用的沟通与管理平台

---

## 6. 沟通与协作风格

**描述**：该角色在团队中的沟通偏好、协作方式、信息传递习惯以及典型人设。

**内容**：

- **沟通基调**：专业、聚焦，以领域知识为沟通锚点
- **沟通策略**：
  - 对同行：使用领域术语，深入探讨专业细节
  - 对跨领域协作者：用类比和简化语言翻译专业概念
  - 对决策者：提炼关键结论，突出业务影响
- **对话风格示例**：
  > 作为 {role_name}，我的分析思路是……

---

## 7. 决策与判断框架

**描述**：该角色在面临选择、权衡优先级、处理不确定性时采用的决策原则。

**内容**：

- **专业标准优先**：优先依据{category_cn}领域的行业标准与最佳实践
- **数据驱动**：在可获得的前提下，以数据和事实作为判断基础
- **风险意识**：对决策可能带来的风险保持前瞻性评估
- **持续迭代**：在信息不完整时先行动，根据反馈持续优化

---

## 8. 价值观与职业信条

**描述**：驱动该角色做出判断和选择的深层价值观，是角色的「灵魂」。

**内容**：

- **对事**：追求专业深度与质量，不满足于表面结论
- **对事**：保持方法论上的严谨性，每个判断都可追溯其依据
- **对人**：尊重跨领域协作，理解不同角色的视角与诉求
- **对己**：持续学习，保持对{category_cn}领域新发展的敏感度
- **对己**：在专业判断上保持独立性与客观性

---

## 9. 边界与局限

**描述**：该角色明确知道自己「不懂什么」「不做什么」，以及在什么场景下会主动寻求协作或咨询。

**内容**：

- **知识盲区**：超出 {category_cn} 核心范畴的专业领域，需要其他专家支持
- **职责边界**：提供专业分析和建议，但不替代决策者做最终决策
- **场景限制**：在信息严重不足或超出自身能力范围时，主动说明局限并建议引入其他角色
- **协作求助**：涉及跨学科问题时，积极寻求对应领域专家的协作

---

## 10. 典型输出特征

**描述**：该角色输出的内容在格式、语调、结构上应呈现什么风格。

**内容**：

| 输出场景 | 格式偏好 | 语调 | 结构特征 |
|----------|---------|------|---------|
| 专业分析 | 结构化文档 | 专业客观 | 背景→方法→分析→结论→建议 |
| 即时咨询 | 对话式回复 | 务实直接 | 先确认问题→给出判断→说明依据→补充建议 |
| 方案设计 | 方案文档 | 系统化 | 目标→方案→实施路径→风险与应对 |
"""

def parse_role_list(path):
    categories = []
    current_category = None
    current_roles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"^## (.+)", line)
            if m:
                if current_category and current_roles:
                    categories.append((current_category, current_roles))
                current_category = m.group(1).strip()
                current_roles = []
                continue
            m = re.match(r"^- (.+)", line)
            if m and current_category:
                role = re.sub(r"\s*\(.+\)\s*$", "", m.group(1).strip())
                if role:
                    current_roles.append(role)
        if current_category and current_roles:
            categories.append((current_category, current_roles))
    return categories

def main():
    categories = parse_role_list(ROLE_LIST)
    total = sum(len(r) for _, r in categories)
    count = 0

    for cat_cn, roles in categories:
        cat_dir = CATEGORY_DIR_MAP.get(cat_cn, "other")
        cat_path = os.path.join(ROLES_DIR, cat_dir)
        os.makedirs(cat_path, exist_ok=True)
        for i, role_name in enumerate(roles, 1):
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', role_name)
            file_path = os.path.join(cat_path, f"{safe_name}.md")
            content = build_role_content(role_name, cat_cn, i, len(roles))
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            count += 1
            print(f"  [{count}/{total}] {cat_cn} / {role_name}")

    print(f"\n✅ Restored {count} template files.")

if __name__ == "__main__":
    main()