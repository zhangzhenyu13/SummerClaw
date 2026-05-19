"""Role selector —— 按配置从 resources/roles 中选取角色并复制到工作目录。

工作流程：
  1. 检查配置中的 role_selector 是否启用
  2. 如果启用，扫描 resources/roles 下所有角色文件
  3. 调用 LLM 根据需求文档选择指定数量的角色
  4. 将选中的角色复制到 workspace/roles/selected/
  5. 如果 selected 目录已存在且有内容，则跳过
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from summerclaw.config.loader import load_config
from summerclaw.config.paths import get_workspace_path
from summerclaw.config.schema import Config

# ── 常量 ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCES_ROLES = PROJECT_ROOT / "resources" / "roles"

DEFAULT_MODEL_OVERRIDE: str | None = None
REQUEST_TIMEOUT = 120  # 秒
MAX_RETRIES = 3

# 默认需求文件内容
DEFAULT_REQUIREMENTS_CONTENT = """# 角色选择需求文档

请根据以下需求选择合适的角色：

需要数据分析师、软件工程师、UI设计师、金融分析师、LLM大模型工程师
"""


# ── Provider 工厂（复用 generate_roles.py 的逻辑） ──────────────────
def _make_provider(config: Config, model_override: str | None = None):
    """创建 LLM provider 实例。"""
    from summerclaw.providers.base import GenerationSettings
    from summerclaw.providers.registry import find_by_name

    model = model_override or config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    if backend == "openai_codex":
        from summerclaw.providers.openai_codex_provider import OpenAICodexProvider
        provider = OpenAICodexProvider(default_model=model)
    elif backend == "github_copilot":
        from summerclaw.providers.github_copilot_provider import GitHubCopilotProvider
        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "azure_openai":
        from summerclaw.providers.azure_openai_provider import AzureOpenAIProvider
        provider = AzureOpenAIProvider(
            api_key=p.api_key, api_base=p.api_base, default_model=model,
        )
    elif backend == "anthropic":
        from summerclaw.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from summerclaw.providers.openai_compat_provider import OpenAICompatProvider
        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=None,
    )
    return provider, model


# ── 扫描角色 ────────────────────────────────────────────────────────
def scan_available_roles(roles_dir: Path = RESOURCES_ROLES) -> list[dict[str, str]]:
    """扫描 resources/roles 下所有角色文件，返回 [(category, role_name, file_path), ...]。"""
    roles = []
    if not roles_dir.exists():
        logger.warning(f"Roles directory not found: {roles_dir}")
        return roles

    for category_dir in sorted(roles_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category_name = category_dir.name.replace("_", " ").title()
        for role_file in sorted(category_dir.glob("*.md")):
            role_name = role_file.stem
            roles.append({
                "category": category_name,
                "role_name": role_name,
                "file_path": str(role_file),
            })

    logger.info(f"Scanned {len(roles)} roles from {roles_dir}")
    return roles


# ── 检查是否需要选择 ────────────────────────────────────────────────
def should_select_roles(workspace_path: Path) -> tuple[bool, Path]:
    """检查工作目录下是否已有选中的角色。
    
    Returns:
        (need_select, selected_dir): need_select 为 True 表示需要选择；
        selected_dir 为 selected 目录路径。
    """
    selected_dir = workspace_path / "roles" / "selected"
    if not selected_dir.exists():
        return True, selected_dir

    # 检查是否有 .md 文件
    md_files = list(selected_dir.glob("*.md"))
    if len(md_files) == 0:
        return True, selected_dir

    logger.info(
        f"角色已存在: {selected_dir} 中有 {len(md_files)} 个已选角色，跳过选择。"
        f"如需重新选择，请删除 {selected_dir} 后重试。"
    )
    return False, selected_dir


# ── 确保需求文件存在 ─────────────────────────────────────────────────
def ensure_requirements_file(workspace_path: Path, requirements_path: str) -> Path:
    """确保需求文件存在，如果不存在则创建默认内容。
    
    Args:
        workspace_path: 工作目录路径
        requirements_path: 需求文件路径（相对于workspace）
        
    Returns:
        需求文件的绝对路径
    """
    # 如果是相对路径，转换为绝对路径
    req_path = Path(requirements_path)
    if not req_path.is_absolute():
        req_path = workspace_path / req_path
    
    # 如果文件不存在，创建默认内容
    if not req_path.exists():
        logger.info(f"Requirements file not found, creating default: {req_path}")
        req_path.parent.mkdir(parents=True, exist_ok=True)
        req_path.write_text(DEFAULT_REQUIREMENTS_CONTENT, encoding="utf-8")
        logger.info(f"Created default requirements file: {req_path}")
    else:
        logger.info(f"Requirements file exists: {req_path}")
    
    return req_path


# ── 读取需求文档 ─────────────────────────────────────────────────────
def load_requirements(workspace_path: Path, requirements_path: str) -> str:
    """读取需求文件内容。
    
    Args:
        workspace_path: 工作目录路径
        requirements_path: 需求文件路径（相对于workspace）
        
    Returns:
        需求文档内容
    """
    req_path = ensure_requirements_file(workspace_path, requirements_path)
    
    try:
        content = req_path.read_text(encoding="utf-8")
        logger.info(f"Loaded requirements from {req_path} ({len(content)} chars)")
        return content
    except Exception as e:
        logger.error(f"Failed to read requirements file {req_path}: {e}")
        raise


# ── 构建 LLM 提示词 ─────────────────────────────────────────────────
def build_selection_prompt(
    available_roles: list[dict[str, str]],
    requirements: str,
    count: int,
) -> str:
    """构建角色选择的 user prompt。"""
    roles_summary = "\n".join([
        f"- [{r['category']}] {r['role_name']}"
        for r in available_roles
    ])

    return f"""请从以下可用角色列表中，根据需求文档选择 {count} 个最合适的角色。

## 可用角色列表（共 {len(available_roles)} 个）

{roles_summary}

## 需求文档

{requirements}

## 输出要求

请以 JSON 数组格式返回选中的角色名称（仅返回角色名，不包含类别），例如：

```json
["数据分析师", "产品经理", "UI设计师"]
```

请确保：
1. 恰好返回 {count} 个角色
2. 角色名称必须与上面列表中的完全一致
3. 角色应该最能满足需求文档中描述的场景
4. 只返回 JSON，不要其他解释
"""


# ── 调用 LLM 选择角色 ───────────────────────────────────────────────
async def select_roles_with_llm(
    provider,
    model: str,
    available_roles: list[dict[str, str]],
    requirements: str,
    count: int,
) -> list[str]:
    """调用 LLM 选择角色，返回选中的角色名称列表。"""
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的角色选择助手，负责根据需求文档从可用角色列表中选择最合适的角色。",
        },
        {
            "role": "user",
            "content": build_selection_prompt(available_roles, requirements, count),
        },
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            from summerclaw.providers.base import LLMResponse

            response: LLMResponse = await asyncio.wait_for(
                provider.chat_with_retry(
                    messages=messages,
                    model=model,
                    retry_mode="standard",
                ),
                timeout=REQUEST_TIMEOUT,
            )

            if response.finish_reason == "error":
                logger.warning(f"LLM error (attempt {attempt}): {response.content[:200]}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)
                    continue
                return []

            # 解析 JSON 响应
            content = response.content or ""
            # 尝试提取 JSON 数组
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                json_str = json_match.group(0)
                selected_roles = json.loads(json_str)
                if isinstance(selected_roles, list) and len(selected_roles) == count:
                    logger.info(f"LLM selected {len(selected_roles)} roles: {selected_roles}")
                    return selected_roles
                else:
                    logger.warning(f"Invalid role count: expected {count}, got {len(selected_roles)}")
            else:
                logger.warning(f"No JSON array found in LLM response")

            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 * attempt)
                continue

            return []

        except asyncio.TimeoutError as e:
            logger.warning(f"LLM timeout (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 * attempt)
                continue
            return []
        except Exception as e:
            logger.warning(f"LLM exception (attempt {attempt}): {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 * attempt)
                continue
            return []

    return []


# ── 复制角色文件 ────────────────────────────────────────────────────
def copy_selected_roles(
    selected_roles: list[str],
    available_roles: list[dict[str, str]],
    workspace_path: Path,
) -> int:
    """将选中的角色文件复制到工作目录。"""
    selected_dir = workspace_path / "roles" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)

    # 构建角色名到文件路径的映射
    role_file_map = {r["role_name"]: r["file_path"] for r in available_roles}

    copied_count = 0
    for role_name in selected_roles:
        if role_name not in role_file_map:
            logger.warning(f"Role not found: {role_name}")
            continue

        src_path = Path(role_file_map[role_name])
        dst_path = selected_dir / f"{role_name}.md"

        if dst_path.exists():
            logger.info(f"Role already exists: {role_name}, skipping")
            continue

        shutil.copy2(src_path, dst_path)
        logger.info(f"Copied role: {role_name}")
        copied_count += 1

    logger.info(f"Copied {copied_count} roles to {selected_dir}")
    return copied_count


# ── 主流程 ──────────────────────────────────────────────────────────
async def select_roles(
    config: Config | None = None,
    workspace_path: Path | None = None,
    model_override: str | None = None,
) -> bool:
    """执行角色选择流程。

    Args:
        config: 配置对象，如果为 None 则从文件加载
        workspace_path: 工作目录路径，如果为 None 则从配置获取
        model_override: 覆盖配置中的模型名

    Returns:
        bool: 是否成功执行了角色选择
    """
    # 加载配置
    if config is None:
        config = load_config()

    # 获取工作目录
    if workspace_path is None:
        workspace_path = get_workspace_path(config.workspace_path)

    # 检查角色选择是否启用
    role_selector_config = config.agents.defaults.role_selector
    if not role_selector_config.enabled:
        logger.info("Role selector is not enabled, skipping")
        return False

    # 检查是否已经选择过角色
    need_select, selected_dir = should_select_roles(workspace_path)
    if not need_select:
        return False

    # 获取配置参数
    requirements_path = getattr(role_selector_config, "requirements", "roles/requirements.md")
    count = getattr(role_selector_config, "count", 5)

    # 从文件加载需求文档
    try:
        requirements = load_requirements(workspace_path, requirements_path)
    except Exception as e:
        logger.error(f"Failed to load requirements: {e}")
        return False

    logger.info(f"Role selector enabled: selecting {count} roles based on requirements")

    # 扫描可用角色
    available_roles = scan_available_roles()
    if not available_roles:
        logger.warning("No available roles found")
        return False

    # 创建 provider
    provider, model = _make_provider(config, model_override)
    logger.info(f"Using model: {model}")

    # 调用 LLM 选择角色
    selected_roles = await select_roles_with_llm(
        provider, model, available_roles, requirements, count
    )

    if not selected_roles:
        logger.warning("Failed to select roles with LLM")
        return False

    # 复制角色文件
    copied_count = copy_selected_roles(selected_roles, available_roles, workspace_path)
    logger.info(f"Role selection completed: {copied_count} roles copied to workspace")

    return True


# ── 同步入口（用于非异步上下文） ─────────────────────────────────────
def select_roles_sync(
    config: Config | None = None,
    workspace_path: Path | None = None,
    model_override: str | None = None,
) -> bool:
    """同步版本的 role selector 入口。"""
    return asyncio.run(select_roles(config, workspace_path, model_override))


# ── CLI 入口 ────────────────────────────────────────────────────────
def main():
    """CLI 入口，用于手动触发角色选择。"""
    import argparse

    parser = argparse.ArgumentParser(description="Select roles from resources/roles based on requirements")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("-w", "--workspace", type=str, default=None, help="Workspace directory")
    parser.add_argument("-m", "--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()

    config = None
    workspace = None

    if args.config:
        from summerclaw.config.loader import load_config, set_config_path
        config_path = Path(args.config).expanduser().resolve()
        set_config_path(config_path)
        config = load_config(config_path)

    if args.workspace:
        workspace = Path(args.workspace).expanduser().resolve()

    success = select_roles_sync(config, workspace, args.model)
    if success:
        print("✓ Role selection completed successfully")
    else:
        print("✗ Role selection skipped or failed")


if __name__ == "__main__":
    main()
