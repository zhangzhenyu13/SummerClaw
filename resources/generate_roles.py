#!/usr/bin/env python3
"""批量生成角色定义文件 —— 使用项目 Provider 请求 LLM 完成。

从 role_list.md 解析分类与角色，对每个角色调用配置的 LLM，按照
build_role.md 中定义的 10 维度模板生成完整角色定义。

特性：
  - 受控并发（asyncio.Semaphore，默认 5 并发）
  - 断点续传（跳过 Content-Length ≥ MIN_FILE_BYTES 的文件）
  - 内置重试（复用 provider 的 chat_with_retry）
  - 超时保护 & 异常隔离
  - 实时进度 & 速率统计
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

# ── 确保项目根在 sys.path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from summerclaw.config.loader import load_config
from summerclaw.providers.base import GenerationSettings, LLMResponse
from summerclaw.providers.registry import find_by_name

# ── 常量 ────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROLE_LIST   = SCRIPT_DIR / "role_list.md"
BUILD_ROLE  = SCRIPT_DIR / "build_role.md"
ROLES_DIR   = SCRIPT_DIR / "roles"

DEFAULT_CONCURRENCY = 5
MIN_FILE_BYTES      = 600          # 小于此值的文件视为未完成，重新生成
REQUEST_TIMEOUT     = 120          # 单次 LLM 请求超时（秒）
MAX_RETRIES         = 3

CATEGORY_DIR_MAP = {
    "科学研究类":            "scientific_research",
    "数据分析类":            "data_analysis",
    "文学创作类":            "literary_creation",
    "工程技术类":            "engineering_technology",
    "AI与计算机科学类":      "ai_computer_science",
    "商业与管理类":          "business_management",
    "设计类":                "design",
    "金融与会计类":          "finance_accounting",
    "医疗健康类":            "healthcare",
    "教育类":                "education",
    "法律类":                "legal",
    "传媒与传播类":          "media_communication",
    "其他专业领域":          "other_professional",
    "农业与食品类":          "agriculture_food",
    "体育与健身类":          "sports_fitness",
    "环境与可持续发展类":    "environment_sustainability",
    "交通与物流类":          "transport_logistics",
    "房地产与建筑类":        "real_estate_construction",
    "娱乐与艺术类":          "entertainment_arts",
    "公共服务与社会工作类":  "public_service_social_work",
    "制造业与生产类":        "manufacturing_production",
    "新兴职业类":            "emerging_professions",
}

# ── Prompt 模板 ─────────────────────────────────────────────────────
def _load_template() -> str:
    """将 build_role.md 正文转为 LLM 的 system prompt 模板。"""
    raw = BUILD_ROLE.read_text(encoding="utf-8")
    # 去掉标题行 "# Role 构建 Prompt"
    raw = re.sub(r"^#\s+.*\n+", "", raw, count=1)
    # 去掉配置分隔符 `---`
    raw = re.sub(r"\n---\n", "\n", raw)
    return raw.strip()


SYSTEM_TEMPLATE = _load_template()

USER_MESSAGE_TEMPLATE = """请为以下角色生成完整的角色定义（严格按 10 个维度输出）：

角色名称：{role_name}
所属类别：{category_cn}

请直接输出 Markdown 格式的完整角色定义，以 "# {role_name}" 开头。"""


# ── 辅助工具 ────────────────────────────────────────────────────────
def _parsed_or_none(model: str, content: str | None) -> str | None:
    """简单检查 LLM 返回的内容是否看起来像合法的角色定义。"""
    if not content or len(content) < MIN_FILE_BYTES:
        return None
    return content


def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', name)


# ── 解析 role_list.md ───────────────────────────────────────────────
def parse_role_list(path: str) -> list[tuple[str, list[str]]]:
    categories: list[tuple[str, list[str]]] = []
    current_category: str | None = None
    current_roles: list[str] = []
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


# ── Provider 工厂（复刻 summerclaw._make_provider） ──────────────────
def _make_provider(config: object, model_override: str | None = None):
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
        reasoning_effort=None,  # 角色生成不需要深度推理
    )
    return provider, model


# ── 单个角色生成 ────────────────────────────────────────────────────
async def generate_one(
    sem: asyncio.Semaphore,
    provider,
    model: str,
    role_name: str,
    category_cn: str,
) -> str | None:
    """为单个角色调用 LLM 并返回 Markdown 内容；失败返回 None。"""
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": USER_MESSAGE_TEMPLATE.format(
            role_name=role_name, category_cn=category_cn,
        )},
    ]

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response: LLMResponse = await asyncio.wait_for(
                    provider.chat_with_retry(
                        messages=messages,
                        model=model,
                        retry_mode="standard",
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
            except asyncio.TimeoutError as e:
                print(f"\n  ⚠ [{role_name}] 超时 (attempt {attempt}): {e}", flush=True)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)
                    continue
                return None
            except Exception as e:
                print(f"\n  ⚠ [{role_name}] 异常 (attempt {attempt}): {type(e).__name__}: {e}", flush=True)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)
                    continue
                return None

            if response.finish_reason == "error":
                print(f"\n  ⚠ [{role_name}] LLM错误 (attempt {attempt}):"
                      f" {response.content[:200] if response.content else 'n/a'}", flush=True)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)
                    continue
                return None

            return _parsed_or_none(model, response.content)

    return None


# ── 进度追踪 ────────────────────────────────────────────────────────
class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.skipped = 0
        self.failed = 0
        self.start = time.monotonic()
        self.lock = asyncio.Lock()

    async def add(self, kind: str) -> None:
        async with self.lock:
            if kind == "done":
                self.done += 1
            elif kind == "skip":
                self.skipped += 1
            elif kind == "fail":
                self.failed += 1
            self._print()

    def _print(self) -> None:
        elapsed = max(time.monotonic() - self.start, 1)
        rate = (self.done + self.failed) / elapsed
        print(
            f"\r  [{self.done + self.skipped + self.failed}/{self.total}]"
            f" ✅{self.done} ⏭{self.skipped} ❌{self.failed}"
            f"  {rate:.1f}/s"
            f"  {_eta(self.total - self.done - self.skipped - self.failed, rate)}   ",
            end="", flush=True,
        )


def _eta(remaining: int, rate: float) -> str:
    if rate <= 0:
        return "ETA: --"
    s = int(remaining / rate)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"ETA: {h}h{m:02d}m{s:02d}s"


# ── 主流程 ──────────────────────────────────────────────────────────
async def amain(concurrency: int, model_override: str | None = None):
    categories = parse_role_list(str(ROLE_LIST))
    total = sum(len(r) for _, r in categories)
    print(f"📋 {len(categories)} 类别 · {total} 角色\n")

    # 加载配置 & 创建 provider
    config = load_config()
    provider, model = _make_provider(config, model_override=model_override)
    print(f"🤖 模型: {model}  |  并发: {concurrency}  |  超时: {REQUEST_TIMEOUT}s\n")

    ROLES_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    prog = Progress(total)

    index_lines: list[str] = ["# 角色索引\n\n"]

    async def process_one(role_name: str, category_cn: str, cat_dir: str):
        """处理单个角色：检查→跳过 或 调用LLM。失败不覆盖已有文件。"""
        safe_name = _sanitize_filename(role_name)
        cat_path = ROLES_DIR / cat_dir
        cat_path.mkdir(parents=True, exist_ok=True)
        out = cat_path / f"{safe_name}.md"

        # 断点续传：已有足够内容的跳过
        if out.exists() and out.stat().st_size >= MIN_FILE_BYTES:
            await prog.add("skip")
            return

        # 已有内容但不足 → 视作待重试的失败文件，保留原内容不动
        content = await generate_one(sem, provider, model, role_name, category_cn)
        if content and len(content) >= MIN_FILE_BYTES:
            out.write_text(content, encoding="utf-8")
            await prog.add("done")
        else:
            # 仅当文件不存在时才写入占位标记；已有文件保留不动
            if not out.exists():
                out.write_text(
                    f"# {role_name}\n\n> ⚠ LLM 生成失败，待重试。\n",
                    encoding="utf-8",
                )
            await prog.add("fail")

    # 逐类别逐角色处理：用信号量控制并发但保证处理完整
    for cat_cn, roles in categories:
        cat_dir = CATEGORY_DIR_MAP.get(cat_cn, "other")
        index_lines.append(f"## {cat_cn} ({cat_dir})\n")

        # 创建异步任务表 → 信号量自动排队
        tasks = [asyncio.ensure_future(process_one(r, cat_cn, cat_dir)) for r in roles]
        await asyncio.gather(*tasks)

        for r in roles:
            safe = _sanitize_filename(r)
            index_lines.append(f"- [{r}]({cat_dir}/{safe}.md)")
        index_lines.append("")

    # 写入索引
    (ROLES_DIR / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    elapsed = time.monotonic() - prog.start
    print(f"\n\n✅ 完成  done={prog.done}  skip={prog.skipped}  fail={prog.failed}"
          f"  ⏱ {elapsed:.0f}s\n")


# ── CLI 入口 ────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量调用 LLM 生成角色定义")
    parser.add_argument("-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"并发数（默认 {DEFAULT_CONCURRENCY}）")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="覆盖配置中的模型名")
    args = parser.parse_args()
    asyncio.run(amain(args.concurrency, args.model))


if __name__ == "__main__":
    main()