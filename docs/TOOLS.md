# Tools in SummerClaw

SummerClaw ships with a rich set of built-in tools that the agent can invoke via function calling. Tools are managed by a unified `ToolRegistry` — each tool implements the `Tool` abstract base class, providing JSON Schema parameters, automatic type casting, and validation.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       ToolRegistry                              │
│  register(Tool) · get(name) · execute(name, params)            │
│  get_definitions() → OpenAI function schemas (sorted + cached)  │
└───────────────────────────┬────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  Built-in    │  │  Conditional │  │  MCP Dynamic │
   │  (always on) │  │  (config)    │  │  (server)    │
   └──────────────┘  └──────────────┘  └──────────────┘
```

Each tool provides:
- `name` / `description` — exposed to the LLM via function-calling
- `parameters` — JSON Schema for type-safe parameter validation
- `read_only` — marks side-effect-free tools safe for parallel execution
- `exclusive` — marks tools that must run alone (serialized)

---

## Built-in Tools (Always Available)

### Filesystem

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line-based pagination. Supports text and images (JPEG/PNG/WebP). Output format: `LINE_NUM|CONTENT`. Use `offset`/`limit` for large files. Reads exceeding ~128K chars are truncated. |
| `write_file` | Write content to a file. Creates parent directories as needed. Overwrites if the file already exists. For partial edits, prefer `edit_file`. |
| `edit_file` | Edit a file by replacing `old_text` with `new_text`. Tolerates minor whitespace/indentation differences and curly/straight quote mismatches. If `old_text` matches multiple times, provide more context or set `replace_all=true`. Shows a diff of the closest match on failure. |
| `list_dir` | List directory contents. Set `recursive=true` for nested exploration. Common noise directories (`.git`, `node_modules`, `__pycache__`, etc.) are auto-ignored. |

### Search

| Tool | Description |
|------|-------------|
| `glob` | Find files matching a glob pattern (e.g. `*.py`, `tests/**/test_*.py`). Results sorted by modification time (newest first). Skips noise directories. Use `head_limit`/`offset` for pagination. |
| `grep` | Search file contents with a regex pattern. Default `output_mode` is `files_with_matches` (file paths only); use `content` mode for matching lines with context. Skips binary and files >2 MB. Supports `glob`/`type` filtering. |

### Notebook

| Tool | Description |
|------|-------------|
| `notebook_edit` | Edit a Jupyter notebook (`.ipynb`) cell. Modes: `replace` (default), `insert` (after target index), `delete`. `cell_index` is 0-based. `cell_type`: `code` or `markdown`. |

### Interaction

| Tool | Description |
|------|-------------|
| `message` | Send a message to the user, optionally with file attachments via the `media` parameter. This is the ONLY way to deliver files (images, documents, audio, video) to the user. Do NOT use `read_file` to send files — that only reads content for agent analysis. |
| `ask_user` | Pause execution and ask the user a question. The agent waits for the user's reply before continuing. Use when clarification, confirmation, or a choice is needed. Provide `candidates` for option lists. The user's answer appears as a user message in the next turn. |
| `spawn` | Spawn a subagent to handle a task in the background. Use for complex or time-consuming tasks that can run independently. The subagent reports back when done. Inspect the workspace first and use a dedicated subdirectory when helpful. |

---

## Conditional Tools (Config-Controlled)

These tools are gated by configuration keys. They are only registered when the corresponding `enable` flag is `true`.

### Shell Execution

| Config Key | Default | Description |
|------------|---------|-------------|
| `tools.exec.enable` | `true` | Enable `exec` tool |

| Tool | Description |
|------|-------------|
| `exec` | Execute a shell command and return its output. Prefer `read_file`/`write_file`/`edit_file` over `cat`/`echo`/`sed`, and `grep`/`glob` over shell `find`/`grep`. Use `-y` or `--yes` flags for non-interactive mode. Output truncated at 10 000 chars; timeout defaults to 60s. |

**Safety:** Dangerous commands are blocked by default (`rm -rf`, `format`, `dd`, `shutdown`, fork bombs, writes to `history.jsonl`/`.dream_cursor`, etc.). Optional `bwrap` sandbox available via `exec.sandbox = "bwrap"` in Docker environments.

**Configuration:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tools.exec.timeout` | int | `60` | Per-command timeout in seconds |
| `tools.exec.sandbox` | string | `""` | Sandbox backend: `""` (none) or `"bwrap"` |
| `tools.exec.restrict_to_workspace` | bool | `false` | Limit file access to workspace directory |

### Runtime Self-Inspection

| Config Key | Default | Description |
|------------|---------|-------------|
| `tools.my.enable` | `true` | Enable `my` tool (read-only) |
| `tools.my.allow_set` | `false` | Allow the agent to modify its own configuration |

| Tool | Description |
|------|-------------|
| `my` | Check and set the agent's own runtime state. Actions: `check`, `set`. Use `check` (no key) for full config overview. Use `check(key)` with dot-paths (e.g. `_last_usage.prompt_tokens`, `web_config.enable`). Use `set(key, value)` to change config or store notes in a scratchpad that persists across turns (but not restarts). Blocked: sensitive keys (`api_key`, `secret`, etc.) and core infrastructure. |

> See also [My Tool](MY_TOOL.md) for detailed usage guide.

### Web Search & Fetch

| Config Key | Default | Description |
|------------|---------|-------------|
| `tools.web.enable` | `false` | Enable web tools (`web_search`, `web_fetch`) |

| Tool | Description |
|------|-------------|
| `web_search` | Search the web. Returns titles, URLs, and snippets. `count` defaults to 5 (max 10). Supports multiple backends: DuckDuckGo (no API key), Brave, Tavily, SearXNG, Jina, Kagi. Falls back to DuckDuckGo when API keys are missing. |
| `web_fetch` | Fetch a URL and extract readable content (HTML → markdown/text). Output capped at `maxChars` (default 50 000). Works for most web pages and docs; may fail on login-walled or JS-heavy sites. |

**Configuration:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tools.web.search.provider` | string | `"duckduckgo"` | Search backend: `duckduckgo` \| `brave` \| `tavily` \| `searxng` \| `jina` \| `kagi` |
| `tools.web.search.apiKey` | string | `""` | API key for providers that require one |
| `tools.web.search.baseUrl` | string | `""` | Base URL for SearXNG instances |
| `tools.web.search.maxResults` | int | `5` | Hard cap on results returned |
| `tools.web.search.timeout` | int | `30` | Per-request timeout in seconds |
| `tools.web.proxy` | string | `""` | HTTP proxy for web requests |

### Browser Automation (Playwright)

| Config Key | Default | Description |
|------------|---------|-------------|
| `tools.browser.enable` | `false` | Enable browser tools (`browser_search`, `browser_fetch`) |
| `tools.browser.enable_control` | `false` | Enable interactive browser control (`browser_navigate`, `browser_snapshot`, `browser_execute_js`) |

| Tool | Description |
|------|-------------|
| `browser_search` | Search the web using a headless Chromium browser (Playwright). Supports Baidu (default), Bing, and DuckDuckGo — no API key required. Prefer this over `web_search` when API keys are unavailable or content is geo-restricted. |
| `browser_fetch` | Fetch a URL using a headless browser (crawl4ai preferred, Playwright fallback). Handles JavaScript-heavy and dynamically rendered pages that `web_fetch` cannot. Output capped at `maxChars` (default 50 000). |
| `browser_navigate` | Control a persistent headless browser: navigate to URLs (`go`), list open tabs (`tabs`), switch the active tab (`switch`), or open a new tab (`new_tab`). The browser session persists across calls. |
| `browser_snapshot` | Capture the accessibility tree of the current browser page. Provides a structured, compact view: headings, links, buttons, form fields, and text — far more token-efficient than raw HTML. |
| `browser_execute_js` | Execute JavaScript on the current browser page and return the result. Automatically wrapped in an async IIFE. Use to click elements, fill forms, scroll, extract data, etc. |

**Dependencies:** `pip install playwright && playwright install chromium`. For `browser_fetch`, `crawl4ai` is optional but recommended for better markdown extraction.

### Task Scheduling

| Config Key | Default | Description |
|------------|---------|-------------|
| `cron.enable` | `true` | Enable `cron` tool (requires cron service) |

| Tool | Description |
|------|-------------|
| `cron` | Schedule reminders and recurring tasks. Actions: `add`, `list`, `remove`. Supports `every_seconds` for simple intervals or `cron_expr` for complex schedules. If `tz` is omitted, defaults to the agent's configured timezone. |

**Configuration:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cron.enable` | bool | `true` | Enable the cron service and `cron` tool |

---

## MCP Tools (Dynamic)

SummerClaw supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing dynamic tool discovery from external servers. MCP tools are automatically wrapped and registered with the `mcp_<server>_` prefix.

| Wrapper Type | Naming Convention | Description |
|-------------|-------------------|-------------|
| `MCPToolWrapper` | `mcp_<server>_<tool>` | Wraps an MCP server tool with parameter schema normalization and timeout handling |
| `MCPResourceWrapper` | `mcp_<server>_resource_<name>` | Exposes MCP resource URIs as read-only tools |
| `MCPPromptWrapper` | `mcp_<server>_prompt_<name>` | Exposes MCP prompts as read-only workflow guide generators |

MCP tools are sorted after built-in tools in the function definitions list, keeping the prompt cache stable.

**Configuration** (via `mcpServers` in config):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    }
  }
}
```

---

## Tool Safety Properties

| Property | Meaning |
|----------|---------|
| `read_only = true` | Side-effect free; can run in parallel with other read-only tools |
| `exclusive = true` | Must run alone; serialized even when concurrency is enabled |
| `concurrency_safe` | `read_only and not exclusive` — can be batched |

### Tools by Safety Classification

| Read-Only | Exclusive | Write |
|-----------|-----------|-------|
| `read_file` | `exec` | `write_file` |
| `list_dir` | `ask_user` | `edit_file` |
| `glob` | `browser_search` | `notebook_edit` |
| `grep` | `browser_fetch` | `message` |
| `web_search` | `browser_navigate` | `spawn` |
| `web_fetch` | `browser_snapshot` | `cron` |
| `browser_snapshot` | `browser_execute_js` | `my` (when `allow_set=true`) |
| MCP resources (read-only) | `web_search` (DuckDuckGo only) | |

---

## Tool Registration Flow

```
AgentLoop.__init__()
  │
  ├─ _register_default_tools()
  │   ├─ ReadFileTool          (always)
  │   ├─ WriteFileTool         (always)
  │   ├─ EditFileTool          (always)
  │   ├─ ListDirTool           (always)
  │   ├─ GlobTool              (always)
  │   ├─ GrepTool              (always)
  │   ├─ NotebookEditTool      (always)
  │   ├─ MessageTool           (always)
  │   ├─ AskUserTool           (always)
  │   ├─ SpawnTool             (always)
  │   │
  │   ├─ ExecTool              (tools.exec.enable)
  │   ├─ WebSearchTool         (tools.web.enable)
  │   ├─ WebFetchTool          (tools.web.enable)
  │   ├─ BrowserSearchTool     (tools.browser.enable)
  │   ├─ BrowserFetchTool      (tools.browser.enable)
  │   ├─ BrowserNavigateTool   (tools.browser.enable_control)
  │   ├─ BrowserSnapshotTool   (tools.browser.enable_control)
  │   ├─ BrowserExecuteJSTool  (tools.browser.enable_control)
  │   └─ CronTool              (cron.enable)
  │
  ├─ MyTool                    (tools.my.enable)
  │
  └─ _connect_mcp()
      └─ MCP wrappers          (mcpServers config)
```
