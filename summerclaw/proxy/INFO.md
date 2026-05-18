# proxy — IP 代理池模块

## 概述

`proxy` 模块为 summerclaw 提供**IP 代理池**功能，通过维护一组可用的代理服务器，使 web/browser 工具在发起网络请求时自动轮换 IP 地址，从而规避目标站点的频率限制和 IP 封禁。模块包含代理收集、健康检查、延迟排序、磁盘缓存和自动补给的完整生命周期管理。

## 架构

```
┌───────────────┐     collect()      ┌─────────────────┐
│ ProxyCollector│ ─────────────────► │                 │
│ (公共源抓取)   │                    │   ProxyPool     │
└───────────────┘                    │                 │
                                     │  ┌───────────┐  │   get_proxy()
┌───────────────┐   add_proxies()    │  │ 健康检查   │  │ ──────────────► 工具层
│ 配置文件/手动  │ ────────────────► │  │ (定时循环) │  │    (web/browser)
│ initial_proxies│                    │  └───────────┘  │
└───────────────┘                    │                 │
                                     │  ┌───────────┐  │
┌───────────────┐   load/save        │  │ 磁盘缓存   │  │
│ ~/.summerclaw/   │ ◄────────────────►│  │ (JSON)     │  │
│ proxy_cache   │                    │  └───────────┘  │
└───────────────┘                    └─────────────────┘
```

- **ProxyCollector**：从多个公共免费代理源并发抓取代理列表，去重后返回。
- **ProxyPool**：管理代理生命周期——维护可用代理队列、定时健康检查、延迟排序择优、失败淘汰、磁盘缓存持久化、自动补给。
- **ProxyInfo**：单个代理的数据模型，追踪延迟、失败次数、最后检查时间等状态。

## 文件说明

| 文件 | 职责 |
|------|------|
| `__init__.py` | 模块入口，导出 `ProxyPool`、`ProxyCollector`、`ProxyInfo` |
| `models.py` | `ProxyInfo` 数据模型，代理状态追踪 |
| `collector.py` | `ProxyCollector`，从公共源抓取免费代理 |
| `pool.py` | `ProxyPool`，代理池核心 — 健康检查、延迟排序、磁盘缓存、自动收集 |

## 核心 API

### `ProxyInfo` (models.py)

单个代理服务器的数据模型。

| 字段 | 类型 | 说明 |
|------|------|------|
| `url` | `str` | 完整代理 URL，如 `http://1.2.3.4:8080` 或 `socks5://1.2.3.4:1080` |
| `protocol` | `str` | 协议类型：`http`、`https`、`socks4`、`socks5`（默认 `http`） |
| `latency_ms` | `float` | 最近一次测量的往返延迟（毫秒），0 表示未检查 |
| `fail_count` | `int` | 连续失败次数，成功时重置为 0 |
| `last_checked` | `float` | 最近一次健康检查的 Unix 时间戳 |
| `source` | `str` | 代理来源标识（如来源 URL 或 `"manual"`、`"cache"`） |

**方法：**

| 方法 | 说明 |
|------|------|
| `is_available(max_fail_count=3)` | 判断代理是否仍可用（失败次数未达阈值） |
| `mark_success(latency_ms)` | 记录成功：重置 fail_count，更新延迟和检查时间 |
| `mark_failure()` | 记录失败：fail_count +1，更新检查时间 |
| `to_url()` | 返回代理 URL 字符串 |
| `to_playwright_proxy()` | 返回 Playwright 兼容的代理字典 `{"server": url}` |

### `ProxyCollector` (collector.py)

从公共免费代理源并发抓取代理列表。

**构造参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `test_timeout` | `int` | `10` | 源抓取和代理测试超时（秒） |
| `check_url` | `str` | `"https://httpbin.org/ip"` | 验证代理可用性的 URL |
| `max_collect` | `int` | `50` | 单次收集最大返回代理数 |

**方法：**

| 方法 | 说明 |
|------|------|
| `collect()` | 并发抓取所有配置的公共源，去重后返回代理 URL 列表（如 `["1.2.3.4:8080", "socks5://5.6.7.8:1080"]`） |

**公共代理源（8 个）：**

- **ProxyScrape**（HTTP + SOCKS5）
- **TheSpeedX GitHub 列表**（HTTP + SOCKS4 + SOCKS5）
- **Proxy-List.download**（HTTP + HTTPS）
- **GeoNode API**（HTTP/SOCKS，JSON 格式，`limit=50`）

### `ProxyPool` (pool.py)

线程安全的异步代理池，提供健康检查、延迟择优和自动补给。

**构造参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `ProxyPoolConfig` | 代理池配置对象 |

**属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `available_count` | `int` | 当前可用代理数 |
| `total_count` | `int` | 池中代理总数（含暂时失败） |
| `is_fallback` | `bool` | 是否处于降级模式（池已启用但无可用代理，直接连接） |

**生命周期方法：**

| 方法 | 说明 |
|------|------|
| `start()` | 启动代理池：加载磁盘缓存、验证初始代理、启动后台健康检查和收集任务 |
| `stop()` | 停止代理池：取消后台任务、持久化当前代理到磁盘缓存 |

**代理获取方法：**

| 方法 | 说明 |
|------|------|
| `get_proxy()` | 返回下一个可用代理 URL（按延迟择优 + 随机选择），无可用时代返回 `None` 并进入 fallback 模式 |
| `get_playwright_proxy()` | 返回 Playwright 兼容的代理字典 `{"server": url}`，无可用时代返回 `None` |

**状态标记方法：**

| 方法 | 说明 |
|------|------|
| `mark_good(proxy_url, latency_ms)` | 标记代理请求成功，重置失败计数并更新延迟 |
| `mark_bad(proxy_url)` | 标记代理请求失败，失败次数达标后自动从池中移除 |

**管理方法：**

| 方法 | 说明 |
|------|------|
| `add_proxies(raw_urls, source)` | 解析、去重、验证并添加新代理，返回新增数量 |

## 配置 (`ProxyPoolConfig`)

配置路径：`config.proxy_pool`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | `bool` | `False` | 是否启用代理池 |
| `min_pool_size` | `int` | `5` | 最小可用代理数，低于此值触发自动收集（范围 1–100） |
| `max_pool_size` | `int` | `20` | 代理池最大容量（范围 5–200） |
| `health_check_interval` | `int` | `300` | 健康检查间隔（秒），设 0 禁用（范围 ≥30） |
| `health_check_url` | `str` | `"https://httpbin.org/ip"` | 代理可用性验证 URL |
| `proxy_test_timeout` | `int` | `10` | 代理验证请求超时（秒，范围 3–30） |
| `max_fail_count` | `int` | `3` | 连续失败次数阈值，超过后标记为失效（范围 1–10） |
| `collect_interval` | `int` | `600` | 池大小检查和自动收集间隔（秒，范围 ≥60） |
| `initial_proxies` | `list[str]` | `[]` | 启动时加载的代理 URL 列表 |
| `proxy_cache_enabled` | `bool` | `True` | 是否启用磁盘缓存持久化 |
| `proxy_cache_path` | `str` | `""` | 缓存文件路径（默认 `~/.summerclaw/proxy_cache.json`） |

**配置示例：**

```json
{
  "proxy_pool": {
    "enabled": true,
    "min_pool_size": 5,
    "max_pool_size": 30,
    "health_check_interval": 300,
    "health_check_url": "https://httpbin.org/ip",
    "proxy_test_timeout": 10,
    "max_fail_count": 3,
    "collect_interval": 600,
    "initial_proxies": ["http://1.2.3.4:8080"],
    "proxy_cache_enabled": true,
    "proxy_cache_path": ""
  }
}
```

## 数据流

### 启动流程

1. `ProxyPool.start()` 被调用
2. 若启用了磁盘缓存，从 `proxy_cache.json` 加载缓存代理，对过期条目重新健康检查
3. 加载配置中的 `initial_proxies`（去重后验证）
4. 保存初始状态到磁盘缓存
5. 启动后台**健康检查循环**（按 `health_check_interval` 周期执行）
6. 若池为空，立即执行一次**初始收集**；否则启动**定期收集循环**

### 代理获取流程

1. 工具层（`web.py` / `browser.py` / `browser_control.py`）调用 `pool.get_proxy()`
2. `ProxyPool` 从可用代理中按延迟排序，取前 50% 随机选择一个
3. 若无可用代理，进入 **fallback 模式**：返回 `None`，工具层使用直连方式请求
4. 后台健康检查和收集任务继续运行，一旦代理恢复就自动退出 fallback

### 健康检查流程

1. 后台任务按 `health_check_interval` 间隔触发
2. 遍历池中所有代理，通过 `_test_proxy()` 发起 HTTP 请求验证
3. 成功的代理：重置 fail_count，更新延迟；失败的代理：fail_count +1
4. 达到 `max_fail_count` 的代理从池中移除（移至 dead 列表）
5. 每 4 次健康检查输出一次完整状态快照（总数、可用数、延迟分布、Top-5 快速代理）
6. 每次检查完成自动保存磁盘缓存

### 自动收集流程

1. 后台任务按 `collect_interval` 间隔检查池大小
2. 当 `available_count < min_pool_size` 时触发收集
3. `ProxyCollector.collect()` 从 8 个公共源并发抓取代理
4. 新代理通过 `add_proxies()` 验证后加入池中
5. 若有新代理加入，自动清除 fallback 状态

### 磁盘缓存流程

- **保存时机**：新代理加入后、健康检查完成后、服务停止时
- **加载时机**：`start()` 启动时
- **缓存格式**：JSON 文件，包含 `updated_at` 时间戳和代理列表
- **过期策略**：加载时检查 `last_checked` 时间，超过 `health_check_interval` 的条目重新验证
- **原子写入**：先写入 `.tmp` 临时文件，再 `replace` 到目标路径，防止写入中断损坏缓存

## 日志层级

模块采用严格的分层日志策略：

| 层级 | 使用场景 |
|------|----------|
| `INFO` | 关键结果与汇总：启动、停止、收集结果、健康检查周期汇总、fallback 状态切换、代理淘汰 |
| `DEBUG` | 详细信息：收集开始、状态快照详情、Top 来源排行、缓存加载详情 |
| `TRACE` | 原子操作：单条代理测试结果（成功/失败及原因）、单个源的抓取产出、缓存读写操作 |

## 集成点

### 消费方

| 模块 | 使用方式 |
|------|----------|
| `agent/tools/web.py` | `web_search` / `web_fetch` 通过 `pool.get_proxy()` 获取代理 URL，传入 `httpx.AsyncClient(proxy=...)` |
| `agent/tools/browser.py` | `browser_search` / `browser_fetch` 通过 `pool.get_playwright_proxy()` 获取 Playwright 代理字典 |
| `agent/tools/browser_control.py` | `BrowserManager` 通过代理池配置 Playwright 持久化浏览器 |

### 初始化方

| 模块 | 职责 |
|------|------|
| `agent/loop.py` | `AgentLoop.__init__()` 中根据 `config.proxy_pool.enabled` 创建 `ProxyPool` 实例，注入到工具层 |
| `agent/subagent.py` | `SubagentManager` 中创建独立的 `ProxyPool` 实例供子 Agent 工具使用 |
| `summerclaw.py` | 从 `config.proxy_pool` 传递配置到 `AgentLoop` |

## 设计原则

1. **降级友好**：代理池耗尽时不阻塞请求，自动 fallback 到直连模式，后台持续寻找新代理。
2. **延迟择优**：代理选择基于延迟排序 + 随机采样，兼顾速度与负载分散。
3. **零阻塞启动**：初始代理验证和收集在后台异步执行，不影响 Agent 主循环启动。
4. **磁盘持久化**：通过 JSON 缓存文件跨重启保留已验证代理，减少冷启动收集开销。
5. **分层日志**：INFO / DEBUG / TRACE 三级日志严格区分，既保证关键信息可见，又避免日志泛滥。
6. **线程安全**：所有代理队列操作通过 `asyncio.Lock` 保护，支持并发获取和修改。
7. **多协议支持**：支持 HTTP、HTTPS、SOCKS4、SOCKS5 四种代理协议，覆盖不同使用场景。
8. **原子缓存写入**：先写临时文件再替换，防止进程崩溃导致缓存文件损坏。

## 与 summerclaw 其他模块的关系

```
summerclaw/
├── proxy/              ← 本模块（IP 代理池）
├── config/schema.py    → ProxyPoolConfig 配置定义
├── agent/
│   ├── loop.py         → AgentLoop 创建 ProxyPool 实例，注入工具层
│   ├── subagent.py     → SubagentManager 创建独立 ProxyPool 实例
│   └── tools/
│       ├── web.py      → web_search/web_fetch 通过 get_proxy() 使用代理
│       ├── browser.py  → browser_search/browser_fetch 通过 get_playwright_proxy() 使用代理
│       └── browser_control.py → BrowserManager 集成代理池
└── summerclaw.py          → 从 config.proxy_pool 传递配置
```