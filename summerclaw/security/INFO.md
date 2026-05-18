# security — 网络安全模块

## 概述

`security` 模块是 summerclaw 的安全防护层，核心职责是**防止 SSRF（Server-Side Request Forgery）攻击**。当 Agent 执行 Web 抓取、浏览器自动化、Shell 命令等工具操作时，该模块对目标 URL 进行网络层校验，拦截指向内网/私有地址的请求，防止攻击者通过 Agent 探测或攻击内部基础设施。

## 架构

```
┌───────────────┐     validate_url_target()      ┌──────────────────┐
│  Web 抓取工具  │ ──────────────────────────────► │                  │
│  (web.py)     │                                  │  security/       │
├───────────────┤     validate_resolved_url()      │  network.py      │
│  浏览器工具    │ ──────────────────────────────► │                  │
│  (browser.py) │                                  │  ┌────────────┐ │
├───────────────┤     contains_internal_url()      │  │ 默认屏蔽段  │ │
│  Shell 工具   │ ──────────────────────────────► │  │ (RFC1918   │ │
│  (shell.py)   │                                  │  │  链路本地   │ │
├───────────────┤                                  │  │  回环地址)  │ │
│  Channel 入口 │     validate_url_target()        │  │             │ │
│  (telegram/qq)│ ──────────────────────────────► │  │ 白名单机制  │ │
└───────────────┘                                  │  └────────────┘ │
                                                   └──────────────────┘
                                                           ▲
                                                   configure_ssrf_whitelist()
                                                           │
                                                   ┌───────┴───────┐
                                                   │ config/loader  │
                                                   │ (启动时注入)    │
                                                   └───────────────┘
```

- **工具层**：Web/浏览器工具在执行请求前调用 `validate_url_target()` 校验目标 URL；Shell 工具在运行命令前调用 `contains_internal_url()` 扫描命令中的 URL。
- **Channel 层**：部分 Channel（如 Telegram、QQ）在消息预处理阶段调用 `validate_url_target()` 对用户提供的链接做安全性检查。
- **配置层**：`config/loader.py` 在加载配置后调用 `configure_ssrf_whitelist()`，将用户配置的 CIDR 白名单注入到安全模块。

## 文件说明

| 文件 | 职责 |
|------|------|
| `__init__.py` | 模块入口，当前为空（导出通过 `network` 子模块直接引用） |
| `network.py` | 核心实现：SSRF 防护逻辑、URL 校验、命令扫描、白名单管理 |

## 核心 API

### 常量与配置

| 符号 | 类型 | 说明 |
|------|------|------|
| `_BLOCKED_NETWORKS` | `list[IPv4Network\|IPv6Network]` | 默认屏蔽的私有/内部网络 CIDR 列表 |
| `_allowed_networks` | `list[IPv4Network\|IPv6Network]` | 用户配置的白名单 CIDR（通过 `configure_ssrf_whitelist()` 设置） |
| `_URL_RE` | `re.Pattern` | 用于从命令字符串中提取 URL 的正则表达式 |

**默认屏蔽的地址范围：**

| CIDR | 说明 |
|------|------|
| `0.0.0.0/8` | 零地址 |
| `10.0.0.0/8` | RFC 1918 私有地址（A 类） |
| `100.64.0.0/10` | Carrier-Grade NAT（含 Tailscale 等） |
| `127.0.0.0/8` | 回环地址 |
| `169.254.0.0/16` | 链路本地地址 / 云元数据服务端点 |
| `172.16.0.0/12` | RFC 1918 私有地址（B 类） |
| `192.168.0.0/16` | RFC 1918 私有地址（C 类） |
| `::1/128` | IPv6 回环地址 |
| `fc00::/7` | IPv6 唯一本地地址（ULA） |
| `fe80::/10` | IPv6 链路本地地址 |

### `configure_ssrf_whitelist(cidrs: list[str]) -> None`

配置 SSRF 白名单，允许特定 CIDR 范围绕过屏蔽。

- **参数**：`cidrs` — CIDR 字符串列表（如 `["100.64.0.0/10"]` 允许 Tailscale 网段）
- **行为**：无效的 CIDR 会被静默忽略（`ValueError` 时跳过）
- **典型用法**：在 `config/loader.py` 中根据 `config.tools.ssrf_whitelist` 配置项调用

### `validate_url_target(url: str) -> tuple[bool, str]`

校验一个 URL 是否可以安全访问。执行三重检查：**协议 → 主机名 → DNS 解析并验证 IP**。

- **返回**：`(ok, error_message)` 二元组，`ok=True` 时 `error_message` 为空
- **校验流程**：
  1. 解析 URL，仅允许 `http` / `https` 协议
  2. 检查是否有有效主机名（`hostname`）
  3. 通过 DNS 解析主机名得到所有 IP 地址
  4. 逐一检查每个 IP 是否落在 `_BLOCKED_NETWORKS` 中（白名单 IP 除外）
- **阻塞场景**：非 http/https 协议、缺失主机名、DNS 解析失败、解析到私有地址

### `validate_resolved_url(url: str) -> tuple[bool, str]`

校验一个已完成重定向的 URL。与 `validate_url_target()` 的区别在于：**对于已是 IP 地址的 hostname 直接检查，跳过 DNS 解析**。

- **适用场景**：HTTP 重定向后的落地 URL 二次校验
- **容错设计**：解析失败或无法解析时返回 `(True, "")`（放行），避免因中间环节异常阻断合法请求

### `contains_internal_url(command: str) -> bool`

扫描命令字符串，检测是否包含指向内网/私有地址的 URL。

- **返回**：`True` 表示命令中包含内网 URL，应被拦截
- **内部机制**：用正则提取命令中的所有 URL，逐一调用 `validate_url_target()` 校验
- **典型用法**：Shell 工具在执行用户命令前调用，防止如 `curl http://169.254.169.254/` 等云元数据窃取攻击

## 集成点

### 工具层 — Web 抓取

| 调用位置 | 函数 | 说明 |
|----------|------|------|
| `agent/tools/web.py#fetch_url()` | `validate_url_target()` | 发起 HTTP 请求前校验目标 URL |
| `agent/tools/web.py#_handle_redirect()` | `validate_resolved_url()` | 重定向后对落地 URL 二次校验 |

### 工具层 — 浏览器

| 调用位置 | 函数 | 说明 |
|----------|------|------|
| `agent/tools/browser.py#navigate()` | `validate_url_target()` | 浏览器导航前校验目标 URL |
| `agent/tools/browser_control.py` | `validate_url_target()` | 浏览器自动化控制前校验 |

### 工具层 — Shell

| 调用位置 | 函数 | 说明 |
|----------|------|------|
| `agent/tools/shell.py#SafetyGuard.check()` | `contains_internal_url()` | 命令执行前的安全扫描 |

### Channel 层

| 调用位置 | 函数 | 说明 |
|----------|------|------|
| `channels/telegram.py` | `validate_url_target()` | 对 Telegram 消息中的链接做安全校验 |
| `channels/qq.py` | `validate_url_target()` | 对 QQ 消息中的链接做安全校验 |

### 配置注入

| 调用位置 | 函数 | 说明 |
|----------|------|------|
| `config/loader.py#_apply_ssrf_whitelist()` | `configure_ssrf_whitelist()` | 启动时将 `config.tools.ssrf_whitelist` 注入安全模块 |
| `config/schema.py#ToolsConfig` | `ssrf_whitelist: list[str]` | 配置模型中的 SSRF 白名单字段 |

## 测试覆盖

测试文件：`tests/security/test_security_network.py`（13 个测试用例）

| 分类 | 测试用例 | 覆盖点 |
|------|----------|--------|
| 协议/域名校验 | `test_rejects_non_http_scheme` | 拒绝非 HTTP 协议 |
| | `test_rejects_missing_domain` | 拒绝缺少域名 |
| 私有 IPv4 屏蔽 | `test_blocks_private_ipv4` | 7 组参数化：回环、RFC1918、metadata、零地址 |
| 私有 IPv6 屏蔽 | `test_blocks_ipv6_loopback` | 拒绝 IPv6 回环地址 `::1` |
| 公网放行 | `test_allows_public_ip` | 允许公网 IP |
| | `test_allows_normal_https` | 允许正常 HTTPS 请求 |
| Shell 命令扫描 | `test_detects_curl_metadata` | 检测 `curl` 提取元数据的命令 |
| | `test_detects_wget_localhost` | 检测 `wget` 访问 localhost |
| | `test_allows_normal_curl` | 正常 curl 命令放行 |
| | `test_no_urls_returns_false` | 无 URL 的命令返回 False |
| SSRF 白名单 | `test_blocks_cgnat_by_default` | 默认屏蔽 CGNAT 地址 |
| | `test_whitelist_allows_cgnat` | 白名单后允许 CGNAT |
| | `test_whitelist_does_not_affect_other_blocked` | 白名单不影响其他屏蔽段 |
| | `test_whitelist_invalid_cidr_ignored` | 无效 CIDR 静默忽略 |

## 设计原则

1. **纵深防御**：URL 校验不仅检查协议和域名，还进行 DNS 解析并逐 IP 验证，防止 DNS 重绑定攻击。
2. **双重校验**：区分请求前（`validate_url_target`）和重定向后（`validate_resolved_url`），覆盖 HTTP 重定向绕过场景。
3. **命令级扫描**：`contains_internal_url()` 从 Shell 命令字符串中提取 URL 并校验，防止攻击者通过管道、环境变量等方式注入内网请求。
4. **白名单机制**：支持通过配置文件 `ssrf_whitelist` 字段指定豁免的 CIDR 范围（如 Tailscale 的 `100.64.0.0/10`），满足企业内部网络场景。
5. **零外部依赖**：完全基于 Python 标准库（`ipaddress`、`socket`、`urllib.parse`、`re`）实现，无第三方依赖。
6. **静默容错**：无效 CIDR 配置被静默忽略；重定向校验中解析失败时放行（不阻塞合法请求）。

## 与 summerclaw 其他模块的关系

```
summerclaw/
├── security/              ← 本模块（网络安全防护）
├── agent/tools/
│   ├── web.py             → Web 工具执行前调用 validate_url_target / validate_resolved_url
│   ├── shell.py           → Shell 工具执行前调用 contains_internal_url
│   ├── browser.py         → 浏览器工具执行前调用 validate_url_target
│   └── browser_control.py → 浏览器自动化控制前调用 validate_url_target
├── channels/
│   ├── telegram.py        → 消息预处理阶段调用 validate_url_target
│   └── qq.py              → 消息预处理阶段调用 validate_url_target
├── config/
│   ├── schema.py          → ToolsConfig.ssrf_whitelist 配置字段定义
│   └── loader.py          → 启动时调用 configure_ssrf_whitelist() 注入白名单
└── tests/security/
    └── test_security_network.py → 13 个测试用例覆盖所有核心 API
```