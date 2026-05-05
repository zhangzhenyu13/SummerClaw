---
name: web-search
description: 通用网络搜索技能，支持多引擎搜索（百度、必应、DuckDuckGo），全部使用 Playwright 实现，无需API密钥即可获取实时信息
version: 2.0.0
author: UnionClaw
license: MIT
tags:
  - search
  - web
  - internet
  - baidu
  - bing
  - playwright
---

# Web Search Skill

A powerful web search skill supporting multiple search engines using Playwright browser automation without requiring API keys.

## Features

- 🔍 **Multi-Engine Support**: Baidu, Bing, DuckDuckGo (全部使用 Playwright)
- 🌐 **No API Key Required**: Uses Playwright browser automation
- 🔄 **Smart Fallback**: Automatically switches engines when one fails
- 📊 **Structured Results**: Returns clean search results with title, URL, and snippet
- 🚀 **High Performance**: Playwright browser automation with headless mode

## Usage

### Basic Search

```python
# 使用百度搜索（默认）
result = main({
    "action": "search",
    "query": "Python tutorial",
    "num_results": 5,
    "engine": "baidu"
})

# 使用必应搜索
result = main({
    "action": "search",
    "query": "Python tutorial",
    "num_results": 5,
    "engine": "bing"
})

# 使用 DuckDuckGo 搜索
result = main({
    "action": "search",
    "query": "Python tutorial",
    "num_results": 5,
    "engine": "duckduckgo"
})
```

### Deep Search

```python
result = main({
    "action": "deep_search",
    "query": "machine learning latest research",
    "num_results": 5
})
```

### Web Page Crawling

```python
result = main({
    "action": "crawl",
    "url": "https://example.com"
})
```

## Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| action | string | Yes | Operation type: "search", "deep_search", or "crawl" |
| query | string | Conditional | Search query (required for search/deep_search) |
| url | string | Conditional | Target URL (required for crawl) |
| num_results | int | No | Number of results, default 5, max 20 |
| region | string | No | Region code, default 'cn-zh' |
| engine | string | No | Search engine: "baidu", "bing", "duckduckgo", default "baidu" |

## Output Format

### Search Result

```python
{
    "success": True,
    "query": "search query",
    "engine": "baidu+playwright",
    "num_results": 5,
    "results": [
        {
            "title": "Result title",
            "href": "https://...",
            "body": "Snippet content"
        }
    ],
    "message": "Search completed"
}
```

### Deep Search Result

```python
{
    "success": True,
    "query": "search query",
    "search_results": [...],
    "detailed_info": {
        "extracted_content": "..."
    },
    "message": "Deep search completed"
}
```

## Execution

**type**: script
**script_path**: scripts/web_search.py
**entry_point**: main
**dependencies**: 
  - uv>=0.1.0
  - crawl4ai>=0.8.0
  - playwright>=1.40.0

## Search Engine Selection

通过 `engine` 参数指定搜索引擎：

- **baidu** (默认): 百度搜索，适合中文内容
- **bing**: 必应搜索，适合国际内容
- **duckduckgo**: DuckDuckGo 搜索，隐私友好

## Notes

1. **First Run**: Playwright will download Chromium browser on first use (~100MB)
2. **Rate Limiting**: Be mindful of search frequency to avoid temporary blocks
3. **Network**: Requires internet connection
4. **Results**: May vary based on search engine algorithms and location

## Error Handling
- Do not call this tool with high concurrency to avoid IP ban.
- If frequently timeout met, please decrease calling frequency and intervals.
- Returns `{"success": False, "message": "..."}` on errors
- Automatically retries with fallback engines
- Graceful degradation when optional dependencies are missing
