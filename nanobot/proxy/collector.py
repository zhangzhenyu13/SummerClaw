"""Proxy collector: fetches free proxy lists from public sources."""

from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING

import httpx
from loguru import logger

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "Mozilla/5.0 (compatible; nanobot-proxy-collector/1.0)"

# Known public proxy list sources
# Each source is a URL that returns a plaintext list of proxy addresses.
_SOURCES = [
    # ProxyScrape — HTTP proxies
    "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text&protocol=http&timeout=5000",
    # ProxyScrape — SOCKS5 proxies
    "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text&protocol=socks5&timeout=5000",
    # TheSpeedX GitHub proxy list — HTTP
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    # TheSpeedX GitHub proxy list — SOCKS4
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks4.txt",
    # TheSpeedX GitHub proxy list — SOCKS5
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
    # Proxy-List.download — HTTP
    "https://www.proxy-list.download/api/v1/get?type=http",
    # Proxy-List.download — HTTPS
    "https://www.proxy-list.download/api/v1/get?type=https",
    # proxylist.geonode.com — HTTP
    "https://proxylist.geonode.com/api/proxy-list?limit=50&page=1&sort_by=lastChecked&sort_type=desc&protocols=http",
]

# Pattern to extract ip:port (with optional protocol prefix)
_IP_PORT_RE = re.compile(
    r"(?:(?:https?|socks[45])://)?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{2,5})",
)


def _extract_ip_port(text: str) -> list[str]:
    """Extract ip:port strings from raw text."""
    matches = _IP_PORT_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _parse_geonode_proxies(data: list[dict]) -> list[str]:
    """Parse GeoNode JSON API response."""
    results: list[str] = []
    for entry in data:
        ip = entry.get("ip", "")
        port = entry.get("port", "")
        protocols = entry.get("protocols", [])
        if ip and port:
            proto = "http"
            if "socks5" in protocols:
                proto = "socks5"
            elif "socks4" in protocols:
                proto = "socks4"
            results.append(f"{proto}://{ip}:{port}")
    return results


# ---------------------------------------------------------------------------
# ProxyCollector
# ---------------------------------------------------------------------------


class ProxyCollector:
    """Collects free proxy addresses from public sources.

    Usage::

        collector = ProxyCollector()
        proxies = await collector.collect()
        # proxies = ["http://1.2.3.4:8080", "socks5://5.6.7.8:1080", ...]
    """

    def __init__(
        self,
        test_timeout: int = 10,
        check_url: str = "https://httpbin.org/ip",
        max_collect: int = 50,
    ) -> None:
        """Args:
            test_timeout: Timeout in seconds for both source fetch and proxy test.
            check_url: URL used to validate proxy candidates.
            max_collect: Maximum number of raw proxies to return per collection.
        """
        self._test_timeout = test_timeout
        self._check_url = check_url
        self._max_collect = max_collect

    async def collect(self) -> list[str]:
        """Fetch and parse proxy lists from all configured sources.

        Returns deduplicated list of proxy URLs (without protocol prefix, e.g. ``1.2.3.4:8080``).
        """
        all_proxies: list[str] = []

        async def _fetch_one(url: str, idx: int) -> tuple[int, list[str]]:
            try:
                async with httpx.AsyncClient(
                    timeout=self._test_timeout,
                    follow_redirects=True,
                ) as client:
                    resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
                    resp.raise_for_status()

                text = resp.text

                # If JSON, try GeoNode parser
                if url.startswith("https://proxylist.geonode.com"):
                    try:
                        import json
                        data = json.loads(text)
                        if isinstance(data, dict):
                            data = data.get("data", data.get("results", []))
                        if isinstance(data, list):
                            parsed = _parse_geonode_proxies(data)
                            logger.trace(
                                "ProxyCollector: source #{} [geonode] → {} proxy(s)", idx, len(parsed),
                            )
                            return (idx, parsed)
                    except Exception:
                        pass

                parsed = _extract_ip_port(text)
                logger.trace(
                    "ProxyCollector: source #{} [{}] → {} proxy(s)",
                    idx, url[:60], len(parsed),
                )
                return (idx, parsed)

            except Exception as e:
                logger.trace("ProxyCollector: source #{} [{}] failed: {}", idx, url[:60], e)
                return (idx, [])

        # Fetch all sources concurrently
        logger.debug(
            "ProxyCollector: fetching from {} sources (timeout={}s)...",
            len(_SOURCES), self._test_timeout,
        )
        tasks = [_fetch_one(url, i) for i, url in enumerate(_SOURCES)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Track per-source yield for summary
        source_stats: list[tuple[str, int]] = []
        seen: set[str] = set()
        for result in results:
            if isinstance(result, Exception):
                continue
            idx, proxies = result
            new_unique = 0
            for proxy in proxies:
                if proxy not in seen:
                    seen.add(proxy)
                    all_proxies.append(proxy)
                    new_unique += 1
            if new_unique > 0:
                source_stats.append((_SOURCES[idx], new_unique))

        before_cap = len(all_proxies)
        # Cap at max_collect
        if len(all_proxies) > self._max_collect:
            all_proxies = all_proxies[:self._max_collect]

        # Rich summary
        active = len([s for s in source_stats if s[1] > 0])
        logger.info(
            "ProxyCollector: collected {} unique proxy(s) from {}/{} active source(s){}",
            len(all_proxies), active, len(_SOURCES),
            f" (capped from {before_cap})" if before_cap > self._max_collect else "",
        )
        if source_stats:
            best = sorted(source_stats, key=lambda x: x[1], reverse=True)[:3]
            details = ", ".join(
                f"{url.split('/')[2] if '://' in url else url[:40]}: {n}"
                for url, n in best
            )
            logger.debug("ProxyCollector: top sources — {}", details)

        return all_proxies
