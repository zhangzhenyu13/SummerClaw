"""Proxy data models for the IP proxy pool."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class ProxyInfo:
    """Information about a single proxy server."""

    url: str
    """Full proxy URL, e.g. ``http://1.2.3.4:8080`` or ``socks5://1.2.3.4:1080``."""

    protocol: str = "http"
    """Protocol: ``http``, ``https``, ``socks4``, or ``socks5``."""

    latency_ms: float = 0.0
    """Last measured round-trip latency in milliseconds (0 = unchecked)."""

    fail_count: int = 0
    """Consecutive failure count. Reset to 0 on each successful check."""

    last_checked: float = 0.0
    """Unix timestamp of last health check."""

    source: str = ""
    """Where this proxy was collected from (e.g. URL or ``"manual"``)."""

    def is_available(self, max_fail_count: int = 3) -> bool:
        """Return True if this proxy is still considered usable."""
        return self.fail_count < max_fail_count

    def mark_success(self, latency_ms: float = 0.0) -> None:
        """Record a successful use: reset fail_count and update latency/check time."""
        self.fail_count = 0
        if latency_ms > 0:
            self.latency_ms = latency_ms
        self.last_checked = time.time()

    def mark_failure(self) -> None:
        """Record a failure: increment fail_count."""
        self.fail_count += 1
        self.last_checked = time.time()

    def to_url(self) -> str:
        """Return the proxy URL string."""
        return self.url

    def to_playwright_proxy(self) -> dict[str, str]:
        """Return a Playwright-compatible proxy dict."""
        return {"server": self.url}
