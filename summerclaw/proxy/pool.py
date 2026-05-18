"""Proxy pool: maintains a collection of usable proxy servers with health checks."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from summerclaw.proxy.models import ProxyInfo

if TYPE_CHECKING:
    from summerclaw.config.schema import ProxyPoolConfig
    from summerclaw.proxy.collector import ProxyCollector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "Mozilla/5.0 (compatible; summerclaw-proxy-pool/1.0)"


def _parse_proxy_url(raw: str) -> str | None:
    """Parse a raw proxy string into a canonical URL.

    Accepts:
        - ``http://1.2.3.4:8080`` (already canonical)
        - ``1.2.3.4:8080`` (assume http)
        - ``socks5://1.2.3.4:1080``

    Returns None if the string is clearly invalid.
    """
    raw = raw.strip()
    if not raw:
        return None
    if "://" in raw:
        return raw
    # Assume http if no protocol
    return f"http://{raw}"


def _extract_protocol(url: str) -> str:
    """Extract protocol from a proxy URL."""
    if url.startswith("socks5://"):
        return "socks5"
    if url.startswith("socks4://"):
        return "socks4"
    if url.startswith("https://"):
        return "https"
    return "http"


# ---------------------------------------------------------------------------
# ProxyPool
# ---------------------------------------------------------------------------


class ProxyPool:
    """Thread-safe async proxy pool with health checking and auto-collection.

    Usage::

        pool = ProxyPool(config)
        await pool.start()

        proxy_url = await pool.get_proxy()
        # ... use proxy_url in httpx / Playwright ...

        # If the request fails due to the proxy:
        pool.mark_bad(proxy_url)

        await pool.stop()
    """

    def __init__(self, config: ProxyPoolConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._proxies: deque[ProxyInfo] = deque()
        self._dead: list[ProxyInfo] = []  # permanently dead proxies
        self._index: int = 0  # round-robin cursor
        self._started = False

        # Fallback tracking — when pool is enabled but empty, we operate in
        # direct-connection mode while background tasks keep replenishing.
        self._in_fallback: bool = False
        self._fallback_log_seq: int = 0  # throttle fallback log to every N calls

        # Cache path
        self._cache_path: Path | None = None
        if config.proxy_cache_enabled:
            if config.proxy_cache_path:
                self._cache_path = Path(config.proxy_cache_path).expanduser()
            else:
                self._cache_path = Path.home() / ".summerclaw" / "proxy_cache.json"

        # Background tasks
        self._health_task: asyncio.Task | None = None
        self._collect_task: asyncio.Task | None = None
        self._collector: ProxyCollector | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the pool: load cache, validate initial proxies, launch background tasks."""
        if self._started:
            return
        self._started = True

        try:
            # Load cached proxies from previous run
            initial_raw = list(self._config.initial_proxies)
            if self._cache_path is not None:
                cached = await self._load_cache()
                if cached:
                    cached_urls = {p.url for p in cached}
                    initial_raw = [url for url in initial_raw if url not in cached_urls]
                    logger.info(
                        "ProxyPool: loaded {} proxy(s) from cache, {} additional from config",
                        len(cached), len(initial_raw),
                    )
                    async with self._lock:
                        for info in cached:
                            self._proxies.append(info)

            # Load and validate initial proxies
            if initial_raw:
                logger.info("ProxyPool: validating {} initial proxy(s)...", len(initial_raw))
                validated = await self._validate_batch(initial_raw)
                async with self._lock:
                    for info in validated:
                        self._proxies.append(info)
                logger.info("ProxyPool: {} initial proxy(s) available", len(validated))

            # Save initial state to cache
            if self._cache_path is not None:
                self._save_cache()

            # Start background health checker
            if self._config.health_check_interval > 0:
                self._health_task = asyncio.create_task(self._health_check_loop())

            # Start background collector — if pool is empty, run initial collection
            # first; otherwise start the periodic loop directly
            if self.available_count == 0:
                logger.info(
                    "ProxyPool: pool empty after startup, scheduling initial collection "
                    "(agent loop will not be blocked)"
                )
                self._collect_task = asyncio.create_task(self._run_initial_collection())
            else:
                self._collect_task = asyncio.create_task(self._collection_loop())

        except Exception:
            logger.exception("ProxyPool: startup failed — pool will remain disabled")
            self._started = False
            # Clean up any tasks that were created
            await self.stop()

    async def stop(self) -> None:
        """Stop background tasks."""
        self._started = False
        for task in (self._health_task, self._collect_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._health_task = None
        self._collect_task = None

        # Persist current pool to cache
        if self._cache_path is not None:
            self._save_cache()

        logger.info("ProxyPool: stopped")

    # -- proxy retrieval -----------------------------------------------------

    @property
    def available_count(self) -> int:
        """Number of currently available proxies."""
        return sum(1 for p in self._proxies if p.is_available(self._config.max_fail_count))

    @property
    def total_count(self) -> int:
        """Total proxies in pool (including temporarily failed)."""
        return len(self._proxies)

    async def get_proxy(self) -> str | None:
        """Return the next available proxy URL, or None if the pool is exhausted.

        When the pool is enabled but empty, returns None so callers can fall
        back to direct connection.  Background health-check and collection
        tasks continue running to replenish the pool.
        """
        async with self._lock:
            available = [
                p for p in self._proxies
                if p.is_available(self._config.max_fail_count)
            ]
            if not available:
                self._track_fallback()
                return None

            # Pick with preference for lower latency, but with some randomness
            # Sort by latency, then pick randomly from the top half
            available.sort(key=lambda p: p.latency_ms if p.latency_ms > 0 else 99999)
            top_n = max(1, len(available) // 2)
            chosen = random.choice(available[:top_n])
            self._clear_fallback()
            return chosen.to_url()

    async def get_playwright_proxy(self) -> dict[str, str] | None:
        """Return a Playwright-compatible proxy dict, or None."""
        async with self._lock:
            available = [
                p for p in self._proxies
                if p.is_available(self._config.max_fail_count)
            ]
            if not available:
                self._track_fallback()
                return None
            available.sort(key=lambda p: p.latency_ms if p.latency_ms > 0 else 99999)
            top_n = max(1, len(available) // 2)
            chosen = random.choice(available[:top_n])
            self._clear_fallback()
            return chosen.to_playwright_proxy()

    def mark_bad(self, proxy_url: str) -> None:
        """Mark a proxy as having failed. Removes from pool after too many failures."""
        for p in self._proxies:
            if p.url == proxy_url:
                p.mark_failure()
                if not p.is_available(self._config.max_fail_count):
                    logger.info(
                        "ProxyPool: proxy {} exhausted ({} consecutive failures), removed from pool",
                        proxy_url, p.fail_count,
                    )
                    # Remove from pool (health check loop also removes, but
                    # we remove here for immediate effect)
                    try:
                        self._proxies.remove(p)
                    except ValueError:
                        pass
                return

    # -- status snapshot -----------------------------------------------------

    def _log_status_snapshot(self) -> None:
        """Log a multi-line summary of the entire pool state."""
        total = self.total_count
        available = self.available_count
        dead = len(self._dead)
        all_latencies = [p.latency_ms for p in self._proxies if p.latency_ms > 0]
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0

        logger.info(
            "ProxyPool status — total: {}, available: {}, dead: {}, "
            "avg latency: {:.0f}ms, fallback: {}",
            total, available, dead, avg_latency, self._in_fallback,
        )

        # Top 5 best latency proxies
        ranked = sorted(
            [p for p in self._proxies if p.is_available(self._config.max_fail_count) and p.latency_ms > 0],
            key=lambda p: p.latency_ms,
        )[:5]
        if ranked:
            best = ", ".join(
                f"{p.url} ({p.latency_ms:.0f}ms)" for p in ranked
            )
            logger.debug("ProxyPool top {} by latency: {}", len(ranked), best)

    # -- disk cache ----------------------------------------------------------

    def _save_cache(self) -> None:
        """Persist available proxies to disk as JSON."""
        if self._cache_path is None:
            return
        try:
            available = [
                p for p in self._proxies
                if p.is_available(self._config.max_fail_count)
            ]
            data = {
                "updated_at": time.time(),
                "proxies": [
                    {
                        "url": p.url,
                        "protocol": p.protocol,
                        "latency_ms": p.latency_ms,
                        "fail_count": p.fail_count,
                        "last_checked": p.last_checked,
                        "source": p.source,
                    }
                    for p in available
                ],
            }
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._cache_path)
            logger.trace("ProxyPool: cache saved — {} proxy(s)", len(available))
        except Exception:
            logger.debug("ProxyPool: failed to save cache")
            logger.trace("ProxyPool: cache save error", exc_info=True)

    async def _load_cache(self) -> list[ProxyInfo]:
        """Load cached proxies from disk and re-validate stale entries.

        Returns the list of ProxyInfo objects that are still valid.
        Entries whose last_checked time exceeds the staleness threshold
        are re-tested; others are trust-loaded immediately.
        """
        if self._cache_path is None:
            return []
        try:
            raw = self._cache_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            entries: list[dict] = data.get("proxies", [])
            if not entries:
                return []

            # Separate fresh from stale.  Staleness threshold = health-check
            # interval so that reloads soon after restart trust the data,
            # while older caches get re-validated.
            now = time.time()
            stale_threshold = self._config.health_check_interval
            fresh: list[ProxyInfo] = []
            stale_urls: list[str] = []

            for entry in entries:
                url = entry.get("url", "")
                if not url:
                    continue
                info = ProxyInfo(
                    url=url,
                    protocol=entry.get("protocol", "http"),
                    latency_ms=entry.get("latency_ms", 0.0),
                    fail_count=0,  # reset on load — cached entries are trusted
                    last_checked=entry.get("last_checked", 0.0),
                    source=entry.get("source", "cache"),
                )
                age = now - info.last_checked
                if age < stale_threshold:
                    fresh.append(info)
                else:
                    stale_urls.append(url)

            if stale_urls:
                logger.debug(
                    "ProxyPool: re-validating {} stale cached proxy(s)...",
                    len(stale_urls),
                )
                revalidated = await self._validate_batch(stale_urls, source="cache")
                fresh.extend(revalidated)

            return fresh
        except FileNotFoundError:
            logger.trace("ProxyPool: no cache file found at {}", self._cache_path)
            return []
        except Exception:
            logger.debug("ProxyPool: failed to load cache")
            logger.trace("ProxyPool: cache load error", exc_info=True)
            return []

    def mark_good(self, proxy_url: str, latency_ms: float = 0.0) -> None:
        """Mark a proxy as working successfully."""
        for p in self._proxies:
            if p.url == proxy_url:
                p.mark_success(latency_ms)
                return

    # -- fallback tracking ----------------------------------------------------

    _FALLBACK_LOG_EVERY = 20  # throttle: log every N-th fallback hit

    def _track_fallback(self) -> None:
        """Record a fallback hit; log at info level when entering fallback, debug thereafter."""
        if not self._in_fallback:
            self._in_fallback = True
            self._fallback_log_seq = 0
            logger.warning(
                "ProxyPool: pool exhausted — falling back to direct connection "
                "(health-check + collector keep running in background)"
            )
            return
        self._fallback_log_seq += 1
        if self._fallback_log_seq % self._FALLBACK_LOG_EVERY == 0:
            logger.info(
                "ProxyPool: still in fallback mode ({} direct requests so far), "
                "collector is searching for new proxies...",
                self._fallback_log_seq,
            )

    def _clear_fallback(self) -> None:
        """Clear fallback state when proxies become available again."""
        if self._in_fallback:
            self._in_fallback = False
            available = self.available_count
            logger.info(
                "ProxyPool: recovered — {} proxy(s) now available, resuming proxied requests",
                available,
            )

    @property
    def is_fallback(self) -> bool:
        """True when the pool is enabled but has no available proxies (direct mode)."""
        return self._started and self._in_fallback

    # -- proxy addition ------------------------------------------------------

    async def add_proxies(self, raw_urls: list[str], source: str = "manual") -> int:
        """Parse, deduplicate, validate, and add new proxies. Returns count of newly added."""
        # Parse and deduplicate
        existing = {p.url for p in self._proxies}
        new_urls: list[str] = []
        for raw in raw_urls:
            url = _parse_proxy_url(raw)
            if url and url not in existing:
                new_urls.append(url)

        if not new_urls:
            return 0

        # Validate
        validated = await self._validate_batch(new_urls, source=source)

        async with self._lock:
            current = {p.url for p in self._proxies}
            added = 0
            for info in validated:
                if info.url not in current:
                    self._proxies.append(info)
                    added += 1

        if added:
            self._clear_fallback()
            logger.info("ProxyPool: added {} new proxy(s), pool size now {}", added, self.total_count)
            self._save_cache()
        return added

    # -- health checking -----------------------------------------------------

    async def _validate_batch(
        self, urls: list[str], source: str = "manual"
    ) -> list[ProxyInfo]:
        """Test a batch of proxy URLs concurrently. Return validated ProxyInfo list."""
        batch_size = len(urls)
        logger.info(
            "ProxyPool: validating {} proxy candidate(s) from '{}' (timeout={}s)...",
            batch_size, source, self._config.proxy_test_timeout,
        )

        async def _check_one(url: str) -> ProxyInfo | None:
            info = ProxyInfo(
                url=url,
                protocol=_extract_protocol(url),
                source=source,
            )
            latency = await self._test_proxy(url)
            if latency is not None:
                info.mark_success(latency)
                return info
            return None

        tasks = [_check_one(u) for u in urls]
        results = await asyncio.gather(*tasks)
        passed = [r for r in results if r is not None]
        failed = batch_size - len(passed)
        logger.info(
            "ProxyPool: validation complete — {} passed, {} failed ({} tested) from '{}'",
            len(passed), failed, batch_size, source,
        )
        return passed

    async def _test_proxy(self, proxy_url: str) -> float | None:
        """Test if a proxy URL works by making a request through it.

        Returns latency in milliseconds on success, None on failure.
        """
        timeout = self._config.proxy_test_timeout
        check_url = self._config.health_check_url
        try:
            start = time.monotonic()
            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=timeout,
                follow_redirects=False,
            ) as client:
                resp = await client.get(
                    check_url,
                    headers={"User-Agent": _USER_AGENT},
                )
                # Accept any 2xx/3xx as success
                if 200 <= resp.status_code < 400:
                    elapsed = (time.monotonic() - start) * 1000
                    logger.trace("ProxyPool: proxy {} OK — {:.0f}ms", proxy_url, elapsed)
                    return elapsed
            logger.trace("ProxyPool: proxy {} FAIL — HTTP {}", proxy_url, resp.status_code)
            return None
        except asyncio.TimeoutError:
            logger.trace("ProxyPool: proxy {} FAIL — timeout ({:.0f}s)", proxy_url, timeout)
            return None
        except Exception as e:
            logger.trace("ProxyPool: proxy {} FAIL — {}: {}", proxy_url, type(e).__name__, e)
            return None

    async def _health_check_loop(self) -> None:
        """Background task: periodically re-validate all proxies."""
        cycle: int = 0
        while self._started:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                if not self._started:
                    break

                async with self._lock:
                    proxies = list(self._proxies)

                if not proxies:
                    continue

                cycle += 1
                logger.debug(
                    "ProxyPool: health check cycle #{} starting — {} proxy(s) to test",
                    cycle, len(proxies),
                )

                # Check each proxy, removing dead ones
                alive: list[ProxyInfo] = []
                dead: list[ProxyInfo] = []
                for p in proxies:
                    latency = await self._test_proxy(p.url)
                    if latency is not None:
                        p.mark_success(latency)
                        alive.append(p)
                    else:
                        p.mark_failure()
                        if p.is_available(self._config.max_fail_count):
                            alive.append(p)
                        else:
                            dead.append(p)

                async with self._lock:
                    self._proxies = deque(alive)
                    self._dead.extend(dead)

                # Summary with latency stats
                latencies = [p.latency_ms for p in alive if p.latency_ms > 0]
                if dead:
                    logger.info(
                        "ProxyPool: health check #{} done — {} alive, {} dead, "
                        "latency min/avg/max {:.0f}/{:.0f}/{:.0f}ms",
                        cycle, len(alive), len(dead),
                        min(latencies) if latencies else 0,
                        sum(latencies) / len(latencies) if latencies else 0,
                        max(latencies) if latencies else 0,
                    )
                else:
                    logger.info(
                        "ProxyPool: health check #{} done — {} alive, "
                        "latency min/avg/max {:.0f}/{:.0f}/{:.0f}ms",
                        cycle, len(alive),
                        min(latencies) if latencies else 0,
                        sum(latencies) / len(latencies) if latencies else 0,
                        max(latencies) if latencies else 0,
                    )

                # Periodic status snapshot every 4 cycles
                if cycle % 4 == 0:
                    self._log_status_snapshot()

                # Persist updated pool to cache
                self._save_cache()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("ProxyPool: health check error")

    # -- collection ----------------------------------------------------------

    async def _run_initial_collection(self) -> None:
        """Run the first collection, then hand off to the normal collection loop."""
        logger.info(
            "ProxyPool: initial collection starting — fetching from public proxy sources..."
        )
        try:
            added = await self._trigger_collection()
            logger.info(
                "ProxyPool: initial collection done — added {} proxy(s), available now {}",
                added, self.available_count,
            )
        except asyncio.CancelledError:
            logger.info("ProxyPool: initial collection cancelled")
            return
        except Exception:
            logger.exception("ProxyPool: initial collection failed")
        finally:
            # Replace this task with the regular periodic collection loop
            if self._started:
                self._collect_task = asyncio.create_task(self._collection_loop())

    async def _collection_loop(self) -> None:
        """Background task: periodically check pool size and trigger collection."""
        while self._started:
            try:
                await asyncio.sleep(self._config.collect_interval)
                if not self._started:
                    break

                available = self.available_count
                if available < self._config.min_pool_size:
                    logger.info(
                        "ProxyPool: available={} < min={}, triggering collection",
                        available, self._config.min_pool_size,
                    )
                    added = await self._trigger_collection()
                    if added > 0:
                        logger.info(
                            "ProxyPool: collection added {} proxy(s), available now {}",
                            added, self.available_count,
                        )
                    else:
                        logger.info(
                            "ProxyPool: collection found no new proxies, available still {}",
                            self.available_count,
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("ProxyPool: collection loop error")

    async def _trigger_collection(self) -> int:
        """Collect proxies from external sources and add them to the pool. Returns count added."""
        if self._collector is None:
            from summerclaw.proxy.collector import ProxyCollector
            self._collector = ProxyCollector(
                test_timeout=self._config.proxy_test_timeout,
                check_url=self._config.health_check_url,
                max_collect=self._config.max_pool_size,
            )

        try:
            raw_proxies = await self._collector.collect()
            if raw_proxies:
                return await self.add_proxies(raw_proxies, source="collector")
            return 0
        except Exception:
            logger.exception("ProxyPool: collection failed")
            return 0
