"""IP proxy pool module.

Provides:
- :class:`ProxyPool` — maintains a collection of usable proxy servers with health checks.
- :class:`ProxyCollector` — fetches free proxy lists from public sources.
- :class:`ProxyInfo` — data model for a single proxy.
"""

from nanobot.proxy.collector import ProxyCollector
from nanobot.proxy.models import ProxyInfo
from nanobot.proxy.pool import ProxyPool

__all__ = ["ProxyPool", "ProxyCollector", "ProxyInfo"]
