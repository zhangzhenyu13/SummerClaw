"""IP proxy pool module.

Provides:
- :class:`ProxyPool` — maintains a collection of usable proxy servers with health checks.
- :class:`ProxyCollector` — fetches free proxy lists from public sources.
- :class:`ProxyInfo` — data model for a single proxy.
"""

from summerclaw.proxy.collector import ProxyCollector
from summerclaw.proxy.models import ProxyInfo
from summerclaw.proxy.pool import ProxyPool

__all__ = ["ProxyPool", "ProxyCollector", "ProxyInfo"]
