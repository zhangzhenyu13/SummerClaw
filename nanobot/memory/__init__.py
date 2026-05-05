"""Memory system — pluggable memory algorithms for nanobot.

The default algorithm is ``naive_memory``, which uses file-based storage
(MEMORY.md, history.jsonl, SOUL.md, USER.md) with token-budget-triggered
consolidation and cron-scheduled Dream processing.

``nemori_memory`` implements Nemori's self-organising long-term memory with
episode segmentation, Predict-Calibrate semantic extraction, and episode merging.

Usage::

    from nanobot.memory import MemoryStore, Consolidator, Dream, AutoCompact
    from nanobot.memory.registry import MemoryRegistry
    from nanobot.memory.naive_memory import NaiveMemoryAlgorithm

    registry = MemoryRegistry()
    registry.register(NaiveMemoryAlgorithm())
    algo = registry.get("naive_memory")
    components = algo.build(...)
"""

from nanobot.memory.base import MemoryAlgorithm, MemoryComponents
from nanobot.memory.naive_memory.auto_compact import AutoCompact
from nanobot.memory.naive_memory.consolidator import Consolidator
from nanobot.memory.naive_memory.dream import Dream
from nanobot.memory.naive_memory.store import MemoryStore
from nanobot.memory.nemori_memory import NemoriMemoryAlgorithm
from nanobot.memory.nemori_memory.consolidator import NemoriConsolidator
from nanobot.memory.nemori_memory.dream import NemoriDream
from nanobot.memory.nemori_memory.store import NemoriStore
from nanobot.memory.registry import MemoryRegistry

# Layerga memory algorithm — GenericAgent-style L0-L4 layered memory
# Zero external dependencies, pure Python implementation.
from nanobot.memory.layerga_memory import LayergaMemoryAlgorithm  # noqa: F401
from nanobot.memory.layerga_memory.auto_compact import LayergaAutoCompact  # noqa: F401
from nanobot.memory.layerga_memory.consolidator import LayergaConsolidator  # noqa: F401
from nanobot.memory.layerga_memory.dream import LayergaDream  # noqa: F401
from nanobot.memory.layerga_memory.store import LayergaStore  # noqa: F401

# Optional ReMe memory algorithm — requires `pip install nanobot-ai[reme]`
# If PyPI does not yet host >=0.3.1.9, install from source:
#   git clone https://github.com/agentscope-ai/ReMe.git && cd ReMe && pip install -e ".[light]"
try:
    import reme  # noqa: F401  # verify the external reme package is installed

    # Only import the adapter subpackage when reme is actually available
    from nanobot.memory.remem_memory import ReMeMemoryAlgorithm  # noqa: F401
    from nanobot.memory.remem_memory.auto_compact import ReMeAutoCompact  # noqa: F401
    from nanobot.memory.remem_memory.consolidator import ReMeConsolidator  # noqa: F401
    from nanobot.memory.remem_memory.dream import ReMeDream  # noqa: F401
    from nanobot.memory.remem_memory.store import ReMeStore  # noqa: F401

    _HAS_REME = True
except ImportError:
    _HAS_REME = False

# Optional EMem memory algorithm — requires `pip install nanobot-ai[emem]`
try:
    from nanobot.memory.emem_memory import EMemMemoryAlgorithm  # noqa: F401
    from nanobot.memory.emem_memory.auto_compact import EMemAutoCompact  # noqa: F401
    from nanobot.memory.emem_memory.consolidator import EMemConsolidator  # noqa: F401
    from nanobot.memory.emem_memory.dream import EMemDream  # noqa: F401
    from nanobot.memory.emem_memory.store import EMemStore  # noqa: F401

    _HAS_EMEM = True
except ImportError:
    _HAS_EMEM = False

__all__ = [
    "AutoCompact",
    "Consolidator",
    "Dream",
    "LayergaAutoCompact",
    "LayergaConsolidator",
    "LayergaDream",
    "LayergaMemoryAlgorithm",
    "LayergaStore",
    "MemoryAlgorithm",
    "MemoryComponents",
    "MemoryRegistry",
    "MemoryStore",
    "NemoriConsolidator",
    "NemoriDream",
    "NemoriMemoryAlgorithm",
    "NemoriStore",
]

if _HAS_REME:
    __all__.extend([
        "ReMeAutoCompact",
        "ReMeConsolidator",
        "ReMeDream",
        "ReMeMemoryAlgorithm",
        "ReMeStore",
    ])

if _HAS_EMEM:
    __all__.extend([
        "EMemAutoCompact",
        "EMemConsolidator",
        "EMemDream",
        "EMemMemoryAlgorithm",
        "EMemStore",
    ])
