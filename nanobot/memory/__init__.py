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
from nanobot.memory.embedding_store import EmbeddingStore, batch_cosine_np  # noqa: F401
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

# Mem0V3 memory algorithm — zero external dependencies, pure Python implementation.
from nanobot.memory.mem0v3_memory import Mem0V3MemoryAlgorithm  # noqa: F401
from nanobot.memory.mem0v3_memory.store import Mem0V3Store  # noqa: F401
from nanobot.memory.mem0v3_memory.consolidator import Mem0V3Consolidator  # noqa: F401
from nanobot.memory.mem0v3_memory.dream import Mem0V3Dream  # noqa: F401
from nanobot.memory.mem0v3_memory.auto_compact import Mem0V3AutoCompact  # noqa: F401

# Supermemory memory algorithm — chunk-based memory with relational versioning,
# temporal grounding, and hybrid search. Zero external dependencies.
from nanobot.memory.supermemory_memory import SupermemoryMemoryAlgorithm  # noqa: F401
from nanobot.memory.supermemory_memory.store import SupermemoryStore  # noqa: F401
from nanobot.memory.supermemory_memory.consolidator import SupermemoryConsolidator  # noqa: F401
from nanobot.memory.supermemory_memory.dream import SupermemoryDream  # noqa: F401
from nanobot.memory.supermemory_memory.auto_compact import SupermemoryAutoCompact  # noqa: F401

# REME memory algorithm — removed (see task: remem_memory算法全量下线)
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

# Hindsight memory algorithm — requires `pip install hindsight-client`
# and a running Hindsight server (https://hindsight.vectorize.io/)
from nanobot.memory.hindsight_memory import HindsightMemoryAlgorithm  # noqa: F401
from nanobot.memory.hindsight_memory.auto_compact import HindsightAutoCompact  # noqa: F401
from nanobot.memory.hindsight_memory.consolidator import HindsightConsolidator  # noqa: F401
from nanobot.memory.hindsight_memory.dream import HindsightDream  # noqa: F401
from nanobot.memory.hindsight_memory.store import HindsightStore  # noqa: F401

# MastraOM memory algorithm — Observational Memory (Observer/Reflector pipeline)
from nanobot.memory.mastra_om_memory import MastraOMMemoryAlgorithm  # noqa: F401
from nanobot.memory.mastra_om_memory.store import MastraOMStore  # noqa: F401
from nanobot.memory.mastra_om_memory.consolidator import MastraOMConsolidator  # noqa: F401
from nanobot.memory.mastra_om_memory.dream import MastraOMDream  # noqa: F401
from nanobot.memory.mastra_om_memory.auto_compact import MastraOMAutoCompact  # noqa: F401

__all__ = [
    "AutoCompact",
    "Consolidator",
    "Dream",
    "EmbeddingStore",
    "batch_cosine_np",
    "HindsightAutoCompact",
    "HindsightConsolidator",
    "HindsightDream",
    "HindsightMemoryAlgorithm",
    "HindsightStore",
    "LayergaAutoCompact",
    "LayergaConsolidator",
    "LayergaDream",
    "LayergaMemoryAlgorithm",
    "LayergaStore",
    "Mem0V3AutoCompact",
    "Mem0V3Consolidator",
    "Mem0V3Dream",
    "Mem0V3MemoryAlgorithm",
    "Mem0V3Store",
    "MemoryAlgorithm",
    "MemoryComponents",
    "MemoryRegistry",
    "MemoryStore",
    "NemoriConsolidator",
    "NemoriDream",
    "NemoriMemoryAlgorithm",
    "NemoriStore",
    "SupermemoryAutoCompact",
    "SupermemoryConsolidator",
    "SupermemoryDream",
    "SupermemoryMemoryAlgorithm",
    "SupermemoryStore",
    "MastraOMMemoryAlgorithm",
    "MastraOMStore",
    "MastraOMConsolidator",
    "MastraOMDream",
    "MastraOMAutoCompact",
]

if _HAS_EMEM:
    __all__.extend([
        "EMemAutoCompact",
        "EMemConsolidator",
        "EMemDream",
        "EMemMemoryAlgorithm",
        "EMemStore",
    ])
