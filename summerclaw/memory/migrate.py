"""Memory data migration utilities — copy legacy shared files to algorithm-specific directories.

Each helper is **idempotent**: if the target path already exists the operation is
skipped, so multiple store initialisations are safe.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger


def migrate_file(src: Path, dst: Path) -> bool:
    """Copy a single file from *src* to *dst* if *src* exists and *dst* does not.

    Returns ``True`` if a copy was performed, ``False`` otherwise.
    """
    if not src.exists():
        return False
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Migrated {} -> {}", src, dst)
    return True


def migrate_dir(src: Path, dst: Path) -> bool:
    """Recursively copy a directory from *src* to *dst* if *dst* does not exist.

    Returns ``True`` if a copy was performed, ``False`` otherwise.
    """
    if not src.is_dir():
        return False
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    logger.info("Migrated directory {} -> {}", src, dst)
    return True


def maybe_migrate_legacy_files(
    memory_dir: Path,
    old_memory_dir: Path,
    old_workspace: Path,
    files: list[str],
    dirs: list[str] | None = None,
) -> None:
    """Migrate legacy shared files into the algorithm-specific *memory_dir*.

    Args:
        memory_dir: The new algorithm-specific data directory
            (``workspace/memory/<algo_name>/``).
        old_memory_dir: The old shared memory directory
            (``workspace/memory/``).
        old_workspace: The old workspace root where SOUL.md, USER.md etc.
            may still live.
        files: Relative file paths to migrate.  Each is first checked under
            *old_memory_dir*, then under *old_workspace*.
        dirs: Relative directory paths to migrate under *old_memory_dir*.
    """
    for rel in files:
        # Try old_memory_dir first, then old_workspace
        src = old_memory_dir / rel
        if not src.exists():
            src = old_workspace / rel
        dst = memory_dir / rel
        migrate_file(src, dst)

    for rel in (dirs or []):
        src = old_memory_dir / rel
        dst = memory_dir / rel
        migrate_dir(src, dst)
