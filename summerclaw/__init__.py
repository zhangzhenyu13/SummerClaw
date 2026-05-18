"""
summerclaw - A lightweight AI agent framework
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


def _read_pyproject_version() -> str | None:
    """Read the source-tree version when package metadata is unavailable."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data.get("project", {}).get("version")


def _resolve_version() -> str:
    try:
        return _pkg_version("summerclaw-ai")
    except PackageNotFoundError:
        # Source checkouts often import summerclaw without installed dist-info.
        return _read_pyproject_version() or "0.2.3"


__version__ = _resolve_version()
__logo__ = "🐈"

from summerclaw.summerclaw import SummerClaw, RunResult

__all__ = ["SummerClaw", "RunResult"]
