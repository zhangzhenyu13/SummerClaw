"""Training data loader — standard split directory format.

Supports the standard train/val/test split layout::

    data/my_tasks/
    ├── train/items.json
    ├── val/items.json
    └── test/items.json

Each item is a JSON object with at least::

    {
        "id": "task_001",
        "question": "User question text",
        "answers": ["candidate answer 1", "candidate answer 2"],
        "context": "Optional context",
        "scorer": "exact_match | llm_judge | custom"
    }

- `answers`: list of candidate answers (any one is accepted as correct).
- `scorer`: optional, defaults to `exact_match`. Use `custom` with a
  `custom-scorer.py` placed in the task output directory.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class DataSplit:
    """A named split of training items."""

    def __init__(self, name: str, items: list[dict[str, Any]]):
        self.name = name
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, n: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Sample *n* items without replacement.

        If *n* >= len(self), returns all items shuffled.
        """
        rng = random.Random(seed)
        pool = list(self.items)
        rng.shuffle(pool)
        return pool[:n]

    def iter_batches(
        self,
        batch_size: int,
        seed: int | None = None,
    ):
        """Yield successive batches of *batch_size* items (shuffled)."""
        pool = list(self.items)
        rng = random.Random(seed)
        rng.shuffle(pool)
        for i in range(0, len(pool), batch_size):
            yield pool[i : i + batch_size]


class DataLoader:
    """Load training data from a split directory structure.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing ``train/``, ``val/``, ``test/`` sub-dirs.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._splits: dict[str, DataSplit] = {}
        self._load()

    def _load(self) -> None:
        for split_name in ("train", "val", "test"):
            items_path = self.data_dir / split_name / "items.json"
            if not items_path.exists():
                continue
            with open(items_path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError(
                    f"{items_path}: expected a JSON array of items, "
                    f"got {type(raw).__name__}"
                )
            self._splits[split_name] = DataSplit(split_name, raw)

    def get_split(self, name: str) -> DataSplit:
        """Return a named split.

        Raises
        ------
        KeyError
            If the split does not exist.
        """
        split = self._splits.get(name)
        if split is None:
            available = ", ".join(sorted(self._splits)) or "(none)"
            raise KeyError(f"Split '{name}' not found. Available: {available}")
        return split

    @property
    def train(self) -> DataSplit:
        return self.get_split("train")

    @property
    def val(self) -> DataSplit:
        return self.get_split("val")

    @property
    def test(self) -> DataSplit:
        return self.get_split("test")

    @property
    def split_names(self) -> list[str]:
        return sorted(self._splits)

    def summary(self) -> dict[str, int]:
        """Return {split_name: item_count} for all loaded splits."""
        return {name: len(split) for name, split in self._splits.items()}
