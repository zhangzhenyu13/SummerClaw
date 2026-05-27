"""Data parsing and splitting utilities for the training dashboard.

Supports JSON, JSONL, and XLSX file formats with configurable
train/val/test split ratios.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


# ── File format requirements ──────────────────────────────────────────────

FORMAT_REQUIREMENTS = """
**Supported file formats:**

| Format | Extension | Description |
|--------|-----------|-------------|
| JSON   | `.json`   | A JSON array of objects |
| JSONL  | `.jsonl`  | One JSON object per line |
| Excel  | `.xlsx`   | One row per item, headers = field names |

**Required fields per item:**

| Field      | Type         | Required | Description |
|------------|--------------|----------|-------------|
| `id`       | string       | Yes      | Unique item identifier |
| `question` | string       | Yes      | User question / input text |
| `answers`  | list[string] | Yes      | List of candidate answers (any one is accepted as correct) |
| `context`  | string       | No       | Optional additional context for the question |
| `scorer`   | string       | No       | Scoring method: `exact_match` (default), `llm_judge`, or `custom` |

**Custom Scorer:**

When `scorer` is set to `custom`, a `custom-scorer.py` file must be uploaded alongside the data.
The script must define a function `score(sample: dict, predicted: str) -> float` that returns a score between 0 and 1.

- `sample`: the full data item dict (contains `id`, `question`, `answers`, `context`, etc.)
- `predicted`: the agent's predicted answer string
- Returns: a float in `[0.0, 1.0]`

**Examples:**

JSON:
```json
[
  {
    "id": "t1",
    "question": "What is 2+2?",
    "answers": ["4", "four"],
    "context": "Basic arithmetic",
    "scorer": "exact_match"
  }
]
```

JSONL (one object per line):
```
{"id":"t1","question":"What is 2+2?","answers":["4","four"],"scorer":"exact_match"}
{"id":"t2","question":"Capital of France?","answers":["Paris"]}
```

XLSX: columns `id`, `question`, `answers` (JSON-encoded list), optional: `context`, `scorer`

**custom-scorer.py example:**
```python
def score(sample: dict, predicted: str) -> float:
    # Custom scoring logic
    answers = sample.get("answers", [])
    predicted_lower = predicted.strip().lower()
    for ans in answers:
        if ans.lower() in predicted_lower:
            return 1.0
    return 0.0
```
"""


# ── File parsers ─────────────────────────────────────────────────────────

def parse_json_file(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON file containing an array of items."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def parse_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file (one JSON object per line)."""
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no}: invalid JSON — {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no}: expected JSON object, got {type(obj).__name__}")
            items.append(obj)
    return items


def parse_xlsx_file(path: Path) -> list[dict[str, Any]]:
    """Parse an XLSX file where each row is an item and headers are field names."""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for XLSX support: pip install openpyxl")
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 2:
        raise ValueError("XLSX file must have at least a header row and one data row")
    headers = [str(h).strip().lower() if h else f"col_{i}" for i, h in enumerate(rows[0])]
    items: list[dict[str, Any]] = []
    for row in rows[1:]:
        if not any(cell is not None and str(cell).strip() for cell in row):
            continue  # Skip empty rows
        item = {}
        for header, cell in zip(headers, row):
            if cell is not None:
                item[header] = str(cell).strip() if isinstance(cell, str) else cell
            else:
                item[header] = ""
        items.append(item)
    wb.close()
    return items


def parse_file(path: Path) -> list[dict[str, Any]]:
    """Auto-detect file format and parse items.

    Supported: .json, .jsonl, .xlsx
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_json_file(path)
    elif suffix == ".jsonl":
        return parse_jsonl_file(path)
    elif suffix == ".xlsx":
        return parse_xlsx_file(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported: .json, .jsonl, .xlsx"
        )


# ── Splitting ────────────────────────────────────────────────────────────

def split_items(
    items: list[dict[str, Any]],
    train_ratio: float = 7,
    val_ratio: float = 2,
    test_ratio: float = 1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split items into train/val/test with given ratios.

    Parameters
    ----------
    items : list
        Items to split.
    train_ratio, val_ratio, test_ratio : float
        Relative ratios (do not need to sum to 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {"train": [...], "val": [...], "test": [...]}
    """
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Ratios must sum to a positive value")

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio / total)
    n_val = round(n * val_ratio / total)
    n_test = n - n_train - n_val  # Remainder goes to test

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def split_with_test(
    main_items: list[dict[str, Any]],
    test_items: list[dict[str, Any]],
    train_ratio: float = 6,
    val_ratio: float = 4,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split main items into train/val, use separate test items as-is.

    Parameters
    ----------
    main_items : list
        Items to split into train/val.
    test_items : list
        Items used directly as the test split.
    train_ratio, val_ratio : float
        Relative ratios for train/val split.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {"train": [...], "val": [...], "test": [...]}
    """
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError("Ratios must sum to a positive value")

    rng = random.Random(seed)
    shuffled = list(main_items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio / total)
    n_val = n - n_train

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": list(test_items),
    }


# ── Write splits to disk ────────────────────────────────────────────────

def write_splits(
    splits: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> dict[str, int]:
    """Write split data to the standard directory layout.

    Creates::
        out_dir/
        ├── train/items.json
        ├── val/items.json
        └── test/items.json

    Returns {split_name: count}.
    """
    summary: dict[str, int] = {}
    for name, items in splits.items():
        if not items:
            continue
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        items_path = split_dir / "items.json"
        with open(items_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        summary[name] = len(items)
    return summary
