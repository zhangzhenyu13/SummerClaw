"""Data management routes — upload, download, list data splits."""
import json
from pathlib import Path
from typing import Annotated, TYPE_CHECKING

from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse
from loguru import logger
from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

if TYPE_CHECKING:
    from fastapi import APIRouter


def register(router: "APIRouter", state: _DashboardState) -> None:
    """Register data management routes on *router*."""
    from summerclaw.agent_trainer.engine.trainer import _load_json

    # ------------------------------------------------------------------
    # Load data from directory
    # ------------------------------------------------------------------

    @router.post("/api/tasks/{task_id}/load_data")
    async def load_data(task_id: str, body: dict = None):
        body = body or {}
        data_dir = body.get("data_dir", "")
        if not data_dir:
            return {"error": "data_dir required"}
        path = Path(data_dir).expanduser()
        if not path.exists():
            return {"error": f"directory not found: {data_dir}"}
        from summerclaw.agent_trainer.datasets.loader import DataLoader
        loader = DataLoader(str(path))
        if not loader.split_names:
            return {"error": "no valid splits found"}
        # Point engine at the task dir so subsequent operations associate correctly.
        if task_id:
            state._maybe_restore_task(task_id)
        state.engine._ensure_out_dir()
        state.engine.set_data_loader(loader)
        if task_id and state.scheduler:
            state.scheduler.register_idle(task_id)
        return {"status": "loaded", "splits": loader.summary()}

    # ------------------------------------------------------------------
    # File upload (multipart form)
    # ------------------------------------------------------------------

    @router.post("/api/upload/file")
    async def upload_file(
        task_id: str = Form(""),
        main_file: Annotated[UploadFile | None, File()] = None,
        test_file: Annotated[UploadFile | None, File()] = None,
        scorer_mode: str = Form("exact_match"),
        scorer_file: Annotated[UploadFile | None, File()] = None,
        train_ratio: float = Form(7.0),
        val_ratio: float = Form(2.0),
        test_ratio: float = Form(1.0),
        seed: int = Form(42),
    ):
        """Upload file(s) and auto-split."""
        from summerclaw.agent_trainer.datasets.splitter import (
            parse_file, split_items, split_with_test, write_splits,
        )
        from summerclaw.agent_trainer.datasets.loader import DataLoader

        if not main_file:
            return {"error": "main_file required"}

        if scorer_mode == "custom" and not (scorer_file and scorer_file.filename):
            return {"error": "scorer_mode='custom' requires a custom-scorer.py file"}

        # Determine the target directory for this upload.
        # When task_id is provided, always write to the task's own directory.
        # Do NOT touch engine state here — the scheduler may be modifying it
        # concurrently. We only update the engine after data is safely on disk.
        if task_id:
            _effective_dir = state.train_root / task_id
            _effective_dir.mkdir(parents=True, exist_ok=True)
        else:
            state.engine._ensure_out_dir()
            _effective_dir = state.engine.out_dir

        logger.info("[Upload] target dir: {} (task_id={})", _effective_dir, task_id)

        temp_dir = _effective_dir / "_temp_upload"
        temp_dir.mkdir(parents=True, exist_ok=True)

        main_path = temp_dir / f"main{Path(main_file.filename).suffix}"
        main_path.write_bytes(await main_file.read())

        try:
            main_items = parse_file(main_path)
        except (ValueError, ImportError) as e:
            logger.error("[Upload] parse failed: {}", e)
            return {"error": f"parsing main file: {e}"}

        if not main_items:
            return {"error": "main file is empty"}

        if scorer_mode and scorer_mode != "exact_match":
            for item in main_items:
                if not item.get("scorer"):
                    item["scorer"] = scorer_mode

        if scorer_file and scorer_file.filename:
            scorer_path = _effective_dir / "custom-scorer.py"
            scorer_path.write_bytes(await scorer_file.read())

        if test_file and test_file.filename:
            test_path = temp_dir / f"test{Path(test_file.filename).suffix}"
            test_path.write_bytes(await test_file.read())
            try:
                test_items = parse_file(test_path)
            except (ValueError, ImportError) as e:
                return {"error": f"parsing test file: {e}"}
            splits = split_with_test(
                main_items, test_items,
                train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
            )
            split_info = (
                f"main({len(main_items)}) -> train:val = "
                f"{train_ratio}:{val_ratio}, test({len(test_items)}) separate"
            )
        else:
            splits = split_items(
                main_items,
                train_ratio=train_ratio, val_ratio=val_ratio,
                test_ratio=test_ratio, seed=seed,
            )
            split_info = (
                f"auto-split train:val:test = "
                f"{train_ratio}:{val_ratio}:{test_ratio}"
            )

        data_root = _effective_dir / "uploaded_data"
        summary = write_splits(splits, data_root)
        logger.info("[Upload] wrote splits to {}: {}", data_root, summary)

        # Now that data is on disk, update engine state.
        # Only set engine's data_loader if the engine is pointed at this task.
        loader = DataLoader(str(data_root))
        if loader.split_names and task_id and str(state.engine.out_dir) == str(_effective_dir):
            state.engine.set_data_loader(loader)
        elif loader.split_names and not task_id:
            state.engine.set_data_loader(loader)

        # Re-register with scheduler so it picks up the newly uploaded data
        # immediately (the idle-delay retry loop would also find it, but this
        # makes the response snappy).
        if task_id and state.scheduler:
            state.scheduler.register_idle(task_id)

        return {
            "status": "uploaded",
            "split_info": split_info,
            "splits": summary,
            "total": len(main_items),
        }

    # ------------------------------------------------------------------
    # Download data split
    # ------------------------------------------------------------------

    @router.get("/api/tasks/{task_id}/data/{split}")
    async def download_task_data(task_id: str, split: str):
        """Download data file for a specific split (train/val/test)."""
        task_dir = state.train_root / task_id
        # Only serve data from THIS task's own uploaded_data directory.
        candidates = [
            task_dir / "uploaded_data" / split / "items.json",
        ]
        # Also allow engine's out_dir if it's pointed at this task
        if str(state.engine.out_dir) == str(task_dir):
            c = state.engine.out_dir / "uploaded_data" / split / "items.json"
            if c not in candidates:
                candidates.append(c)
        for p in candidates:
            if p.is_file():
                return FileResponse(
                    str(p),
                    media_type="application/json",
                    filename=f"{split}.json",
                )
        return {"error": f"No data found for split '{split}' in task '{task_id}'"}

    # ------------------------------------------------------------------
    # List data splits
    # ------------------------------------------------------------------

    @router.get("/api/tasks/{task_id}/data")
    async def list_task_data(task_id: str):
        """List available data splits and their sizes."""
        task_dir = state.train_root / task_id
        # Only look at THIS task's own uploaded_data directory.
        data_dirs = [
            task_dir / "uploaded_data",
        ]
        # Also allow engine's out_dir if it's pointed at this task
        if str(state.engine.out_dir) == str(task_dir):
            d = state.engine.out_dir / "uploaded_data"
            if d not in data_dirs:
                data_dirs.append(d)

        splits: dict[str, int] = {}
        for data_root in data_dirs:
            if not data_root.is_dir():
                continue
            for split_dir in sorted(data_root.iterdir()):
                if not split_dir.is_dir():
                    continue
                items_path = split_dir / "items.json"
                if items_path.is_file():
                    try:
                        data = _load_json(str(items_path)) or []
                        splits[split_dir.name] = len(data)
                    except Exception:
                        splits[split_dir.name] = -1
            if splits:
                break  # Use first non-empty data source
        return {"splits": splits, "task_id": task_id}

    # ------------------------------------------------------------------
    # Upload data as JSON
    # ------------------------------------------------------------------

    @router.post("/api/upload/data")
    async def upload_data_json(body: dict = None):
        """Upload training data as JSON array."""
        state.engine._ensure_out_dir()
        items = (body or {}).get("items", [])
        split_name = (body or {}).get("split", "train")
        if not items:
            return {"error": "items array required"}
        out_dir = state.engine.out_dir / "uploaded_data" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        data_root = state.engine.out_dir / "uploaded_data"
        from summerclaw.agent_trainer.datasets.loader import DataLoader
        loader = DataLoader(str(data_root))
        if loader.split_names:
            state.engine.set_data_loader(loader)
        return {"status": "uploaded", "split": split_name, "count": len(items)}
