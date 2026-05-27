"""FastAPI REST endpoints for training status and control.

Provides external API access for:
  - Training status queries
  - Log retrieval
  - History and skill content
  - Training control (start, cancel)
  - Data loading and uploading
  - Skill deployment
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.engine.trainer import TrainerEngine


def _create_api(engine: TrainerEngine):
    """Create FastAPI APIRouter with training status endpoints."""
    try:
        from fastapi import APIRouter
    except ImportError:
        logger.warning("FastAPI not installed; dashboard API disabled")
        return None

    router = APIRouter()

    @router.get("/api/status")
    async def get_status():
        """Current training status."""
        return {
            "is_running": engine.is_running,
            "current_score": engine.current_score if hasattr(engine, '_current_score') else 0,
            "best_score": engine.best_score,
            "best_step": engine._best_step,
            "total_steps": engine.history.total_steps,
            "total_epochs": engine.history.total_epochs,
            "task_dir": str(engine.out_dir),
            "has_data": engine.has_data(),
        }

    @router.get("/api/logs")
    async def get_logs():
        """Get recent log events."""
        with engine._events_lock:
            recent = list(engine._events[-100:])
        return {"logs": recent}

    @router.get("/api/history")
    async def get_history():
        """Full training history."""
        return engine.history.to_dict()

    @router.get("/api/best_skill")
    async def get_best_skill():
        """Best skill content."""
        return {"content": engine.best_skill, "chars": len(engine.best_skill)}

    @router.get("/api/current_skill")
    async def get_current_skill():
        """Current skill content."""
        return {"content": engine.current_skill, "chars": len(engine.current_skill)}

    @router.post("/api/cancel")
    async def cancel_training():
        """Request training cancellation."""
        engine.request_cancel()
        return {"status": "cancel_requested"}

    @router.post("/api/start")
    async def start_training(body: dict = None):
        """Start training (called from dashboard UI).

        Accepts an optional ``skill_init_path`` in the request body to
        set/override the initial skill file path before training begins.
        If not provided and no initial skill is loaded, the engine will
        attempt LLM-based auto-generation.

        Also accepts an optional ``task_id`` to point the engine at an
        existing task directory (useful after restart to restore data).
        """
        body = body or {}

        # If a task_id is given, point engine at that existing task
        task_id = body.get("task_id", "")
        if task_id:
            # Resolve train_root from the engine's initial out_dir or default
            _train_root = getattr(engine, '_train_root', None)
            if _train_root is None:
                _train_root = Path.home() / ".summerclaw" / "train-algs"
            task_dir = Path(_train_root) / task_id
            if task_dir.is_dir() and str(task_dir) != str(engine.out_dir):
                engine._set_task_dir(task_dir)

        engine._ensure_out_dir()

        # Log config source
        cfg = engine._trainer_cfg
        if cfg:
            logger.info(
                "[TRAIN] Starting with config: epochs={}, batch={}, lr={}, "
                "scheduler={}, update_mode={}, slow_update={}, meta_skill={}",
                cfg.get('num_epochs', '?'), cfg.get('batch_size', '?'),
                cfg.get('edit_budget', '?'),
                cfg.get('lr_scheduler', cfg.get('lr_mode', 'constant')),
                cfg.get('update_mode', cfg.get('skill_update_mode', 'patch')),
                cfg.get('use_slow_update', True),
                cfg.get('use_meta_skill', True),
            )

        # Allow setting skill_init_path from the UI
        skill_path = body.get("skill_init_path", "")
        if skill_path:
            engine.skill_init_path = skill_path
            # If the file is readable, pre-load content immediately
            p = Path(skill_path).expanduser()
            if p.is_file():
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    engine.skill_init = content
                    engine._current_skill = content
                    engine._best_skill = content
        result = engine.start_training_async()
        return {"status": result}

    @router.post("/api/load_data")
    async def load_data(body: dict = None):
        """Load training data from a directory path."""
        data_dir = (body or {}).get("data_dir", "")
        if not data_dir:
            return {"error": "data_dir required"}
        path = Path(data_dir).expanduser()
        if not path.exists():
            return {"error": f"directory not found: {data_dir}"}
        from summerclaw.agent_trainer.datasets.loader import DataLoader
        loader = DataLoader(str(path))
        if not loader.split_names:
            return {"error": "no valid splits (train/val/test) found"}
        engine.set_data_loader(loader)
        return {"status": "loaded", "splits": loader.summary()}

    @router.post("/api/upload_data")
    async def upload_data(body: dict = None):
        """Upload training data as JSON array (legacy endpoint)."""
        engine._ensure_out_dir()
        items = (body or {}).get("items", [])
        split_name = (body or {}).get("split", "train")
        if not items:
            return {"error": "items array required"}
        # Write to workspace
        out_dir = engine.out_dir / "uploaded_data" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        # Reload data loader
        data_root = engine.out_dir / "uploaded_data"
        from summerclaw.agent_trainer.datasets.loader import DataLoader
        loader = DataLoader(str(data_root))
        if loader.split_names:
            engine.set_data_loader(loader)
        return {"status": "uploaded", "split": split_name, "count": len(items)}

    @router.post("/api/upload_file")
    async def upload_file(
        main_file: "UploadFile" = None,
        test_file: "UploadFile" = None,
        scorer_mode: str = "exact_match",
        scorer_file: "UploadFile" = None,
        train_ratio: float = 7.0,
        val_ratio: float = 2.0,
        test_ratio: float = 1.0,
        seed: int = 42,
    ):
        """Upload file(s) and auto-split.

        Accepts multipart/form-data with .json, .jsonl, or .xlsx files.
        Optionally accepts a custom-scorer.py file for scorer='custom'.
        scorer_mode: 'exact_match' (default), 'llm_judge', or 'custom'.
        """
        from summerclaw.agent_trainer.datasets.splitter import (
            parse_file, split_items, split_with_test, write_splits,
        )
        from summerclaw.agent_trainer.datasets.loader import DataLoader

        if not main_file:
            return {"error": "main_file required"}

        # Validate: custom scorer requires a scorer file
        if scorer_mode == "custom" and not (scorer_file and scorer_file.filename):
            return {"error": "scorer_mode='custom' requires a custom-scorer.py file"}

        engine._ensure_out_dir()

        # Save uploaded file to temp location
        temp_dir = engine.out_dir / "_temp_upload"
        temp_dir.mkdir(parents=True, exist_ok=True)

        main_path = temp_dir / f"main{Path(main_file.filename).suffix}"
        main_path.write_bytes(await main_file.read())

        try:
            main_items = parse_file(main_path)
        except (ValueError, ImportError) as e:
            return {"error": f"parsing main file: {e}"}

        if not main_items:
            return {"error": "main file is empty"}

        # Apply scorer override to items without scorer field
        if scorer_mode and scorer_mode != "exact_match":
            for item in main_items:
                if not item.get("scorer"):
                    item["scorer"] = scorer_mode

        # Save custom scorer if provided
        if scorer_file and scorer_file.filename:
            scorer_path = engine.out_dir / "custom-scorer.py"
            scorer_path.write_bytes(await scorer_file.read())

        # Check for test file
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
            split_info = f"main({len(main_items)}) -> train:val = {train_ratio}:{val_ratio}, test({len(test_items)}) separate"
        else:
            splits = split_items(
                main_items,
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                seed=seed,
            )
            split_info = f"auto-split train:val:test = {train_ratio}:{val_ratio}:{test_ratio}"

        # Write splits
        data_root = engine.out_dir / "uploaded_data"
        summary = write_splits(splits, data_root)

        # Reload
        loader = DataLoader(str(data_root))
        if loader.split_names:
            engine.set_data_loader(loader)

        return {
            "status": "uploaded",
            "split_info": split_info,
            "splits": summary,
            "total": len(main_items),
        }

    @router.post("/api/deploy")
    async def deploy_skill(body: dict = None):
        """Deploy best skill to target path."""
        target = (body or {}).get("target_path", "")
        if not target:
            return {"error": "target_path required"}
        content = await engine.deploy_skill(target)
        return {"status": "deployed", "path": target, "chars": len(content)}

    return router
