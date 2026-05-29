"""Task CRUD routes — create, delete, config, eval, deploy, YAML management."""
from __future__ import annotations

import shutil
from datetime import datetime as _dt
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from fastapi import APIRouter
    from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def _get_split_count(engine, split: str) -> int:
    """Get number of items in a data split."""
    try:
        return len(engine.data_loader.get_split(split).items)
    except (KeyError, AttributeError):
        return -1


def _compute_breakdown_stats(per_item_results: list) -> dict:
    """Compute breakdown stats from per-item rollout results.

    Returns dict with n_total, n_correct, n_timeout, n_error, success_rate.
    """
    n_total = len(per_item_results)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_correct": 0,
            "n_timeout": 0,
            "n_error": 0,
            "success_rate": 0.0,
        }

    n_correct = sum(1 for r in per_item_results if r.hard == 1)
    n_timeout = sum(
        1 for r in per_item_results
        if r.fail_reason and r.fail_reason.startswith("rollout_timeout")
    )
    n_error = sum(
        1 for r in per_item_results
        if r.fail_reason and not r.fail_reason.startswith("rollout_timeout") and r.hard == 0
    )

    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_timeout": n_timeout,
        "n_error": n_error,
        "success_rate": round(n_correct / n_total, 6) if n_total > 0 else 0.0,
    }


def _compute_comparison(eval_dir: Path, split: str) -> dict | None:
    """Compute comparison metrics when both with_skill and no_skill results exist.

    Returns dict with:
    - all_items: score on all items (timeouts = 0)
    - completed_items: score only on items where BOTH runs completed (no timeout/error)
    - n_total, n_completed_both, n_both_ok, n_at_least_one_ok
    - no_skill_success_rate, with_skill_success_rate
    """
    no_skill_file = eval_dir / f"{split}_no_skill_items.json"
    with_skill_file = eval_dir / f"{split}_with_skill_items.json"
    no_skill_result_file = eval_dir / f"{split}_no_skill.json"
    with_skill_result_file = eval_dir / f"{split}_with_skill.json"

    if not (no_skill_file.is_file() and with_skill_file.is_file()):
        return None
    if not (no_skill_result_file.is_file() and with_skill_result_file.is_file()):
        return None

    from summerclaw.agent_trainer.engine.trainer import _load_json

    no_skill_items = _load_json(str(no_skill_file)) or []
    with_skill_items = _load_json(str(with_skill_file)) or []
    no_skill_data = _load_json(str(no_skill_result_file)) or {}
    with_skill_data = _load_json(str(with_skill_result_file)) or {}

    # Build lookup by item id
    no_skill_map = {item["id"]: item for item in no_skill_items}
    with_skill_map = {item["id"]: item for item in with_skill_items}

    n_total = len(no_skill_items)

    # Items where BOTH runs completed (no timeout/error in either)
    completed_both_ids = [
        item_id for item_id in no_skill_map.keys()
        if item_id in with_skill_map
        and not no_skill_map[item_id].get("fail_reason", "")
        and not with_skill_map[item_id].get("fail_reason", "")
    ]
    n_completed_both = len(completed_both_ids)

    # Score on completed items
    if n_completed_both > 0:
        completed_no_skill_score = sum(
            no_skill_map[item_id].get("hard", 0)
            for item_id in completed_both_ids
        ) / n_completed_both
        completed_with_skill_score = sum(
            with_skill_map[item_id].get("hard", 0)
            for item_id in completed_both_ids
        ) / n_completed_both
    else:
        completed_no_skill_score = 0.0
        completed_with_skill_score = 0.0

    # Items where both succeeded
    both_ok_ids = [
        item_id for item_id in no_skill_map.keys()
        if item_id in with_skill_map
        and no_skill_map[item_id].get("hard") == 1
        and with_skill_map[item_id].get("hard") == 1
    ]
    n_both_ok = len(both_ok_ids)

    # Items where at least one succeeded
    at_least_one_ok_ids = set()
    for item_id, item in no_skill_map.items():
        if item.get("hard") == 1:
            at_least_one_ok_ids.add(item_id)
    for item_id, item in with_skill_map.items():
        if item.get("hard") == 1:
            at_least_one_ok_ids.add(item_id)
    n_at_least_one_ok = len(at_least_one_ok_ids)

    return {
        "all_items": {
            "no_skill_score": no_skill_data.get("score", 0.0),
            "with_skill_score": with_skill_data.get("score", 0.0),
            "n_total": n_total,
        },
        "completed_items": {
            "no_skill_score": round(completed_no_skill_score, 6),
            "with_skill_score": round(completed_with_skill_score, 6),
            "n_items": n_completed_both,
        },
        "n_both_ok": n_both_ok,
        "n_at_least_one_ok": n_at_least_one_ok,
        "no_skill_stats": no_skill_data.get("stats", {}),
        "with_skill_stats": with_skill_data.get("stats", {}),
    }


def register(router: APIRouter, state: _DashboardState) -> None:
    """Register task CRUD routes on *router*."""
    from summerclaw.agent_trainer.engine.trainer import _load_json, _save_json
    from summerclaw.agent_trainer.dashboard.task_utils import (
        _find_default_yaml_template,
        _generate_default_yaml,
        _apply_params_to_yaml,
    )

    # ------------------------------------------------------------------
    # Create task
    # ------------------------------------------------------------------

    @router.post("/api/tasks")
    async def create_task(body: dict):
        try:
            task_name = body.get("name", "")
            algorithm = body.get("algorithm", "skillopt")
            if not task_name:
                return {"error": "Task name is required"}

            # Auto-generate task_id with timestamp (e.g. skillopt-20260528-143022)
            ts = _dt.now().strftime("%Y%m%d-%H%M%S")
            task_id = f"{algorithm}-{ts}"
            task_dir = state.train_root / task_id
            if task_dir.exists():
                return {"error": f"Task directory already exists: {task_id}"}
            task_dir.mkdir(parents=True, exist_ok=True)

            cfg = {
                "algorithm": algorithm,
                "task_id": task_id,
                "name": task_name,
                "num_epochs": int(body.get("epochs", 3)),
                "batch_size": int(body.get("batch_size", 5)),
                "workers": int(body.get("workers", 0)),
                "seed": int(body.get("seed", 42)),
                "created_at": _dt.now().isoformat(),
            }
            # Memory algorithm: "null"/""/None → disable memory; otherwise store name
            _mem_algo = body.get("memory_algorithm", None)
            if _mem_algo in (None, "", "null", "none"):
                cfg["memory_algorithm"] = None
            else:
                cfg["memory_algorithm"] = str(_mem_algo)
            # Enabled tools: empty list = all defaults; otherwise selective
            _tools = body.get("enabled_tools", None)
            if isinstance(_tools, list):
                cfg["enabled_tools"] = _tools
            if body.get("description"):
                cfg["description"] = body["description"]
            if body.get("data_dir"):
                cfg["data_dir"] = body["data_dir"]
            if body.get("skill_path"):
                cfg["skill_init"] = body["skill_path"]
            _save_json(str(task_dir / "config.json"), cfg)
            (task_dir / "skills").mkdir(exist_ok=True)

            # Copy initial skill: copy_from takes priority, then explicit skill_path
            copy_from_id = body.get("copy_from", "")
            _skill_copied_from_source = False
            _data_copied_from_source = False
            if copy_from_id:
                _src_task_dir = state.train_root / copy_from_id
                # --- Copy initial skill ---
                _src_skill = _src_task_dir / "skills" / "skill_v0000.md"
                if _src_skill.exists():
                    shutil.copy2(str(_src_skill), str(task_dir / "skills" / "skill_v0000.md"))
                    _copied_path = str(task_dir / "skills" / "skill_v0000.md")
                    cfg["skill_init"] = _copied_path
                    _save_json(str(task_dir / "config.json"), cfg)
                    _skill_copied_from_source = True
                # --- Copy uploaded data ---
                _src_data = _src_task_dir / "uploaded_data"
                if _src_data.is_dir():
                    _dst_data = task_dir / "uploaded_data"
                    shutil.copytree(str(_src_data), str(_dst_data), dirs_exist_ok=True)
                    _data_copied_from_source = True
                # --- Copy custom-scorer.py ---
                _src_scorer = _src_task_dir / "custom-scorer.py"
                if _src_scorer.is_file():
                    shutil.copy2(str(_src_scorer), str(task_dir / "custom-scorer.py"))
            if not _skill_copied_from_source:
                skill_path = body.get("skill_path", "")
                if skill_path and Path(skill_path).exists():
                    shutil.copy2(skill_path, str(task_dir / "skills" / "skill_v0000.md"))

            # Create / copy skillopt.yaml
            yaml_dst = task_dir / "skillopt.yaml"
            yaml_content = body.get("yaml_content", "")
            if yaml_content:
                # Use provided YAML content (e.g. from copy-from-source)
                import yaml as _yaml
                try:
                    _yaml.safe_load(yaml_content)
                    yaml_dst.write_text(yaml_content, encoding="utf-8")
                except Exception as yaml_exc:
                    logger.warning(
                        "Invalid YAML provided for task {}, falling back to template: {}",
                        task_id, yaml_exc,
                    )
                    template = _find_default_yaml_template()
                    if template:
                        shutil.copy2(str(template), str(yaml_dst))
                    else:
                        yaml_dst.write_text(_generate_default_yaml(), encoding="utf-8")
            else:
                template = _find_default_yaml_template()
                if template:
                    shutil.copy2(str(template), str(yaml_dst))
                else:
                    yaml_dst.write_text(_generate_default_yaml(), encoding="utf-8")

            _apply_params_to_yaml(yaml_dst, {
                "num_epochs": int(body.get("epochs", 3)),
                "batch_size": int(body.get("batch_size", 5)),
                "workers": int(body.get("workers", 0)),
                "seed": int(body.get("seed", 42)),
                "edit_budget": int(body.get("learning_rate", 4)),
                "lr_scheduler": body.get("lr_scheduler", "constant"),
                "skill_update_mode": body.get("update_mode", "patch"),
                "use_slow_update": bool(body.get("slow_update", True)),
                "use_meta_skill": bool(body.get("meta_skill", True)),
                "reasoning_effort": body.get("reasoning_effort", "medium"),
                "memory_algorithm": cfg.get("memory_algorithm"),
                "enabled_tools": cfg.get("enabled_tools", []),
            })

            logger.info(
                "Task created: {} | algorithm: {} | memory: {} | tools: {}",
                task_id,
                algorithm,
                cfg.get("memory_algorithm") or "disabled",
                ", ".join(cfg.get("enabled_tools", [])) or "all",
            )

            # Register with scheduler for auto-scheduling (idle → running/queued after 10s)
            if state.scheduler:
                state.scheduler.register_idle(task_id)

            return {
                "status": "created",
                "task_id": task_id,
                "path": str(task_dir),
                "copied_skill_path": cfg.get("skill_init", "") if _skill_copied_from_source else "",
                "data_copied": _data_copied_from_source,
            }
        except Exception as exc:
            return {"error": f"Error creating task: {exc}"}

    # ------------------------------------------------------------------
    # Delete task
    # ------------------------------------------------------------------

    @router.delete("/api/tasks/{task_id}")
    async def delete_task(task_id: str):
        try:
            task_dir = state.train_root / task_id
            if not task_dir.exists():
                return {"error": f"Task directory not found: {task_id}"}
            shutil.rmtree(str(task_dir))
            return {"status": "deleted", "task_id": task_id}
        except Exception as exc:
            return {"error": f"Error deleting task: {exc}"}

    # ------------------------------------------------------------------
    # Get task config (for pre-filling create form)
    # ------------------------------------------------------------------

    @router.get("/api/tasks/{task_id}/config")
    async def get_task_config(task_id: str):
        """Return the task's config.json + skillopt.yaml for pre-filling create form."""
        task_dir = state.train_root / task_id
        if not task_dir.is_dir():
            return {"error": f"Task directory not found: {task_id}"}
        config = _load_json(str(task_dir / "config.json")) or {}
        yaml_path = task_dir / "skillopt.yaml"
        yaml_content = ""
        flat: dict = {}
        if yaml_path.is_file():
            yaml_content = yaml_path.read_text(encoding="utf-8")
            try:
                from summerclaw.agent_trainer.config import load_config, flatten_config
                flat = flatten_config(load_config(str(yaml_path)))
            except Exception as exc:
                logger.warning("Failed to parse skillopt.yaml for task {}: {}", task_id, exc)
        return {
            "config": config,
            "yaml_content": yaml_content,
            "flat": flat,
            "task_id": task_id,
            "has_skill": (task_dir / "skills" / "skill_v0000.md").is_file(),
            "has_data": (task_dir / "uploaded_data").is_dir(),
        }

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------

    @router.post("/api/tasks/{task_id}/eval_test")
    async def run_test_evaluation(task_id: str, body: dict = None):
        """Run val+test evaluation (with & without skill) on a completed task."""
        body = body or {}
        try:
            if state.engine.is_running:
                return {"error": "Cannot run eval while training is in progress."}

            state._maybe_restore_task(task_id)
            state.engine._ensure_out_dir()

            # Auto-load data from task directory if not already loaded
            if not state.engine.has_data():
                task_dir = state.train_root / task_id
                data_root = task_dir / "uploaded_data"
                if not data_root.is_dir():
                    # Fallback: scan all subdirs for uploaded_data
                    for entry in sorted(state.train_root.iterdir()):
                        candidate = entry / "uploaded_data"
                        if candidate.is_dir():
                            data_root = candidate
                            break
                if data_root.is_dir():
                    from summerclaw.agent_trainer.datasets.loader import DataLoader
                    loader = DataLoader(str(data_root))
                    if loader.split_names:
                        state.engine.set_data_loader(loader)

            if not state.engine.has_data():
                return {"error": "No data loaded for this task. Upload data first."}

            # Load best skill from disk for this task
            task_dir = state.train_root / task_id
            best_skill_path = task_dir / "best_skill.md"
            if best_skill_path.is_file():
                content = best_skill_path.read_text(encoding="utf-8")
                state.engine._best_skill = content
                state.engine._current_skill = content
            else:
                skill_dir = task_dir / "skills"
                if skill_dir.is_dir():
                    files = sorted(skill_dir.glob("*.md"))
                    if files:
                        content = files[-1].read_text(encoding="utf-8")
                        state.engine._best_skill = content
                        state.engine._current_skill = content

            summary = await state.engine._run_test_evaluation()
            return {"status": "done", "task_id": task_id, "summary": summary}
        except Exception as exc:
            return {"error": f"Test evaluation failed: {exc}"}

    @router.get("/api/tasks/{task_id}/eval_test")
    async def get_test_evaluation(task_id: str):
        """Return val+test evaluation results if available."""
        task_dir = state.train_root / task_id
        p = task_dir / "test_evaluation" / "test_summary.json"
        if p.is_file():
            data = _load_json(str(p)) or {}
            return {"status": "done", "summary": data}
        return {"status": "not_found"}

    # ------------------------------------------------------------------
    # Single evaluation (one split × with/without skill)
    # ------------------------------------------------------------------

    @router.post("/api/tasks/{task_id}/eval_single")
    async def run_single_evaluation(task_id: str, body: dict = None):
        """Run a single evaluation: one split with or without skill.

        Body params:
          - split: "val" | "test"  (default "val")
          - with_skill: bool       (default False)

        Returns per-item breakdown stats and comparison when both runs exist.
        """
        body = body or {}
        split = body.get("split", "val")
        with_skill = bool(body.get("with_skill", False))

        try:
            if state.engine.is_running:
                return {"error": "Cannot run eval while training is in progress."}

            state._maybe_restore_task(task_id)
            state.engine._ensure_out_dir()

            # Auto-load data from task directory if not already loaded
            if not state.engine.has_data():
                task_dir = state.train_root / task_id
                data_root = task_dir / "uploaded_data"
                if not data_root.is_dir():
                    for entry in sorted(state.train_root.iterdir()):
                        candidate = entry / "uploaded_data"
                        if candidate.is_dir():
                            data_root = candidate
                            break
                if data_root.is_dir():
                    from summerclaw.agent_trainer.datasets.loader import DataLoader
                    loader = DataLoader(str(data_root))
                    if loader.split_names:
                        state.engine.set_data_loader(loader)

            if not state.engine.has_data():
                return {"error": "No data loaded for this task. Upload data first."}

            # Load best skill from disk for this task
            task_dir = state.train_root / task_id
            best_skill_path = task_dir / "best_skill.md"
            if best_skill_path.is_file():
                content = best_skill_path.read_text(encoding="utf-8")
                state.engine._best_skill = content
                state.engine._current_skill = content
            else:
                skill_dir = task_dir / "skills"
                if skill_dir.is_dir():
                    files = sorted(skill_dir.glob("*.md"))
                    if files:
                        content = files[-1].read_text(encoding="utf-8")
                        state.engine._best_skill = content
                        state.engine._current_skill = content

            # Run single eval
            score = await state.engine.eval_only(
                split=split, with_skill=with_skill,
            )

            # Compute breakdown stats from per-item results
            per_item_results = getattr(state.engine.algorithm, '_last_rollout_results', [])
            stats = _compute_breakdown_stats(per_item_results)

            # Persist result to disk
            eval_dir = task_dir / "eval_single"
            eval_dir.mkdir(parents=True, exist_ok=True)
            result_key = f"{split}_{'with_skill' if with_skill else 'no_skill'}"
            result_file = eval_dir / f"{result_key}.json"

            # Save per-item results for comparison
            per_item_file = eval_dir / f"{result_key}_items.json"
            per_item_data = [
                {"id": r.id, "hard": r.hard, "fail_reason": r.fail_reason}
                for r in per_item_results
            ]
            _save_json(str(per_item_file), per_item_data)

            result_data = {
                "split": split,
                "with_skill": with_skill,
                "score": round(score, 6),
                "n_items": stats["n_total"],
                "stats": stats,
            }

            # Compute comparison when both skill/no_skill results exist for this split
            comparison = _compute_comparison(eval_dir, split)
            if comparison:
                result_data["comparison"] = comparison

            _save_json(str(result_file), result_data)

            return {"status": "done", "task_id": task_id, "result": result_data}
        except Exception as exc:
            return {"error": f"Single evaluation failed: {exc}"}

    @router.get("/api/tasks/{task_id}/eval_single")
    async def get_single_evaluation(task_id: str):
        """Return all cached single-eval results."""
        task_dir = state.train_root / task_id
        eval_dir = task_dir / "eval_single"
        results = {}
        if eval_dir.is_dir():
            for f in sorted(eval_dir.glob("*.json")):
                # Skip per-item data files (e.g. val_with_skill_items.json)
                if f.stem.endswith("_items"):
                    continue
                key = f.stem  # e.g. "val_with_skill"
                data = _load_json(str(f))
                if data:
                    results[key] = data
        return {"status": "done" if results else "not_found", "results": results}

    # ------------------------------------------------------------------
    # Deploy skill
    # ------------------------------------------------------------------

    @router.post("/api/tasks/{task_id}/deploy")
    async def deploy_skill(task_id: str, body: dict = None):
        body = body or {}
        skill_name = body.get("skill_name", "")
        if not skill_name:
            return {"error": "skill_name required"}
        try:
            target_dir = Path.home() / ".summerclaw" / "workspace" / "skills"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = str(target_dir / f"{skill_name}.md")
            # Restore task if needed
            state._maybe_restore_task(task_id)
            content = await state.engine.deploy_skill(target_path)
            return {"status": "deployed", "path": target_path, "chars": len(content)}
        except Exception as exc:
            return {"error": f"Deploy failed: {exc}"}

    # ------------------------------------------------------------------
    # YAML management
    # ------------------------------------------------------------------

    @router.get("/api/tasks/{task_id}/yaml")
    async def download_task_yaml(task_id: str):
        yaml_path = state.train_root / task_id / "skillopt.yaml"
        if not yaml_path.is_file():
            return {"error": "skillopt.yaml not found in task directory"}
        content = yaml_path.read_text(encoding="utf-8")
        return {"content": content, "filename": yaml_path.name}

    @router.post("/api/tasks/{task_id}/yaml")
    async def upload_task_yaml(task_id: str, body: dict = None):
        """Upload YAML content (JSON body with 'content' field)."""
        body = body or {}
        content = body.get("content", "")
        if not content:
            return {"error": "content required"}
        try:
            task_dir = state.train_root / task_id
            if not task_dir.is_dir():
                return {"error": f"Task directory not found: {task_id}"}
            dst = task_dir / "skillopt.yaml"
            # Validate YAML
            import yaml as _yaml
            try:
                _yaml.safe_load(content)
            except Exception as ve:
                return {"error": f"YAML syntax error: {ve}"}
            dst.write_text(content, encoding="utf-8")
            # If this task is currently loaded, re-apply to engine
            if str(task_dir) == str(state.engine.out_dir):
                state._apply_yaml_to_engine(task_dir)
            return {"status": "uploaded", "path": str(dst)}
        except Exception as exc:
            return {"error": f"Error uploading YAML: {exc}"}

    @router.get("/api/yaml/template")
    async def yaml_template():
        template = _find_default_yaml_template()
        if template:
            content = template.read_text(encoding="utf-8")
        else:
            content = _generate_default_yaml()
        return {"content": content, "filename": "skillopt.yaml"}
