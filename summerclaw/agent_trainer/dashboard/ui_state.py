"""UI state management and callback functions for the dashboard.

Encapsulates all mutable state, event handlers, and data-fetching callbacks
used by the Gradio interface. Keeps the layout module (``ui.py``) free of
business logic.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
from datetime import datetime as _dt
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore[assignment]

from loguru import logger

from summerclaw.agent_trainer.engine.trainer import TrainerEngine, _load_json, _save_json
from summerclaw.agent_trainer.dashboard.task_utils import (
    _default_train_root,
    _load_task_history,
    _parse_task_created,
    _scan_all_tasks_cached,
)


class UIState:
    """Holds all mutable state and callback functions for the Gradio dashboard."""

    # ── Table constants ──────────────────────────────────────────────────
    _TABLE_HEADERS = [
        "Task ID", "Algorithm", "Status", "Steps",
        "Best Score", "Config", "Created",
    ]
    _PER_PAGE = 10

    def __init__(
        self,
        engine: TrainerEngine,
        train_root: Path | None = None,
        active_sessions: dict | None = None,
    ):
        self.engine = engine
        self.train_root = train_root or _default_train_root()
        self.active_sessions: dict = active_sessions or {}

        # Stop / completion tracking
        self._stop_requested: bool = False
        self._was_running: bool = False

        # Cached task list state (for Gradio 6 .select with no inputs)
        self._last_filtered_tasks: list[dict] = []
        self._last_page: int = 1

        # Wire progress callback
        engine.set_progress_callback(self._on_progress)

    # ── Progress hook (legacy) ──────────────────────────────────────────

    def _on_progress(self, event_type: str, payload: dict) -> None:
        """Called by TrainerEngine's progress callback (legacy hook)."""
        pass  # Events are now stored centrally on engine._events

    # ── Notification helper ─────────────────────────────────────────────

    def _fire_notify(self, message: str) -> None:
        """Push a notification to channels (thread-safe, best-effort)."""
        try:
            _nf = None
            for _sess in self.active_sessions.values():
                _nf = _sess.get("notify_fn")
                if _nf:
                    break
            if _nf:
                import asyncio as _aio
                _loop = _sess.get("main_loop")
                if _loop and _loop.is_running():
                    _aio.run_coroutine_threadsafe(_nf(message), _loop)
                else:
                    _loop2 = _aio.get_event_loop()
                    if _loop2.is_running():
                        _loop2.create_task(_nf(message))
        except Exception:
            pass  # best-effort, never block UI

    # ── Data display helpers ────────────────────────────────────────────

    def get_status_text(self) -> str:
        hist = self.engine.history
        if self._stop_requested and self.engine.is_running:
            _status = 'Stopping...'
        else:
            _status = 'Running' if self.engine.is_running else 'Idle'
        return (
            f"**Status**: {_status}\n\n"
            f"**Best Score**: {self.engine.best_score:.4f} (step {self.engine._best_step})\n\n"
            f"**Total Steps**: {hist.total_steps}\n\n"
            f"**Total Epochs**: {hist.total_epochs}\n\n"
        )

    def get_history_table(self) -> list[list]:
        steps = self.engine.history.steps
        if not steps:
            return []
        return [
            [s.step, s.epoch, f"{s.score:.4f}", s.action, s.skill_hash,
             s.n_edits_applied, s.n_edits_rejected]
            for s in steps
        ]

    def get_event_log(self) -> str:
        with self.engine._events_lock:
            recent = list(self.engine._events[-50:])
        lines = []
        for e in recent:
            ts = e.get('time', '')
            ev = e.get('event', '')
            if ev == 'log':
                level = e.get('level', 'INFO')
                tag = e.get('module', '')
                msg = e.get('message', '')
                lines.append(f"[{ts}] [{level}] {tag}: {msg}")
            else:
                extra = {k: v for k, v in e.items() if k not in ('time', 'event')}
                extra_str = json.dumps(extra, ensure_ascii=False) if extra else ""
                lines.append(f"[{ts}] {ev} {extra_str}")
        return "\n".join(lines)

    def get_score_chart(self) -> dict:
        steps = self.engine.history.steps
        if not steps:
            return {}
        return {
            "Step": [s.step for s in steps],
            "Score": [s.score for s in steps],
        }

    def get_data_status(self) -> str:
        if self.engine.has_data():
            summary = self.engine.data_loader.summary()
            info = ", ".join(f"{k}={v}" for k, v in summary.items())
            return f"**Data**: Loaded ({info})"
        return "**Data**: Not loaded"

    def get_task_info(self) -> str:
        return f"**Task Dir**: `{self.engine.out_dir}`"

    def get_log_lines(self) -> str:
        with self.engine._events_lock:
            recent = list(self.engine._events[-100:])
        lines = []
        for e in recent:
            ts = e.get('time', '')
            ev = e.get('event', '')
            if ev == 'log':
                level = e.get('level', 'INFO')
                tag = e.get('module', '')
                msg = e.get('message', '')
                lines.append(f"[{ts}] [{level}] {tag}: {msg}")
            else:
                extra = {k: v for k, v in e.items() if k not in ('time', 'event')}
                extra_str = json.dumps(extra, ensure_ascii=False) if extra else ""
                lines.append(f"[{ts}] {ev} {extra_str}")
        return "\n".join(lines)

    # ── Training control actions ─────────────────────────────────────────

    def do_cancel(self):
        self.engine.request_cancel()
        self._stop_requested = True
        task_id = Path(self.engine.out_dir).name
        msg = f"⏹ Training stop requested — task: {task_id}"
        print(msg)
        self._fire_notify(msg)
        return (
            "Cancel requested... (waiting for current step to finish)",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def _maybe_restore_task(self, selected_task_id: str) -> None:
        """Point the engine at the selected task if it differs from current."""
        if not selected_task_id:
            return
        selected_dir = self.train_root / selected_task_id
        if selected_dir.is_dir() and str(selected_dir) != str(self.engine.out_dir):
            self.engine._set_task_dir(selected_dir)

    def do_start_training(self, skill_path, selected_task_id):
        try:
            if self.engine.is_running:
                return "Training already in progress.", gr.update(), gr.update()
            self._maybe_restore_task(selected_task_id)
            if not self.engine.has_data():
                return "No training data loaded. Upload data first.", gr.update(), gr.update()
            self.engine._ensure_out_dir()
            if skill_path:
                self.engine.skill_init_path = skill_path
                p = Path(skill_path).expanduser()
                if p.is_file():
                    content = p.read_text(encoding="utf-8").strip()
                    if content:
                        self.engine.skill_init = content
                        self.engine._current_skill = content
                        self.engine._best_skill = content
            self.engine._cancel_requested = False
            self._stop_requested = False

            def _run():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.engine.train())
                except Exception as exc:
                    logger.error("Training failed: {}", exc, exc_info=True)
                finally:
                    loop.close()

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            task_id = Path(self.engine.out_dir).name
            msg = f"▶️ Training started — task: {task_id}"
            print(msg)
            self._fire_notify(msg)
            return (
                "Training started.",
                gr.update(interactive=False),
                gr.update(interactive=True),
            )
        except Exception as exc:
            return f"Error starting training: {exc}", gr.update(), gr.update()

    # ── Data actions ────────────────────────────────────────────────────

    def do_load_data(self, data_dir):
        try:
            if not data_dir:
                return "Error: data directory path required"
            path = Path(data_dir).expanduser()
            if not path.exists():
                return f"Error: directory not found: {data_dir}"
            from summerclaw.agent_trainer.datasets.loader import DataLoader
            loader = DataLoader(str(path))
            if not loader.split_names:
                return "Error: no valid splits found (need train/items.json at minimum)"
            self.engine._ensure_out_dir()
            self.engine.set_data_loader(loader)
            summary = loader.summary()
            info = ", ".join(f"{k}={v}" for k, v in summary.items())
            return f"Data loaded: {info}"
        except Exception as exc:
            return f"Error loading data: {exc}"

    def do_upload_file(self, main_file, test_file, scorer_mode, scorer_file,
                       train_ratio, val_ratio, test_ratio, seed):
        """Upload training file(s) and auto-split."""
        try:
            from summerclaw.agent_trainer.datasets.splitter import (
                parse_file, split_items, split_with_test, write_splits,
            )
            from summerclaw.agent_trainer.datasets.loader import DataLoader

            if not main_file:
                return "Error: please upload a main data file (.json, .jsonl, or .xlsx)"

            if scorer_mode == "custom" and not scorer_file:
                return "Error: scorer is set to 'custom' — please upload a custom-scorer.py file"

            main_path = Path(main_file.name) if hasattr(main_file, 'name') else Path(main_file)
            try:
                main_items = parse_file(main_path)
            except (ValueError, ImportError) as e:
                return f"Error parsing main file: {e}"

            if not main_items:
                return "Error: main file is empty"

            if scorer_mode and scorer_mode != "exact_match":
                for item in main_items:
                    if not item.get("scorer"):
                        item["scorer"] = scorer_mode

            self.engine._ensure_out_dir()

            if scorer_file:
                scorer_src = Path(scorer_file.name) if hasattr(scorer_file, 'name') else Path(scorer_file)
                scorer_dst = self.engine.out_dir / "custom-scorer.py"
                shutil.copy2(str(scorer_src), str(scorer_dst))

            try:
                r_train = float(train_ratio) if train_ratio else 7.0
                r_val = float(val_ratio) if val_ratio else 2.0
                r_test = float(test_ratio) if test_ratio else 1.0
            except ValueError:
                return "Error: split ratios must be numbers"

            try:
                r_seed = int(seed) if seed else 42
            except ValueError:
                r_seed = 42

            if test_file:
                test_path = Path(test_file.name) if hasattr(test_file, 'name') else Path(test_file)
                try:
                    test_items = parse_file(test_path)
                except (ValueError, ImportError) as e:
                    return f"Error parsing test file: {e}"
                splits = split_with_test(
                    main_items, test_items,
                    train_ratio=r_train, val_ratio=r_val, seed=r_seed,
                )
                split_info = f"main({len(main_items)}) → train:val = {r_train}:{r_val} | test({len(test_items)}) separate"
            else:
                splits = split_items(
                    main_items,
                    train_ratio=r_train, val_ratio=r_val, test_ratio=r_test,
                    seed=r_seed,
                )
                split_info = f"auto-split → train:val:test = {r_train}:{r_val}:{r_test}"

            data_root = self.engine.out_dir / "uploaded_data"
            summary = write_splits(splits, data_root)

            loader = DataLoader(str(data_root))
            if loader.split_names:
                self.engine.set_data_loader(loader)

            parts = [f"{k}={v}" for k, v in summary.items()]
            return f"Loaded {len(main_items)} items. {split_info}\nSplits: {', '.join(parts)}"
        except Exception as exc:
            return f"Error uploading file: {exc}"

    def do_apply_scorer(self, scorer_mode, scorer_file):
        """Apply scorer setting independently (without uploading data)."""
        try:
            if scorer_mode == "custom" and not scorer_file:
                return "Error: scorer is 'custom' — upload a custom-scorer.py file"

            self.engine._ensure_out_dir()

            if scorer_file:
                scorer_src = Path(scorer_file.name) if hasattr(scorer_file, 'name') else Path(scorer_file)
                scorer_dst = self.engine.out_dir / "custom-scorer.py"
                shutil.copy2(str(scorer_src), str(scorer_dst))

            # Persist scorer setting to engine for later use during upload
            self.engine._pending_scorer_mode = scorer_mode
            return f"Scorer set to: {scorer_mode}" + (" (custom script saved)" if scorer_file else "")
        except Exception as exc:
            return f"Error applying scorer: {exc}"

    def do_apply_split(self, train_ratio, val_ratio, test_ratio, seed):
        """Apply split ratio setting independently."""
        try:
            r_train = float(train_ratio) if train_ratio else 7.0
            r_val = float(val_ratio) if val_ratio else 2.0
            r_test = float(test_ratio) if test_ratio else 1.0
            r_seed = int(seed) if seed else 42

            # Persist to engine for later use during upload
            self.engine._pending_split = {
                "train": r_train, "val": r_val, "test": r_test, "seed": r_seed,
            }
            return f"Split ratios saved: train={r_train}, val={r_val}, test={r_test}, seed={r_seed}"
        except Exception as exc:
            return f"Error applying split: {exc}"

    def do_deploy(self, target_path):
        if not target_path:
            return "Error: target path required"
        try:
            loop = asyncio.new_event_loop()
            content = loop.run_until_complete(self.engine.deploy_skill(target_path))
            loop.close()
            return f"Deployed to {target_path} ({len(content)} chars)"
        except Exception as exc:
            return f"Deploy failed: {exc}"

    # ── Task list helpers ───────────────────────────────────────────────

    def _tasks_to_rows(self, tasks):
        rows = []
        for t in tasks:
            bs = t["best_score"]
            bs_str = f"{bs:.4f}" if isinstance(bs, (int, float)) and bs >= 0 else "\u2014"
            rows.append([
                t["task_id"], t["algorithm"], t["status"],
                t["total_steps"], bs_str,
                f"{t['epochs']}e / bs={t['batch_size']}", t["created"],
            ])
        return rows

    def _get_filtered_tasks(self, search="", status_filter="all",
                            sort_field="created", sort_asc=False):
        if not status_filter:
            status_filter = "all"
        if not sort_field:
            sort_field = "created"
        tasks = _scan_all_tasks_cached(self.train_root, self.active_sessions)
        if search:
            q = search.lower()
            tasks = [t for t in tasks if
                     q in t["task_id"].lower() or
                     q in t["algorithm"].lower() or
                     q in t.get("notes", "").lower()]
        if status_filter and status_filter != "all":
            tasks = [t for t in tasks if t["status"] == status_filter]
        if sort_field in ("created", "best_score", "total_steps", "algorithm"):
            tasks.sort(key=lambda x: x.get(sort_field, ""),
                       reverse=not sort_asc)
        self._last_filtered_tasks.clear()
        self._last_filtered_tasks.extend(tasks)
        return tasks

    def _paginate(self, tasks, page):
        total = len(tasks)
        total_pages = max(1, -(-total // self._PER_PAGE))
        page = max(1, min(page, total_pages))
        start = (page - 1) * self._PER_PAGE
        end = min(start + self._PER_PAGE, total)
        return self._tasks_to_rows(tasks[start:end]), page, total_pages, start, end, total

    def _page_info_text(self, page, total_pages, start, end, total):
        if total == 0:
            return "No tasks found"
        return f"Page {page}/{total_pages} | Showing {start+1}\u2013{end} of {total}"

    def _page_btn_states(self, page, total_pages):
        return (
            gr.update(interactive=page > 1),
            gr.update(interactive=page < total_pages),
        )

    # ── Task detail helpers ─────────────────────────────────────────────

    def get_task_detail_md(self, task_id):
        if not task_id:
            return ""
        task_dir = self.train_root / task_id
        config = _load_json(str(task_dir / "config.json")) or {}
        state = _load_json(str(task_dir / "runtime_state.json")) or {}
        has_history = (task_dir / "history.json").exists()
        created = _parse_task_created(task_id)
        bs = state.get("best_score", -1)
        bs_str = f"{bs:.4f}" if isinstance(bs, (int, float)) and bs >= 0 else "\u2014"
        total_steps = state.get("last_completed_step", 0)
        total_epochs = config.get("num_epochs", 0) if has_history else 0
        return (
            f"**Task**: `{task_id}`\n\n"
            f"**Algorithm**: {config.get('algorithm', '?')}\n\n"
            f"**Created**: {created}\n\n"
            f"**Status**: {'Completed' if has_history else 'Idle'}\n\n"
            f"**Best Score**: {bs_str}\n\n"
            f"**Total Steps**: {total_steps}\n\n"
            f"**Total Epochs**: {total_epochs}\n\n"
        )

    def get_readonly_history(self, task_id):
        if not task_id:
            return []
        hist = _load_task_history(self.train_root / task_id)
        steps = hist.get("steps", [])
        if not steps:
            return []
        return [
            [s.get("step"), s.get("epoch"), f"{s.get('score', 0):.4f}",
             s.get("action"), s.get("skill_hash", ""),
             s.get("n_edits_applied", 0), s.get("n_edits_rejected", 0)]
            for s in steps
        ]

    def get_readonly_score_chart(self, task_id):
        if not task_id:
            return {}
        hist = _load_task_history(self.train_root / task_id)
        steps = hist.get("steps", [])
        if not steps:
            return {}
        return {
            "Step": [s.get("step") for s in steps],
            "Score": [s.get("score", 0) for s in steps],
        }

    # ── Task management actions ─────────────────────────────────────────

    def do_delete_task(self, selected_task_id):
        try:
            if not selected_task_id:
                return "Error: no task selected"
            task_dir = self.train_root / selected_task_id
            if not task_dir.exists():
                return f"Error: task directory not found: {selected_task_id}"
            shutil.rmtree(str(task_dir))
            return f"Task **{selected_task_id}** deleted."
        except Exception as exc:
            return f"Error deleting task: {exc}"

    def do_copy_task(self, selected_task_id):
        try:
            if not selected_task_id:
                return "Error: no task selected"
            src = self.train_root / selected_task_id
            if not src.exists():
                return f"Error: task directory not found: {selected_task_id}"
            ts = _dt.now().strftime("%Y%m%d-%H%M%S")
            algo = selected_task_id.rsplit("-", 2)[0] if "-" in selected_task_id else selected_task_id
            new_id = f"{algo}-copy-{ts}"
            dst = self.train_root / new_id
            shutil.copytree(str(src), str(dst))
            return f"Task copied to **{new_id}**"
        except Exception as exc:
            return f"Error copying task: {exc}"

    def do_create_task(self, algorithm, data_dir, epochs, batch_size, seed, skill_path):
        try:
            if not algorithm:
                return "Error: select an algorithm"
            ts = _dt.now().strftime("%Y%m%d-%H%M%S")
            task_id = f"{algorithm}-{ts}"
            task_dir = self.train_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "algorithm": algorithm,
                "num_epochs": int(epochs) if epochs else 3,
                "batch_size": int(batch_size) if batch_size else 5,
                "seed": int(seed) if seed else 42,
            }
            if data_dir:
                cfg["data_dir"] = data_dir
            if skill_path:
                cfg["skill_init"] = skill_path
            _save_json(str(task_dir / "config.json"), cfg)
            (task_dir / "skills").mkdir(exist_ok=True)
            if skill_path and Path(skill_path).exists():
                shutil.copy2(skill_path, str(task_dir / "skills" / "skill_v0000.md"))
            return f"Task **{task_id}** created at `{task_dir}`"
        except Exception as exc:
            return f"Error creating task: {exc}"

    # ── Refresh / pagination ────────────────────────────────────────────

    def refresh_task_list(self, page, search, status_filter, sort_field, sort_asc):
        tasks = self._get_filtered_tasks(search, status_filter, sort_field, sort_asc)
        rows, cur_page, total_pages, start, end, total = self._paginate(tasks, page or 1)
        self._last_page = cur_page
        return (
            rows,
            self._page_info_text(cur_page, total_pages, start, end, total),
            *self._page_btn_states(cur_page, total_pages),
            cur_page,
        )

    def on_task_select(self, evt: gr.SelectData):
        """Gradio 6.0+: .select() has no inputs; use cached task list."""
        row_idx = evt.index[0] if evt.index else 0
        offset = (self._last_page - 1) * self._PER_PAGE
        idx = offset + row_idx
        if not self._last_filtered_tasks or idx >= len(self._last_filtered_tasks):
            return "", "", gr.update(visible=False)
        tid = self._last_filtered_tasks[idx]["task_id"]
        return tid, self.get_task_detail_md(tid), gr.update(visible=True)

    def do_refresh_all(self, selected_task_id, page, search, status_filter,
                       sort_field, sort_asc):
        tasks = self._get_filtered_tasks(search, status_filter, sort_field, sort_asc)
        rows, cur_page, total_pages, start, end, total = self._paginate(tasks, page or 1)
        pinfo = self._page_info_text(cur_page, total_pages, start, end, total)
        prev_s, next_s = self._page_btn_states(cur_page, total_pages)

        running = self.engine.is_running

        # Detect training completion
        if self._was_running and not running:
            task_id = Path(self.engine.out_dir).name
            if self._stop_requested:
                msg = f"⏹ Training stopped — task: {task_id} (best={self.engine.best_score:.4f}, steps={self.engine.history.total_steps})"
            else:
                msg = f"✅ Training completed — task: {task_id} (best={self.engine.best_score:.4f}, steps={self.engine.history.total_steps})"
            print(msg)
            self._fire_notify(msg)
        self._was_running = running

        if self._stop_requested:
            if running:
                start_btn_state = gr.update(interactive=False)
                cancel_btn_state = gr.update(interactive=False)
            else:
                self._stop_requested = False
                start_btn_state = gr.update(interactive=True)
                cancel_btn_state = gr.update(interactive=False)
        else:
            start_btn_state = gr.update(interactive=not running)
            cancel_btn_state = gr.update(interactive=running)

        if not selected_task_id:
            return (rows, "", self.get_status_text(), self.get_data_status(),
                    self.get_task_info(), self.get_history_table(),
                    self.get_score_chart(), self.get_log_lines(),
                    pinfo, prev_s, next_s, cur_page,
                    gr.update(visible=False),
                    start_btn_state, cancel_btn_state)

        active_dir = str(self.engine.out_dir)
        selected_dir = str(self.train_root / selected_task_id)
        detail_md = self.get_task_detail_md(selected_task_id)

        if selected_task_id and not self.engine.has_data():
            self._maybe_restore_task(selected_task_id)

        if selected_dir == active_dir or self.engine.has_data():
            return (rows, detail_md, self.get_status_text(), self.get_data_status(),
                    self.get_task_info(), self.get_history_table(),
                    self.get_score_chart(), self.get_log_lines(),
                    pinfo, prev_s, next_s, cur_page,
                    gr.update(visible=True),
                    start_btn_state, cancel_btn_state)

        hist = _load_task_history(self.train_root / selected_task_id)
        steps = hist.get("steps", [])
        hist_rows = [
            [s.get("step"), s.get("epoch"), f"{s.get('score', 0):.4f}",
             s.get("action"), s.get("skill_hash", ""),
             s.get("n_edits_applied", 0), s.get("n_edits_rejected", 0)]
            for s in steps
        ]
        chart = ({"Step": [s.get("step") for s in steps],
                  "Score": [s.get("score", 0) for s in steps]}
                 if steps else {})
        return (rows, detail_md,
                "**Status**: Read-only (historical task)",
                "**Data**: N/A (historical)",
                f"**Task Dir**: `{selected_dir}`",
                hist_rows, chart, "",
                pinfo, prev_s, next_s, cur_page,
                gr.update(visible=True),
                start_btn_state, cancel_btn_state)
