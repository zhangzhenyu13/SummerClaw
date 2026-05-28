"""Concurrency-aware task scheduler for the training dashboard.

Implements a workers-budget scheduler that controls how many training
tasks run concurrently based on ``maxConcurrency`` from system config.
"""
from __future__ import annotations

import asyncio
import threading as _threading
from pathlib import Path

from loguru import logger

from summerclaw.agent_trainer.engine.trainer import TrainerEngine, _load_json, _save_json
from summerclaw.agent_trainer.dashboard.task_utils import _is_task_actually_running


class _TaskScheduler:
    """Workers-budget-aware task scheduler.

    maxConcurrency (= provider.max_concurrency) is the total workers budget
    pool.  New tasks sit idle for *idle_delay_s* seconds, then the scheduler
    decides whether to start them immediately (enough budget) or queue them.

    When a running task finishes, the scheduler tries to promote queued
    tasks in FIFO order.
    """

    IDLE_DELAY_S = 10.0
    TICK_S = 2.0

    def __init__(
        self,
        engine: TrainerEngine,
        train_root: Path,
        active_sessions: dict,
    ):
        self.engine = engine
        self.train_root = train_root
        self.active_sessions = active_sessions
        self._idle_pending: dict[str, float] = {}   # task_id → created_ts
        self._queued: list[str] = []                  # FIFO order
        self._bg_task: asyncio.Task | None = None
        self._running = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._bg_task is None or self._bg_task.done():
            self._running = True
            self._bg_task = asyncio.ensure_future(self._loop())

    def stop(self) -> None:
        self._running = False
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()

    async def _loop(self) -> None:
        logger.info("[Scheduler] started (maxConcurrency={})", self.max_concurrency)
        while self._running:
            try:
                await asyncio.sleep(self.TICK_S)
                if not self._running:
                    break
                self._tick_finished()
                await self._promote_queued()
                await self._check_idle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[Scheduler] tick error: {}", exc, exc_info=True)
        logger.info("[Scheduler] stopped")

    # -- public API -------------------------------------------------------

    @property
    def max_concurrency(self) -> int:
        """Read maxConcurrency from system config → provider → fallback 20.

        Priority:
          1. System config.json ``agents.defaults.maxConcurrency``
          2. ``engine.env.provider.max_concurrency`` (runtime value)
          3. Hardcoded fallback: 20
        """
        # 1) Try system config (authoritative source)
        try:
            from summerclaw.config.loader import load_config as _load_sys_cfg
            sys_cfg = _load_sys_cfg()
            mc = getattr(sys_cfg.agents.defaults, "max_concurrency", 0)
            if mc > 0:
                return mc
        except Exception:
            pass
        # 2) Fallback to provider runtime value
        env = getattr(self.engine, "env", None)
        if env is not None:
            provider = getattr(env, "provider", None)
            if provider:
                pmc = getattr(provider, "max_concurrency", 0) or 0
                if pmc > 0:
                    return pmc
        return 20

    def used_workers(self) -> int:
        """Sum of effective workers consumed by currently running tasks.

        Checks both active_sessions and PID+heartbeat fallback.
        """
        total = 0
        seen_dirs: set[str] = set()

        # 1) From active_sessions
        for info in self.active_sessions.values():
            rtd = info.get("running_task_dir")
            eng = info.get("engine")
            if rtd and eng and getattr(eng, "is_running", False):
                td = Path(rtd) if not isinstance(rtd, Path) else rtd
                td_str = str(td)
                if td_str not in seen_dirs:
                    seen_dirs.add(td_str)
                    total += self._effective_workers_for_dir(td)

        # 2) Fallback: scan train_root for tasks with fresh PID+heartbeat
        #    (covers cases where active_sessions is empty, e.g. after restart)
        if self.train_root.is_dir():
            for entry in self.train_root.iterdir():
                if not entry.is_dir() or entry.name.startswith("_"):
                    continue
                td_str = str(entry)
                if td_str in seen_dirs:
                    continue  # already counted
                if _is_task_actually_running(entry, self.active_sessions):
                    seen_dirs.add(td_str)
                    total += self._effective_workers_for_dir(entry)

        return total

    def left_budget(self) -> int:
        return max(0, self.max_concurrency - self.used_workers())

    def register_idle(self, task_id: str) -> None:
        import time as _t
        self._idle_pending[task_id] = _t.monotonic()
        logger.info("[Scheduler] registered idle task: {}", task_id)

    def unregister(self, task_id: str) -> None:
        self._idle_pending.pop(task_id, None)
        if task_id in self._queued:
            self._queued.remove(task_id)

    def on_task_finished(self, task_id: str) -> None:
        """Called when a training run finishes — triggers promotion."""
        self.unregister(task_id)
        logger.info("[Scheduler] task finished: {}, budget freed", task_id)
        # Schedule promotion on next tick (don't block the caller)

    def get_task_workers(self, task_id: str) -> int:
        return self._effective_workers_for_dir(self.train_root / task_id)

    def status_info(self) -> dict:
        # Per-running-task workers breakdown
        running_tasks: dict[str, int] = {}  # task_id -> effective_workers
        seen_dirs: set[str] = set()

        # 1) From active_sessions
        for info in self.active_sessions.values():
            rtd = info.get("running_task_dir")
            eng = info.get("engine")
            if rtd and eng and getattr(eng, "is_running", False):
                td = Path(rtd) if not isinstance(rtd, Path) else rtd
                td_str = str(td)
                if td_str not in seen_dirs:
                    seen_dirs.add(td_str)
                    tid = td.name
                    running_tasks[tid] = self._effective_workers_for_dir(td)

        # 2) Fallback: scan train_root for tasks with fresh PID+heartbeat
        if self.train_root.is_dir():
            for entry in self.train_root.iterdir():
                if not entry.is_dir() or entry.name.startswith("_"):
                    continue
                td_str = str(entry)
                if td_str in seen_dirs:
                    continue
                if _is_task_actually_running(entry, self.active_sessions):
                    seen_dirs.add(td_str)
                    running_tasks[entry.name] = self._effective_workers_for_dir(entry)

        return {
            "max_concurrency": self.max_concurrency,
            "used_workers": self.used_workers(),
            "left_budget": self.left_budget(),
            "idle_pending": dict(self._idle_pending),
            "queued": list(self._queued),
            "running_tasks": running_tasks,
        }

    # -- internals -------------------------------------------------------

    def _effective_workers_for_dir(self, task_dir: Path) -> int:
        config = _load_json(str(task_dir / "config.json")) or {}
        w = int(config.get("workers", 0))
        if w > 0:
            return w
        mc = self.max_concurrency
        return max(1, int(mc * 0.8)) if mc > 0 else 4

    def _tick_finished(self) -> None:
        """Detect finished running tasks and clean up."""
        for info in list(self.active_sessions.values()):
            rtd = info.get("running_task_dir")
            eng = info.get("engine")
            if rtd and eng and not getattr(eng, "is_running", False):
                tid = Path(rtd).name
                self.unregister(tid)
                logger.info("[Scheduler] detected task finished: {}", tid)

    async def _check_idle(self) -> None:
        import time as _t
        now = _t.monotonic()
        to_remove: list[str] = []
        for tid, ts in list(self._idle_pending.items()):
            if (now - ts) < self.IDLE_DELAY_S:
                continue
            # Verify task is still idle (not already started manually)
            task_dir = self.train_root / tid
            if not task_dir.is_dir():
                to_remove.append(tid)
                continue
            state = _load_json(str(task_dir / "runtime_state.json")) or {}
            if state.get("status") not in ("", None):
                to_remove.append(tid)
                continue  # already started or otherwise non-idle

            # Guard: do not auto-start until training data is available.
            # Re-arm the idle timer so we retry after data is uploaded.
            if not self._task_has_data(task_dir):
                self._idle_pending[tid] = now
                logger.debug(
                    "[Scheduler] skipping auto-start for {} (no data yet)", tid,
                )
                continue

            to_remove.append(tid)
            needed = self._effective_workers_for_dir(task_dir)
            if self.left_budget() >= needed:
                await self._start_task(tid)
            else:
                self._queued.append(tid)
                self._write_status(tid, "queued")
                logger.info(
                    "[Scheduler] queued {} (need {}w, budget left {}w)",
                    tid, needed, self.left_budget(),
                )
        for tid in to_remove:
            self._idle_pending.pop(tid, None)

    async def _promote_queued(self) -> None:
        promoted: list[str] = []
        for tid in list(self._queued):
            task_dir = self.train_root / tid
            if not task_dir.is_dir():
                promoted.append(tid)
                continue
            # Keep queued until data is available (don't promote to a guaranteed failure)
            if not self._task_has_data(task_dir):
                break  # FIFO — later items arrived no earlier, so stop
            needed = self._effective_workers_for_dir(task_dir)
            if self.left_budget() >= needed:
                self._queued.remove(tid)
                promoted.append(tid)
                await self._start_task(tid)
            else:
                break  # FIFO — stop at first that can't fit
        # Clean up removed entries
        for tid in promoted:
            if tid in self._queued:
                self._queued.remove(tid)

    def _task_has_data(self, task_dir: Path) -> bool:
        """Return True when the task directory has training data ready to load."""
        uploaded = task_dir / "uploaded_data"
        if uploaded.is_dir():
            try:
                for entry in uploaded.iterdir():
                    if entry.is_dir() and (entry / "items.json").is_file():
                        return True
            except OSError:
                pass
        # Also honor a data_dir reference in config.json (external data path)
        cfg = _load_json(str(task_dir / "config.json")) or {}
        data_dir = cfg.get("data_dir")
        if data_dir:
            p = Path(data_dir).expanduser()
            if p.is_dir():
                try:
                    for entry in p.iterdir():
                        if entry.is_dir() and (entry / "items.json").is_file():
                            return True
                except OSError:
                    pass
        return False

    def _write_status(self, task_id: str, status: str) -> None:
        task_dir = self.train_root / task_id
        rt_path = task_dir / "runtime_state.json"
        state = _load_json(str(rt_path)) or {}
        state["status"] = status
        _save_json(str(rt_path), state)

    async def _start_task(self, task_id: str) -> dict:
        """Programmatically start training for a task (scheduler path)."""
        logger.info("[Scheduler] starting task: {}", task_id)
        self._write_status(task_id, "running")

        try:
            task_dir = self.train_root / task_id
            engine = self.engine

            # Point engine at this task
            if str(task_dir) != str(engine.out_dir):
                engine._set_task_dir(task_dir)

            # Apply YAML config
            yaml_path = task_dir / "skillopt.yaml"
            if yaml_path.is_file():
                try:
                    from summerclaw.agent_trainer.config import load_config, flatten_config
                    cfg = load_config(str(yaml_path))
                    flat = flatten_config(cfg)
                    _CORE = {
                        "num_epochs": ("num_epochs", int),
                        "batch_size": ("batch_size", int),
                        "edit_budget": ("edit_budget", int),
                        "seed": ("seed", int),
                        "workers": ("workers", int),
                        "eval_test": ("eval_test", bool),
                    }
                    for fk, (attr, cast) in _CORE.items():
                        if fk in flat:
                            setattr(engine, attr, cast(flat[fk]))
                    env = getattr(engine, "env", None)
                    if env is not None and hasattr(env, "workers"):
                        yw = flat.get("workers", 0)
                        if yw > 0:
                            env.workers = int(yw)
                        else:
                            pm = getattr(env.provider, "max_concurrency", 0) or 0
                            env.workers = max(1, int(pm * 0.8)) if pm > 0 else 4

                    # Reconfigure memory algorithm and tools from task YAML
                    if env is not None and hasattr(env, "reconfigure_for_task"):
                        _mem_algo = flat.get("memory_algorithm")
                        if _mem_algo in (None, "", "null", "none", "Null", "None"):
                            _mem_algo = None
                        _enabled_tools = flat.get("enabled_tools")
                        if isinstance(_enabled_tools, list) and len(_enabled_tools) > 0:
                            pass  # keep as-is
                        else:
                            _enabled_tools = None  # None = all tools
                        env.reconfigure_for_task(_mem_algo, _enabled_tools)

                    # Apply algorithm-level params
                    algo = getattr(engine, "algorithm", None)
                    if algo is not None:
                        _ALGO = {
                            "lr_scheduler": ("lr_scheduler_type", str),
                            "lr_mode": ("lr_mode", str),
                            "edit_budget": ("edit_budget", int),
                            "min_edit_budget": ("min_lr", int),
                            "skill_update_mode": ("update_mode", str),
                            "update_mode": ("update_mode", str),
                            "use_slow_update": ("use_slow_update", bool),
                            "slow_update_samples": ("slow_update_samples", int),
                            "use_meta_skill": ("use_meta_skill", bool),
                            "longitudinal_pair_policy": ("longitudinal_pair_policy", str),
                            "minibatch_size": ("minibatch_size", int),
                            "merge_batch_size": ("merge_batch_size", int),
                            "max_analyst_rounds": ("max_analyst_rounds", int),
                            "analyst_workers": ("analyst_workers", int),
                            "aggregate_workers": ("aggregate_workers", int),
                            "evaluate_workers": ("evaluate_workers", int),
                            "reasoning_effort": ("reasoning_effort", str),
                            "rewrite_reasoning_effort": ("rewrite_reasoning_effort", str),
                            "rewrite_max_completion_tokens": ("rewrite_max_completion_tokens", int),
                        }
                        for fk, (attr, cast) in _ALGO.items():
                            if fk in flat and hasattr(algo, attr):
                                setattr(algo, attr, cast(flat[fk]))

                    engine._trainer_cfg.update(flat)
                    logger.info("[Scheduler] applied skillopt.yaml from {} to engine", task_dir)
                except Exception as exc:
                    logger.error("[Scheduler] Failed to apply skillopt.yaml from {}: {}", task_dir, exc, exc_info=True)

            engine._ensure_out_dir()
            engine._cancel_requested = False

            # Defensive: ensure data_loader is set from the task's uploaded_data.
            # _set_task_dir above normally restores it, but if the engine was
            # pointed at a different task in between, the loader may be None.
            if not getattr(engine, "data_loader", None):
                _uploaded = task_dir / "uploaded_data"
                if _uploaded.is_dir():
                    try:
                        from summerclaw.agent_trainer.datasets.loader import DataLoader
                        _loader = DataLoader(str(_uploaded))
                        if _loader.split_names:
                            engine.set_data_loader(_loader)
                            logger.info(
                                "[Scheduler] restored data_loader for {} : {}",
                                task_id, _loader.summary(),
                            )
                    except Exception as _dlexc:
                        logger.warning(
                            "[Scheduler] failed to restore data for {}: {}",
                            task_id, _dlexc,
                        )

            if not getattr(engine, "data_loader", None):
                # Don't launch a thread that would immediately fail.
                msg = (
                    f"No training data for task {task_id}. "
                    f"Upload data before starting training."
                )
                logger.error("[Scheduler] {}", msg)
                self._write_status(task_id, "failed")
                self.unregister(task_id)
                return {"error": msg}

            # Record running_task_dir
            _tds = str(engine.out_dir)
            for _sess in self.active_sessions.values():
                if _sess.get("engine") is engine:
                    _sess["running_task_dir"] = _tds

            def _run():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(engine.train())
                except Exception as exc:
                    logger.error(
                        "[Scheduler] training failed for {}: {}",
                        task_id, exc, exc_info=True,
                    )
                finally:
                    for _sess in self.active_sessions.values():
                        if _sess.get("engine") is engine:
                            _sess.pop("running_task_dir", None)
                            _sess.pop("stop_requested", None)
                    # Let the scheduler promote queued tasks now that budget is freed.
                    self.on_task_finished(task_id)
                    loop.close()

            t = _threading.Thread(target=_run, daemon=True)
            t.start()
            logger.info("[Scheduler] task {} started successfully", task_id)
            return {"status": "started", "task_id": task_id}

        except Exception as exc:
            logger.error("[Scheduler] failed to start {}: {}", task_id, exc, exc_info=True)
            self._write_status(task_id, "queued")
            if task_id not in self._queued:
                self._queued.append(task_id)
            return {"error": str(exc)}
