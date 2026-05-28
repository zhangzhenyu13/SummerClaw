"""Algorithms, config, engine-level, and scheduler info routes."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter
    from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def register(router: APIRouter, state: _DashboardState) -> None:
    """Register algorithms / config / engine / scheduler routes on *router*."""

    # ------------------------------------------------------------------
    # Algorithms & tools
    # ------------------------------------------------------------------

    @router.get("/api/algorithms")
    async def list_algorithms():
        from summerclaw.agent_trainer.registry import list_algorithms
        algos = list_algorithms() or ["skillopt"]
        return {"algorithms": algos}

    @router.get("/api/memory-algorithms")
    async def list_memory_algorithms():
        return {"algorithms": [
            "naive_memory", "nemori_memory", "layerga_memory",
            "mem0v3_memory", "supermemory_memory", "hindsight_memory",
            "mastra_om_memory",
        ]}

    @router.get("/api/tools")
    async def list_tools():
        return {"categories": [
            {
                "key": "filesystem",
                "label": "文件系统 Filesystem",
                "default_excluded": False,
                "tools": ["read_file", "write_file", "edit_file", "list_dir"],
            },
            {
                "key": "search",
                "label": "搜索 Search",
                "default_excluded": False,
                "tools": ["glob", "grep"],
            },
            {
                "key": "shell",
                "label": "Shell 执行",
                "default_excluded": False,
                "tools": ["exec"],
            },
            {
                "key": "web",
                "label": "Web 工具",
                "default_excluded": False,
                "tools": ["web_search", "web_fetch"],
            },
            {
                "key": "browser",
                "label": "浏览器 Browser",
                "default_excluded": False,
                "tools": ["browser_search", "browser_fetch", "browser_navigate", "browser_snapshot", "browser_execute_js"],
            },
            {
                "key": "scheduling",
                "label": "调度 & 自动化",
                "default_excluded": False,
                "tools": ["spawn", "cron"],
            },
            {
                "key": "communication",
                "label": "通信 & 交互",
                "default_excluded": True,
                "tools": ["message", "ask_user"],
            },
            {
                "key": "self_state",
                "label": "自身状态 & 笔记本",
                "default_excluded": True,
                "tools": ["my", "notebook_edit"],
            },
            {
                "key": "dynamic",
                "label": "动态扩展 MCP",
                "default_excluded": True,
                "tools": ["mcp"],
            },
        ]}

    # ------------------------------------------------------------------
    # Config & status
    # ------------------------------------------------------------------

    @router.get("/api/config")
    async def get_config():
        cfg = getattr(state.engine, "_trainer_cfg", {})
        return {
            "epochs": cfg.get("num_epochs", 3),
            "batch_size": cfg.get("batch_size", 5),
            "learning_rate": cfg.get("edit_budget", 4),
            "seed": cfg.get("seed", 42),
            "lr_scheduler": cfg.get("lr_scheduler", cfg.get("lr_mode", "constant")),
            "update_mode": cfg.get("update_mode", cfg.get("skill_update_mode", "patch")),
            "slow_update": cfg.get("use_slow_update", True),
            "meta_skill": cfg.get("use_meta_skill", True),
            "reasoning_effort": cfg.get("reasoning_effort", "medium"),
            "task_dir": str(state.engine.out_dir),
            "has_data": state.engine.has_data(),
        }

    @router.get("/api/status")
    async def get_status():
        return {
            "is_running": state.engine.is_running,
            "current_score": state.engine.current_score if hasattr(state.engine, "_current_score") else 0,
            "best_score": state.engine.best_score,
            "best_step": state.engine._best_step,
            "total_steps": state.engine.history.total_steps,
            "total_epochs": state.engine.history.total_epochs,
            "task_dir": str(state.engine.out_dir),
            "has_data": state.engine.has_data(),
        }

    @router.get("/api/logs")
    async def get_logs():
        with state.engine._events_lock:
            recent = list(state.engine._events[-100:])
        return {"logs": recent}

    @router.get("/api/history")
    async def get_engine_history():
        return state.engine.history.to_dict()

    @router.get("/api/best_skill")
    async def get_best_skill():
        return {"content": state.engine.best_skill, "chars": len(state.engine.best_skill)}

    @router.get("/api/current_skill")
    async def get_current_skill():
        return {"content": state.engine.current_skill, "chars": len(state.engine.current_skill)}

    # ------------------------------------------------------------------
    # Engine-level control
    # ------------------------------------------------------------------

    @router.post("/api/cancel")
    async def cancel_engine():
        state.engine.request_cancel()
        return {"status": "cancel_requested"}

    @router.post("/api/start")
    async def start_engine(body: dict = None):
        body = body or {}
        task_id = body.get("task_id", "")
        if task_id:
            _train_root = getattr(state.engine, "_train_root", None)
            if _train_root is None:
                _train_root = Path.home() / ".summerclaw" / "train-algs"
            task_dir = Path(_train_root) / task_id
            if task_dir.is_dir() and str(task_dir) != str(state.engine.out_dir):
                state.engine._set_task_dir(task_dir)
        state.engine._ensure_out_dir()
        skill_path = body.get("skill_init_path", "")
        if skill_path:
            state.engine.skill_init_path = skill_path
            p = Path(skill_path).expanduser()
            if p.is_file():
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    state.engine.skill_init = content
                    state.engine._current_skill = content
                    state.engine._best_skill = content
        # Pin running_task_dir so _scan_all_tasks finds the right task
        _task_dir_str = str(state.engine.out_dir)
        for _sess in state.active_sessions.values():
            if _sess.get("engine") is state.engine:
                _sess["running_task_dir"] = _task_dir_str
        result = state.engine.start_training_async()
        return {"status": result}

    @router.post("/api/deploy")
    async def deploy_engine(body: dict = None):
        target = (body or {}).get("target_path", "")
        if not target:
            return {"error": "target_path required"}
        content = await state.engine.deploy_skill(target)
        return {"status": "deployed", "path": target, "chars": len(content)}

    # ------------------------------------------------------------------
    # Scheduler info
    # ------------------------------------------------------------------

    @router.get("/api/scheduler")
    async def get_scheduler_info():
        if not state.scheduler:
            return {"enabled": False}
        info = state.scheduler.status_info()
        info["enabled"] = True
        return info

    @router.post("/api/scheduler/stop")
    async def scheduler_stop():
        if state.scheduler:
            state.scheduler.stop()
            return {"status": "stopped"}
        return {"error": "scheduler not initialized"}

    @router.post("/api/scheduler/start")
    async def scheduler_start():
        if state.scheduler:
            state.scheduler.start()
            return {"status": "started"}
        return {"error": "scheduler not initialized"}
