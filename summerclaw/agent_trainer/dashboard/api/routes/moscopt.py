"""MOSCOPT pool state routes — pool info, history, Q-scores.

Exposes MOSCOPT-specific data for dashboard visualization (Section 9.3):
  - GET /api/tasks/{task_id}/pool         — current pool snapshot
  - GET /api/tasks/{task_id}/pool/history  — per-epoch pool history
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter
    from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def _get_moscopt_algo(state: _DashboardState, task_id: str):
    """Return the MOSCOPTAlgorithm instance for a task, or None."""
    # Check scheduler-managed engines first
    if state.scheduler is not None:
        _te = state.scheduler.get_task_engine(task_id)
        if _te is not None:
            algo = _te.algorithm
            if getattr(algo, "name", "") == "moscopt":
                return algo

    # Fall back to active engine
    algo = state.engine.algorithm
    if getattr(algo, "name", "") == "moscopt":
        return algo
    return None


def register(router: "APIRouter", state: "_DashboardState") -> None:
    """Register MOSCOPT pool routes on *router*."""

    @router.get("/api/tasks/{task_id}/pool")
    async def get_pool(task_id: str):
        """Return current MOSCOPT pool state."""
        algo = _get_moscopt_algo(state, task_id)
        if algo is None:
            return {"error": "not_moscopt", "message": "Task is not using MOSCOPT algorithm"}

        pool = algo._pool
        return {
            "pool_size": pool.size,
            "n": pool.n,
            "k": pool.k,
            "epoch": pool.epoch,
            "q_scores": {sid: round(q, 4) for sid, q in pool.q_scores.items()},
            "activation_counts": dict(pool.activation_counts),
            "cooccurrence": {
                si: {sj: c for sj, c in partners.items()}
                for si, partners in pool.cooccurrence.items()
            },
            "gate": pool.gate[:500] if pool.gate else "",
            "summaries": {
                sid: {
                    "label": s.get("label", ""),
                    "q_score": round(s.get("q_score", 0.0), 4),
                    "activation_count": s.get("activation_count", 0),
                }
                for sid, s in pool.summaries.items()
            },
            "converged": algo.converged,
            "gating_granularity": algo.gating_granularity,
            "diversity_threshold": algo.diversity_threshold,
        }

    @router.get("/api/tasks/{task_id}/pool/history")
    async def get_pool_history(task_id: str):
        """Return per-epoch pool history for charting."""
        algo = _get_moscopt_algo(state, task_id)
        if algo is None:
            return {"error": "not_moscopt", "message": "Task is not using MOSCOPT algorithm"}

        history = getattr(algo, "_pool_history", [])
        # Format for charting: q_score curves per skill + pool size over time
        epochs = [h["epoch"] for h in history]

        # Build per-skill Q-score series
        all_sids: set[str] = set()
        for h in history:
            all_sids.update(h.get("q_scores", {}).keys())

        q_series = {}
        for sid in sorted(all_sids, key=lambda s: int(s) if s.isdigit() else s):
            q_series[sid] = [
                round(h.get("q_scores", {}).get(sid, 0.0), 4)
                for h in history
            ]

        # Pool size over time
        pool_sizes = [h.get("pool_size", 0) for h in history]

        # Skill membership changes
        membership = [
            {
                "epoch": h["epoch"],
                "skill_ids": h.get("skill_ids", []),
                "pool_size": h.get("pool_size", 0),
            }
            for h in history
        ]

        return {
            "epochs": epochs,
            "q_score_series": q_series,
            "pool_sizes": pool_sizes,
            "membership": membership,
            "converged_epochs": [
                h["epoch"] for h in history if h.get("converged")
            ],
        }
