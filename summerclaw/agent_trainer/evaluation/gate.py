"""Validation gate — accept / reject candidate skills.

Analogous to validation-based early stopping and model selection in neural
network training: compares the candidate's score against the current and
best scores, then returns an accept/reject decision.

The trainer owns side-effects (cache lookup, rollout, printing, state
mutation).  This module is the pure decision function.
"""
from __future__ import annotations

from summerclaw.agent_trainer.types import GateAction, GateResult


def evaluate_gate(
    candidate_skill: str,
    cand_hard: float,
    current_skill: str,
    current_score: float,
    best_skill: str,
    best_score: float,
    best_step: int,
    global_step: int,
) -> GateResult:
    """Pure gate decision: compare candidate score to current/best.

    Returns a *GateResult* with updated state; the caller decides what
    to do with it (print, mutate trainer state, log, etc.).

    Parameters
    ----------
    candidate_skill : str
        The candidate skill document content.
    cand_hard : float
        Candidate hard accuracy on validation set.
    current_skill : str
        Current skill document content.
    current_score : float
        Current skill's hard accuracy.
    best_skill : str
        Best skill document content seen so far.
    best_score : float
        Best hard accuracy seen so far.
    best_step : int
        Global step at which the best score was achieved.
    global_step : int
        Current global step number.

    Returns
    -------
    GateResult
        Decision with updated skill/score state.
    """
    if cand_hard > current_score:
        new_current_skill = candidate_skill
        new_current_score = cand_hard
        if cand_hard > best_score:
            return GateResult(
                action="accept_new_best",
                current_skill=new_current_skill,
                current_score=new_current_score,
                best_skill=candidate_skill,
                best_score=cand_hard,
                best_step=global_step,
            )
        return GateResult(
            action="accept",
            current_skill=new_current_skill,
            current_score=new_current_score,
            best_skill=best_skill,
            best_score=best_score,
            best_step=best_step,
        )
    return GateResult(
        action="reject",
        current_skill=current_skill,
        current_score=current_score,
        best_skill=best_skill,
        best_score=best_score,
        best_step=best_step,
    )
