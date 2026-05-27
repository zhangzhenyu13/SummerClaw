"""SkillOpt algorithm — structured skill optimization via reflection.

6-stage per-step pipeline:
  1. Rollout   — execute episodes with current skill
  2. Reflect   — minibatch trajectory analysis → patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — LLM-driven edit ranking + budget selection
  5. Update    — apply selected edits to skill document
  6. Evaluate  — validation gate (accept/reject candidate)

Epoch-level hooks:
  - Slow Update — LLM-driven longitudinal analysis → protected skill region
  - Meta Skill  — cross-epoch optimizer memory → injected into all LLM calls

Supported update modes:
  - patch                    — standard incremental edits
  - rewrite_from_suggestions — suggestion-based rewrite proposals
  - full_rewrite_minibatch   — minibatch-level full skill rewrite candidates

LR schedulers:
  - constant   — fixed edit budget
  - linear     — linear decay from max to min
  - cosine     — cosine annealing from max to min
  - autonomous — LLM-driven adaptive edit budget
"""
from summerclaw.agent_trainer.algorithms.skillopt.algorithm import SkillOptAlgorithm
from summerclaw.agent_trainer.algorithms.skillopt.initial_skill import (
    generate_initial_skill_from_data,
    load_skill_from_file,
    resolve_initial_skill,
)

__all__ = [
    "SkillOptAlgorithm",
    "generate_initial_skill_from_data",
    "load_skill_from_file",
    "resolve_initial_skill",
]
