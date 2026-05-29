"""MOSCOPT algorithm — Mixture-of-Skill Collective Optimization.

Extends SkillOpt with multi-skill pool management, text-based gating,
and collective evolution mechanisms.  Backward compatible: when
pool_size=1 and activate_count=1, degenerates to standard SkillOpt.

6-stage per-step pipeline (inherited from SkillOpt):
  1. Rollout   — gate selects K skills → execute with activated skills
  2. Reflect   — analyze trajectories, generate skill + gate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — LLM-driven edit ranking + budget selection
  5. Update    — apply edits to skill pool or gate document
  6. Evaluate  — validation gate (accept/reject candidate)

Epoch-level hooks:
  - Slow Update — LLM-driven longitudinal analysis → protected skill region
  - Meta Skill  — cross-epoch optimizer memory → injected into all LLM calls
  - Collective Evolution — cull + breed + merge (every E epochs)

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
from summerclaw.agent_trainer.algorithms.moscopt.algorithm import MOSCOPTAlgorithm
from summerclaw.agent_trainer.algorithms.moscopt.pool import (
    SkillPool,
    distill_merged_skills,
    distill_top_skill,
    extract_routing_table,
    generate_diverse_pool,
    generate_gate_prompt,
    inject_foreign_gene,
    parse_pool,
    serialize_pool,
)
from summerclaw.agent_trainer.algorithms.skillopt.initial_skill import (
    generate_initial_skill_from_data,
    load_skill_from_file,
    resolve_initial_skill,
)

__all__ = [
    "MOSCOPTAlgorithm",
    "SkillPool",
    "distill_merged_skills",
    "distill_top_skill",
    "extract_routing_table",
    "generate_diverse_pool",
    "generate_gate_prompt",
    "inject_foreign_gene",
    "parse_pool",
    "serialize_pool",
    "generate_initial_skill_from_data",
    "load_skill_from_file",
    "resolve_initial_skill",
]
