"""Agent Trainer — pluggable skill optimization framework for SummerClaw.

A zero-intrusion training module that optimizes agent skills via
pluggable algorithms (e.g. SkillOpt) while keeping the online agent
running normally.

Key features:
  - **Data isolation**: Training uses an independent workspace
    (``train-outputs/<alg>-<task>/``) with its own memory store,
    session, and output files.
  - **Runtime consistency**: Training agent uses the same tools,
    memory algorithm type, and model config as the online agent.
  - **Zero intrusion**: No modifications to existing agent core files.
  - **Pluggable algorithms**: BaseAlgorithm + registry pattern.
  - **Channel-driven**: Start via ``/train <algorithm>`` from any channel.
  - **Dashboard**: Gradio web UI for monitoring and control.
"""
