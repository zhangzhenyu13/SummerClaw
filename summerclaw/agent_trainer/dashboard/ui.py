"""Gradio Blocks layout for the training dashboard.

Builds the two-level navigation:
  - **Task List**: search, filter, paginate all training tasks
  - **Task Detail**: training controls, data management, history, deploy

All mutable state and callbacks live in ``UIState`` (``ui_state.py``).
Data-tab-specific widgets live in ``ui_data.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore[assignment]

from loguru import logger

from summerclaw.agent_trainer.engine.trainer import TrainerEngine
from summerclaw.agent_trainer.dashboard.task_utils import _default_train_root
from summerclaw.agent_trainer.dashboard.ui_state import UIState
from summerclaw.agent_trainer.dashboard.ui_data import build_data_tab


def create_gradio_app(
    engine: TrainerEngine,
    train_root: Path | None = None,
    active_sessions: dict | None = None,
):
    """Create and return the Gradio Blocks application.

    Returns ``None`` if Gradio is not installed.
    """
    try:
        import gradio as gr
    except ImportError:
        logger.warning("Gradio not installed; dashboard UI disabled")
        return None

    if train_root is None:
        train_root = _default_train_root()
    if active_sessions is None:
        active_sessions = {}

    state = UIState(engine, train_root, active_sessions)

    # ── Build initial table for first render ────────────────────────────
    _init_tasks = state._get_filtered_tasks()
    _init_rows, _init_page, _init_tp, _init_s, _init_e, _init_total = state._paginate(_init_tasks, 1)
    _init_pinfo = state._page_info_text(_init_page, _init_tp, _init_s, _init_e, _init_total)

    with gr.Blocks(title="Agent Trainer Dashboard") as demo:

        # ── Shared state ────────────────────────────────────────────────
        selected_task_state = gr.State("")
        page_state = gr.State(1)

        # ═════════════════════════════════════════════════════════════════
        # Task List
        # ═════════════════════════════════════════════════════════════════
        gr.Markdown("# Agent Trainer Dashboard")

        with gr.Row():
            search_input = gr.Textbox(
                label="Search", placeholder="Search tasks...",
                scale=3, min_width=200,
            )
            status_filter = gr.Dropdown(
                choices=["all", "running", "completed", "idle"],
                value="all", label="Status", scale=1, min_width=120,
                allow_custom_value=True,
            )
            sort_field = gr.Dropdown(
                choices=["created", "best_score", "total_steps", "algorithm"],
                value="created", label="Sort by", scale=1, min_width=120,
                allow_custom_value=True,
            )
            sort_asc = gr.Checkbox(label="Ascending", value=False, scale=0)
            new_task_btn = gr.Button("+ New Task", variant="primary", scale=0, min_width=100)

        task_list_table = gr.Dataframe(
            headers=UIState._TABLE_HEADERS,
            value=_init_rows,
            interactive=False,
            wrap=True,
        )

        with gr.Row():
            page_info_md = gr.Markdown(_init_pinfo)
            with gr.Row(scale=0):
                prev_page_btn = gr.Button("< Prev", size="sm", interactive=False)
                next_page_btn = gr.Button("Next >", size="sm",
                                          interactive=_init_tp > 1)
            refresh_list_btn = gr.Button("Refresh", size="sm", scale=0)
            delete_btn = gr.Button("Delete", variant="stop", size="sm", scale=0)
            copy_btn = gr.Button("Copy", size="sm", scale=0)

        action_output = gr.Textbox(label="Action Result", visible=True)

        # ═════════════════════════════════════════════════════════════════
        # New Task (collapsible)
        # ═════════════════════════════════════════════════════════════════
        with gr.Accordion("Create New Task", open=False, visible=True) as new_task_acc:
            from summerclaw.agent_trainer.registry import list_algorithms as _list_algos
            _avail = _list_algos() or ["skillopt"]
            new_algo_dd = gr.Dropdown(
                choices=_avail, value=_avail[0] if _avail else None,
                label="Algorithm",
            )
            new_data_dir = gr.Textbox(label="Data Directory (optional)", placeholder="/path/to/train-data")
            with gr.Row():
                new_epochs = gr.Number(label="Epochs", value=3, precision=0)
                new_batch = gr.Number(label="Batch Size", value=5, precision=0)
                new_seed = gr.Number(label="Seed", value=42, precision=0)
            new_skill_path = gr.Textbox(
                label="Initial Skill Path (optional)",
                placeholder="/path/to/skill.md",
            )
            create_task_btn = gr.Button("Create Task", variant="primary")

        # ═════════════════════════════════════════════════════════════════
        # Task Detail (collapsible)
        # ═════════════════════════════════════════════════════════════════
        with gr.Accordion("Task Detail", open=True, visible=False) as detail_acc:
            task_detail_header_md = gr.Markdown(value="")
            task_info_md = gr.Markdown(value=state.get_task_info)

            with gr.Row():
                with gr.Column(scale=1):
                    status_md = gr.Markdown(value=state.get_status_text)
                    data_status_md = gr.Markdown(value=state.get_data_status)

                    # ── Config info notice ─────────────────────────────
                    cfg = getattr(engine, '_trainer_cfg', {})
                    _cfg_parts: list[str] = []
                    _cfg_parts.append(
                        f"**训练配置**: epochs={cfg.get('num_epochs', 3)}, "
                        f"batch={cfg.get('batch_size', 5)}, "
                        f"lr={cfg.get('edit_budget', 4)}, "
                        f"scheduler={cfg.get('lr_scheduler', cfg.get('lr_mode', 'constant'))}"
                    )
                    if cfg.get('use_slow_update'):
                        _cfg_parts.append("slow_update=ON")
                    if cfg.get('use_meta_skill'):
                        _cfg_parts.append("meta_skill=ON")
                    _cfg_notice = " | ".join(_cfg_parts)
                    gr.Markdown(
                        f"---\n"
                        f"{_cfg_notice}\n\n"
                        f"💡 修改项目根目录的 `skillopt.yaml` 可调整训练参数，保存后重启训练生效"
                    )

                    with gr.Row():
                        start_btn = gr.Button(
                            "Start Training", variant="primary",
                            interactive=not engine.is_running,
                        )
                        cancel_btn = gr.Button(
                            "Stop Training", variant="stop",
                            interactive=engine.is_running,
                        )
                    control_output = gr.Textbox(label="Status")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Data"):
                            build_data_tab(state)

                        with gr.Tab("History"):
                            history_table = gr.Dataframe(
                                headers=["Step", "Epoch", "Score", "Action",
                                         "Hash", "Edits", "Rejected"],
                                value=state.get_history_table,
                            )
                            score_plot = gr.LinePlot(
                                value=state.get_score_chart, x="Step", y="Score",
                                title="Score Progress",
                            )
                        with gr.Tab("Deploy"):
                            deploy_input = gr.Textbox(
                                label="Target Path",
                                placeholder="/path/to/skills/my_skill.md",
                            )
                            deploy_btn = gr.Button("Deploy Best Skill")
                            deploy_output = gr.Textbox(label="Result")

            gr.Markdown("---")
            gr.Markdown("### Training Log")
            log_window = gr.Textbox(
                value=state.get_log_lines, lines=15, max_lines=50,
                label="Output Log", interactive=False,
            )

        # ── Shared output / input lists ─────────────────────────────────

        _list_outs = [task_list_table, page_info_md,
                      prev_page_btn, next_page_btn, page_state]
        _filter_ins = [page_state, search_input, status_filter,
                       sort_field, sort_asc]
        _all_timer_outs = (
            [task_list_table, task_detail_header_md,
             status_md, data_status_md, task_info_md,
             history_table, score_plot, log_window,
             page_info_md, prev_page_btn, next_page_btn,
             page_state, detail_acc,
             start_btn, cancel_btn]
        )

        # ── Wire up events ──────────────────────────────────────────────

        # Task selection (Gradio 6.0+: no inputs on .select)
        task_list_table.select(
            state.on_task_select,
            outputs=[selected_task_state, task_detail_header_md, detail_acc],
        )

        # Training controls
        start_btn.click(
            state.do_start_training,
            inputs=[new_skill_path, selected_task_state],
            outputs=[control_output, start_btn, cancel_btn],
        )
        cancel_btn.click(
            state.do_cancel,
            outputs=[control_output, start_btn, cancel_btn],
        )

        # Deploy
        deploy_btn.click(state.do_deploy, inputs=[deploy_input],
                         outputs=[deploy_output])

        # Task management
        delete_btn.click(
            state.do_delete_task, inputs=[selected_task_state],
            outputs=[action_output],
        ).then(state.refresh_task_list, inputs=_filter_ins, outputs=_list_outs)

        copy_btn.click(
            state.do_copy_task, inputs=[selected_task_state],
            outputs=[action_output],
        ).then(state.refresh_task_list, inputs=_filter_ins, outputs=_list_outs)

        create_task_btn.click(
            state.do_create_task,
            inputs=[new_algo_dd, new_data_dir, new_epochs,
                    new_batch, new_seed, new_skill_path],
            outputs=[action_output],
        ).then(state.refresh_task_list, inputs=_filter_ins, outputs=_list_outs)

        # New task button toggles accordion
        new_task_btn.click(
            lambda: gr.update(open=True), outputs=[new_task_acc],
        )

        # Search / filter / sort changes -> reset to page 1 and refresh
        def _on_filter_change(search, status, sfield, sasc):
            return state.refresh_task_list(1, search, status, sfield, sasc)

        for comp in (search_input, status_filter, sort_field, sort_asc):
            comp.change(
                _on_filter_change,
                inputs=[search_input, status_filter, sort_field, sort_asc],
                outputs=_list_outs,
            )

        # Pagination
        prev_page_btn.click(
            lambda p, s, st, sf, sa: state.refresh_task_list(max(1, (p or 1) - 1), s, st, sf, sa),
            inputs=_filter_ins, outputs=_list_outs,
        )
        next_page_btn.click(
            lambda p, s, st, sf, sa: state.refresh_task_list((p or 1) + 1, s, st, sf, sa),
            inputs=_filter_ins, outputs=_list_outs,
        )

        refresh_list_btn.click(
            state.refresh_task_list, inputs=_filter_ins, outputs=_list_outs,
        )

        # Auto-refresh every 5 seconds using gr.Timer (Gradio 6.0+)
        refresh_timer = gr.Timer(5)
        refresh_timer.tick(
            state.do_refresh_all,
            inputs=[selected_task_state] + _filter_ins,
            outputs=_all_timer_outs,
        )

    return demo
