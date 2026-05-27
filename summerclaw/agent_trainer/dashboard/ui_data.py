"""Data Tab — independent module for data upload, scorer, and split settings.

The Data Tab is split into three independent sections that do not affect
each other:
  1. **Upload Data** (top): file upload + directory loading
  2. **Scorer Settings** (middle): scorer mode + custom script
  3. **Split Ratios** (bottom): train/val/test ratio config

Each section has its own confirm button and status indicator. If a section
was previously configured, its last value is displayed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from summerclaw.agent_trainer.dashboard.ui_state import UIState


def build_data_tab(state: UIState):
    """Build the Data Tab UI and wire up event handlers.

    Returns a dict of component references so the parent layout can
    include them in timer outputs if needed.
    """

    # ── Determine initial data status ────────────────────────────────────
    has_data = state.engine.has_data()
    _initial_data_info = _build_data_info(state) if has_data else ""

    # ═══════════════════════════════════════════════════════════════════════
    # Section 1: Upload Data (horizontal two-column layout)
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Row(equal_height=False):

        # ── Left column: Data status / Load from Directory ───────────────
        with gr.Column(scale=1, min_width=280):
            data_status_box = gr.Markdown(
                value=_initial_data_info,
                elem_classes=["data-status-box"],
            )

            # "Load from Directory" sub-section
            with gr.Accordion("Load from Directory", open=False):
                data_dir_input = gr.Textbox(
                    label="Data Directory Path",
                    placeholder="/path/to/data (train/items.json, val/items.json, ...)",
                )
                load_data_btn = gr.Button("Load Directory", size="sm")
                load_data_output = gr.Textbox(label="Result", lines=2, visible=True)

        # ── Right column: File Upload ───────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            with gr.Accordion("Upload Data", open=not has_data):
                gr.Markdown(
                    "Upload a file — the system will auto-split. "
                    "Optionally provide a separate test file."
                )
                with gr.Accordion("File Format", open=False):
                    gr.Markdown(
                        "**Supported:** `.json`, `.jsonl`, `.xlsx`\n\n"
                        "**Required fields:**\n"
                        "- `id` (string): unique identifier\n"
                        "- `question` (string): user input\n"
                        "- `answers` (list[string]): candidate answers\n"
                        "- `context` (string, optional)\n"
                        "- `scorer` (string, optional): `exact_match` / `llm_judge` / `custom`\n\n"
                        "**JSON example:**\n"
                        '```json\n[{"id":"t1","question":"What is 2+2?",'
                        '"answers":["4","four"],"scorer":"exact_match"}]\n```\n\n'
                        "**JSONL example:**\n"
                        '```\n{"id":"t1","question":"2+2?","answers":["4","four"]}\n```\n\n'
                        "**XLSX:** columns `id`, `question`, `answers` (JSON list)"
                    )
                main_file = gr.File(
                    label="Main Data File (required)",
                    file_types=[".json", ".jsonl", ".xlsx"],
                    type="filepath",
                )
                test_file = gr.File(
                    label="Test File (optional)",
                    file_types=[".json", ".jsonl", ".xlsx"],
                    type="filepath",
                )
                upload_btn = gr.Button("Upload & Split", variant="primary", size="sm")
                upload_output = gr.Textbox(label="Upload Result", lines=3)

    # ═══════════════════════════════════════════════════════════════════════
    # Section 2: Scorer Settings (independent)
    # ═══════════════════════════════════════════════════════════════════════
    gr.Markdown("---")
    with gr.Accordion("Scorer Settings", open=False):
        with gr.Row():
            scorer_mode = gr.Dropdown(
                choices=["exact_match", "llm_judge", "custom"],
                value=getattr(state.engine, '_pending_scorer_mode', 'exact_match'),
                label="Scorer Mode",
                info="Applied to items without a 'scorer' field.",
                scale=2,
            )
            scorer_file = gr.File(
                label="Custom Scorer Script (.py)",
                file_types=[".py"],
                type="filepath",
                visible=False,
                scale=1,
            )
        scorer_apply_btn = gr.Button("Apply Scorer", size="sm")
        scorer_status = gr.Textbox(label="Scorer Status", lines=1, interactive=False)

    # ═══════════════════════════════════════════════════════════════════════
    # Section 3: Split Ratios (independent)
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Accordion("Split Ratios", open=False):
        with gr.Row():
            # Restore previous values if available
            _prev = getattr(state.engine, '_pending_split', {})
            train_ratio = gr.Number(label="Train", value=_prev.get('train', 7), precision=0)
            val_ratio = gr.Number(label="Val", value=_prev.get('val', 2), precision=0)
            test_ratio = gr.Number(label="Test", value=_prev.get('test', 1), precision=0)
            split_seed = gr.Number(label="Seed", value=_prev.get('seed', 42), precision=0)
        split_apply_btn = gr.Button("Apply Split", size="sm")
        split_status = gr.Textbox(label="Split Status", lines=1, interactive=False)

    # ═══════════════════════════════════════════════════════════════════════
    # Wire events
    # ═══════════════════════════════════════════════════════════════════════

    # Scorer mode change -> show/hide custom scorer file
    # Gradio 6.x: .change() without explicit inputs auto-passes the component value
    scorer_mode.change(
        lambda mode: gr.update(visible=(mode == "custom")),
        outputs=[scorer_file],
    )

    # Upload & Split (uses current scorer + split settings)
    upload_btn.click(
        state.do_upload_file,
        inputs=[main_file, test_file, scorer_mode, scorer_file,
                train_ratio, val_ratio, test_ratio, split_seed],
        outputs=[upload_output],
    ).then(
        lambda: _build_data_info(state),
        outputs=[data_status_box],
    )

    # Load from directory
    load_data_btn.click(
        state.do_load_data,
        inputs=[data_dir_input],
        outputs=[load_data_output],
    ).then(
        lambda: _build_data_info(state),
        outputs=[data_status_box],
    )

    # Apply scorer independently
    scorer_apply_btn.click(
        state.do_apply_scorer,
        inputs=[scorer_mode, scorer_file],
        outputs=[scorer_status],
    )

    # Apply split independently
    split_apply_btn.click(
        state.do_apply_split,
        inputs=[train_ratio, val_ratio, test_ratio, split_seed],
        outputs=[split_status],
    )

    return {
        "data_status_box": data_status_box,
        "main_file": main_file,
        "test_file": test_file,
        "scorer_mode": scorer_mode,
        "scorer_file": scorer_file,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_seed": split_seed,
    }


def _build_data_info(state: UIState) -> str:
    """Build a Markdown string showing current data loading status."""
    if not state.engine.has_data():
        return "**Data**: Not loaded"

    loader = state.engine.data_loader
    summary = loader.summary()
    info = ", ".join(f"**{k}**={v}" for k, v in summary.items())

    # Try to show the data path
    data_path = getattr(loader, '_root', None) or getattr(loader, 'root', '')
    path_str = f"\n\n**Path**: `{data_path}`" if data_path else ""

    return f"**Data**: Loaded\n\n{info}{path_str}"
