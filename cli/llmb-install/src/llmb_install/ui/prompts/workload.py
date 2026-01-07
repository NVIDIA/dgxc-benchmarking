# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


"""Workload selection prompts for LLMB installer."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llmb_install.core.exemplar import get_exemplar_workloads, validate_exemplar_workloads
from llmb_install.ui.interface import UIInterface


def prompt_workload_selection(
    ui: UIInterface,
    workloads: Dict[str, Dict[str, Any]],
    default_selected: Optional[List[str]] = None,
    show_install_all: bool = True,
    allow_empty: bool = False,
    show_exemplar_option: bool = False,
    default_mode: str = 'custom',
    llmb_repo: Optional[Path] = None,
    gpu_type: Optional[str] = None,
) -> Tuple[Optional[List[str]], str]:
    """Prompt the user to select workloads to install.

    Args:
        ui: UI interface for user interaction
        workloads: Dictionary of available workloads
        default_selected: List of workload keys to pre-select
        show_install_all: Whether to show "Install all workloads" option
        allow_empty: Whether to allow selecting no workloads (for resume scenarios)
        show_exemplar_option: Whether to show "Exemplar Cloud" selection option
        default_mode: Default selection mode ('custom' or 'exemplar')
        llmb_repo: Path to LLMB repository root (required for exemplar mode)
        gpu_type: GPU type (required for exemplar mode)

    Returns:
        Tuple[Optional[List[str]], str]: (List of selected workload keys or None if cancelled, selection mode)
    """
    if not workloads:
        ui.log("No workloads found!")
        return None, 'custom'

    selection_mode = 'custom'

    if show_exemplar_option:
        ui.print_section("Workload Selection")

        choice = ui.prompt_select(
            "Workload Selection Mode:",
            choices=[
                {
                    'name': "Exemplar Cloud (Install all recipes required for Exemplar Cloud certification)",
                    'value': "exemplar",
                },
                {'name': "Custom (Manually select specific recipes)", 'value': "custom"},
            ],
            default=default_mode,
        )

        if choice is None:
            return None, 'custom'  # User cancelled

        selection_mode = choice

        if choice == "exemplar":
            # Use exemplar.yaml to select workloads
            if not llmb_repo or not gpu_type:
                ui.log("Error: llmb_repo and gpu_type are required for Exemplar Cloud mode", level='error')
                return None, 'custom'

            try:
                # Convert to Path if it's a string
                llmb_repo_path = Path(llmb_repo) if isinstance(llmb_repo, str) else llmb_repo
                base_keys = get_exemplar_workloads(llmb_repo_path, gpu_type)
                selected = validate_exemplar_workloads(base_keys, workloads, gpu_type)
                return selected, selection_mode
            except ValueError as e:
                ui.log(f"Error: {e}", level='error')
                ui.log("")  # Blank line for readability
                ui.log("Falling back to custom selection.")
                selection_mode = 'custom'

    selected = _prompt_manual_workload_selection(ui, workloads, default_selected, show_install_all, allow_empty)

    return selected, selection_mode


def _prompt_manual_workload_selection(
    ui: UIInterface,
    workloads: Dict[str, Dict[str, Any]],
    default_selected: Optional[List[str]] = None,
    show_install_all: bool = True,
    allow_empty: bool = False,
) -> Optional[List[str]]:
    """Handle manual checkbox selection of workloads."""
    choices = []

    # Add "Install all workloads" option if requested
    if show_install_all:
        choices.append({'name': "Install all workloads", 'value': "all"})

    for key, _data in sorted(workloads.items()):
        choices.append({'name': key, 'value': key})

    def validate_selection(selected: List[str]) -> str:
        if len(selected) > 0 or allow_empty:
            return "valid"
        return "invalid:Please select at least one workload"

    # Use default selections if provided, otherwise standard prompt
    prompt_message = "Select workloads to install (use space to select, enter to confirm):"

    # Determine which defaults are actually available in the workloads
    actual_defaults = []
    if default_selected:
        actual_defaults = [w for w in default_selected if w in workloads]

    selected = ui.prompt_checkbox(
        prompt_message,
        choices=choices,
        validate=validate_selection,
        default=actual_defaults,
    )

    if selected is None:
        return None  # User cancelled

    if "all" in selected:
        return list(workloads.keys())

    return selected  # Return empty list if no selections (when allow_empty=True)
