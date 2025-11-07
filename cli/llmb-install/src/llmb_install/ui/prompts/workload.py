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

from typing import Any, Dict, List, Optional

from llmb_install.ui.interface import UIInterface


def prompt_workload_selection(
    ui: UIInterface,
    workloads: Dict[str, Dict[str, Any]],
    default_selected: Optional[List[str]] = None,
    show_install_all: bool = True,
    allow_empty: bool = False,
) -> Optional[List[str]]:
    """Prompt the user to select workloads to install.

    Args:
        ui: UI interface for user interaction
        workloads: Dictionary of available workloads
        default_selected: List of workload keys to pre-select
        show_install_all: Whether to show "Install all workloads" option
        allow_empty: Whether to allow selecting no workloads (for resume scenarios)

    Returns:
        Optional[List[str]]: List of selected workload keys, or None if cancelled
    """
    if not workloads:
        ui.log("No workloads found!")
        return None

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
