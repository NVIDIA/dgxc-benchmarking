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


"""Installation configuration prompts for LLMB installer."""

import os
from pathlib import Path
from typing import Optional

from llmb_install.container.image import is_enroot_installed
from llmb_install.ui.interface import UIInterface


def prompt_install_location(ui: UIInterface, default: Optional[str] = None) -> Optional[str]:
    """Prompt the user for the installation location.

    Args:
        ui: UI interface for user interaction
        default: Default installation path to suggest

    Returns:
        Optional[str]: Installation path, or None if cancelled
    """
    ui.log("\nInstallation Location")
    ui.log("--------------------")
    ui.log("Note: Workloads can be quite large and may require significant disk space.")
    ui.log("It is recommended to install on a high-performance file system (e.g., Lustre, GPFS).")
    ui.log("Example paths: /lustre/your/username/llmb or /gpfs/your/username/llmb\n")

    def validate_path(path: str) -> str:
        if not path:
            return "valid"

        # Remove any quotes that might be present
        path = path.strip('"\'')

        # Skip validation for incomplete paths during typing
        if path.endswith(os.sep):
            return "valid"

        parent_dir = os.path.dirname(path)
        if parent_dir and not os.access(parent_dir, os.W_OK):
            return "invalid:No write permission in the parent directory."
        return "valid"

    location = ui.prompt_path(
        "Where would you like to install the workloads?", default=default or "", validate=validate_path
    )

    if location is None:
        return None

    # Remove any quotes that might be present
    location = location.strip('"\'')

    # Expand user path and get absolute path using pathlib
    location = str(Path(location).expanduser().resolve())

    ui.log(f"Installation location: {location}")
    return location


def prompt_install_method(ui: UIInterface, default: Optional[str] = None, express_mode: bool = False) -> str:
    """Prompt the user to select the installation method (local or slurm).

    Args:
        ui: UI interface for user interaction
        default: Default installation method from system config (if available)
        express_mode: Whether this is being called from express mode (shows default messages)

    Returns:
        str: The selected installation method ('local' or 'slurm')
    """
    ui.log("\nInstallation Method")
    ui.log("--------------------")
    ui.log(
        "Please select how you would like to perform longer running tasks like container image fetching and dataset downloads."
    )
    ui.log(" 'local': Tasks will be run directly on the current machine. Requires enroot to be available.")
    ui.log(
        " 'slurm': Tasks will be submitted as SLURM jobs. This is recommended for clusters where interactive nodes may have limited resources or network access."
    )

    # Check if we're running within a SLURM job
    running_in_slurm_job = 'SLURM_JOB_ID' in os.environ

    enroot_available = is_enroot_installed()

    if running_in_slurm_job:
        ui.log("\nDetected: Running within a SLURM job (SLURM_JOB_ID found in environment).")
        ui.log("SLURM installation method is not supported when running within a SLURM job.")

        if not enroot_available:
            ui.log("\nError: Cannot proceed with installation.")
            ui.log("You are running within a SLURM job, but enroot is not available on this system.")
            ui.log("Local installation requires enroot for container image downloading.")
            ui.log("Please either:")
            ui.log("  1. Run the installer from outside a SLURM job, or")
            ui.log("  2. Ensure enroot is available on the compute nodes")
            raise SystemExit(1)

        ui.log("Forcing local installation method.")
        return "local"

    # Determine available options
    options = []
    if enroot_available:
        options.append("local")
    options.append("slurm")  # SLURM is always available as an option

    if not enroot_available:
        ui.log("\nNote: Enroot is not available on this system.")
        ui.log("Local installation is not available. Using SLURM method.")
        return "slurm"

    # Prompt for choice
    # Use provided default if valid, otherwise no default
    if default and default in options:
        selected_default = default
        if express_mode:
            ui.log(f"Using saved default: {default}")
    else:
        selected_default = None

    selected = ui.prompt_select("Select installation method:", options, default=selected_default)

    if selected == "slurm":
        ui.log("\nNote: SLURM method will submit jobs for container downloads and other tasks.")
        ui.log("Ensure that compute nodes have internet access and enroot available.")
    else:
        ui.log("\nNote: Local method will run tasks directly on this machine.")
        ui.log("Ensure this machine has sufficient resources and network access.")

    return selected
