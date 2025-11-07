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


"""Environment configuration prompts for LLMB installer."""

import os
import subprocess
import sys
from typing import Dict, Optional

from llmb_install.constants import MAX_PYTHON_VERSION_TUPLE, MIN_PYTHON_VERSION_TUPLE
from llmb_install.environment.detector import (
    detect_virtual_environment,
    get_system_python_path,
    get_system_python_version,
    has_active_conda_environment,
    is_conda_installed,
    is_uv_installed,
    is_venv_installed,
)
from llmb_install.environment.venv_manager import get_clean_environment_for_subprocess
from llmb_install.ui.interface import UIInterface


def prompt_environment_type(ui: UIInterface, default: Optional[str] = None, express_mode: bool = False) -> str:
    """Prompt the user to select their preferred environment type (uv, venv, or conda).

    Args:
        ui: UI interface for user interaction
        default: Default environment type from system config (if available)
        express_mode: Whether this is being called from express mode (shows default messages)

    Returns:
        str: Selected environment type ('uv', 'venv', or 'conda')
    """
    # Detect system Python version (clean environment)
    if detect_virtual_environment():
        env = get_clean_environment_for_subprocess()
        current_python_version = get_system_python_version(env)
        system_python_path = get_system_python_path(env)
        if current_python_version is not None:
            venv_available = (
                subprocess.run(['python3', '-m', 'venv', '--help'], env=env, capture_output=True, text=True).returncode
                == 0
            )
        else:
            venv_available = False  # No system Python detected
    else:
        current_python_version = sys.version_info[:3]
        system_python_path = sys.executable
        venv_available = is_venv_installed()

    # Environment availability checks
    conda_available = is_conda_installed()
    uv_available = is_uv_installed()

    # Check compatibility
    if current_python_version is not None:
        python_compatible = MIN_PYTHON_VERSION_TUPLE <= current_python_version < MAX_PYTHON_VERSION_TUPLE
        can_use_venv = venv_available and python_compatible
    else:
        python_compatible = False
        can_use_venv = False  # Can't use venv if no system Python detected
    can_use_conda = conda_available
    can_use_uv = uv_available

    # Print environment status
    ui.log("Environment Configuration")
    ui.log("------------------------")
    if current_python_version is not None:
        ui.log(f"System Python version: {'.'.join(map(str, current_python_version))}")
        ui.log(f"System Python path: {system_python_path}")
    else:
        ui.log("System Python version: Not detected")

    ui.log(
        f"Supported version range: [{'.'.join(map(str, MIN_PYTHON_VERSION_TUPLE))}, {'.'.join(map(str, MAX_PYTHON_VERSION_TUPLE))})"
    )
    ui.log("")

    # Determine available options
    options = []
    if can_use_uv:
        options.append("uv")
    if can_use_venv:
        options.append("venv")
    if can_use_conda:
        options.append("conda")

    if not options:
        if not python_compatible and current_python_version is not None:
            ui.log(f"Error: System Python version {'.'.join(map(str, current_python_version))} is not supported.")
            ui.log(
                f"Please use Python version in range [{'.'.join(map(str, MIN_PYTHON_VERSION_TUPLE))}, {'.'.join(map(str, MAX_PYTHON_VERSION_TUPLE))})."
            )
        else:
            ui.log("Error: No supported environment managers found.")
            ui.log("Please install at least one of: uv, venv (Python standard library), or conda.")
        raise SystemExit(1)

    # Display available options
    if len(options) > 1:
        ui.log("Multiple environment options available:")

        # Use provided default if valid, otherwise no default
        if default and default in options:
            selected_default = default
            if express_mode:
                ui.log(f"Using saved default: {default}")
        else:
            selected_default = None

        selected = ui.prompt_select("Select your preferred environment type:", options, default=selected_default)
        if selected is None:
            # User cancelled (Ctrl-C)
            ui.log("\nInstallation cancelled by user.")
            raise SystemExit(0)
    else:
        selected = options[0]
        ui.log(f"Using {selected} (only available option)")

    # Validate selection and warn if necessary
    if selected == "venv" and detect_virtual_environment():
        ui.log("")
        ui.log("Warning: You are currently in a virtual environment.")
        ui.log("The installer will use system Python to create new virtual environments.")

    if selected == "conda" and detect_virtual_environment() and not has_active_conda_environment():
        ui.log("")
        ui.log("Warning: You are in a non-conda virtual environment.")
        ui.log("Consider deactivating it before using conda for installation.")

    return selected


def prompt_environment_variables(
    ui: UIInterface, defaults: Optional[Dict[str, str]] = None, express_mode: bool = False
) -> Optional[Dict[str, str]]:
    """Prompt the user for environment variables.

    Currently prompts for HF_TOKEN, but designed to be easily expandable
    for additional environment variables in the future.

    Args:
        ui: UI interface for user interaction
        defaults: Default environment variables from system config (if available)
        express_mode: Whether this is being called from express mode

    Returns:
        Dict[str, str]: Dictionary containing the environment variables, or None if cancelled
    """
    # In express mode, if we have HF_TOKEN in defaults, use it without prompting
    if express_mode and defaults and 'HF_TOKEN' in defaults and defaults['HF_TOKEN']:
        ui.log("Using saved HF_TOKEN from system configuration.")
        return {'HF_TOKEN': defaults['HF_TOKEN']}

    ui.log("\nEnvironment Variables")
    ui.log("--------------------")
    ui.log("")

    env_vars = {}

    # HF_TOKEN validation function - now optional
    def validate_hf_token(token: str) -> bool | str:
        if not token:
            return True  # Allow empty token
        if not token.startswith('hf_'):
            return "HF_TOKEN must start with 'hf_' if provided."
        return True

    # Determine default HF_TOKEN (prefer saved config over environment)
    default_hf_token = ""
    source_message = ""

    if defaults and 'HF_TOKEN' in defaults and defaults['HF_TOKEN']:
        default_hf_token = defaults['HF_TOKEN']
        source_message = f"Found saved HF_TOKEN: {default_hf_token[:10]}..."
    else:
        env_hf_token = os.environ.get('HF_TOKEN', '')
        if env_hf_token:
            default_hf_token = env_hf_token
            source_message = f"Found existing HF_TOKEN in environment: {env_hf_token[:10]}..."

    if source_message:
        # Token found - provide brief context
        ui.log("=" * 70)
        ui.log("Hugging Face Token (HF_TOKEN)")
        ui.log("=" * 70)
        ui.log("")
        ui.log("Most workloads require a Hugging Face token to download models and datasets.")
        ui.log("")
        ui.log(source_message)
        ui.log("Press Enter to use this token, or enter a new one below.")
        ui.log("")
    else:
        # No existing token - make it clear this is important
        ui.log("=" * 70)
        ui.log("Hugging Face Token (HF_TOKEN)")
        ui.log("=" * 70)
        ui.log("")
        ui.log("Most workloads require a Hugging Face token to download models and datasets.")
        ui.log("")
        ui.log("ACTION REQUIRED:")
        ui.log("  1. Get your token: https://huggingface.co/settings/tokens")
        ui.log("  2. Enter it below")
        ui.log("")
        ui.log("Skip only if you know your workloads don't require Hugging Face access.", level='warning')
        ui.log("")

    hf_token = ui.prompt_text(
        f"Enter HF_TOKEN{' (press Enter to use existing)' if default_hf_token else ''}:",
        default=default_hf_token,
        validate=validate_hf_token,
    )

    if hf_token is None:  # User cancelled
        return None

    if hf_token:
        env_vars['HF_TOKEN'] = hf_token
        ui.log("✓ HF_TOKEN configured.", level='success')
    elif default_hf_token:
        env_vars['HF_TOKEN'] = default_hf_token
        ui.log("✓ Using existing HF_TOKEN.", level='success')
    else:
        ui.log("⚠ No HF_TOKEN provided - workloads requiring Hugging Face will fail.", level='warning')

    return env_vars
