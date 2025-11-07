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


"""Cache directory management utilities for LLMB installer."""

import os
import subprocess
import sys
from pathlib import Path

from llmb_install.environment.detector import is_uv_installed


def setup_cache_directories(install_path: str, venv_type: str) -> None:
    """Configure cache directories for the selected environment type with fallback defaults.

    Args:
        install_path: The LLMB installation path
        venv_type: The selected virtual environment type ('uv', 'venv', or 'conda')
    """

    # Default cache locations
    default_pip_cache = os.path.join(install_path, '.cache', 'pip')
    default_uv_cache = os.path.join(install_path, '.cache', 'uv')

    # Always handle PIP_CACHE_DIR since all environment types may use pip
    pip_cache_dir = os.environ.get('PIP_CACHE_DIR')
    if not pip_cache_dir:
        os.environ['PIP_CACHE_DIR'] = default_pip_cache
        os.makedirs(default_pip_cache, exist_ok=True)
        print(f"Set PIP_CACHE_DIR to: {default_pip_cache}")
    elif _is_under_home_directory(pip_cache_dir):
        print(f"\nWARNING: PIP_CACHE_DIR is under home directory: {pip_cache_dir}")
        print("This may cause space issues. Consider using a more performant storage.")

    # Handle UV_CACHE_DIR if uv is available (regardless of primary venv_type)
    if venv_type == 'uv' or is_uv_installed():
        uv_cache_dir = os.environ.get('UV_CACHE_DIR')
        if not uv_cache_dir:
            os.environ['UV_CACHE_DIR'] = default_uv_cache
            os.makedirs(default_uv_cache, exist_ok=True)
            print(f"Set UV_CACHE_DIR to: {default_uv_cache}")
        elif _is_under_home_directory(uv_cache_dir):
            print(f"\nWARNING: UV_CACHE_DIR is under home directory: {uv_cache_dir}")
            print("This may cause space issues. Consider using a more performant storage.")


def check_pip_cache_location() -> None:
    """Check the pip cache directory location and warn if it's under /home."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "dir"], capture_output=True, text=True, check=True
        )
        pip_cache_dir = result.stdout.strip()

        if _is_under_home_directory(pip_cache_dir):
            print("\nWARNING: Your pip cache directory is located under home directory")
            print(f"Current location: {pip_cache_dir}")
            print("This may cause space issues when installing large packages.")
            print("Consider setting PIP_CACHE_DIR environment variable to a location with more space.")
    except subprocess.CalledProcessError:
        print("Could not determine pip cache location.")


def _is_under_home_directory(path: str) -> bool:
    """Check if a path is under the user's home directory.

    Args:
        path: Path to check

    Returns:
        True if path is under home directory, False otherwise
    """
    try:
        home_dir = Path.home()
        check_path = Path(path).resolve()
        return home_dir in check_path.parents or check_path == home_dir
    except (OSError, ValueError):
        # Fallback to simple string check if Path operations fail
        return path.startswith('/home')
