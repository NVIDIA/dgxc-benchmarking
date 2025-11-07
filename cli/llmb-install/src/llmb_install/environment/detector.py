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


"""Environment detection utilities for LLMB installer."""

import importlib.util
import os
import shutil
import subprocess
import sys
from typing import Dict, Optional


def detect_virtual_environment() -> bool:
    """Detect if running in a virtual environment.

    Returns:
        True or False
    """
    # Check for conda environment
    if os.environ.get('CONDA_PREFIX') or os.environ.get('CONDA_DEFAULT_ENV'):
        return True

    # Check for venv environment
    if os.environ.get('VIRTUAL_ENV'):
        return True

    return False


def has_active_conda_environment() -> bool:
    """
    Return True if a conda environment is currently active (i.e., CONDA_PREFIX or CONDA_DEFAULT_ENV is set).

    Does not mean the conda environment is the top most venv.
    """
    return bool(os.environ.get('CONDA_PREFIX') or os.environ.get('CONDA_DEFAULT_ENV'))


def is_venv_installed() -> bool:
    """Check if current python process has venv installed."""
    return importlib.util.find_spec("venv") is not None


def is_conda_installed() -> bool:
    """Check if conda is available in PATH."""
    return shutil.which('conda') is not None


def is_uv_installed() -> bool:
    """Check if uv is available in PATH."""
    return shutil.which('uv') is not None


def get_system_python_version(clean_env: Optional[Dict[str, str]] = None) -> Optional[tuple]:
    """Attempt to get the system Python version using clean environment.

    Note: This attempts to detect the system Python by removing virtual environment
    pollution from PATH, but cannot guarantee it's truly the system Python if
    multiple Python installations exist or custom PATH modifications are present.

    Args:
        clean_env: Clean environment dict, will create if None

    Returns:
        tuple: Python version tuple (major, minor, patch), or None if not detected
    """

    if clean_env is None:
        clean_env = get_clean_environment_for_subprocess()

    try:
        result = subprocess.run(['python3', '--version'], env=clean_env, capture_output=True, text=True, check=True)
        version_str = result.stdout.strip().split(' ')[1]
        return tuple(map(int, version_str.split('.')[:3]))
    except (subprocess.CalledProcessError, IndexError, ValueError):
        # Return None if no system Python is detected
        return None


# Grouped here to avoid a circular import issue when it was in venv_manager.
def get_clean_environment_for_subprocess() -> Dict[str, str]:
    """Create a clean environment for subprocess execution without virtual environment pollution.

    This function removes virtual environment paths and variables that can cause
    architecture mismatch issues when running commands on different nodes via SLURM.

    Returns:
        Dict[str, str]: Clean environment variables for subprocess execution
    """
    env = os.environ.copy()

    if not detect_virtual_environment():
        return env

    # Clean conda environment if present
    if has_active_conda_environment():
        # Remove conda-specific environment variables
        conda_prefix = env.get('CONDA_PREFIX')
        conda_vars_to_remove = [
            'CONDA_PREFIX',
            'CONDA_DEFAULT_ENV',
            'CONDA_PROMPT_MODIFIER',
            'CONDA_SHLVL',
            'CONDA_PYTHON_EXE',
            'CONDA_EXE',
            '_CE_CONDA',
            '_CE_M',
        ]
        for var in list(env.keys()):
            if var.startswith('CONDA_PREFIX_'):
                conda_vars_to_remove.append(var)

        for var in conda_vars_to_remove:
            env.pop(var, None)

        # Clean PATH by removing conda bin directory (consistent with venv approach)
        if conda_prefix and 'PATH' in env:
            conda_bin = os.path.join(conda_prefix, 'bin')
            path_parts = env['PATH'].split(os.pathsep)
            cleaned_path_parts = [p for p in path_parts if p != conda_bin]
            env['PATH'] = os.pathsep.join(cleaned_path_parts)

        # Ensure base conda environment not activated during srun/sbatch.
        # Mainly a problem for systems where the login and compute nodes have different architectures.
        env['CONDA_AUTO_ACTIVATE'] = 'false'

    # Clean venv environment if present
    if env.get('VIRTUAL_ENV'):
        # Remove venv-specific environment variables
        venv_path = env.pop('VIRTUAL_ENV', None)
        env.pop('VIRTUAL_ENV_PROMPT', None)

        # Clean PATH by removing venv bin directory
        if venv_path and 'PATH' in env:
            venv_bin = os.path.join(venv_path, 'bin')
            path_parts = env['PATH'].split(os.pathsep)
            cleaned_path_parts = [p for p in path_parts if p != venv_bin]
            env['PATH'] = os.pathsep.join(cleaned_path_parts)

    # Remove Python-specific variables that might cause issues
    python_vars_to_remove = ['PYTHONHOME', 'PYTHONPATH']
    for var in python_vars_to_remove:
        env.pop(var, None)

    return env


def get_system_python_path(clean_env: Optional[Dict[str, str]] = None) -> str:
    """Attempt to get the system Python executable path using clean environment.

    Note: This attempts to detect the system Python by removing virtual environment
    pollution from PATH, but cannot guarantee it's truly the system Python if
    multiple Python installations exist or custom PATH modifications are present.

    Args:
        clean_env: Clean environment dict, will create if None

    Returns:
        str: Path to python3 executable (system if detectable, fallback to current)
    """

    if clean_env is None:
        clean_env = get_clean_environment_for_subprocess()

    try:
        python_path = shutil.which('python3', path=clean_env.get('PATH'))
        if python_path:
            return python_path
        else:
            # Fallback to current executable if not found in PATH
            return sys.executable
    except Exception:
        # Fallback to current executable if detection fails
        return sys.executable
