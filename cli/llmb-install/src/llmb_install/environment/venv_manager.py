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


"""Virtual environment management utilities for LLMB installer."""

import os
import shutil
import subprocess

from llmb_install.constants import MIN_PYTHON_VERSION
from llmb_install.environment.detector import (
    get_clean_environment_for_subprocess,
)


def create_virtual_environment(venv_path: str, venv_type: str) -> None:
    """Create a virtual environment at the specified path using the given environment type.

    Args:
        venv_path: Path where the virtual environment should be created
        venv_type: Type of virtual environment to create ('venv', 'conda', or 'uv')

    Raises:
        ValueError: If venv_type is not supported
        subprocess.CalledProcessError: If virtual environment creation fails.
    """
    print(f"Creating virtual environment at {venv_path}...")

    if venv_type == 'venv':
        # Use system python3 instead of current environment's python
        env = get_clean_environment_for_subprocess()
        try:
            subprocess.run(
                ['python3', '-m', 'venv', '--clear', venv_path], env=env, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error creating venv environment: {e}")
            raise

    elif venv_type == 'conda':
        # If the conda env path already exists, remove it to prevent errors
        if os.path.exists(venv_path):
            print(f"  Removing existing conda environment at {venv_path}")
            shutil.rmtree(venv_path)

        try:
            # Create conda environment with minimum required Python version
            subprocess.run(
                ['conda', 'create', '-p', venv_path, f'python={MIN_PYTHON_VERSION}', '--yes'],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error creating conda environment: {e}")
            raise

    elif venv_type == 'uv':
        try:
            subprocess.run(
                [
                    'uv',
                    'venv',
                    '--clear',
                    '--python',
                    MIN_PYTHON_VERSION,
                    venv_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error creating uv environment: {e}")
            raise

        # uv does not install pip in new venvs. Installing it manually handles legacy scripts where 'pip install' is used.
        uv_venv = get_venv_environment(venv_path, 'uv')
        try:
            subprocess.run(
                [
                    'uv',
                    'pip',
                    'install',
                    'pip',
                ],
                env=uv_venv,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error installing pip in uv environment: {e}")
            raise
    else:
        raise ValueError(f"Unsupported environment type: {venv_type}")

    print(f"Created virtual environment at: {venv_path}")


def get_venv_environment(venv_path: str, venv_type: str) -> dict:
    """Prepare environment variables for running commands in a virtual environment.

    Args:
        venv_path: Path to the virtual environment
        venv_type: Type of virtual environment ('venv', 'conda', or 'uv')

    Returns:
        dict: Modified environment variables for use with subprocess

    Raises:
        ValueError: If python3 executable is not found in the virtual environment
    """
    env = os.environ.copy()

    bin_dir = os.path.join(venv_path, 'bin')
    python_path = os.path.join(bin_dir, 'python3')
    if not os.path.exists(python_path):
        raise ValueError(f"Invalid virtual environment: python3 executable not found at {python_path}")

    env['PATH'] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.pop('PYTHONHOME', None)

    if venv_type == 'venv' or venv_type == 'uv':
        env['VIRTUAL_ENV'] = venv_path
    elif venv_type == 'conda':
        env['CONDA_PREFIX'] = venv_path
        # UV uses this to distinguish activated conda envs from system conda installations
        env['CONDA_DEFAULT_ENV'] = venv_path

    return env
