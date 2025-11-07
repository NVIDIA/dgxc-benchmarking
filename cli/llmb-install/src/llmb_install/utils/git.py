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

"""Git utilities for LLMB installation.

This module provides functions for checking git and git-lfs availability
and configuration.
"""

import shutil
import subprocess
from functools import lru_cache

from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def is_git_lfs_installed() -> bool:
    """Check if Git LFS is installed and available in the system PATH.

    Returns:
        bool: True if git-lfs command is available, False otherwise
    """
    return shutil.which('git-lfs') is not None


def ensure_git_lfs_configured() -> None:
    """Ensure Git LFS is configured for the current user.

    Checks if Git LFS is already configured by looking for the lfs filter
    in git config. If not configured, runs 'git lfs install --skip-repo' to set it up.

    Uses --skip-repo to configure global filters without modifying the llmb-repo itself,
    since the installer runs from within the llmb-repo.

    This function is idempotent and safe to call multiple times.

    Raises:
        subprocess.CalledProcessError: If git lfs install fails
    """
    try:
        # Check if already configured by looking for the lfs process filter
        result = subprocess.run(
            ['git', 'config', '--global', '--get', 'filter.lfs.process'],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.debug("Git LFS already configured")
            return

        # Not configured, run git lfs install
        logger.debug("Configuring Git LFS")
        subprocess.run(
            ['git', 'lfs', 'install', '--skip-repo'],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug("Git LFS configured successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure Git LFS: {e.stderr if e.stderr else str(e)}")
        raise
