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

"""Task management for workload execution."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger('llmb_run.tasks')


@dataclass
class WorkloadTask:
    workload_key: str  # Full workload key (e.g., 'pretraining_nemotron')
    model_size: str
    dtype: str
    scale: int
    profile: bool = False
    env_overrides: dict = field(default_factory=dict)
    model_overrides: dict = field(default_factory=dict)
    extra_slurm_params: Dict[str, Any] = field(default_factory=dict)


def format_task_output(task, prefix="", suffix=""):
    """Format task details in a consistent way with aligned columns."""
    # Fixed width fields for alignment
    workload_field = f"{task.workload_key}_{task.model_size}"
    dtype_field = f"dtype={task.dtype}"
    scale_field = f"scale={task.scale}"
    profile_field = f"profile={task.profile}"

    # Build the base output with aligned fields
    output = f"{prefix}{workload_field:<30} {dtype_field:<12} {scale_field:<12} {profile_field:<15}"

    # Add optional fields if they exist
    if task.env_overrides:
        output += f" env={task.env_overrides}"
    if task.model_overrides:
        output += f" params={task.model_overrides}"
    if suffix:
        output += f" {suffix}"
    return output


def print_tasks(task_list):
    """Print task details in a consistent format."""
    for task in task_list:
        logger.info(format_task_output(task, prefix="Task: "))
