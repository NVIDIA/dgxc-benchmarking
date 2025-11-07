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

"""SLURM utilities for job management."""

import logging
import shlex
import subprocess
from dataclasses import dataclass

logger = logging.getLogger('llmb_run.slurm_utils')


@dataclass
class SlurmJob:
    job_id: int | None
    job_status: str = None
    job_workdir: str = None


def get_slurm_job_status(jobid: int):
    """Get the status of a SLURM job by job ID.

    Args:
        jobid: SLURM job ID

    Returns:
        str: Job status string, or None if error occurred
    """
    cmd = f"sacct -X --format=State --noheader -j {jobid}"
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
        job_status = result.stdout.strip()
        logger.debug(f"Job {jobid} status: {job_status}")
        return job_status
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running sacct for job {jobid}: {e.stderr}")
        return None


def get_cluster_name():
    """Get the cluster name from SLURM configuration.

    Returns:
        str: Cluster name from SLURM config, or None if not found or error occurred
    """
    cmd = "scontrol show config"
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith('ClusterName'):
                # Extract the value after the '=' sign
                parts = line.split('=', 1)
                if len(parts) == 2:
                    cluster_name = parts[1].strip()
                    logger.debug(f"Found cluster name from SLURM config: {cluster_name}")
                    return cluster_name

        logger.debug("ClusterName not found in SLURM config output")
        return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running scontrol show config: {e.stderr}")
        return None
