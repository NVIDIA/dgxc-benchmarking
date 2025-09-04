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

"""Configuration management for llmb-run."""

import logging
import os
import pathlib

import yaml

logger = logging.getLogger('llmb_run.config_manager')

def get_cluster_config():
    """
    Load and validate cluster configuration from cluster_config.yaml.

    It will check for the cluster_config.yaml file in the current directory and the path specified by the LLMB_INSTALL environment variable.
    Priority is given to the current directory.
    """
    config_file_name = 'cluster_config.yaml'
    config_path_in_cwd = pathlib.Path.cwd() / config_file_name
    config_path = None

    llmb_install_path = os.environ.get('LLMB_INSTALL')

    # Configuration file in the current directory takes precedence.
    if config_path_in_cwd.exists():
        config_path = config_path_in_cwd
        if llmb_install_path:
            logger.debug(f"Found '{config_file_name}' in current directory, which takes precedence over LLMB_INSTALL.")
    # If not in CWD, check the path specified by the LLMB_INSTALL environment variable.
    elif llmb_install_path:
        llmb_config_path = pathlib.Path(llmb_install_path) / config_file_name
        if llmb_config_path.exists():
            config_path = llmb_config_path

    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing cluster configuration file '{config_path}': {e}")
    else:
        raise FileNotFoundError(f"Cluster configuration file '{config_file_name}' not found. Looked in the current directory and under the path specified by the LLMB_INSTALL environment variable.")

    # Validate required fields
    if 'launcher' not in config:
        raise ValueError("Missing required 'launcher' section in cluster configuration.")
    
    launcher_config = config['launcher']
    if 'gpu_type' not in launcher_config:
        raise ValueError("Missing required 'launcher.gpu_type' field in cluster configuration.")
    
    if 'llmb_install' not in launcher_config:
        raise ValueError("Missing required 'launcher.llmb_install' field in cluster configuration.")
    
    if 'llmb_repo' not in launcher_config:
        raise ValueError("Missing required 'launcher.llmb_repo' field in cluster configuration.")
    
    # Add current working directory for output logs
    config['cwd'] = pathlib.Path.cwd()
    
    return config

def get_slurm_env_vars(config, gpu_partition=True):
    """Convert slurm config section to SBATCH_ environment variables."""
    slurm_env = {}
    slurm_config = config.get('slurm', {})
    
    if slurm_config.get('account'):
        slurm_env['SBATCH_ACCOUNT'] = str(slurm_config['account'])
    if gpu_partition:
        if slurm_config.get('gpu_partition'):
            slurm_env['SBATCH_PARTITION'] = str(slurm_config['gpu_partition'])
        if slurm_config.get('gpu_gres'):
            slurm_env['SBATCH_GPUS_PER_NODE'] = str(slurm_config['gpu_gres'])
    else:
        if slurm_config.get('cpu_partition'):
            slurm_env['SBATCH_PARTITION'] = str(slurm_config['cpu_partition'])
        if slurm_config.get('cpu_gres'):
            slurm_env['SBATCH_GPUS_PER_NODE'] = str(slurm_config['cpu_gres'])
    
    return slurm_env

def get_workload_config(config, workload_key):
    """Get workload-specific configuration from cluster config."""
    workload_configs = config.get('workloads', {}).get('config', {})
    return workload_configs.get(workload_key, {})
