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


"""Cluster configuration management for LLMB installer."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from llmb_install.cluster.slurm import get_cluster_name
from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)


def load_cluster_config(install_path: str) -> Optional[Dict[str, Any]]:
    """Load cluster_config.yaml from an installation directory.

    Args:
        install_path: Path to the installation directory

    Returns:
        Dict containing config, or None if not found/invalid
    """
    cluster_config_path = Path(install_path) / "cluster_config.yaml"
    if not cluster_config_path.exists():
        return None

    try:
        with open(cluster_config_path, 'r') as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        logger.warning(f"Could not load cluster config: {e}")
        return None


def create_cluster_config(
    install_path: str,
    root_dir: str,
    selected_workloads: List[str],
    slurm_info: dict,
    env_vars: dict,
    gpu_type: str,
    venv_type: str,
    workload_venvs: Dict[str, str],
    node_architecture: str = 'x86_64',
    install_method: str = 'local',
    image_folder: Optional[str] = None,
    existing_cluster_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Create or update cluster_config.yaml file with all installation configuration.

    Args:
        install_path: Base installation directory
        root_dir: Path to the LLMB repository root
        selected_workloads: List of selected workload keys
        slurm_info: SLURM configuration dictionary
        env_vars: Environment variables dictionary
        gpu_type: Selected GPU type
        venv_type: Type of virtual environment used ('venv' or 'conda')
        workload_venvs: Dictionary mapping workload keys to their venv paths
        node_architecture: Node architecture (e.g., 'x86_64', 'arm64')
        install_method: Installation method ('local' or 'slurm')
        image_folder: Optional path to container image folder
        existing_cluster_config: Optional existing config for incremental updates
    """
    # If updating, merge with existing workloads
    if existing_cluster_config:
        existing_workloads = existing_cluster_config.get('workloads', {}).get('installed', [])
        all_workloads = list(set(existing_workloads + selected_workloads))

        # Merge venv configs - existing configs take precedence
        existing_configs = existing_cluster_config.get('workloads', {}).get('config', {})
        merged_workload_venvs = dict(existing_configs)
        # Add new workload venvs
        for wl_name in selected_workloads:
            if wl_name not in merged_workload_venvs:
                venv_path = workload_venvs.get(wl_name)
                if venv_path:
                    merged_workload_venvs[wl_name] = {
                        'venv_path': venv_path,
                        'venv_type': venv_type,
                    }
    else:
        all_workloads = selected_workloads
        merged_workload_venvs = {}
        for workload_key in selected_workloads:
            venv_path = workload_venvs.get(workload_key)
            if venv_path:
                merged_workload_venvs[workload_key] = {
                    'venv_path': venv_path,
                    'venv_type': venv_type,
                }

    launcher_config = {
        'llmb_repo': root_dir,
        'llmb_install': install_path,
        'gpu_type': gpu_type,
        'node_architecture': node_architecture,  # Keep for backward compatibility
    }

    # Add cluster name if available
    # For incremental installs, preserve existing cluster_name if current detection fails
    cluster_name = get_cluster_name()
    if cluster_name:
        launcher_config['cluster_name'] = cluster_name
    elif existing_cluster_config:
        # Preserve existing cluster_name if we can't detect it now
        existing_cluster_name = existing_cluster_config.get('launcher', {}).get('cluster_name')
        if existing_cluster_name:
            launcher_config['cluster_name'] = existing_cluster_name

    # Build install metadata section
    # For incremental installs, merge with existing install section to preserve all fields
    if existing_cluster_config:
        # Start with existing install section (preserves all fields including image_folder)
        install_section = existing_cluster_config.get('install', {}).copy()
        # Update with new values (these should match existing for incremental, but update anyway)
        install_section['venv_type'] = venv_type
        install_section['method'] = install_method
        install_section['node_architecture'] = node_architecture
        # Update image_folder if explicitly provided (allows override via -i flag)
        if image_folder is not None:
            install_section['image_folder'] = image_folder
        # If image_folder is None and not in existing, don't add it
    else:
        # New installation - create fresh install section
        install_section = {
            'venv_type': venv_type,
            'method': install_method,
            'node_architecture': node_architecture,  # Canonical location
        }
        if image_folder:
            install_section['image_folder'] = image_folder

    cluster_config = {
        'launcher': launcher_config,
        'install': install_section,  # New section for install-specific metadata
        'environment': env_vars,
        'slurm': {
            'account': slurm_info['slurm']['account'],
            'gpu_partition': slurm_info['slurm']['gpu_partition'],
            'gpu_gres': slurm_info['slurm'].get('gpu_partition_gres'),
            'cpu_partition': slurm_info['slurm']['cpu_partition'],
            'cpu_gres': slurm_info['slurm'].get('cpu_partition_gres'),
        },
        'workloads': {'installed': all_workloads, 'config': merged_workload_venvs},
    }

    # Write the cluster config file
    config_path = os.path.join(install_path, "cluster_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(cluster_config, f, default_flow_style=False, sort_keys=False)

    print(f"Created cluster configuration: {config_path}")
