# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Configuration data models for LLMB Install."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SlurmConfig:
    """SLURM cluster configuration."""

    account: str
    gpu_partition: str
    cpu_partition: str
    gpu_partition_gres: Optional[int] = None
    cpu_partition_gres: Optional[int] = None


@dataclass
class InstallConfig:
    """Central configuration object for LLMB installation."""

    # Required fields (no defaults)
    install_path: str
    venv_type: str  # 'uv', 'venv', 'conda'
    gpu_type: str  # 'h100', 'gb200', 'b200'
    node_architecture: str  # 'x86_64', 'aarch64'

    # Optional fields (with defaults)
    slurm: Optional[SlurmConfig] = None
    selected_workloads: List[str] = field(default_factory=list)
    workload_selection_mode: str = 'custom'  # 'custom' or 'exemplar'
    install_method: str = 'local'  # 'local' or 'slurm'
    ui_mode: str = 'simple'  # 'simple', 'rich', 'express'
    environment_vars: Dict[str, str] = field(default_factory=dict)
    cache_dirs_configured: bool = False
    image_folder: Optional[str] = None  # Shared container image folder
    dev_mode: bool = False  # Development mode: skip repo copying, use original location
    llmb_repo: Optional[str] = None  # Path to LLMB repository (original or copied)
    is_incremental_install: bool = (
        False  # True if this is an incremental install (adding workloads to existing installation)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {
            'install_path': self.install_path,
            'venv_type': self.venv_type,
            'gpu_type': self.gpu_type,
            'node_architecture': self.node_architecture,
            'selected_workloads': self.selected_workloads,
            'workload_selection_mode': self.workload_selection_mode,
            'install_method': self.install_method,
            'ui_mode': self.ui_mode,
            'environment_vars': self.environment_vars,
            'cache_dirs_configured': self.cache_dirs_configured,
            'image_folder': self.image_folder,
            'dev_mode': self.dev_mode,
            'llmb_repo': self.llmb_repo,
            'is_incremental_install': self.is_incremental_install,
        }

        if self.slurm:
            result['slurm'] = {
                'account': self.slurm.account,
                'gpu_partition': self.slurm.gpu_partition,
                'cpu_partition': self.slurm.cpu_partition,
                'gpu_partition_gres': self.slurm.gpu_partition_gres,
                'cpu_partition_gres': self.slurm.cpu_partition_gres,
            }
        else:
            result['slurm'] = None

        return result

    def to_play_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for headless playfiles."""
        data = self.to_dict()
        denylist = {
            'llmb_repo',
            'dev_mode',
            'ui_mode',
            'cache_dirs_configured',
            'is_incremental_install',
        }
        data = {key: value for key, value in data.items() if key not in denylist}
        if data.get('image_folder') is None:
            data.pop('image_folder', None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstallConfig':
        """Create config from dictionary."""
        slurm_data = data.get('slurm')
        slurm_config = None
        if slurm_data:
            slurm_config = SlurmConfig(
                account=slurm_data['account'],
                gpu_partition=slurm_data['gpu_partition'],
                cpu_partition=slurm_data['cpu_partition'],
                gpu_partition_gres=slurm_data.get('gpu_partition_gres'),
                cpu_partition_gres=slurm_data.get('cpu_partition_gres'),
            )

        return cls(
            install_path=data['install_path'],
            venv_type=data['venv_type'],
            gpu_type=data['gpu_type'],
            node_architecture=data['node_architecture'],
            slurm=slurm_config,
            selected_workloads=data.get('selected_workloads', []),
            workload_selection_mode=data.get('workload_selection_mode', 'custom'),
            install_method=data.get('install_method', 'local'),
            ui_mode=data.get('ui_mode', 'simple'),
            environment_vars=data.get('environment_vars', {}),
            cache_dirs_configured=data.get('cache_dirs_configured', False),
            image_folder=data.get('image_folder'),
            dev_mode=data.get('dev_mode', False),
            llmb_repo=data.get('llmb_repo'),
            is_incremental_install=data.get('is_incremental_install', False),
        )

    def get_remaining_workloads(self, completed: List[str]) -> List[str]:
        """Get workloads that still need to be installed.

        Args:
            completed: List of completed workload names

        Returns:
            List of workload names that still need to be installed
        """
        return [w for w in self.selected_workloads if w not in completed]

    @property
    def locked_fields_for_resume(self) -> List[str]:
        """Get list of fields that cannot be changed during resume edit.

        Returns:
            List of field names that are locked during resume
        """
        return ['install_path', 'gpu_type', 'node_architecture', 'venv_type', 'llmb_repo', 'dev_mode', 'image_folder']

    @property
    def editable_fields_for_resume(self) -> List[str]:
        """Get list of fields that can be changed during resume edit.

        Returns:
            List of field names that can be edited during resume
        """
        return ['slurm', 'install_method', 'selected_workloads']

    def get_slurm_dict(self) -> Dict[str, Any]:
        """Convert SLURM config to dictionary format for legacy code compatibility.

        Returns:
            Dictionary containing SLURM configuration, or empty dict if no SLURM config
        """
        if not self.slurm:
            return {}

        return {
            'slurm': {
                'account': self.slurm.account,
                'gpu_partition': self.slurm.gpu_partition,
                'cpu_partition': self.slurm.cpu_partition,
                'gpu_partition_gres': self.slurm.gpu_partition_gres,
                'cpu_partition_gres': self.slurm.cpu_partition_gres,
            }
        }
