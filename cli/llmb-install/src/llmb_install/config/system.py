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

"""System configuration persistence for LLMB Install.

Handles saving and loading of sanitized system configuration that persists
stable settings across installs while excluding per-install varying fields.
"""

import os
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from llmb_install.config.models import InstallConfig, SlurmConfig
from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SystemConfig:
    """Sanitized system configuration that persists across installs.

    Contains stable system settings but excludes per-install variables like
    install_path and selected_workloads.
    """

    # Core system settings (stable)
    venv_type: str  # 'uv', 'venv', 'conda'
    install_method: str  # 'local' or 'slurm'
    gpu_type: str  # 'h100', 'gb200', 'b200'
    node_architecture: str  # 'x86_64', 'aarch64'

    # SLURM settings (stable for a cluster)
    slurm: Optional[SlurmConfig] = None

    # Environment variables (potentially stable)
    environment_vars: Dict[str, str] = None

    # Shared container image folder (stable)
    image_folder: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.environment_vars is None:
            self.environment_vars = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'venv_type': self.venv_type,
            'install_method': self.install_method,
            'gpu_type': self.gpu_type,
            'node_architecture': self.node_architecture,
            'environment_vars': self.environment_vars,
            'image_folder': self.image_folder,
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary."""
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
            venv_type=data['venv_type'],
            install_method=data['install_method'],
            gpu_type=data['gpu_type'],
            node_architecture=data['node_architecture'],
            slurm=slurm_config,
            environment_vars=data.get('environment_vars', {}),
            image_folder=data.get('image_folder'),
        )

    @classmethod
    def from_install_config(cls, install_config: InstallConfig) -> 'SystemConfig':
        """Extract system config from a complete install config.

        This sanitizes the install config by keeping only stable system settings
        and excluding per-install variables.
        """
        return cls(
            venv_type=install_config.venv_type,
            install_method=install_config.install_method,
            gpu_type=install_config.gpu_type,
            node_architecture=install_config.node_architecture,
            slurm=install_config.slurm,
            environment_vars=install_config.environment_vars.copy(),
            image_folder=install_config.image_folder,
        )


def _get_system_config_dir() -> Path:
    """Get XDG-compliant system config directory."""
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
    if xdg_config_home:
        return Path(xdg_config_home) / "llmb"
    else:
        return Path.home() / ".config" / "llmb"


class SystemConfigManager:
    """Manages persistent system configuration.

    Handles saving sanitized system settings on successful installs and
    loading them to provide defaults for subsequent installs.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize system config manager.

        Args:
            config_path: Custom path for system config. If None, uses XDG default.
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = _get_system_config_dir() / "system_config.yaml"

    def save(self, install_config: InstallConfig) -> None:
        """Save sanitized system config from successful install.

        Args:
            install_config: Complete install config to sanitize and save.
        """
        system_config = SystemConfig.from_install_config(install_config)

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write to temporary file first (atomic operation)
            temp_path = self.config_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                yaml.safe_dump(system_config.to_dict(), f, default_flow_style=False, indent=2)

            # Set restrictive permissions (0600) for security
            os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)

            # Atomic move to final location
            temp_path.replace(self.config_path)

            logger.info(f"Saved system config to {self.config_path}")
            logger.debug(
                "Persisted fields: venv_type, install_method, gpu_type, " "node_architecture, slurm, environment_vars"
            )
            logger.debug(
                "Excluded changing fields: install_path, selected_workloads, " "venv_path, cache_dirs_configured"
            )

        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Failed to save system config to {self.config_path}: {e}")

    def load(self) -> Optional[SystemConfig]:
        """Load system config for use as defaults.

        Returns:
            SystemConfig if file exists and is valid, None otherwise.
        """
        if not self.config_path.exists():
            logger.debug(f"No system config found at {self.config_path}")
            return None

        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"System config file is empty: {self.config_path}")
                return None

            system_config = SystemConfig.from_dict(data)
            logger.info(f"Loaded system config from {self.config_path}")
            logger.debug(f"Available defaults: {list(data.keys())}")
            return system_config

        except (yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load system config from {self.config_path}: {e}")
            return None

    def exists(self) -> bool:
        """Check if system config file exists."""
        return self.config_path.exists()

    def delete(self) -> None:
        """Delete system config file if it exists."""
        if self.config_path.exists():
            self.config_path.unlink()
            logger.info(f"Deleted system config: {self.config_path}")

    def get_path(self) -> Path:
        """Get the path to the system config file."""
        return self.config_path


# Global instance for easy access
_system_config_manager = SystemConfigManager()


def save_system_config(install_config: InstallConfig) -> None:
    """Save system config from successful install (convenience function)."""
    _system_config_manager.save(install_config)


def load_system_config() -> Optional[SystemConfig]:
    """Load system config for defaults (convenience function)."""
    return _system_config_manager.load()


def system_config_exists() -> bool:
    """Check if system config exists (convenience function)."""
    return _system_config_manager.exists()


def get_system_config_path() -> Path:
    """Get system config file path (convenience function)."""
    return _system_config_manager.get_path()


class InstallStateManager:
    """Manages installation state for resume functionality.

    Handles saving and loading installation state to enable resuming failed
    installations from where they left off. Uses same robust patterns as
    SystemConfigManager for atomic writes and error handling.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize install state manager.

        Args:
            config_path: Custom path for install state. If None, uses XDG default.
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = _get_system_config_dir() / "install_state.yaml"

    def save_install_state(
        self,
        config: InstallConfig,
        completed_workloads: List[str],
        workload_venvs: Optional[Dict[str, str]] = None,
        existing_cluster_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save installation state for resume functionality.

        Args:
            config: Complete install config
            completed_workloads: List of individual workload names that completed successfully
            workload_venvs: Mapping of workload names to their venv paths
            existing_cluster_config: For incremental installs, the original cluster config
        """
        state_data = {
            'install_config': config.to_dict(),
            'completed_workloads': completed_workloads,
            'workload_venvs': workload_venvs or {},
            'timestamp': datetime.now().isoformat(),
        }

        # Store existing_cluster_config for incremental installs
        if existing_cluster_config is not None:
            state_data['existing_cluster_config'] = existing_cluster_config

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write to temporary file first (atomic operation)
            temp_path = self.config_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                yaml.safe_dump(state_data, f, default_flow_style=False, indent=2)

            # Set restrictive permissions (0600) for security
            os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)

            # Atomic move to final location
            temp_path.replace(self.config_path)

            logger.info(f"Saved install state to {self.config_path}")
            logger.debug(f"Completed workloads: {completed_workloads}")

        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Failed to save install state to {self.config_path}: {e}")

    def load_install_state(self) -> Optional[Tuple[InstallConfig, List[str], Dict[str, str], Optional[Dict[str, Any]]]]:
        """Load installation state for resume functionality.

        Returns:
            Tuple of (InstallConfig, completed_workloads, workload_venvs, existing_cluster_config) if valid state exists, None otherwise.
            Returns None if state is stale (>7 days) or invalid.
            The existing_cluster_config is None for normal installs, populated for incremental installs.
        """
        if not self.config_path.exists():
            logger.debug(f"No install state found at {self.config_path}")
            return None

        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"Install state file is empty: {self.config_path}")
                return None

            # Check if state is stale (>7 days)
            timestamp_str = data.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if datetime.now() - timestamp > timedelta(days=7):
                        logger.info(f"Install state is stale ({timestamp}), ignoring")
                        return None
                except ValueError:
                    logger.warning(f"Invalid timestamp in install state: {timestamp_str}")
                    return None

            # Extract and validate data
            install_config_data = data.get('install_config')
            completed_workloads = data.get('completed_workloads', [])
            workload_venvs = data.get('workload_venvs', {})
            existing_cluster_config = data.get('existing_cluster_config')  # May be None

            if not install_config_data:
                logger.warning("Install state missing install_config data")
                return None

            # Reconstruct InstallConfig
            install_config = InstallConfig.from_dict(install_config_data)

            # Validate that install directory still exists
            if not os.path.exists(install_config.install_path):
                logger.debug(
                    f"Install directory no longer exists, clearing resume state: {install_config.install_path}"
                )
                self.clear_install_state()
                return None

            logger.info(f"Loaded install state from {self.config_path}")
            logger.debug(f"Completed workloads: {completed_workloads}")
            logger.debug(f"Workload venvs: {len(workload_venvs)} mappings")
            if existing_cluster_config:
                logger.debug("Incremental install detected (existing_cluster_config present)")

            return (install_config, completed_workloads, workload_venvs, existing_cluster_config)

        except (yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load install state from {self.config_path}: {e}")
            return None

    def clear_install_state(self) -> None:
        """Clear installation state after successful completion."""
        if self.config_path.exists():
            self.config_path.unlink()
            logger.info(f"Cleared install state: {self.config_path}")

    def exists(self) -> bool:
        """Check if install state file exists."""
        return self.config_path.exists()

    def get_path(self) -> Path:
        """Get the path to the install state file."""
        return self.config_path


# Global instance for easy access
_install_state_manager = InstallStateManager()


def save_install_state(
    config: InstallConfig,
    completed_workloads: List[str],
    workload_venvs: Optional[Dict[str, str]] = None,
    existing_cluster_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save install state (convenience function)."""
    _install_state_manager.save_install_state(config, completed_workloads, workload_venvs, existing_cluster_config)


def load_install_state() -> Optional[Tuple[InstallConfig, List[str], Dict[str, str], Optional[Dict[str, Any]]]]:
    """Load install state for resume (convenience function)."""
    return _install_state_manager.load_install_state()


def clear_install_state() -> None:
    """Clear install state after completion (convenience function)."""
    _install_state_manager.clear_install_state()


def install_state_exists() -> bool:
    """Check if install state exists (convenience function)."""
    return _install_state_manager.exists()
