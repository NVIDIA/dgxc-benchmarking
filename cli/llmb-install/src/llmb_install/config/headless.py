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


"""Configuration file loading and saving for headless installation."""

import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def save_installation_config(config_file: str, config_data: Dict[str, Any]) -> None:
    """Save installation configuration to a YAML file.

    Args:
        config_file: Path to the configuration file to save
        config_data: Dictionary containing all installation configuration
    """
    try:
        # Ensure the destination directory exists
        resolved_path = Path(config_file).resolve()
        os.makedirs(resolved_path.parent, exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        os.chmod(config_file, 0o600)
        print(f"✓ Configuration saved to: {config_file}")
    except Exception as e:
        print(f"Error saving configuration to {config_file}: {e}")
        raise SystemExit(1) from e


def _missing_required_fields(data: Dict[str, Any], fields: Iterable[str]) -> list[str]:
    return [field for field in fields if field not in data]


def _validate_required_strings(data: Dict[str, Any], fields: Iterable[str]) -> None:
    blank_fields = []
    for field in fields:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            blank_fields.append(field)
    if blank_fields:
        raise ValueError(f"Configuration fields cannot be blank: {blank_fields}")


def load_installation_config(config_file: str) -> Dict[str, Any]:
    """Load installation configuration from a YAML file.

    Args:
        config_file: Path to the configuration file to load

    Returns:
        Dict containing all installation configuration

    Raises:
        SystemExit: If the configuration file cannot be loaded or is invalid
    """
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a dictionary")

        # TODO: Remove this deprecated-key check after next public release.
        deprecated_keys = [key for key in ('slurm_info', 'env_vars') if key in config_data]
        if deprecated_keys:
            raise ValueError(
                "Playfiles must use top-level slurm and environment_vars; "
                "slurm_info/env_vars are no longer supported."
            )

        # Validate required fields
        required_fields = [
            'install_path',
            'venv_type',
            'gpu_type',
            'node_architecture',
            'install_method',
            'selected_workloads',
        ]

        missing_fields = _missing_required_fields(config_data, required_fields)
        if missing_fields:
            raise ValueError(f"Configuration file is missing required fields: {missing_fields}")

        _validate_required_strings(config_data, ['install_path', 'venv_type', 'gpu_type', 'node_architecture'])

        selected_workloads = config_data.get('selected_workloads')
        if not isinstance(selected_workloads, list):
            raise ValueError("selected_workloads must be a list")
        if not selected_workloads:
            raise ValueError("selected_workloads cannot be empty")
        invalid_workloads = [w for w in selected_workloads if not isinstance(w, str) or not w.strip()]
        if invalid_workloads:
            raise ValueError("selected_workloads must be a list of non-empty strings")

        env_vars = config_data.get('environment_vars')
        if env_vars is not None and not isinstance(env_vars, dict):
            raise ValueError("environment_vars must be a dictionary when provided")

        # Validate SLURM configuration when provided or required.
        slurm_config = config_data.get('slurm')
        if slurm_config is not None:
            if not isinstance(slurm_config, dict):
                raise ValueError("slurm must be a dictionary when provided")

            _validate_required_strings(
                slurm_config,
                ['account', 'gpu_partition', 'cpu_partition'],
            )

        if config_data.get('install_method') == 'slurm' and not slurm_config:
            raise ValueError("slurm configuration is required when install_method is 'slurm'")

        print(f"✓ Configuration loaded from: {config_file}")
        return config_data

    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        raise SystemExit(1) from None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file {config_file}: {e}")
        raise SystemExit(1) from e
    except ValueError as e:
        print(f"Error: Invalid configuration in {config_file}: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"Error loading configuration from {config_file}: {e}")
        raise SystemExit(1) from e
