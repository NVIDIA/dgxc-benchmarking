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


"""Configuration file loading and saving for headless installation."""

import os
from pathlib import Path
from typing import Any, Dict

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

        # Validate required fields
        required_fields = [
            'venv_type',
            'install_path',
            'slurm_info',
            'gpu_type',
            'node_architecture',
            'install_method',
            'selected_workloads',
            'env_vars',
        ]

        missing_fields = [field for field in required_fields if field not in config_data]
        if missing_fields:
            raise ValueError(f"Configuration file is missing required fields: {missing_fields}")

        # Validate SLURM partition configuration
        slurm_info = config_data.get('slurm_info', {})
        if 'slurm' in slurm_info:
            slurm_config = slurm_info['slurm']
            gpu_partition = slurm_config.get('gpu_partition', '').strip()
            cpu_partition = slurm_config.get('cpu_partition', '').strip()

            if not gpu_partition:
                raise ValueError("GPU partition cannot be blank in headless configuration")
            if not cpu_partition:
                raise ValueError("CPU partition cannot be blank in headless configuration")

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
