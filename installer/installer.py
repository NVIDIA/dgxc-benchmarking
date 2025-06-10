#!/usr/bin/env python3

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

import os
import yaml
import shutil
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import questionary

# Define minimum Python version required for virtual environments
MIN_PYTHON_VERSION = "3.12"
MIN_PYTHON_VERSION_TUPLE = tuple(map(int, MIN_PYTHON_VERSION.split('.')))

def find_metadata_files(root_dir: str) -> list[Path]:
    """Find all metadata.yaml files in the given directory and its subdirectories, excluding deprecated folder."""
    metadata_files = []
    for path in Path(root_dir).rglob("metadata.yaml"):
        if 'deprecated' not in path.parts:
            metadata_files.append(path)
    return metadata_files

def check_pip_cache_location() -> None:
    """Check the pip cache directory location and warn if it's under /home."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "dir"],
            capture_output=True,
            text=True,
            check=True
        )
        pip_cache_dir = result.stdout.strip()
        
        if pip_cache_dir.startswith('/home'):
            print("\nWARNING: Your pip cache directory is located under /home")
            print(f"Current location: {pip_cache_dir}")
            print("This may cause space issues when installing large packages.")
            print("Consider setting PIP_CACHE_DIR environment variable to a location with more space.")
    except subprocess.CalledProcessError:
        print("Could not determine pip cache location.")

def parse_metadata_file(file_path: Path) -> Dict[str, Any]:
    """Parse a metadata.yaml file and return its contents."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def build_workload_dict(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a dictionary of available workloads from metadata.yaml files.
    Key format: 'workload_type'_'workload'
    """
    workload_dict = {}
    
    metadata_files = find_metadata_files(root_dir)
    
    for file_path in metadata_files:
        try:
            metadata = parse_metadata_file(file_path)
            
            general = metadata.get('general', {})
            workload_type = general.get('workload_type')
            workload = general.get('workload')
            
            if workload_type and workload:
                key = f"{workload_type}_{workload}"
                
                # Store metadata sections and the parent directory path for later use
                workload_dict[key] = {
                    'general': general,
                    'container': metadata.get('container', {}),
                    'setup': metadata.get('setup', {}),
                    'run': metadata.get('run', {}),
                    'path': str(file_path.parent)
                }
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return workload_dict

def get_supported_gpu_types(workloads: Dict[str, Dict[str, Any]]) -> set[str]:
    """Get all supported GPU types from workload metadata files."""
    gpu_types = set()
    
    for workload_data in workloads.values():
        run_config = workload_data.get('run', {})
        gpu_configs = run_config.get('gpu_configs', {})
        gpu_types.update(gpu_configs.keys())
    
    return gpu_types

def filter_workloads_by_gpu_type(workloads: Dict[str, Dict[str, Any]], gpu_type: str) -> Dict[str, Dict[str, Any]]:
    """Filter workloads to only include those that support the specified GPU type."""
    filtered_workloads = {}
    
    for key, workload_data in workloads.items():
        run_config = workload_data.get('run', {})
        gpu_configs = run_config.get('gpu_configs', {})
        
        # Only include workloads that have configuration for the selected GPU type
        if gpu_type in gpu_configs:
            filtered_workloads[key] = workload_data
    
    return filtered_workloads

def prompt_gpu_type(workloads: Dict[str, Dict[str, Any]]) -> str:
    """Prompt the user to select GPU type based on available workloads.
    
    Returns:
        str: The selected GPU type ('h100' or 'gb200')
    """
    print("\nGPU Type Selection")
    print("------------------")
    print("Please select the GPU type for your cluster.")
    print("This will determine which workloads are available for installation.")
    
    # Get supported GPU types from workloads, but limit to known types
    supported_types = get_supported_gpu_types(workloads)
    known_types = {'h100', 'gb200'}
    available_types = list(supported_types.intersection(known_types))
    
    if not available_types:
        print("No supported GPU types found in workload metadata.")
        raise SystemExit(1)
    
    # Sort for consistent ordering
    available_types.sort()
    
    choices = []
    for gpu_type in available_types:
        choices.append({
            'name': gpu_type.upper(),
            'value': gpu_type
        })
    
    gpu_type = questionary.select(
        "Select GPU type:",
        choices=choices,
        default=choices[0] if choices else None
    ).ask()
    
    if gpu_type is None:
        print("\nInstallation cancelled.")
        raise SystemExit(1)
    
    print(f"Selected GPU type: {gpu_type}")
    return gpu_type

def prompt_node_architecture(gpu_type: str) -> str:
    """Prompt the user for the node CPU architecture, with auto-selection based on GPU type.
    
    Args:
        gpu_type: The selected GPU type ('h100' or 'gb200')
        
    Returns:
        str: The selected node CPU architecture ('x86_64' or 'aarch64')
    """
    print("\nGPU Node - CPU Architecture")
    print("----------------------------")
    
    if gpu_type == 'gb200':
        print("GB200 systems are ARM-based (aarch64). Auto-selecting aarch64 architecture.")
        return 'aarch64'
    
    print("Please select the CPU architecture of your GPU nodes to ensure correct container image downloads.")
    print("Choosing the wrong architecture (e.g., aarch64 for an x86_64 system) will result in 'Exec format error'.")
    
    if gpu_type == 'h100':
        print("\nH100 systems are typically x86_64 based.")
    
    architecture = questionary.select(
        "Select the CPU architecture of your GPU nodes:",
        choices=[
            {'name': "x86_64", 'value': 'x86_64'},
            {'name': "aarch64", 'value': 'aarch64'}
        ],
        default={'name': "x86_64", 'value': 'x86_64'}
    ).ask()

    if architecture is None:
         print("\nInstallation cancelled.")
         raise SystemExit(1)

    print(f"Selected node architecture: {architecture}")
    return architecture

def print_available_workloads(workloads: Dict[str, Dict[str, Any]]) -> None:
    """Print all available workloads and their configurations."""
    print("\nAvailable Workloads:")
    for key, data in workloads.items():
        print(f"\n{key}:")
        print(f"  Path: {data['path']}")
        print(f"  General: {data['general']}")
        print(f"  Container: {data['container']}")
        print(f"  Setup: {data['setup']}")

def prompt_install_location() -> str:
    """Prompt the user for the installation location."""
    print("\nInstallation Location")
    print("--------------------")
    print("Note: Workloads can be quite large and may require significant disk space.")
    print("It is recommended to install on a high-performance file system (e.g., Lustre, GPFS).")
    print("Example paths: /lustre/your/username/llmb or /gpfs/your/username/llmb\n")
    
    def validate_path(path: str) -> bool | str:
        if not path:
            return True
            
        # Remove any quotes that might be present
        path = path.strip('"\'')
        
        # Skip validation for incomplete paths during typing
        if path.endswith(os.sep):
            return True
            
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.access(parent_dir, os.W_OK):
            return "No write permission in the parent directory."
        return True
    
    location = questionary.path(
        "Where would you like to install the workloads?",
        validate=validate_path
    ).ask()
    
    if location is None:
        return None
        
    location = os.path.expanduser(location.strip('"\''))
    return location


def prompt_slurm_info() -> dict:
    """Prompt the user for SLURM information.
    
    Returns:
        dict: Dictionary containing SLURM configuration information
    """
    print("\nSLURM Configuration")
    print("------------------")
    print("Please provide the following SLURM information for your cluster:\n")
    print("Note:This information will also be used for the launcher.")
    
    # Try to get accounts the user is associated with
    try:
        username = os.environ.get('USER') or os.environ.get('USERNAME')
        if not username:
            result = subprocess.run(["whoami"], capture_output=True, text=True, check=True)
            username = result.stdout.strip()
        
        # Get accounts associated with the current user
        result = subprocess.run(
            ["sacctmgr", "show", "assoc", f"user={username}", "format=account", "-p", "-n"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse accounts and remove the pipe character at the end
        accounts = [a.strip().rstrip('|') for a in result.stdout.strip().split('\n') if a.strip()]
        
        accounts = sorted(set(accounts))
        
        if accounts:
            # If there are too many accounts, limit the display to avoid overwhelming the user
            if len(accounts) > 10:
                display_accounts = accounts[:10]
                print(f"Your accounts (showing 10 of {len(accounts)}): {', '.join(display_accounts)}")
                print("Type '?' to see all your accounts.")
            else:
                print(f"Your accounts: {', '.join(accounts)}")
        else:
            print("Could not automatically detect your SLURM accounts. Please enter your account name manually.")
            accounts = []
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Could not retrieve account information. You'll need to enter it manually.")
        accounts = []
    
    def validate_account(account_input):
        if not account_input:
            return True
            
        if account_input == '?' and accounts and len(accounts) > 10:
            # Show all accounts when user types '?'
            print("\nAll available accounts:")
            # Display accounts in a more readable format, 5 per line
            for i in range(0, len(accounts), 5):
                print("  " + ", ".join(accounts[i:i+5]))
            print()
            return "Type your account name"
        
        if accounts and account_input not in accounts:
            return f"Warning: '{account_input}' is not in the list of available accounts. Press Enter to use it anyway, or try another account."
            
        return True
    
    account = questionary.text(
        "SLURM account:",
        default="",
        validate=validate_account
    ).ask()
    print()
    
    if account is None:
        return None
    
    try:
        result = subprocess.run(
            ["sinfo", "--noheader", "-o", "%P"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse the output to identify the default partition (marked with *)
        raw_partitions = result.stdout.strip().split('\n')
        default_partition = None
        partitions = []
        
        for p in raw_partitions:
            p = p.strip()
            if p.endswith('*'):
                default_partition = p.rstrip('*')
                partitions.append(default_partition)
            else:
                partitions.append(p)
        
        partitions = sorted(set(partitions))
        
        if partitions:
            # If there are too many partitions, limit the display
            if len(partitions) > 10:
                display_partitions = partitions[:10]
                print(f"Available partitions (showing 10 of {len(partitions)}): {', '.join(display_partitions)}")
                print("Type '?' to see all available partitions.")
            else:
                print(f"Available partitions: {', '.join(partitions)}")
                
            if default_partition:
                print(f"Default partition: {default_partition}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Could not retrieve partition information. You'll need to enter it manually.")
        partitions = []
    
    def validate_partition(partition_input):
        if not partition_input:
            return True
            
        if partition_input == '?' and partitions and len(partitions) > 10:
            # Show all partitions when user types '?'
            print("\nAll available partitions:")
            # Display partitions in a more readable format, 5 per line
            for i in range(0, len(partitions), 5):
                print("  " + ", ".join(partitions[i:i+5]))
            print()
            return "Type your partition name"
            
        if partitions and partition_input not in partitions:
            return f"Warning: '{partition_input}' is not in the list of available partitions. Press Enter to use it anyway, or try another partition."
            
        return True
    
    print("This is the partition where you want to run the benchmarks.")
    partition = questionary.text(
        "SLURM GPU partition:",
        default=default_partition if default_partition else "",
        validate=validate_partition
    ).ask()
    print()

    if partition is None:
        return None

    cpu_default = None
    cpu_partitions = [p for p in partitions if p.startswith('cpu')]
    if cpu_partitions:
        cpu_default = cpu_partitions[0]

    print("This is the partition for tasks that do not require GPUs. If you only have GPU partitions, use a low priority queue.")
    cpu_partition = questionary.text(
        "SLURM CPU partition:",
        default=cpu_default if cpu_default else "",
        validate=validate_partition
    ).ask()
    print()

    if cpu_partition is None:
        return None

    # Auto-detect GRES information for both partitions
    gpu_partition_gres_value = None
    cpu_partition_gres_value = None

    # Helper function to parse GRES output for GPU count
    def parse_gpu_gres(gres_output: str) -> Optional[int]:
        if gres_output == '(null)' or 'gpu:' not in gres_output:
            return None
        try:
            # Extract the number after 'gpu:'
            # Handle cases like 'gpu:8,mib:100' by taking the first part
            return int(gres_output.split('gpu:')[1].split(',')[0].strip())
        except ValueError:
            return None

    try:
        result = subprocess.run(
            ["sinfo", "--noheader", "-p", f"{partition},{cpu_partition}", "-o", "%P,%G"],
            capture_output=True,
            text=True,
            check=True
        )
        gres_outputs = result.stdout.strip().splitlines()

        # Parse the output lines which are now in the format "partition_name,GRES_info"
        for line in gres_outputs:
            parts = line.split(',', 1) # Split only on the first comma
            if len(parts) == 2:
                p_name, gres_raw = parts
                p_name = p_name.strip().rstrip('*')
                gres_raw = gres_raw.strip()

                if p_name == partition:
                    gpu_partition_gres_value = parse_gpu_gres(gres_raw)
                elif p_name == cpu_partition:
                    cpu_partition_gres_value = parse_gpu_gres(gres_raw)

        if gpu_partition_gres_value is not None:
            print(f"Auto-detected GPU partition GRES: {gpu_partition_gres_value} GPUs per node")
        else:
            print(f"No GPU GRES for partition '{partition}' detected.")
            
        if cpu_partition_gres_value is not None:
            print(f"Auto-detected CPU partition GRES: {cpu_partition_gres_value} GPUs per node")
        else:
            print(f"No GPU GRES detected for CPU partition '{cpu_partition}' (expected)")

    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"Note: Could not determine GRES information for partitions {partition},{cpu_partition}: {e}")

    slurm_config = {
        "slurm": {
            "account": account,
            "gpu_partition": partition,
            "cpu_partition": cpu_partition,
            "gpu_partition_gpus_from_gres": gpu_partition_gres_value,
            "cpu_partition_gpus_from_gres": cpu_partition_gres_value
        }
    }

    return slurm_config

def prompt_workload_selection(workloads: Dict[str, Dict[str, Any]]) -> Optional[list[str]]:
    """Prompt the user to select workloads to install."""
    if not workloads:
        print("No workloads found!")
        return None
    
    choices = [
        {
            'name': "Install all workloads",
            'value': "all"
        }
    ]
    
    for key, data in sorted(workloads.items()):
        choices.append({
            'name': key,
            'value': key
        })
    
    selected = questionary.checkbox(
        "Select workloads to install (use space to select, enter to confirm):",
        choices=choices,
        validate=lambda selected: len(selected) > 0 or "Please select at least one workload"
    ).ask()
    
    if not selected:
        return None
        
    if "all" in selected:
        return list(workloads.keys())
    
    return selected

def is_venv_installed() -> bool:
    """Check if current python process has venv installed."""
    try:
        import venv
        return True
    except ImportError:
        return False
    
def is_conda_installed() -> bool:
    return shutil.which('conda') is not None

def is_enroot_installed() -> bool:
    return shutil.which('enroot') is not None

def create_virtual_environment(venv_path: str, venv_type: str) -> None:
    """Create a virtual environment at the specified path using the given environment type.
    
    Args:
        venv_path: Path where the virtual environment should be created
        venv_type: Type of virtual environment to create ('venv' or 'conda')
    
    Raises:
        ValueError: If venv_type is not supported
        subprocess.CalledProcessError: If conda environment creation fails
    """
    print(f"Creating virtual environment at {venv_path}...")
    
    if venv_type == 'venv':
        import venv
        builder = venv.EnvBuilder(
            system_site_packages=False,
            clear=True,
            with_pip=True
        )
        builder.create(venv_path)

    elif venv_type == 'conda':
        import subprocess
        try:
            # Create conda environment with minimum required Python version
            subprocess.run([
                'conda', 'create',
                '-p', venv_path,
                f'python={MIN_PYTHON_VERSION}',
                '--yes'
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating conda environment: {e}")
            raise
    else:
        raise ValueError(f"Unsupported environment type: {venv_type}")
    
    print(f"Created virtual environment at: {venv_path}")

def get_venv_environment(venv_path: str, venv_type: str) -> dict:
    """Prepare environment variables for running commands in a virtual environment.
    
    Args:
        venv_path: Path to the virtual environment
        venv_type: Type of virtual environment ('venv' or 'conda')
        
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
    
    if venv_type == 'venv':
        env['VIRTUAL_ENV'] = venv_path
    elif venv_type == 'conda':
        env['CONDA_PREFIX'] = venv_path
    
    return env


def install_workload(workload_key: str, workload_data: Dict[str, Any], install_path: str, venv_type: str, node_architecture: str) -> None:
    """Install a single workload to the specified location.
    
    Args:
        workload_key: Unique identifier for the workload
        workload_data: Dictionary containing workload configuration
        install_path: Base installation directory
        venv_type: Type of virtual environment to create ('venv' or 'conda')
        node_architecture: The selected node architecture ('x86_64' or 'aarch64')
    """
    print(f"\n\nInstalling {workload_key}")
    print("--------------------------------")
    target_dir = os.path.join(install_path, "workloads", workload_key)
    os.makedirs(target_dir, exist_ok=True)
    
    env = os.environ.copy()

    setup_config = workload_data.get('setup', {})
    if setup_config.get('venv_req', False):
        venv_name = f"{workload_key}_venv"
        venvs_dir = os.path.join(install_path, "venvs")
        os.makedirs(venvs_dir, exist_ok=True)
        venv_path = os.path.join(venvs_dir, venv_name)
        create_virtual_environment(venv_path, venv_type)
        env = get_venv_environment(venv_path, venv_type)
    else:
        print(f"No virtual environment required for {workload_key}")


    env['LLMB_INSTALL'] = install_path
    # Signal to setup scripts that this is an automated install (prevents automatic sqsh downloads)
    env['MANUAL_INSTALL'] = 'false'

    source_dir = workload_data['path']
    print(f"Installing {workload_key} to {target_dir}")
    
    setup_script = setup_config.get('setup_script')
    if setup_script:
        script_path = os.path.join(source_dir, setup_script)
        print(f"Running setup script: {script_path}")
        try:
            if not os.path.exists(script_path):
                print(f"Warning: Setup script {script_path} not found")
                return
                
            os.chmod(script_path, 0o755)
            
            subprocess.run(
                [script_path],
                env=env,
                cwd=source_dir,
                check=True,
                text=True
            )
            print(f"\n{workload_key} setup script completed successfully.")
                
        except subprocess.CalledProcessError as e:
            print(f"\nError running setup script (return code: {e.returncode})")
            raise

def get_required_images(workloads: Dict[str, Dict[str, Any]], selected_keys: list[str]) -> Dict[str, str]:
    """Get the dictionary of container images required by the selected workloads.
    
    Returns:
        Dict[str, str]: Dictionary mapping image URLs to their desired filenames
    """
    # Use filename as key to deduplicate images that have different URL formats
    # but represent the same image (e.g., nvcr.io/nvidia/nemo vs nvcr.io#nvidia/nemo)
    filename_to_url = {}
    
    for key in selected_keys:
        workload = workloads[key]
        container_config = workload.get('container', {})
        
        # Handle image field which can be a string, list, or list of dicts
        images = container_config.get('images', [])
        
        if isinstance(images, str):
            # Simple string format - generate filename from URL
            filename = _generate_image_filename(images)
            filename_to_url[filename] = images
        elif isinstance(images, list):
            for image in images:
                if isinstance(image, str):
                    # Simple string format - generate filename from URL
                    filename = _generate_image_filename(image)
                    filename_to_url[filename] = image
                elif isinstance(image, dict):
                    url = image.get('url')
                    name = image.get('name')
                    if url:
                        if name:
                            filename = f"{name}.sqsh"
                            filename_to_url[filename] = url
                        else:
                            filename = _generate_image_filename(url)
                            filename_to_url[filename] = url
    
    # Convert back to URL -> filename mapping for compatibility with existing code
    required_images = {url: filename for filename, url in filename_to_url.items()}
    return required_images

def _generate_image_filename(image_url: str) -> str:
    """Generate a filename for a container image URL.
    
    Args:
        image_url: Container image URL (e.g., 'nvcr.io/nvidia/nemo:25.02.01' or 'nvcr.io#nvidia/nemo:sha256:abc123...')
    
    Returns:
        str: Generated filename ending with .sqsh
    """
    try:
        # Normalize URL separators - replace # with / for consistent parsing
        normalized_url = image_url.replace('#', '/')
        
        # Extract parent name, image name and version/digest from image URL
        parts = normalized_url.split('/')
        parent = parts[-2]  # Get 'nvidia' from URL
        name_and_tag = parts[-1]  # Get 'nemo:25.02.01' or 'nemo:sha256:abc123...'
        
        if ':' in name_and_tag:
            name, tag_or_digest = name_and_tag.split(':', 1)
            
            # Check if this is a SHA digest
            if tag_or_digest.startswith('sha256:'):
                # For SHA digests, use first 12 characters of the hash
                hash_part = tag_or_digest.split(':', 1)[1][:12]
                version = f"sha256-{hash_part}"
            else:
                # Regular tag
                version = tag_or_digest
        else:
            # No tag specified, use 'latest'
            name = name_and_tag
            version = "latest"
        
        return f"{parent}+{name}+{version}.sqsh"
        
    except (IndexError, ValueError) as e:
        # Fallback: use a sanitized version of the full URL
        sanitized = image_url.replace('/', '_').replace(':', '_').replace('#', '_')
        return f"{sanitized}.sqsh"

def prompt_environment_type() -> str:
    """Prompt the user to select their preferred environment type (venv or conda).
       Includes checks for Python version compatibility."""

    current_python_version = sys.version_info[:3]
    
    print("Environment Configuration")
    print("------------------------")
    print(f"Current Python version: {'.'.join(map(str, current_python_version))}")
    print(f"Minimum Python version for venvs: {MIN_PYTHON_VERSION}")
    
    venv_available = is_venv_installed()
    conda_available = is_conda_installed()
    
    can_use_venv = venv_available and current_python_version >= MIN_PYTHON_VERSION_TUPLE
    can_use_conda = conda_available
    
    selected_venv_type = None

    if can_use_venv:
        if can_use_conda:
            # Both are available and venv version is ok, offer choice, default to venv
            print("Both venv (with compatible Python version) and conda are available.")
            selected_venv_type = questionary.select(
                "Select your preferred environment type:",
                choices=[
                    {'name': "venv (Python's built-in virtual environment) - Recommended", 'value': 'venv'},
                    {'name': "Conda (Anaconda/Miniconda environment)", 'value': 'conda'}
                ],
                default={'name': "venv (Python's built-in virtual environment) - Recommended", 'value': 'venv'}
            ).ask()
        else:
            # Only venv is available and version is ok, use venv
            print("Only venv (with compatible Python version) is available. Using venv.")
            selected_venv_type = 'venv'
    elif can_use_conda:
        if venv_available and current_python_version < MIN_PYTHON_VERSION_TUPLE:
            # venv is available but Python version is too low, conda is available
            print(f"WARNING: Your current Python version ({'.'.join(map(str, current_python_version))}) is below the minimum required ({MIN_PYTHON_VERSION}) for venv workloads.")
            print("Conda is available and will be used instead.")
        else:
            print("venv is not available. Using conda.")
            
        selected_venv_type = 'conda'
    else:
        # Neither venv nor conda is available, or venv available but wrong version and conda not available
        error_message = "Error: Cannot proceed with installation."
        if venv_available and current_python_version < MIN_PYTHON_VERSION_TUPLE:
             error_message += f"Your current Python version ({'.'.join(map(str, current_python_version))}) is below the minimum required ({MIN_PYTHON_VERSION}) for venv workloads."
        else:
            error_message += "Neither venv nor conda is available on your system."

        error_message += "A compatible virtual environment is required to install the recipes. Please ensure venv (with Python = 3.12.x) or conda is installed."
        print(error_message)
        raise SystemExit(1)

    if selected_venv_type is None:
         print("Environment selection cancelled. Exiting.")
         raise SystemExit(1)

    print(f"Using environment type: {selected_venv_type}")
    return selected_venv_type

def create_cluster_config(install_path: str, root_dir: str, selected_workloads: list[str], 
                         slurm_info: dict, env_vars: dict, gpu_type: str, node_architecture: str, venv_type: str) -> None:
    """Create the cluster_config.yaml file with all installation configuration.
    
    Args:
        install_path: Base installation directory
        root_dir: Path to the LLMB repository root
        selected_workloads: List of selected workload keys
        slurm_info: SLURM configuration dictionary
        env_vars: Environment variables dictionary
        gpu_type: Selected GPU type
        node_architecture: Selected node architecture
        venv_type: Type of virtual environment used ('venv' or 'conda')
    """
    cluster_config = {
        'launcher': {
            'llmb_repo': root_dir,
            'llmb_install': install_path,
            'gpu_type': gpu_type
        },
        'environment': env_vars,
        'slurm': {
            'account': slurm_info['slurm']['account'],
            'gpu_partition': slurm_info['slurm']['gpu_partition'],
            'gpu_gres': slurm_info['slurm'].get('gpu_partition_gpus_from_gres'),
            'cpu_partition': slurm_info['slurm']['cpu_partition'],
            'cpu_gres': slurm_info['slurm'].get('cpu_partition_gpus_from_gres')
        },
        'workloads': {
            'installed': selected_workloads,
            'config': {}
        }
    }
    
    # Add workload-specific configuration (venv paths and types)
    venvs_dir = os.path.join(install_path, "venvs")
    for workload_key in selected_workloads:
        venv_name = f"{workload_key}_venv"
        venv_path = os.path.join(venvs_dir, venv_name)
        
        cluster_config['workloads']['config'][workload_key] = {
            'venv_path': venv_path,
            'venv_type': venv_type
        }
    
    # Write the cluster config file
    config_path = os.path.join(install_path, "cluster_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(cluster_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created cluster configuration: {config_path}")

def fetch_container_images(images: Dict[str, str], install_path: str, node_architecture: str, install_method: str, slurm_info: Dict[str, Any]) -> None:
    """Fetch container images using enroot and save them as sqsh files.
    
    Args:
        images: Dictionary mapping image URLs to desired filenames
        install_path: Directory where the sqsh files should be saved
        node_architecture: The selected node architecture ('x86_64' or 'aarch64')
        install_method: The selected installation method ('local' or 'slurm')
        slurm_info: Dictionary containing SLURM configuration information
    
    The function will:
    1. Use provided filenames or generate them from image URLs
    2. Create sqsh files using enroot import
    3. Skip if sqsh file already exists
    """
    images_dir = os.path.join(install_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for image_url, filename in sorted(images.items()):
        try:
            sqsh_file = os.path.join(images_dir, filename)
            
            if os.path.exists(sqsh_file):
                print(f"Skipping {image_url} -- {sqsh_file} - file already exists.")
                continue
                
            cmd_args = ["enroot", "import", "-o", sqsh_file]
            if node_architecture:
                 cmd_args.extend([f"-a", node_architecture])
            cmd_args.append(f"docker://{image_url}")

            if install_method == 'slurm':
                srun_args = ["srun", f"--account={slurm_info['slurm']['account']}", f"--partition={slurm_info['slurm']['cpu_partition']}", "-t", "60", "--exclusive", "--pty"]
                #if slurm_info['slurm'].get('gpus_per_node'):
                #     srun_args.append(f"--gres=gpu:{slurm_info['slurm']['gpus_per_node']}")

                full_cmd_args = srun_args + cmd_args
                print(f"Submitting download via SLURM: {' '.join(full_cmd_args)}")
            else:
                full_cmd_args = cmd_args
                print(f"Downloading {image_url} locally: {' '.join(full_cmd_args)}")

            try:
                subprocess.run(full_cmd_args, check=True, text=True)
                print(f"Successfully downloaded {sqsh_file}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error: Failed to download {sqsh_file}: {str(e)}\n")
                if install_method == 'slurm':
                    print("Please ensure the compute nodes you selected in the 'CPU_PARTITION' option are configured correctly, or try a local installation (if available).")
                else:
                    print("Ensure your system has appropriate resources or try a SLURM-based installation.")
                raise SystemExit(1)
                
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
            raise SystemExit(1)

def prompt_install_method() -> str:
    """Prompt the user to select the installation method (local or slurm).
    Includes checks for enroot availability and SLURM job detection.

    Returns:
        str: The selected installation method ('local' or 'slurm')
    """
    print("\nInstallation Method")
    print("--------------------")
    print("Please select how you would like to perform longer running tasks like container image fetching and dataset downloads.")
    print(" 'local': Tasks will be run directly on the current machine. Requires enroot to be available.")
    print(" 'slurm': Tasks will be submitted as SLURM jobs. This is recommended for clusters where interactive nodes may have limited resources or network access.")
    
    # Check if we're running within a SLURM job
    running_in_slurm_job = 'SLURM_JOB_ID' in os.environ
    
    enroot_available = is_enroot_installed()
    
    if running_in_slurm_job:
        print("\nDetected: Running within a SLURM job (SLURM_JOB_ID found in environment).")
        print("SLURM installation method is not supported when running within a SLURM job.")
        
        if not enroot_available:
            print("\nError: Cannot proceed with installation.")
            print("You are running within a SLURM job, but enroot is not available on this system.")
            print("Local installation requires enroot for container image downloading.")
            print("Please either:")
            print("  1. Run the installer from outside a SLURM job, or")
            print("  2. Ensure enroot is available on the compute nodes")
            raise SystemExit(1)
        
        print("Automatically defaulting to local installation method.")
        method = 'local'
    elif enroot_available:
        method = questionary.select(
            "Select installation method:",
            choices=[
                {'name': "Local (current machine)", 'value': 'local'},
                {'name': "SLURM", 'value': 'slurm'}
            ],
            default={'name': "Local (current machine)", 'value': 'local'}
        ).ask()
    else:
        print("\nNote: enroot is not available on this system.")
        print("Local installation requires enroot for container image downloading.")
        print("Automatically selecting SLURM-based installation.\n")
        method = 'slurm'

    if method is None:
         print("\nInstallation cancelled.")
         raise SystemExit(1)

    print(f"Selected installation method: {method}")
    return method

def prompt_environment_variables() -> Dict[str, str]:
    """Prompt the user for environment variables.
    
    Currently prompts for HF_TOKEN, but designed to be easily expandable
    for additional environment variables in the future.
    
    Returns:
        Dict[str, str]: Dictionary containing the environment variables
    """
    print("\nEnvironment Variables")
    print("--------------------")
    print("Some workloads require specific environment variables to function properly.")
    print("Please provide the following environment variables:\n")
    
    env_vars = {}
    
    # HF_TOKEN validation function - now optional
    def validate_hf_token(token: str) -> bool | str:
        if not token:
            return True  # Allow empty token
        if not token.startswith('hf_'):
            return "HF_TOKEN must start with 'hf_' if provided."
        return True
    
    # Get current HF_TOKEN from environment if it exists
    current_hf_token = os.environ.get('HF_TOKEN', '')
    
    if current_hf_token:
        print(f"Found existing HF_TOKEN in environment: {current_hf_token[:10]}...")
    
    print("HuggingFace Token (HF_TOKEN) - Some workloads require this for accessing HuggingFace models and datasets.")
    print("You can get your token from: https://huggingface.co/settings/tokens")
    print("Note: If you're sure you don't need HF_TOKEN for your selected workloads, this can be left blank.")
    
    hf_token = questionary.text(
        "Enter your HuggingFace token (HF_TOKEN) or leave blank:",
        default=current_hf_token,
        validate=validate_hf_token
    ).ask()
    
    if hf_token is None:
        print("\nInstallation cancelled.")
        raise SystemExit(1)
    
    # Only add HF_TOKEN to env_vars if it's not empty
    if hf_token.strip():
        env_vars['HF_TOKEN'] = hf_token.strip()
        print("✓ HF_TOKEN configured successfully")
    else:
        print("✓ HF_TOKEN left blank - some workloads may require you to set this manually later")
    
    return env_vars

def create_llmb_run_symlink(root_dir: str, install_path: str) -> None:
    """Create a symbolic link to the llmb-run script in the install_path."""
    print("\nCreating llmb-run symbolic link")
    print("----------------------------")

    llmb_run_source = os.path.join(root_dir, "llmb-run", "llmb-run")
    llmb_run_symlink_target = os.path.join(install_path, "llmb-run")

    try:
        if os.path.exists(llmb_run_source):
            # Remove existing symlink if it exists to avoid FileExistsError
            if os.path.exists(llmb_run_symlink_target) or os.path.islink(llmb_run_symlink_target):
                os.remove(llmb_run_symlink_target)
            os.symlink(llmb_run_source, llmb_run_symlink_target)
            print(f"✓ Created symbolic link to {llmb_run_source} at {llmb_run_symlink_target}")
        else:
            print(f"✗ Warning: llmb-run script not found at {llmb_run_source}. Skipping symlink creation.")
    except Exception as e:
        print(f"✗ Error creating symbolic link: {e}")

def main():
    # Get the LLMB root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    check_pip_cache_location()
    
    workloads = build_workload_dict(root_dir)
    
    #print_available_workloads(workloads)
    
    try:
        install_path = prompt_install_location()
        if not install_path:
            print("\nInstallation cancelled.")
            return
        
        slurm_info = prompt_slurm_info()
        if not slurm_info:
            print("\nInstallation cancelled.")
            return

        gpu_type = prompt_gpu_type(workloads)
        slurm_info['slurm']['node_architecture'] = prompt_node_architecture(gpu_type)
        print()

        # Filter workloads to only show those compatible with selected GPU type
        filtered_workloads = filter_workloads_by_gpu_type(workloads, gpu_type)
        if not filtered_workloads:
            print(f"No workloads found that support {gpu_type} GPU type.")
            return

        venv_type = prompt_environment_type()
        print()

        install_method = prompt_install_method()
        print()
        
        selected = prompt_workload_selection(filtered_workloads)
        if not selected:
            print("\nInstallation cancelled.")
            return

        env_vars = prompt_environment_variables()
        if env_vars is None:
            print("\nInstallation cancelled.")
            return
        print()

    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        return
    
    if not selected:
        print("\nNo workloads selected. Exiting.")
        return
    
    # Setup install directory structure
    os.makedirs(install_path, exist_ok=True)
    os.makedirs(os.path.join(install_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(install_path, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(install_path, "workloads"), exist_ok=True)
    os.makedirs(os.path.join(install_path, "venvs"), exist_ok=True)
    
    create_llmb_run_symlink(root_dir, install_path)
    
    print("\nDownloading required container images.")
    print("--------------------------------")
    required_images = get_required_images(filtered_workloads, selected)
    print("\nRequired container images:")
    for image, filename in sorted(required_images.items()):
        print(f"  - {image} -> {filename}")
    print("\n")
    
    fetch_container_images(required_images, install_path, slurm_info['slurm']['node_architecture'], install_method, slurm_info)
    
    for workload_key in selected:
        install_workload(workload_key, filtered_workloads[workload_key], install_path, venv_type, slurm_info['slurm']['node_architecture'])
    
    create_cluster_config(install_path, root_dir, selected, slurm_info, env_vars, gpu_type, slurm_info['slurm']['node_architecture'], venv_type)
    
    print(f"\nInstallation complete! Workloads have been installed to: {install_path}")

if __name__ == "__main__":
    main() 
