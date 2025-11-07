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


"""Workload management functions for LLMB installation.

This module provides functions for finding, parsing, and processing workload metadata files,
as well as executing workload setup tasks and installation scripts.
"""

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from llmb_install.cluster.slurm import augment_env_for_job_type
from llmb_install.environment.venv_manager import (
    create_virtual_environment,
    get_venv_environment,
)


def find_metadata_files(root_dir: str) -> List[Path]:
    """Find all metadata.yaml files in the given directory and its subdirectories, excluding deprecated folder."""
    metadata_files = []
    for path in Path(root_dir).rglob("metadata.yaml"):
        if 'deprecated' not in path.parts:
            metadata_files.append(path)
    return metadata_files


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
                    'repositories': metadata.get('repositories', {}),
                    'setup': metadata.get('setup', {}),
                    'run': metadata.get('run', {}),
                    'tools': metadata.get('tools', {}),
                    'path': str(file_path.parent),
                }

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return workload_dict


def filter_tools_from_workload_list(workloads: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Filter tools from the workload list."""
    tools = {}
    for key, workload_data in workloads.items():
        if workload_data.get('general', {}).get('workload_type') == 'tools':
            tools[key] = workload_data
    return tools


def get_setup_tasks(workload_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of setup tasks for a workload.

    Behaviour:
    • If `setup.tasks` exists, return that list preserving order.
    • Otherwise, return an empty list - legacy `setup_script` is handled
      elsewhere by `run_post_install_script` for full backward compatibility.
    """
    setup_cfg: Dict[str, Any] = workload_data.get("setup", {}) or {}
    tasks: List[Dict[str, Any]] = setup_cfg.get("tasks", []) or []

    if tasks:
        return tasks

    # No explicit tasks defined
    return []


def run_setup_tasks(
    workload_key: str,
    workload_data: Dict[str, Any],
    venv_path: Optional[str],
    venv_type: Optional[str],
    install_path: str,
    slurm_info: Dict[str, Any],
    global_env_vars: Dict[str, str],
    gpu_type: str,
):
    """Execute setup tasks defined for a workload in serial order.

    Args:
        workload_key: Identifier such as "finetune_llama4-maverick".
        workload_data: Metadata dict for the workload.
        venv_path: Path to the venv to activate for this workload (may be None).
        venv_type: 'venv' or 'conda'.
        install_path: Base installation path ($LLMB_INSTALL).
        slurm_info: Cluster SLURM config as gathered earlier.
        global_env_vars: Env vars collected from the user (e.g. HF_TOKEN).
        gpu_type: GPU type (e.g., 'h100', 'gb200').
    """
    tasks = get_setup_tasks(workload_data)
    if not tasks:
        return

    workload_dir = workload_data["path"]

    for idx, task in enumerate(tasks, start=1):
        name = task.get("name", f"task_{idx}")
        cmd = task.get("cmd")
        if not cmd:
            print(f"Skipping task '{name}' - no cmd provided.")
            continue
        job_type = task.get("job_type", "local").lower()
        requires_gpus = task.get("requires_gpus", False)
        # Ensure all env values are strings to avoid type-related subprocess errors
        task_env_extra_raw = task.get("env", {}) or {}
        task_env_extra: Dict[str, str] = {k: str(v) for k, v in task_env_extra_raw.items()}

        # Compose environment
        if venv_path:
            env = get_venv_environment(venv_path, venv_type)
        else:
            env = os.environ.copy()

        env["LLMB_INSTALL"] = install_path
        env["LLMB_WORKLOAD"] = os.path.join(install_path, "workloads", workload_key)
        env["MANUAL_INSTALL"] = "false"
        env["GPU_TYPE"] = gpu_type
        # user-provided globals
        env.update(global_env_vars)
        # task-specific overrides
        env.update(task_env_extra)
        # SLURM augmentation
        augment_env_for_job_type(env, job_type, slurm_info, requires_gpus)

        banner = f"Running setup task [{workload_key}] - {name} (type: {job_type})"
        print("\n" + banner)
        print("-" * len(banner))

        try:
            if job_type == "sbatch":
                # Submit the sbatch job. Assume `cmd` contains the sbatch script path (and optional args).
                full_cmd = ["sbatch"] + shlex.split(cmd)
                result = subprocess.run(full_cmd, check=True, capture_output=True, text=True, cwd=workload_dir, env=env)

                # Attempt to extract job-id from output (handles both --parsable and default formats)
                stdout = result.stdout.strip()
                job_id_match = re.search(r"(\d+)$", stdout)
                job_id = job_id_match.group(1) if job_id_match else None
                if job_id:
                    print(f"✓ Submitted SBATCH job (id={job_id}) for task '{name}'.")
                else:
                    print(f"✓ Submitted SBATCH job for task '{name}': {stdout or '(no output)'}")
            else:
                # local, nemo2, srun are run inline - split command safely to avoid shell injection
                cmd_list = shlex.split(cmd)
                subprocess.run(cmd_list, check=True, cwd=workload_dir, env=env, capture_output=False)
                print(f"✓ Finished task '{name}' successfully.")

        except subprocess.CalledProcessError as e:
            stderr_msg = (e.stderr or '').strip()
            print(f"Error: setup task '{name}' for {workload_key} failed (return code {e.returncode}).")
            if stderr_msg:
                print(stderr_msg)
            raise


def run_post_install_script(setup_script: str, source_dir: str, env: Dict[str, str]):
    """Run a post-install setup script within the correct environment.

    Distinct from the scripted workload install, as that also creates a venv.

    Args:
        setup_script: The name of the setup script
        source_dir: The directory where the script is located
        env: The environment dictionary for running subprocesses
    """
    print("\n⚠️  WARNING: setup_script functionality is deprecated and will be removed in a future release.")
    print("   Please migrate to the 'tasks' feature in metadata.yaml for setup operations.")
    print("   See documentation: docs/recipe_guide.md#setup-tasks\n")

    script_path = os.path.join(source_dir, setup_script)
    print(f"Running post-install script: {script_path}")
    try:
        if not os.path.exists(script_path):
            print(f"Warning: Post-install script {script_path} not found, skipping.")
            return

        os.chmod(script_path, 0o755)

        subprocess.run([script_path], env=env, cwd=source_dir, check=True, text=True)
        print("\n✓ Post-install script completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"\nError running post-install script (return code: {e.returncode})")
        raise


def install_scripted_workload(
    workload_key: str,
    workload_data: Dict[str, Any],
    install_path: str,
    venv_type: str,
    env_vars: Dict[str, str],
    gpu_type: str,
) -> Optional[str]:
    """Install a workload whose dependencies are defined entirely by a shell script.

    This function is used for workloads that rely on a 'setup_script' to handle
    their setup, rather than declaring dependencies in the metadata. It will
    create a dedicated virtual environment for this workload.

    Args:
        workload_key: The unique identifier for the workload.
        workload_data: The dictionary of metadata for the workload.
        install_path: The base installation directory for all workloads.
        venv_type: The type of virtual environment to create ('venv' or 'conda').
        env_vars: The environment variables to pass to the setup script.
        gpu_type: GPU type (e.g., 'h100', 'gb200').
    Returns:
        The path to the created virtual environment, or None if no venv was required.
    """
    print(f"\n\nInstalling {workload_key} (scripted method)")
    print("-----------------------------------------")
    print("⚠️  WARNING: Scripted workload installation is deprecated and will be removed in a future release.")
    print("   Please migrate to dependency-based installation with 'tasks' feature in metadata.yaml.")
    print("   See documentation: docs/recipe_guide.md#setup-tasks\n")

    target_dir = os.path.join(install_path, "workloads", workload_key)
    os.makedirs(target_dir, exist_ok=True)

    env = os.environ.copy()
    venv_path = None

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
    env['LLMB_WORKLOAD'] = os.path.join(install_path, "workloads", workload_key)
    # Signal to setup scripts that this is an automated install (prevents automatic sqsh downloads)
    env['MANUAL_INSTALL'] = 'false'
    env['GPU_TYPE'] = gpu_type
    if env_vars:
        env_vars_str = {k: str(v) for k, v in env_vars.items()}
        env.update(env_vars_str)  # Ensure things like HF_TOKEN are set in the setup env.

    source_dir = workload_data['path']
    print(f"Installing {workload_key} to {target_dir}")

    setup_script = setup_config.get('setup_script')
    if setup_script:
        script_path = os.path.join(source_dir, setup_script)
        print(f"Running setup script: {script_path}")

        if not os.path.exists(script_path):
            print(f"Error: Setup script {script_path} not found!")
            return venv_path

        os.chmod(script_path, 0o755)

        try:
            subprocess.run([script_path], env=env, cwd=source_dir, check=True, text=True)
            print(f"\n✓ Setup script for {workload_key} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\nError running setup script for {workload_key} (return code: {e.returncode})")
            raise
    else:
        print(f"No setup script specified for {workload_key}")

    return venv_path
