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

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import questionary
import yaml

# Define [minimum, maximum) Python version required for virtual environments
MIN_PYTHON_VERSION = "3.12"
MIN_PYTHON_VERSION_TUPLE = tuple(map(int, MIN_PYTHON_VERSION.split('.')))
MAX_PYTHON_VERSION = "3.13"
MAX_PYTHON_VERSION_TUPLE = tuple(map(int, MAX_PYTHON_VERSION.split('.')))

def _select_from_by_gpu(mapping: Any, gpu_type: str, item_label: str) -> Any:
    """Select a value from a {'by_gpu': {...}} mapping with optional 'default'.

    If mapping does not contain 'by_gpu', it is returned unchanged.
    Raises ValueError when selection is impossible.
    """
    if not isinstance(mapping, dict) or 'by_gpu' not in mapping:
        return mapping
    table = mapping.get('by_gpu') or {}
    if not isinstance(table, dict):
        raise ValueError(f"Invalid by_gpu mapping for {item_label}")
    if gpu_type in table:
        return table[gpu_type]
    if 'default' in table:
        return table['default']
    raise ValueError(
        f"No {item_label} defined for GPU '{gpu_type}' and no 'default' provided."
    )

def _resolve_images_for_gpu(images_field: Any, gpu_type: str) -> list[Any]:
    """Normalize container.images for a specific GPU type.

    Supports:
      - list[str | {url,name}]
      - { by_gpu: { h100|b200|gb200|default: list[str|{url,name}] } }
    Returns a list regardless of input form.
    """
    resolved = _select_from_by_gpu(images_field, gpu_type, "container.images")
    if isinstance(resolved, list):
        return resolved
    return [resolved]

def _resolve_repositories_for_gpu(repos_field: Any, gpu_type: str) -> Dict[str, Dict[str, str]]:
    """Resolve repositories for a specific GPU type.

    Supports a flat mapping {repo_key: {url, commit}} or a top-level
    { by_gpu: { h100|b200|gb200|default: { repo_key: {url, commit} } } }.
    """
    # If empty or None, return empty mapping
    if not repos_field:
        return {}

    # If top-level by_gpu wrapper exists, select for this GPU
    selected = _select_from_by_gpu(repos_field, gpu_type, "repositories")

    if not isinstance(selected, dict):
        raise ValueError("repositories must resolve to a mapping of repo_key -> {url, commit}")

    # Validate and normalize entries
    out: Dict[str, Dict[str, str]] = {}
    for name, entry in selected.items():
        if not isinstance(entry, dict) or 'url' not in entry or 'commit' not in entry:
            raise ValueError(f"Invalid repository entry for '{name}' after GPU resolution")
        out[name] = {'url': entry['url'], 'commit': entry['commit']}
    return out

def resolve_gpu_overrides(
    workloads: Dict[str, Dict[str, Any]], gpu_type: str
) -> Dict[str, Dict[str, Any]]:
    """Resolve GPU-specific image and repository overrides for each workload.

    Returns a deep-copied structure where:
      - container.images is a concrete list
      - repositories is a flat {repo_key: {url, commit}} mapping
    """
    specialized: Dict[str, Dict[str, Any]] = {}
    for key, workload in workloads.items():
        wd_copy = copy.deepcopy(workload)

        # Resolve images
        container_cfg = wd_copy.get('container', {}) or {}
        if 'images' in container_cfg:
            container_cfg['images'] = _resolve_images_for_gpu(container_cfg['images'], gpu_type)
            wd_copy['container'] = container_cfg

        # Resolve repositories (top-level by_gpu wrapper supported)
        repos_cfg = wd_copy.get('repositories', {}) or {}
        if repos_cfg:
            wd_copy['repositories'] = _resolve_repositories_for_gpu(repos_cfg, gpu_type)

        specialized[key] = wd_copy
    return specialized

def _canonicalize_dependencies_for_grouping(dependencies: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Create a canonical representation of dependencies for grouping.

    Goal: group workloads that effectively install the same code even if the
    acquisition method differs (e.g., script vs pip) or wrappers were present.

    Strategy:
      - For pip entries:
          • Strings are kept as canonical strings (prefixed for type).
          • Dicts with url+commit collapse to canonical triplets, ignoring
            editable/install_target/repo_key differences.
      - For git entries:
          • Collapse to canonical url+commit pairs, ignoring install_method.
      - Return an object with sorted, deduplicated lists for stable hashing.
    """
    if not dependencies:
        return None

    canonical_pip: set[str] = set()
    canonical_sources: set[str] = set()
    pip_git_pairs: set[tuple[str, str]] = set()
    git_pairs: set[tuple[str, str]] = set()

    # Pip normalization
    for item in dependencies.get('pip', []) or []:
        if isinstance(item, str):
            canonical_pip.add(f"pip-spec|{item}")
        elif isinstance(item, dict):
            url = item.get('url')
            commit = item.get('commit')
            if url and commit:
                # Track as a source; acquisition method differences are ignored
                pip_git_pairs.add((url, commit))
            else:
                # Fallback to stable dump of the dict excluding non-functional fields
                tmp = {k: v for k, v in item.items() if k not in ('editable', 'install_target', 'repo_key')}
                canonical_pip.add(f"pip-dict|{json.dumps(tmp, sort_keys=True)}")

    # Git normalization
    for _, entry in (dependencies.get('git', {}) or {}).items():
        if not isinstance(entry, dict):
            continue
        url = entry.get('url')
        commit = entry.get('commit')
        if url and commit:
            git_pairs.add((url, commit))

    # Unify sources across channels (pip-git and git)
    for url, commit in pip_git_pairs.union(git_pairs):
        canonical_sources.add(f"src|{url}|{commit}")

    return {
        'sources': sorted(canonical_sources),
        'pip': sorted(canonical_pip),
    }

def _has_editable_pip_dep(dependencies: Optional[Dict[str, Any]]) -> bool:
    """Return True if any pip dependency requests an editable install.

    Editable installs are isolated to their own venv to avoid cross-recipe
    interference from mutable source checkouts.
    """
    if not dependencies:
        return False
    for item in dependencies.get('pip', []) or []:
        if isinstance(item, dict) and item.get('editable'):
            return True
    return False

def detect_virtual_environment() -> Optional[str]:
    """Detect if running in a virtual environment and return its type.
    
    Returns:
        Optional[str]: 'conda', 'venv', or None if not in a virtual environment
    """
    # Check for conda environment
    if os.environ.get('CONDA_PREFIX') or os.environ.get('CONDA_DEFAULT_ENV'):
        return 'conda'
    
    # Check for venv environment
    if os.environ.get('VIRTUAL_ENV'):
        return 'venv'
    
    return None

def get_clean_environment_for_subprocess() -> Dict[str, str]:
    """Create a clean environment for subprocess execution without virtual environment pollution.
    
    This function removes virtual environment paths and variables that can cause
    architecture mismatch issues when running commands on different nodes via SLURM.
    
    Returns:
        Dict[str, str]: Clean environment variables for subprocess execution
    """
    env = os.environ.copy()
    venv_type = detect_virtual_environment()
    
    if venv_type == 'conda':
        # Remove conda-specific environment variables
        conda_prefix = env.get('CONDA_PREFIX')
        conda_vars_to_remove = [
            'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_PROMPT_MODIFIER',
            'CONDA_SHLVL', 'CONDA_PYTHON_EXE', 'CONDA_EXE', '_CE_CONDA',
            '_CE_M'
        ]
        for var in env.keys():
            if var.startswith('CONDA_PREFIX_'):
                conda_vars_to_remove.append(var)
        
        for var in conda_vars_to_remove:
            env.pop(var, None)
        
        # Clean PATH by removing conda bin directory (consistent with venv approach)
        if conda_prefix and 'PATH' in env:
            conda_bin = os.path.join(conda_prefix, 'bin')
            path_parts = env['PATH'].split(os.pathsep)
            cleaned_path_parts = [p for p in path_parts if p != conda_bin]
            env['PATH'] = os.pathsep.join(cleaned_path_parts)
        
        # Ensure base conda environment not activated during srun/sbatch.
        # Mainly a problem for systems where the login and compute nodes have different architectures.
        env['CONDA_AUTO_ACTIVATE'] = 'false'
    
    elif venv_type == 'venv':
        # Remove venv-specific environment variables
        venv_path = env.pop('VIRTUAL_ENV', None)
        env.pop('VIRTUAL_ENV_PROMPT', None)
        
        # Clean PATH by removing venv bin directory
        if venv_path and 'PATH' in env:
            venv_bin = os.path.join(venv_path, 'bin')
            path_parts = env['PATH'].split(os.pathsep)
            cleaned_path_parts = [p for p in path_parts if p != venv_bin]
            env['PATH'] = os.pathsep.join(cleaned_path_parts)
    
    # Remove Python-specific variables that might cause issues
    python_vars_to_remove = ['PYTHONHOME', 'PYTHONPATH']
    for var in python_vars_to_remove:
        env.pop(var, None)
    
    return env

def parse_gpu_gres(gres_output: str) -> Optional[int]:
    """Extract the GPU count from a SLURM GRES string.

    Accepted examples:
        gpu:8
        gpu:a100:8
        gpu:8(S:0-1)
        gpu:a100:8(S:0-1)
        gpu:8,mib:100
        gpu:a100_3g.20gb:2

    Returns an integer within the range 1-8 or *None* when parsing fails.
    """
    if not gres_output or gres_output == "(null)" or "gpu:" not in gres_output:
        return None

    # Keep the substring after "gpu:" then strip any extras after ',' or '('
    gpu_part = gres_output.split("gpu:", 1)[1]
    for sep in (",", "("):
        gpu_part = gpu_part.split(sep, 1)[0]

    # In cases like 'gpu:a100:8' keep only the numeric part after the last ':'
    gpu_part = gpu_part.split(":")[-1].strip()

    if not gpu_part.isdigit():
        return None

    count = int(gpu_part)
    return count if 1 <= count <= 8 else None

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
                    'repositories': metadata.get('repositories', {}),
                    'setup': metadata.get('setup', {}),
                    'run': metadata.get('run', {}),
                    'path': str(file_path.parent)
                }
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return workload_dict

def _resolve_dependencies(workload_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolves repo_key references in a dependency spec."""
    dependencies = copy.deepcopy(workload_data.get('setup', {}).get('dependencies'))
    if not dependencies:
        return None

    repositories = workload_data.get('repositories', {})

    # Resolve git dependencies
    for name, git_dep in dependencies.get('git', {}).items():
        repo_key = git_dep.get('repo_key')
        if repo_key and repo_key in repositories:
            git_dep.update(repositories[repo_key])
            del git_dep['repo_key'] # remove the key after resolving

    # Resolve pip dependencies
    for pip_dep in dependencies.get('pip', []):
        if isinstance(pip_dep, dict):
            repo_key = pip_dep.get('repo_key')
            if repo_key and repo_key in repositories:
                pip_dep.update(repositories[repo_key])
                del pip_dep['repo_key']

    # Add pip deps that require cloning
    for pip_dep in dependencies.get('pip', []):
        if isinstance(pip_dep, dict):
            if pip_dep.get('editable') or pip_dep.get('install_target'):
                git_deps = dependencies.setdefault('git', {})

                try:
                    git_deps[pip_dep['package']] = {
                        'url': pip_dep['url'],
                        'commit': pip_dep['commit'],
                        'install_method': {'type': 'pip'}
                    }
                except KeyError as e:
                    missing_field = e.args[0] if e.args else "unknown"
                    raise KeyError(
                        f"Error in pip dependency '{pip_dep.get('package', '<unknown>')}': "
                        f"required field '{missing_field}' is missing. "
                        f"Each pip dependency that uses 'editable' or 'install_target' must specify both 'url' and 'commit' or use 'repo_key'."
                    )

    # Remove empty git section if no repos were added
    if 'git' in dependencies and not dependencies['git']:
        del dependencies['git']

    return dependencies

def group_workloads_by_dependencies(
    workloads: Dict[str, Dict[str, Any]], selected_keys: list[str]
) -> Dict[Optional[str], list[str]]:
    """Group workloads by their fully resolved dependency specification."""
    dep_groups: Dict[Optional[str], list[str]] = {}
    
    for key in selected_keys:
        workload_data = workloads[key]
        resolved_deps = _resolve_dependencies(workload_data)
        
        if not resolved_deps:
            # Scripted workload installs without explicit dependencies
            if None not in dep_groups:
                dep_groups[None] = []
            dep_groups[None].append(key)
            continue
            
        # Editable installs get isolated venvs regardless of otherwise matching deps.
        if _has_editable_pip_dep(resolved_deps):
            dep_hash = f"editable-isolated::{key}"
            #print(f"[venv-grouping] {key}: forcing individual venv due to editable pip dependency")
        else:
            # Create a canonical representation that unifies equivalent sources
            # across acquisition methods (git vs pip-git) and ignores non-functional fields.
            canonical = _canonicalize_dependencies_for_grouping(resolved_deps)
            dep_string = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
            dep_hash = hashlib.sha256(dep_string.encode('utf-8')).hexdigest()
            #print(f"[venv-grouping] {key}: hash={dep_hash}")
            #print(f"[venv-grouping] {key} canonical={dep_string}")
        
        if dep_hash not in dep_groups:
            dep_groups[dep_hash] = []
        dep_groups[dep_hash].append(key)
        
    return dep_groups

def print_dependency_group_summary(dep_groups: Dict[Optional[str], list[str]]) -> None:
    """Print a user-friendly summary of how workloads are grouped by dependencies."""
    print("\nWorkload Installation Plan")
    print("=========================")
    
    scripted_workloads = dep_groups.get(None, [])
    dependency_groups = {k: v for k, v in dep_groups.items() if k is not None}
    
    # Count individual installations (both scripted and unique dependency workloads)
    individual_count = len(scripted_workloads)
    individual_count += sum(1 for workloads in dependency_groups.values() if len(workloads) == 1)
    
    # Count shared virtual environment groups
    shared_count = sum(len(workloads) for workloads in dependency_groups.values() if len(workloads) > 1)
    shared_groups_count = len([g for g in dependency_groups.values() if len(g) > 1])
    
    if individual_count > 0:
        print(f"\nIndividual installations ({individual_count} workloads):")
        print("Each workload will have its own virtual environment:")
        
        # Show scripted workloads
        for workload in sorted(scripted_workloads):
            print(f"  • {workload}")
        
        # Show workloads with unique dependencies
        for group_workloads in dependency_groups.values():
            if len(group_workloads) == 1:
                print(f"  • {group_workloads[0]}")
    
    if shared_count > 0:
        print(f"\nShared virtual environment groups ({shared_count} workloads in {shared_groups_count} groups):")
        print("These workloads share the same dependencies and will use a common virtual environment:")
        
        for group_workloads in dependency_groups.values():
            if len(group_workloads) > 1:
                print(f"  • {', '.join(sorted(group_workloads))}")
    
    print()

def clone_git_repos(git_deps: Dict[str, Any], target_dir: str):
    """Clone git repositories into the target directory."""
    if not git_deps:
        return
        
    print(f"Cloning git repositories for {os.path.basename(target_dir)}...")
    for name, repo_info in git_deps.items():
        repo_url = repo_info['url']
        commit = repo_info['commit']
        # Derive the directory name from the git URL to preserve the original repo name
        repo_name_from_url = repo_url.split('/')[-1].replace('.git', '')
        clone_path = os.path.join(target_dir, repo_name_from_url)
        
        print(f"  Cloning {name} from {repo_url} into {clone_path}...")
        if not os.path.exists(clone_path):
            try:
                subprocess.run(
                    ['git', 'clone', repo_url, clone_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error cloning {repo_url}: {e.stderr}")
                raise
        
        # Checkout the specific commit
        try:
            subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=clone_path,
                check=True,
                capture_output=True,
                text=True
            )
            subprocess.run(
                ['git', 'checkout', '-f', commit],
                cwd=clone_path,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error checking out commit {commit} in {clone_path}: {e.stderr}")
            raise

    print("✓ Cloning complete.")

def install_shared_dependencies(
    venv_path: str, venv_type: str, dependencies: Dict[str, Any], workload_clone_path: str, env: Dict[str, str]
):
    """Install shared dependencies into a virtual environment.
    
    This function handles two types of dependencies from the workload's metadata:
    1. Git repositories: For repos that need to be installed via a script, this
       function runs the specified script from the locally cloned repository.
    2. Pip packages: Installs all packages listed under the 'pip' key. This
       can include standard packages from PyPI and direct-from-git installs.

    Args:
        venv_path: The path to the shared virtual environment.
        venv_type: The type of virtual environment ('venv' or 'conda').
        dependencies: The fully resolved dependency dictionary.
        workload_clone_path: Path to the directory where git repos are cloned.
        env: The environment dictionary for running subprocesses.
    """
    print(f"Installing dependencies into shared venv: {venv_path}")

    # Install git repositories that require a script first
    git_deps = dependencies.get('git', {})
    if git_deps:
        print("  Installing git repositories via script...")
        for name, repo_info in git_deps.items():
            install_method = repo_info.get('install_method', {})
            install_type = install_method.get('type')

            # Get original repo name from URL
            repo_url = repo_info['url']
            repo_name_from_url = repo_url.split('/')[-1].replace('.git', '')
            repo_path = os.path.join(workload_clone_path, repo_name_from_url)
            
            if install_type == 'script':
                print(f"    - Installing {name} using script '{install_method['path']}'...")
                try:
                    script_path = os.path.join(repo_path, install_method['path'])
                    os.chmod(script_path, 0o755)
                    subprocess.run(
                        [script_path],
                        cwd=repo_path,
                        check=True,
                        env=env
                    )
                except subprocess.CalledProcessError:
                    print(f"Error installing git repository {name} via script.")
                    raise
            elif install_type == 'clone':
                 print(f"    - Git repo '{name}' is set to 'clone' only, skipping install step.")
            elif install_type == 'pip':
                 print(f"    - Git repo '{name}' will be installed via pip.")
            else:
                # We can log if a repo is listed but has no valid install method.
                print(f"    - Skipping git repo '{name}': no valid install method found.")
    
    # Install pip packages
    pip_deps = dependencies.get('pip', [])
    if pip_deps:
        print("  Installing pip packages...")
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        for item in pip_deps:
            editable = False
            cwd = None  # CWD for subprocess, default to None
            package_name = None

            if isinstance(item, str):
                # Compatibility for simple string package definitions ie 'scipy<1.13.0'
                package_str = item
            elif isinstance(item, dict):
                package_name = item['package']
                is_git_repo = 'url' in item and 'commit' in item
                editable = item.get('editable', False)
                install_target = item.get('install_target')

                if is_git_repo:
                    repo_name_from_url = item['url'].split('/')[-1].replace('.git', '')
                    repo_path = os.path.join(workload_clone_path, repo_name_from_url)

                    if install_target:
                        # Install from a local source path (e.g., '.[all]')
                        package_str = install_target
                        cwd = repo_path
                    elif editable:
                        # Editable install from a local path
                        package_str = repo_path
                    else:
                        # Install from git+https URL
                        package_str = f"git+{item['url']}@{item['commit']}#egg={package_name}"
                else:
                    # Standard package from PyPI
                    package_str = package_name
            else:
                continue

            print(f"    - {package_name if package_name else package_str} (Editable: {editable})")
            
            pip_command = [pip_path, 'install', '--no-cache-dir']
            if editable:
                pip_command.append('-e')
            pip_command.append(package_str)

            try:
                subprocess.run(
                    pip_command,
                    check=True,
                    env=env,
                    cwd=cwd
                )
            except subprocess.CalledProcessError:
                print(f"Error installing pip package {package_str}.")
                raise
    
    print("✓ Dependency installation complete.")

def get_setup_tasks(workload_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return the list of setup tasks for a workload.

    Behaviour:
    • If `setup.tasks` exists, return that list preserving order.
    • Otherwise, return an empty list – legacy `setup_script` is handled
      elsewhere by `run_post_install_script` for full backward compatibility.
    """
    setup_cfg: Dict[str, Any] = workload_data.get("setup", {}) or {}
    tasks: list[Dict[str, Any]] = setup_cfg.get("tasks", []) or []

    if tasks:
        return tasks

    # No explicit tasks defined
    return []


def _augment_env_for_job_type(env: Dict[str, str], job_type: str, slurm_info: Dict[str, Any], requires_gpus: bool = False):
    """Inject cluster-specific SBATCH/SLURM variables.

    Task metadata must *not* override these values – they are entirely
    cluster/user-specific.  Therefore we **overwrite** any existing keys.
    
    Args:
        env: Environment dictionary to modify
        job_type: Type of job (local, nemo2, sbatch, srun)
        slurm_info: SLURM configuration dictionary
        requires_gpus: Whether this task requires GPU resources
    """
    account = slurm_info["slurm"].get("account", "")
    
    # Select partition and GRES based on GPU requirements
    if requires_gpus:
        partition = slurm_info["slurm"].get("gpu_partition")
        gres = slurm_info["slurm"].get("gpu_partition_gres")
    else:
        partition = slurm_info["slurm"].get("cpu_partition")
        gres = slurm_info["slurm"].get("cpu_partition_gres")

    if job_type in ("nemo2", "sbatch", "srun"):
        if not account:
            raise ValueError(f"SLURM account must be set for job_type '{job_type}'. Please provide that information during installation.")
        if not partition:
            partition_type = "GPU" if requires_gpus else "CPU"
            raise ValueError(f"SLURM {partition_type} partition must be set for job_type '{job_type}'. Please provide that information during installation.")

    if job_type in ("nemo2", "sbatch"):
        env["SBATCH_ACCOUNT"] = account
        env["SBATCH_PARTITION"] = partition
        if gres is not None:
            env["SBATCH_GPUS_PER_NODE"] = str(gres)
    elif job_type == "srun":
        env["SLURM_ACCOUNT"] = account
        env["SLURM_PARTITION"] = partition
        if gres is not None:
            env["SLURM_GPUS_PER_NODE"] = str(gres)


def run_setup_tasks(
    workload_key: str,
    workload_data: Dict[str, Any],
    venv_path: Optional[str],
    venv_type: Optional[str],
    install_path: str,
    slurm_info: Dict[str, Any],
    global_env_vars: Dict[str, str],
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
    """
    tasks = get_setup_tasks(workload_data)
    if not tasks:
        return

    workload_dir = workload_data["path"]

    for idx, task in enumerate(tasks, start=1):
        name = task.get("name", f"task_{idx}")
        cmd = task.get("cmd")
        if not cmd:
            print(f"Skipping task '{name}' – no cmd provided.")
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
        # user-provided globals
        env.update(global_env_vars)
        # task-specific overrides
        env.update(task_env_extra)
        # SLURM augmentation
        _augment_env_for_job_type(env, job_type, slurm_info, requires_gpus)

        banner = f"Running setup task [{workload_key}] – {name} (type: {job_type})"
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
                # local, nemo2, srun are run inline via shell
                subprocess.run(cmd, shell=True, check=True, cwd=workload_dir, env=env, capture_output=False)
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
    script_path = os.path.join(source_dir, setup_script)
    print(f"Running post-install script: {script_path}")
    try:
        if not os.path.exists(script_path):
            print(f"Warning: Post-install script {script_path} not found, skipping.")
            return
            
        os.chmod(script_path, 0o755)
        
        subprocess.run(
            [script_path],
            env=env,
            cwd=source_dir,
            check=True,
            text=True
        )
        print("\n✓ Post-install script completed successfully.")
            
    except subprocess.CalledProcessError as e:
        print(f"\nError running post-install script (return code: {e.returncode})")
        raise

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

def filter_tools_from_workload_list(workloads: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Filter tools from the workload list."""
    tools = {}
    for key, workload_data in workloads.items():
        if workload_data.get('general', {}).get('workload_type') == 'tools':
            tools[key] = workload_data
    return tools

def prompt_gpu_type(workloads: Dict[str, Dict[str, Any]]) -> str:
    """Prompt the user to select GPU type based on available workloads.
    
    Returns:
        str: The selected GPU type (h100, gb200, b200)
    """
    print("\nGPU Type Selection")
    print("------------------")
    print("Please select the GPU type for your cluster.")
    print("This will determine which workloads are available for installation.")
    
    # Get supported GPU types from workloads, but limit to known types
    supported_types = get_supported_gpu_types(workloads)
    known_types = {'h100', 'gb200', 'b200'}
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
        gpu_type: The selected GPU type (h100, gb200, b200)
        
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
    
    if gpu_type == 'h100' or gpu_type == 'b200':
        print(f"\n{gpu_type.upper()} systems are typically x86_64 based.")
    
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

def prompt_install_location() -> Optional[str]:
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


def prompt_slurm_info() -> Optional[dict]:
    """Prompt the user for SLURM information.
    
    Returns:
        dict: Dictionary containing SLURM configuration information
    """
    print("\nSLURM Configuration")
    print("------------------")
    print("Please provide the following SLURM information for your cluster:\n")
    print("Note: This information will also be used for the launcher.")
    
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
    
    default_account = accounts[0] if len(accounts) == 1 else ""
    account = questionary.text(
        "SLURM account:",
        default=default_account,
        validate=validate_account
    ).ask()
    print()
    
    if account is None:
        return None
    
    default_partition = None # must be outside of try block to not break validation.
    try:
        result = subprocess.run(
            ["sinfo", "--noheader", "-o", "%P"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse the output to identify the default partition (marked with *)
        raw_partitions = result.stdout.strip().split('\n')
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
    gpu_partition = questionary.text(
        "SLURM GPU partition:",
        default=default_partition if default_partition else "",
        validate=validate_partition
    ).ask()
    print()

    if gpu_partition is None:
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

    # Streamlined GRES auto-detection (with rich diagnostics)
    # -------------------------------------------------------------
    
    def _detect_partition_gres(partitions: list[str]) -> Dict[str, Dict[str, Any]]:
        """Detect GRES information for the given partitions using sinfo.
        
        Returns:
            Dict mapping partition names to their GRES info:
            {
                'partition_name': {
                    'gpu_count': Optional[int],     # consolidated GPU count or None
                    'has_gpu_lines': bool,          # at least one line contained 'gpu:'
                    'unparseable_lines': list[str], # gpu lines we failed to parse
                    'is_heterogeneous': bool        # multiple distinct counts detected
                }
            }
        
        Raises:
            SystemExit: If sinfo is not available (hard error)
            RuntimeError: If sinfo command fails for other reasons
        """
        partition_info = {
            partition: {
                "gpu_counts": [],
                "has_gpu_lines": False,
                "unparseable_lines": [],
            }
            for partition in partitions
        }

        try:
            result = subprocess.run(
                ["sinfo", "--noheader", "-p", ",".join(partitions), "-o", "%P,%G"],
                capture_output=True,
                text=True,
                check=True,
            )
            sinfo_lines = result.stdout.strip().splitlines()
        except FileNotFoundError:
            print("\nError: 'sinfo' command not found.")
            print("This installer must be run on a system with SLURM installed.")
            print("Please run this installer on a SLURM login node.")
            raise SystemExit(1)
        except subprocess.CalledProcessError as e:
            print(f"\nWarning: 'sinfo' command failed with return code {e.returncode}")
            if e.stderr:
                print(f"Error output: {e.stderr.strip()}")
            print("Cannot auto-detect GRES information. Will prompt for manual input.")
            raise RuntimeError(f"sinfo command failed: {e}")

        # Parse sinfo output
        for line in sinfo_lines:
            if not line.strip() or "," not in line:
                continue

            partition_name, gres_raw = [s.strip() for s in line.split(",", 1)]
            partition_name = partition_name.rstrip("*")  # Remove default partition marker
            
            if partition_name not in partition_info:
                continue

            if "gpu:" in gres_raw:
                partition_info[partition_name]["has_gpu_lines"] = True

            gpu_count = parse_gpu_gres(gres_raw)
            if gpu_count is not None:
                partition_info[partition_name]["gpu_counts"].append(gpu_count)
            elif "gpu:" in gres_raw:
                # Keep track of lines we failed to parse that should have GPU info
                partition_info[partition_name]["unparseable_lines"].append(gres_raw)

        # Consolidate results for each partition
        for partition, info_dict in partition_info.items():
            unique_counts = list(set(info_dict["gpu_counts"]))
            if not unique_counts:
                info_dict["gpu_count"] = None
                info_dict["is_heterogeneous"] = False
            else:
                # Pick the most common count when heterogeneous
                most_common_count = max(set(info_dict["gpu_counts"]), key=info_dict["gpu_counts"].count)
                info_dict["gpu_count"] = most_common_count
                info_dict["is_heterogeneous"] = len(unique_counts) > 1

        return partition_info

    def _print_gres_detection_result(partition: str, is_gpu_partition: bool, partition_info: dict):
        """Print user-friendly GRES detection results."""
        label = "GPU partition" if is_gpu_partition else "CPU partition"
        gpu_count = partition_info["gpu_count"]
        
        if gpu_count is not None:
            if partition_info["is_heterogeneous"]:
                unique_counts = sorted(set(partition_info["gpu_counts"]))
                print(
                    f"Warning: Partition '{partition}' has nodes with different GPU counts: {unique_counts}. "
                    f"Using most common value: {gpu_count}"
                )
            print(f"✓ Auto-detected {label} GRES: {gpu_count} GPUs per node")
        else:
            if not partition_info["has_gpu_lines"]:
                print(f"• No GPU GRES detected for partition '{partition}'")
            else:
                print(f"• Could not auto-detect GPU GRES for partition '{partition}'")
                if partition_info["unparseable_lines"]:
                    print(f"  Failed to parse: {', '.join(partition_info['unparseable_lines'])}")

    # Perform GRES detection
    try:
        gres_info = _detect_partition_gres([gpu_partition, cpu_partition])
    except RuntimeError:
        # sinfo failed but system has SLURM - fall back to manual prompting
        print("Falling back to manual GRES entry.")
        gres_info = {
            gpu_partition: {"gpu_count": None, "has_gpu_lines": True, "unparseable_lines": [], "is_heterogeneous": False},
            cpu_partition: {"gpu_count": None, "has_gpu_lines": True, "unparseable_lines": [], "is_heterogeneous": False}
        }

    gpu_partition_info = gres_info[gpu_partition]
    cpu_partition_info = gres_info[cpu_partition]

    gpu_partition_gres_value = gpu_partition_info["gpu_count"]
    cpu_partition_gres_value = cpu_partition_info["gpu_count"]

    # Ensure consistency when partitions are identical
    if gpu_partition == cpu_partition:
        cpu_partition_gres_value = gpu_partition_gres_value
        cpu_partition_info = gpu_partition_info

    # Display detection results
    _print_gres_detection_result(gpu_partition, True, gpu_partition_info)
    if cpu_partition != gpu_partition:
        _print_gres_detection_result(cpu_partition, False, cpu_partition_info)

    # --------------------  Manual fallback prompting  -----------------------
    if gpu_partition_gres_value is None and gpu_partition_info["has_gpu_lines"]:
        gpu_partition_gres_value = _prompt_gpu_gres(gpu_partition)

    if (
        cpu_partition != gpu_partition
        and cpu_partition_gres_value is None
        and cpu_partition_info["has_gpu_lines"]
    ):
        cpu_partition_gres_value = _prompt_gpu_gres(cpu_partition)

    # Maintain consistency for identical partitions
    if gpu_partition == cpu_partition:
        cpu_partition_gres_value = gpu_partition_gres_value

    slurm_config = {
        "slurm": {
            "account": account,
            "gpu_partition": gpu_partition,
            "cpu_partition": cpu_partition,
            "gpu_partition_gres": gpu_partition_gres_value,
            "cpu_partition_gres": cpu_partition_gres_value
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
    return importlib.util.find_spec("venv") is not None
    
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
        # If the conda env path already exists, remove it to prevent errors
        if os.path.exists(venv_path):
            print(f"  Removing existing conda environment at {venv_path}")
            shutil.rmtree(venv_path)

        import subprocess
        try:
            # Create conda environment with minimum required Python version
            subprocess.run([
                'conda', 'create',
                '-p', venv_path,
                f'python={MIN_PYTHON_VERSION}',
                '--yes'
            ], check=True, capture_output=True, text=True)
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


def install_scripted_workload(workload_key: str, workload_data: Dict[str, Any], install_path: str, venv_type: str, env_vars: Dict[str, str]) -> Optional[str]:
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
    Returns:
        The path to the created virtual environment, or None if no venv was required.
    """
    print(f"\n\nInstalling {workload_key} (scripted method)")
    print("-----------------------------------------")
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
    # Signal to setup scripts that this is an automated install (prevents automatic sqsh downloads)
    env['MANUAL_INSTALL'] = 'false'
    if env_vars:
        env_vars_str = {k: str(v) for k, v in env_vars.items()}
        env.update(env_vars_str) # Ensure things like HF_TOKEN are set in the setup env.

    source_dir = workload_data['path']
    print(f"Installing {workload_key} to {target_dir}")
    
    setup_script = setup_config.get('setup_script')
    if setup_script:
        script_path = os.path.join(source_dir, setup_script)
        print(f"Running setup script: {script_path}")
        try:
            if not os.path.exists(script_path):
                print(f"Warning: Setup script {script_path} not found")
                return venv_path
                
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
    
    return venv_path

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
        
    except (IndexError, ValueError):
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
    print(f"Supported Python version range for venvs: [{MIN_PYTHON_VERSION}, {MAX_PYTHON_VERSION})")
    
    venv_available = is_venv_installed()
    conda_available = is_conda_installed()

    if detect_virtual_environment() == 'conda':
        # Force conda usage when running inside a conda environment to prevent venv from
        # creating problematic dependencies on the parent conda environment
        print("✓ Detected conda environment - automatically using conda for installation")
        print("  Note: To use venv instead, run this installer outside of any conda environment")
        venv_available = False
    
    can_use_venv = venv_available and MIN_PYTHON_VERSION_TUPLE <= current_python_version < MAX_PYTHON_VERSION_TUPLE
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
        elif venv_available and current_python_version >= MAX_PYTHON_VERSION_TUPLE:
            # venv is available but Python version is too high, conda is available
            print(f"WARNING: Your current Python version ({'.'.join(map(str, current_python_version))}) is above the maximum supported ({MAX_PYTHON_VERSION}) for venv workloads.")
            print("Conda is available and will be used instead.")
        else:
            print("venv is not available. Using conda.")
            
        selected_venv_type = 'conda'
    else:
        # Neither venv nor conda is available, or venv available but wrong version and conda not available
        print("Error: Cannot proceed with installation.")
        print()
        
        if venv_available and current_python_version < MIN_PYTHON_VERSION_TUPLE:
            print(f"Your current Python version ({'.'.join(map(str, current_python_version))}) is below the minimum required ({MIN_PYTHON_VERSION}).")
            print("To fix this issue, please either:")
            print("  1. Install conda/miniconda (recommended)")
            print(f"  2. Upgrade your Python to version {MIN_PYTHON_VERSION} but less than {MAX_PYTHON_VERSION}")
        elif venv_available and current_python_version >= MAX_PYTHON_VERSION_TUPLE:
            print(f"Your current Python version ({'.'.join(map(str, current_python_version))}) is above the maximum supported ({MAX_PYTHON_VERSION}).")
            print("To fix this issue, please either:")
            print("  1. Install conda/miniconda (recommended)")
            print(f"  2. Use Python version {MIN_PYTHON_VERSION} but less than {MAX_PYTHON_VERSION}")
        else:
            print("Neither venv nor conda is available on your system.")
            print("To fix this issue, please either:")
            print("  1. Install conda/miniconda (recommended)")
            print(f"  2. Install Python {MIN_PYTHON_VERSION} with venv support")
        
        raise SystemExit(1)

    if selected_venv_type is None:
         print("Environment selection cancelled. Exiting.")
         raise SystemExit(1)

    print(f"Using environment type: {selected_venv_type}")
    return selected_venv_type

def get_cluster_name() -> Optional[str]:
    """Get the SLURM cluster name if available.
    
    Returns:
        Optional[str]: The cluster name if found, None otherwise
    """
    try:
        result = subprocess.run(
            ["scontrol", "show", "config"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.lower().startswith('clustername'):
                # Parse "ClusterName = cluster" format
                if '=' in line:
                    cluster_name = line.split('=', 1)[1].strip()
                    return cluster_name if cluster_name else None
        
        return None
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def create_cluster_config(install_path: str, root_dir: str, selected_workloads: list[str], 
                         slurm_info: dict, env_vars: dict, gpu_type: str, node_architecture: str, venv_type: str,
                         workload_venvs: Dict[str, str]) -> None:
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
        workload_venvs: Dictionary mapping workload keys to their venv paths
    """
    launcher_config = {
        'llmb_repo': root_dir,
        'llmb_install': install_path,
        'gpu_type': gpu_type
    }
    
    # Add cluster name if available
    cluster_name = get_cluster_name()
    if cluster_name:
        launcher_config['cluster_name'] = cluster_name
    
    cluster_config = {
        'launcher': launcher_config,
        'environment': env_vars,
        'slurm': {
            'account': slurm_info['slurm']['account'],
            'gpu_partition': slurm_info['slurm']['gpu_partition'],
            'gpu_gres': slurm_info['slurm'].get('gpu_partition_gres'),
            'cpu_partition': slurm_info['slurm']['cpu_partition'],
            'cpu_gres': slurm_info['slurm'].get('cpu_partition_gres')
        },
        'workloads': {
            'installed': selected_workloads,
            'config': {}
        }
    }
    
    # Add workload-specific configuration (venv paths and types)
    for workload_key in selected_workloads:
        venv_path = workload_venvs.get(workload_key)
        
        cluster_config['workloads']['config'][workload_key] = {
            'venv_path': venv_path,
            'venv_type': venv_type if venv_path else None
        }
    
    # Write the cluster config file
    config_path = os.path.join(install_path, "cluster_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(cluster_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created cluster configuration: {config_path}")

def fetch_container_images(images: Dict[str, str], install_path: str, node_architecture: str, install_method: str, slurm_info: Dict[str, Any], shared_images_dir: Optional[str] = None) -> None:
    """Fetch container images using enroot and save them as sqsh files.
    
    Args:
        images: Dictionary mapping image URLs to desired filenames
        install_path: Directory where the sqsh files should be saved
        node_architecture: The selected node architecture ('x86_64' or 'aarch64')
        install_method: The selected installation method ('local' or 'slurm')
        slurm_info: Dictionary containing SLURM configuration information
        shared_images_dir: Optional path to a shared directory for images.
    
    The function will:
    1. Use provided filenames or generate them from image URLs
    2. Create sqsh files using enroot import
    3. Skip if sqsh file already exists
    4. If a shared_images_dir is provided, store images there and symlink to install_path/images.
    """
    local_images_dir = os.path.join(install_path, "images")
    os.makedirs(local_images_dir, exist_ok=True)
    
    if shared_images_dir:
        source_images_dir = shared_images_dir
        print(f"\nUsing shared image location: {source_images_dir}")
        os.makedirs(source_images_dir, exist_ok=True)
    else:
        source_images_dir = local_images_dir

    # Detect virtual environment and prepare clean environment for SLURM
    venv_type = detect_virtual_environment()
    if venv_type and install_method == 'slurm':
        print(f"Detected {venv_type} environment. Preparing clean environment for SLURM execution...")
        clean_env = get_clean_environment_for_subprocess()
    else:
        clean_env = None

    for image_url, filename in sorted(images.items()):
        try:
            source_sqsh_file = os.path.join(source_images_dir, filename)
            
            if os.path.exists(source_sqsh_file):
                print(f"Skipping {image_url} -- {source_sqsh_file} - file already exists.")
            else:
                cmd_args = ["enroot", "import", "-o", source_sqsh_file]
                if node_architecture:
                     cmd_args.extend(["-a", node_architecture])
                cmd_args.append(f"docker://{image_url}")

                if install_method == 'slurm':
                    srun_args = ["srun", f"--account={slurm_info['slurm']['account']}", f"--partition={slurm_info['slurm']['cpu_partition']}", "-t", "35", "--exclusive", "--pty"]
                    if slurm_info['slurm'].get('cpu_partition_gres'):
                        srun_args.append(f"--gpus-per-node={slurm_info['slurm']['cpu_partition_gres']}")
                    
                    full_cmd_args = srun_args + cmd_args
                    print(f"Submitting download via SLURM: {' '.join(full_cmd_args)}")
                else:
                    full_cmd_args = cmd_args
                    print(f"Downloading {image_url} locally: {' '.join(full_cmd_args)}")
                    clean_env = None  # Don't use clean env for local execution

                try:
                    subprocess.run(full_cmd_args, check=True, text=True, env=clean_env)
                    print(f"Successfully downloaded {source_sqsh_file}\n")
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to download {source_sqsh_file}: {str(e)}\n")
                    if install_method == 'slurm':
                        print("Please ensure the compute nodes you selected in the 'CPU_PARTITION' option are configured correctly, or try a local installation (if available).")
                        if venv_type:
                            print(f"Note: Running from a {venv_type} environment. Architecture mismatch between login and compute nodes may cause 'Exec format error'.")
                            print("If using conda, please try venv instead with system python.")
                    else:
                        print("Ensure your system has appropriate resources or try a SLURM-based installation.")
                    raise SystemExit(1)

            # If using a shared directory, create/update symlink in the local install
            if shared_images_dir:
                symlink_path = os.path.join(local_images_dir, filename)
                abs_source_sqsh_file = os.path.abspath(source_sqsh_file)

                # Handle existing file/symlink at target
                if os.path.lexists(symlink_path):
                    if os.path.islink(symlink_path) and os.readlink(symlink_path) == abs_source_sqsh_file:
                        continue  # Symlink exists and is correct
                    os.remove(symlink_path)
                
                os.symlink(abs_source_sqsh_file, symlink_path)
                print(f"✓ Linked {filename} to shared image directory.")

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

def save_installation_config(config_file: str, config_data: Dict[str, Any]) -> None:
    """Save installation configuration to a YAML file.
    
    Args:
        config_file: Path to the configuration file to save
        config_data: Dictionary containing all installation configuration
    """
    try:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        os.chmod(config_file, 0o600)
        print(f"✓ Configuration saved to: {config_file}")
    except Exception as e:
        print(f"Error saving configuration to {config_file}: {e}")
        raise SystemExit(1)

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
        required_fields = ['venv_type', 'install_path', 'slurm_info', 'gpu_type', 
                          'node_architecture', 'install_method', 'selected_workloads', 'env_vars']
        
        missing_fields = [field for field in required_fields if field not in config_data]
        if missing_fields:
            raise ValueError(f"Configuration file is missing required fields: {missing_fields}")
        
        # Validate env_vars is a dict of string→string
        env_vars = config_data.get('env_vars', {})
        if not isinstance(env_vars, dict):
            raise ValueError("env_vars must be a dictionary")
        
        for key, value in env_vars.items():
            if not isinstance(key, str):
                raise ValueError(f"env_vars keys must be strings, found {type(key).__name__}: {key}")
            if not isinstance(value, str):
                raise ValueError(f"env_vars values must be strings, found {type(value).__name__}: {value}")
        
        # Validate selected_workloads is a list of strings
        selected_workloads = config_data.get('selected_workloads', [])
        if not isinstance(selected_workloads, list):
            raise ValueError("selected_workloads must be a list")
        
        for i, workload in enumerate(selected_workloads):
            if not isinstance(workload, str):
                raise ValueError(f"selected_workloads items must be strings, found {type(workload).__name__} at index {i}: {workload}")
        
        print(f"✓ Configuration loaded from: {config_file}")
        return config_data
        
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error loading configuration from {config_file}: {e}")
        raise SystemExit(1)

def find_llmb_repo_root() -> str:
    """Find the LLMB repository root directory.
    
    This function supports multiple use cases:
    1. When run from within a repo checkout (pip-installed llmb-install)
    2. When run directly from the repo (python3 installer/installer.py)
    3. When run via install.sh
    
    Returns:
        str: Path to the LLMB repository root directory
        
    Raises:
        SystemExit: If no valid LLMB repository root can be found
    """
    def is_llmb_repo_root(path: Path) -> bool:
        """Check if a directory appears to be the LLMB repository root.
        
        Uses only files that are included in the public release whitelist.
        """
        indicators = [
            'install.sh',
            'README.md', 
            'installer/installer.py'
        ]
        
        return all((path / indicator).exists() for indicator in indicators)
    
    def find_repo_root_upward(start_path: Path) -> Optional[Path]:
        """Search upward from start_path to find the repository root."""
        current = start_path.resolve()
        
        # Limit search
        max_levels = 5
        for _ in range(max_levels):
            if is_llmb_repo_root(current):
                return current
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent
        return None
    
    repo_root = find_repo_root_upward(Path.cwd())
    if repo_root:
        print(f"Found LLMB repository root: {repo_root}")
        return str(repo_root)
       
    print("\nError: Could not find LLMB repository.")
    print("Please run this command from within an LLMB repository checkout.")
    raise SystemExit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLMB Workload Installer")

    parser.add_argument(
        '-i', '--image-folder',
        type=str,
        default=None,
        help="Path to a shared folder for container images. If provided, images will be stored here and symlinked into the installation directory."
    )

    # --play and --record are mutually exclusive.
    group = parser.add_mutually_exclusive_group()

    # Headless execution options
    group.add_argument(
        '--record',
        type=str,
        metavar='CONFIG_FILE',
        help="Record mode: Save user inputs to CONFIG_FILE without performing installation. Use this to create a configuration file for headless installation."
    )

    group.add_argument(
        '--play',
        type=str,
        metavar='CONFIG_FILE',
        help="Play mode: Load user inputs from CONFIG_FILE and perform installation without prompting. Use this for headless installation."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    root_dir = find_llmb_repo_root()
    
    check_pip_cache_location()
    
    workloads = build_workload_dict(root_dir)
    if not workloads:
        print(f"\nError: No workloads found in {root_dir}")
        print("This repository may be missing workload metadata files.")
        raise SystemExit(1)
    
    # Handle play mode (headless installation)
    if args.play:
        print("=== HEADLESS INSTALLATION MODE ===")
        config_data = load_installation_config(args.play)
        
        # Extract configuration values
        venv_type = config_data['venv_type']
        install_path = config_data['install_path']
        slurm_info = config_data['slurm_info']
        gpu_type = config_data['gpu_type']
        node_architecture = config_data['node_architecture']
        install_method = config_data['install_method']
        selected = config_data['selected_workloads']
        env_vars = config_data['env_vars']
        
        # Add GPU_TYPE to env_vars for compatibility
        env_vars['GPU_TYPE'] = gpu_type
        
        print(f"Environment type: {venv_type}")
        print(f"Install path: {install_path}")
        print(f"GPU type: {gpu_type}")
        print(f"Node architecture: {node_architecture}")
        print(f"Install method: {install_method}")
        print(f"Selected workloads: {', '.join(selected)}")
        print()
        
        # Filter workloads based on loaded configuration
        filtered_workloads = filter_workloads_by_gpu_type(workloads, gpu_type)
        if not filtered_workloads:
            print(f"No workloads found that support {gpu_type} GPU type.")
            return
        
        # Add tools to workload list
        tools = filter_tools_from_workload_list(workloads)
        filtered_workloads.update(tools)
        
        # Resolve GPU-specific images/repos for the selected GPU type so that
        # downstream logic (image fetching, dependency grouping) sees concrete values.
        filtered_workloads = resolve_gpu_overrides(filtered_workloads, gpu_type)
        
        # Validate that all selected workloads exist
        missing_workloads = [w for w in selected if w not in filtered_workloads]
        if missing_workloads:
            print(f"Error: Selected workloads not found: {missing_workloads}")
            raise SystemExit(1)
        
        # Continue with installation using loaded configuration
        perform_installation(root_dir, install_path, slurm_info, gpu_type, node_architecture, 
                           install_method, selected, env_vars, venv_type, filtered_workloads, args)
        return
    
    # Handle record mode (save configuration without installation)
    if args.record:
        print("=== RECORD MODE ===")
        print("Collecting user inputs for configuration recording...")
        print()
    
    # Interactive mode (default behavior)
    print() # header spacing
    try:
        venv_type = prompt_environment_type()
        print()

        install_path = prompt_install_location()
        if not install_path:
            print("\nInstallation cancelled.")
            return
        
        slurm_info = prompt_slurm_info()
        if not slurm_info:
            print("\nInstallation cancelled.")
            return

        gpu_type = prompt_gpu_type(workloads)
        node_architecture = prompt_node_architecture(gpu_type)
        slurm_info['slurm']['node_architecture'] = node_architecture
        print()

        # Filter workloads to only show those compatible with selected GPU type
        filtered_workloads = filter_workloads_by_gpu_type(workloads, gpu_type)
        if not filtered_workloads:
            print(f"No workloads found that support {gpu_type} GPU type.")
            return
        
        # Add tools to workload list -- we want to error out first if no benchmarks found.
        tools = filter_tools_from_workload_list(workloads)
        filtered_workloads.update(tools)

        # Resolve GPU-specific images/repos for the selected GPU type so that
        # downstream logic (image fetching, dependency grouping) sees concrete values.
        filtered_workloads = resolve_gpu_overrides(filtered_workloads, gpu_type)

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
        # Add GPU_TYPE, WAR
        # TODO: Remove this once we have a config dict.
        env_vars['GPU_TYPE'] = gpu_type

        # Handle record mode - save configuration and exit
        if args.record:
            config_data = {
                'venv_type': venv_type,
                'install_path': install_path,
                'slurm_info': slurm_info,
                'gpu_type': gpu_type,
                'node_architecture': node_architecture,
                'install_method': install_method,
                'selected_workloads': selected,
                'env_vars': env_vars
            }
            save_installation_config(args.record, config_data)
            print("\nConfiguration recorded successfully!")
            print(f"To perform headless installation, run: ./install.sh --play {args.record}")
            return

    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        return
    
    if not selected:
        print("\nNo workloads selected. Exiting.")
        return
    
    # Perform the actual installation
    perform_installation(root_dir, install_path, slurm_info, gpu_type, node_architecture, 
                       install_method, selected, env_vars, venv_type, filtered_workloads, args)

def perform_installation(root_dir: str, install_path: str, slurm_info: Dict[str, Any], 
                       gpu_type: str, node_architecture: str, install_method: str, 
                       selected: list[str], env_vars: Dict[str, str], venv_type: str, 
                       filtered_workloads: Dict[str, Dict[str, Any]], args) -> None:
    """Perform the actual installation process.
    
    This function contains all the installation logic that was previously in main().
    It can be called from both interactive and headless modes.
    
    Args:
        root_dir: Path to the LLMB repository root
        install_path: Base installation directory
        slurm_info: SLURM configuration dictionary
        gpu_type: Selected GPU type
        node_architecture: Selected node architecture
        install_method: Installation method ('local' or 'slurm')
        selected: List of selected workload keys
        env_vars: Environment variables dictionary
        venv_type: Type of virtual environment ('venv' or 'conda')
        filtered_workloads: Dictionary of available workloads filtered by GPU type
        args: Command line arguments
    """
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
    
    fetch_container_images(required_images, install_path, node_architecture, install_method, slurm_info, args.image_folder)

    workload_venvs = {} # To store venv path for each workload
    dep_groups = group_workloads_by_dependencies(filtered_workloads, selected)
    
    # Show the user how workloads will be grouped
    print_dependency_group_summary(dep_groups)
    
    print("Installing Workloads")
    print("===================")

    for dep_hash, workload_keys in dep_groups.items():
        if dep_hash is None: # Scripted workloads
            print("\n[Individual Installations - Legacy Setup Scripts]")
            print("-" * 60)
            for workload_key in workload_keys:
                venv_path = install_scripted_workload(workload_key, filtered_workloads[workload_key], install_path, venv_type, env_vars)
                workload_venvs[workload_key] = venv_path
                # Execute any additional setup tasks defined for the workload
                run_setup_tasks(workload_key, filtered_workloads[workload_key], venv_path, venv_type, install_path, slurm_info, env_vars)
        else: # New dependency management
            venvs_dir = os.path.join(install_path, "venvs")

            if len(workload_keys) == 1:
                # If group has only one workload, it's still an individual installation
                workload_key = workload_keys[0]
                print("\n[Individual Installation - Unique Dependencies]")
                print(f"Installing: {workload_key}")
                print("-" * 60)
                venv_name = f"{workload_key}_venv"
            else:
                # Otherwise, use a shared name for the group
                print("\n[Shared Virtual Environment Group]")
                print(f"Installing workloads: {', '.join(sorted(workload_keys))}")
                print("-" * 70)
                venv_name = f"shared_venv_{dep_hash[:12]}"
            
            # 1. Create one venv for this group
            venv_path = os.path.join(venvs_dir, venv_name)
            create_virtual_environment(venv_path, venv_type)
            
            # Get resolved dependencies from the first workload in the group
            first_workload_key = workload_keys[0]
            dependencies = _resolve_dependencies(filtered_workloads[first_workload_key])
            
            if dependencies is None:
                print(f"Warning: No dependencies found for workload group containing {first_workload_key}")
                continue
            
            # 2. For each workload, create its folder and clone any repos that need to be local
            git_deps_to_clone = dependencies.get('git', {})
            
            for workload_key in workload_keys:
                workload_dir = os.path.join(install_path, "workloads", workload_key)
                os.makedirs(workload_dir, exist_ok=True)
                
                # Clone all necessary git repos into the workload dir
                clone_git_repos(git_deps_to_clone, workload_dir)
                workload_venvs[workload_key] = venv_path
            
            # 3. Install dependencies into the shared venv using the first workload's clones
            env = get_venv_environment(venv_path, venv_type)
            env['LLMB_INSTALL'] = install_path
            env['MANUAL_INSTALL'] = 'false'
            if env_vars:
                env_vars_str = {k: str(v) for k, v in env_vars.items()}
                env.update(env_vars_str) # Ensure things like HF_TOKEN are set in the setup env.
            
            first_workload_dir = os.path.join(install_path, "workloads", first_workload_key)
            install_shared_dependencies(venv_path, venv_type, dependencies, first_workload_dir, env)

            # 4. For each workload, run post-install script (if any)
            for workload_key in workload_keys:
                workload_data = filtered_workloads[workload_key]
                setup_config = workload_data.get('setup', {})
                setup_script = setup_config.get('setup_script')
                if setup_script:
                    source_dir = workload_data['path']
                    run_post_install_script(setup_script, source_dir, env)
                # Execute new-style setup tasks (if any)
                run_setup_tasks(workload_key, workload_data, venv_path, venv_type, install_path, slurm_info, env_vars)
    
    create_cluster_config(install_path, root_dir, selected, slurm_info, env_vars, gpu_type, node_architecture, venv_type, workload_venvs)
    
    print(f"\nInstallation complete! Workloads have been installed to: {install_path}")

    # Flag to track if any async jobs were submitted
    async_jobs_submitted = False

    # Pre-check for async tasks across all selected workloads (sbatch, nemo2)
    for workload_key in selected:
        workload_data = filtered_workloads[workload_key]
        for task in workload_data.get('setup', {}).get('tasks', []):
            job_type = task.get('job_type', 'local').lower()
            if job_type in ('sbatch', 'nemo2'):
                async_jobs_submitted = True
                break  # No need to check other tasks for this workload
        if async_jobs_submitted: # If found in this workload, no need to check other workloads
            break

    # Add the notice here if async jobs were submitted
    if async_jobs_submitted:
        message_lines = [
            "IMPORTANT: Some installation tasks were submitted as SLURM sbatch jobs.",
            "These jobs may still be queued or running in the background.",
            "To verify their status, please run: 'squeue -u $USER'"
        ]
        max_len = max(len(line) for line in message_lines)
        border = "=" * (max_len + 4) # +4 for padding

        print(f"\n{border}")
        for line in message_lines:
            print(f"  {line.ljust(max_len)}  ")
        print(f"{border}\n")

    print("To run llmb-run from any directory, set the LLMB_INSTALL environment variable:")
    print(f"  export LLMB_INSTALL={install_path}")
    print("Consider adding this to your shell profile (e.g., ~/.bashrc or ~/.bash_aliases) for permanent access.")

def _prompt_gpu_gres(partition_name: str) -> Optional[int]:
    """Interactively ask the user for GPUs per node for *partition_name*."""
    while True:
        resp = questionary.text(
            f"Enter GPUs per node for partition '{partition_name}' (leave blank if not applicable):"
        ).ask()

        if resp is None:
            print("\nInstallation cancelled.")
            raise SystemExit(1)

        resp = resp.strip()
        if resp == "":
            return None
        if resp.isdigit():
            val = int(resp)
            if 1 <= val <= 8:
                return val
            print("Please enter an integer between 1 and 8.")
        else:
            print("Please enter a valid integer.")

if __name__ == "__main__":
    main() 
