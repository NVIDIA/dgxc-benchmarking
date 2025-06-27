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

import re
import glob
import pathlib
import json
import itertools
import argparse
import logging
import sys
import subprocess
import shlex
import os
import ast
from collections import defaultdict
from pprint import pprint
from dataclasses import dataclass, field
from enum import Enum

import yaml

class ValidationErrorType(Enum):
    WORKLOAD_NOT_FOUND = "workload_not_found"
    WORKLOAD_NOT_INSTALLED = "workload_not_installed"
    GPU_TYPE_NOT_SUPPORTED = "gpu_type_not_supported"
    MODEL_SIZE_NOT_SUPPORTED = "model_size_not_supported"
    DTYPE_NOT_SUPPORTED = "dtype_not_supported"
    SCALE_NOT_SUPPORTED = "scale_not_supported"

@dataclass
class WorkloadTask:
    workload_key: str  # Full workload key (e.g., 'pretraining_nemotron')
    model_size: str
    dtype: str
    scale: int
    profile: bool = False
    env_overrides: dict = field(default_factory=dict)
    model_overrides: dict = field(default_factory=dict)

# Convert shortened model parameters to corresponding recipe parameters.
nemo_model_params = {
    "mbs": "model.micro_batch_size",
    "gbs": "model.global_batch_size",
    "cp": "model.context_parallel_size",
    "tp": "model.tensor_model_parallel_size",
    "pp": "model.pipeline_model_parallel_size",
    "vp": "model.virtual_pipeline_model_parallel_size",
}

# Placeholder for JAX model parameters (to be implemented)
jax_model_params = {}

class LevelFormatter(logging.Formatter):
    """Custom formatter that changes format based on log level."""
    def __init__(self, fmt_dict):
        super().__init__()
        self.fmt_dict = fmt_dict

    def format(self, record):
        # Select the format based on the log level
        fmt = self.fmt_dict.get(record.levelno, self.fmt_dict[logging.INFO])
        formatter = logging.Formatter(fmt)
        return formatter.format(record)

# Define log formats for different levels.
formatters = {
    logging.DEBUG: "DEBUG: %(message)s",
    logging.INFO: "%(message)s",
    logging.ERROR: "ERROR: %(message)s",
    logging.CRITICAL: "CRITICAL: %(message)s",
}

logger = logging.getLogger('llmb_run')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(LevelFormatter(formatters))
logger.addHandler(console_handler)

infra_params = ["SCALE"]
w_params = ["DTYPE"]

exclude_workloads = []

def get_cluster_config():
    return yaml.safe_load(open('cluster_config.yaml', 'r'))

def get_workloads(config):
    """Get a list of all workload metadata files in the repo. Extract relevant data."""
    workloads = {}
    # Get list of metadata files using the new launcher config structure
    llmb_repo = config['launcher']['llmb_repo']
    metadata_files = glob.glob(f"{llmb_repo}/**/metadata.yaml", recursive=True)

    for meta_file in metadata_files:
        try:
            with open(meta_file, 'r') as f:
                metadata = yaml.safe_load(f)

            # Extract workload name and type from the 'general' section
            workload = metadata.get('general', {}).get('workload')
            workload_type = metadata.get('general', {}).get('workload_type')
            if not workload or not workload_type:
                logger.warning(f"Metadata file {meta_file} is missing 'general.workload' or 'general.workload_type' field. Skipping.")
                continue

            # Create the full workload key in format workload_type_workload
            workload_key = f"{workload_type}_{workload}"

            if workload in exclude_workloads:
                logger.info(f"Excluding workload {workload} based on exclude_workloads list.")
                continue

            # Store the entire parsed metadata AND the directory path
            workloads[workload_key] = {
                "metadata": metadata,
                "dir": str(pathlib.Path(meta_file).parent),
                "workload": workload,
                "workload_type": workload_type
            }

        except FileNotFoundError:
            logger.error(f"Metadata file not found: {meta_file}")
            continue # or raise, depending on desired behavior
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {meta_file}: {e}")
            continue # or raise
        except Exception as e:
            logger.error(f"An unexpected error occurred processing {meta_file}: {e}")
            continue

    return workloads

def get_venv_environment(venv_path: str, env_type: str = 'venv') -> dict:
    """Prepare environment variables for running commands in a virtual environment.
    
    Args:
        venv_path: Path to the virtual environment
        env_type: Type of virtual environment ('venv' or 'conda')
        
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
    
    if env_type == 'venv':
        env['VIRTUAL_ENV'] = venv_path
    elif env_type == 'conda':
        env['CONDA_PREFIX'] = venv_path
    
    return env

def get_tasks_simple(workloads, input_file, cluster_config=None):
    """Parse a simple format task file into workload configurations.
    
    The simple format is designed for quick task specification with minimal syntax.
    Each workload section starts with a header and contains one or more task lines.
    
    Format:
    workload_modelsize:
    (dtype_list, scale_list, repeats, profile=False)
    
    Example:
    pretraining_grok1_314b:
    (['fp8', 'bf16'], [128, 256], 3)
    ('fp8', [128, 256, 512], 1, True)  # With profiling enabled
    
    Args:
        workloads: Dictionary of available workloads
        input_file: Path to the task specification file
        cluster_config: Optional cluster configuration for validation
        
    Returns:
        dict: Nested dictionary of workload tasks
        
    Raises:
        FileNotFoundError: If input_file does not exist
        ValueError: If task format is invalid
    """
    header_re = re.compile(r'^([\w\.]+)_([\w\.]+):$')  # Capture workload and model size
    task_re = re.compile(r'^\s*\((.*)\)\s*$')  # Match entire task line
    
    workload_tasks = {}
    current_workload_key = None
    current_model_size = None
    current_tasks = []

    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Empty line indicates end of current workload section
                    if current_workload_key:
                        workload_tasks.setdefault(current_workload_key, {})[current_model_size] = current_tasks
                    current_workload_key = current_model_size = None
                    current_tasks = []
                    continue

                if header_match := header_re.match(line):
                    workload_key, model_size = header_match.groups()
                    
                    is_valid, error_type, error_msg, suggestions = validate_workload_with_details(
                        workloads, workload_key, model_size, cluster_config=cluster_config
                    )
                    if not is_valid:
                        user_error = format_validation_error(
                            workload_key, model_size, None, None, 
                            cluster_config.get('launcher', {}).get('gpu_type') if cluster_config else None,
                            error_type, error_msg, suggestions
                        )
                        logger.error(user_error)
                        logger.error(f"Skipping invalid workload specification: {workload_key}_{model_size}")
                        # Reset state to skip tasks for this invalid workload
                        current_workload_key = None
                        current_model_size = None
                        current_tasks = []
                        continue

                    current_workload_key = workload_key
                    current_model_size = model_size
                    current_tasks = []

                elif task_match := task_re.match(line):
                    if not current_workload_key:
                        # Skip tasks for invalid workloads
                        continue
                    try:
                        task_data = ast.literal_eval(task_match.group(1))
                        current_tasks.append((current_workload_key,) + task_data)
                    except (SyntaxError, ValueError) as e:
                        logger.error(f"Invalid task format in line: {line}")
                        raise ValueError(f"Invalid task format: {e}")
        
        # Handle the last workload if file doesn't end with newline
        if current_workload_key:
            workload_tasks.setdefault(current_workload_key, {})[current_model_size] = current_tasks
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing input file: {e}")
        raise
    return workload_tasks

def merge_dicts(default, override):
    """Helper to merge two dictionaries (override takes precedence)."""
    result = default.copy()
    result.update(override)
    return result

def get_tasks_yaml(input_file):
    """Parse an advanced YAML workload file.
    
    The expected YAML format is described in the README.md file.

    Returns a dictionary in the same nested format:
      { workload: { model_size: [ list of tuples ] } }
    where each tuple is:
      (workload, model_size, dtype, scale, profile, env_overrides, model_overrides)
    """
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    workload_tasks = {}
    for header, spec in data.items():
        if "_" not in header:
            raise ValueError(f"Invalid header format: {header}. Expected 'workload_modelsize'.")
        workload, model_size = header.rsplit("_", 1)
        tasks_list = []
        defaults = spec.get("defaults", {})
        default_env = defaults.get("env", {})
        default_params = defaults.get("params", {})

        # Get top level settings
        top_dtypes = spec.get("dtypes")
        top_scales = spec.get("scales")
        top_repeats = spec.get("repeats", 1)
        top_add_profile = spec.get("add_profile", False)

        for task in spec.get("tasks", []):
            # Get task-specific settings, falling back to top level if not specified
            dtypes = task.get("dtypes", top_dtypes)
            if dtypes is None:
                raise ValueError(f"Missing required field 'dtypes' in task: {task}")
            dtypes = [dtypes] if isinstance(dtypes, str) else dtypes
            if not isinstance(dtypes, list):
                raise ValueError(f"'dtypes' must be a string or list, got {type(dtypes)}")

            scales = task.get("scales", top_scales)
            if scales is None:
                raise ValueError(f"Missing required field 'scales' in task: {task}")
            if isinstance(scales, int):
                scales = [scales]
            elif not isinstance(scales, list):
                raise ValueError(f"'scales' must be an integer or list, got {type(scales)}")
            
            repeats = task.get("repeats", top_repeats)
            profile = task.get("profile", False)
            add_profile = task.get("add_profile", top_add_profile)
            
            # Validate that only one profiling mode is set
            if profile and add_profile:
                raise ValueError("Cannot specify both 'profile' and 'add_profile' in the same task")
            
            overrides = task.get("overrides", {})
            
            # Env Overrides
            task_env = merge_dicts(default_env, overrides.get("env", {}))
            
            # Model Specific Overrides
            param_overrides = overrides.get("params", {})
            
            # Find parameters that need to be swept
            regular_params = {}
            sweep_params = {}
            for key, value in param_overrides.items():
                if isinstance(value, list):
                    sweep_params[key] = value
                else:
                    regular_params[key] = value
            
            if sweep_params:
                # Generate all combinations of sweep parameters
                param_names = list(sweep_params.keys())
                param_values = [sweep_params[name] for name in param_names]
                for param_combination in itertools.product(*param_values):
                    # Create a copy of the base params
                    current_params = merge_dicts(default_params, regular_params)
                    
                    # Apply the current combination of sweep parameters
                    for name, value in zip(param_names, param_combination):
                        current_params[name] = value
                    
                    # Generate tasks for this combination
                    for dtype, scale in itertools.product(dtypes, scales):
                        # Add regular performance runs
                        for r in range(repeats):
                            tasks_list.append(
                                (workload, model_size, dtype, scale, profile, task_env, current_params)
                            )
                        # Add one profiling run if requested
                        if add_profile:
                            tasks_list.append(
                                (workload, model_size, dtype, scale, True, task_env, current_params)
                            )
            else:
                # No sweeps needed, proceed as before
                current_params = default_params.copy()
                current_params.update(regular_params)
                for dtype, scale in itertools.product(dtypes, scales):
                    # Add regular performance runs
                    for r in range(repeats):
                        tasks_list.append(
                            (workload, model_size, dtype, scale, profile, task_env, current_params)
                        )
                    # Add one profiling run if requested
                    if add_profile:
                        tasks_list.append(
                            (workload, model_size, dtype, scale, True, task_env, current_params)
                        )
                        
        workload_tasks.setdefault(workload, {})[model_size] = tasks_list
    return workload_tasks

# Dispatcher for task parsing based on file extension.
def get_tasks_wrapper(workloads, input_file, cluster_config=None):
    if input_file.endswith(('.yaml', '.yml')):
        return get_tasks_yaml(input_file)
    else:
        return get_tasks_simple(workloads, input_file, cluster_config)

def gen_tasks(simple_tasks):
    """Convert parsed simple workload spec into a list of WorkloadTasks.
    This function expects that each task in simple_tasks is a tuple of the form:
      (workload_key, dtype_list, scale_list, repeat_count, *optional_params)
    """
    task_list = []
    for workload_key in simple_tasks:
        for model_size in simple_tasks[workload_key]:
            for task in simple_tasks[workload_key][model_size]:
                workload_key_from_task, dtype, scales, repeats, *params = task
                if isinstance(dtype, str):
                    dtype = [dtype]
                for dt, scale in itertools.product(dtype, scales):
                    for r in range(repeats):
                        # If optional params exist, the first is profile.
                        profile = params[0] if params else False
                        task_list.append(WorkloadTask(workload_key_from_task, model_size, dt, scale, profile))
    return task_list

# Flatten tasks from YAML advanced format to WorkloadTask objects.
def flatten_yaml_tasks(advanced_tasks):
    task_list = []
    for workload in advanced_tasks:
        for model_size in advanced_tasks[workload]:
            for t in advanced_tasks[workload][model_size]:
                # t is a tuple: (workload, model_size, dtype, scale, profile, env_overrides, model_overrides)
                w, m, dt, scale, profile, env_overrides, model_overrides = t
                task_list.append(WorkloadTask(w, m, dt, scale, profile, env_overrides=env_overrides, model_overrides=model_overrides))
    return task_list

def validate_workload_with_details(workloads, workload_key, model_size, dtype=None, scale=None, cluster_gpu_type=None, cluster_config=None):
    """Validate workload and return detailed results for better error reporting.
    
    Returns:
        tuple: (is_valid: bool, error_type: ValidationErrorType, error_message: str, suggestions: list)
    """
    installed_workloads = cluster_config.get('workloads', {}).get('installed', [])
    
    if workload_key not in workloads:
        # Filter suggestions by installed workloads
        available_workloads = [w for w in sorted(workloads.keys()) if w in installed_workloads]
        suggestions = [w for w in available_workloads if workload_key.lower() in w.lower() or w.lower() in workload_key.lower()]
        if not suggestions:
            suggestions = available_workloads[:5]  # Show first 5 as examples
        
        error_msg = f"Workload '{workload_key}' not found."
        return False, ValidationErrorType.WORKLOAD_NOT_FOUND, error_msg, suggestions

    if installed_workloads and workload_key not in installed_workloads:
        error_msg = f"Workload '{workload_key}' is not installed on this cluster."
        return False, ValidationErrorType.WORKLOAD_NOT_INSTALLED, error_msg, installed_workloads

    metadata = workloads[workload_key]['metadata']
    
    gpu_configs = metadata.get('run', {}).get('gpu_configs', {})
    
    if cluster_gpu_type and cluster_gpu_type not in gpu_configs:
        supported_gpu_types = list(gpu_configs.keys())
        error_msg = f"GPU type '{cluster_gpu_type}' not supported for workload '{workload_key}'."
        return False, ValidationErrorType.GPU_TYPE_NOT_SUPPORTED, error_msg, supported_gpu_types
    
    if cluster_gpu_type:
        model_configs = gpu_configs.get(cluster_gpu_type, {}).get('model_configs', [])
    else:
        model_configs = []
        for gpu_config in gpu_configs.values():
            model_configs.extend(gpu_config.get('model_configs', []))

    model_config = None
    for config_entry in model_configs:
        if config_entry.get('model_size') == model_size:
            model_config = config_entry
            break

    if not model_config:
        available_sizes = sorted(set(config.get('model_size') for config in model_configs if config.get('model_size')))
        gpu_info = f" for GPU type '{cluster_gpu_type}'" if cluster_gpu_type else ""
        error_msg = f"Model size '{model_size}' not supported for workload '{workload_key}'{gpu_info}."
        return False, ValidationErrorType.MODEL_SIZE_NOT_SUPPORTED, error_msg, available_sizes

    if dtype is not None:
        supported_dtypes = model_config.get('dtypes', [])
        if isinstance(supported_dtypes, str):
            supported_dtypes = [supported_dtypes]
        if dtype not in supported_dtypes:
            error_msg = f"Data type '{dtype}' not supported for {workload_key}_{model_size}."
            return False, ValidationErrorType.DTYPE_NOT_SUPPORTED, error_msg, supported_dtypes

    if scale is not None:
        supported_scales = model_config.get('scales', [])
        exact_scales = model_config.get('exact_scales', False)
        
        try:
            scale_int = int(scale)
            supported_scales_int = [int(s) for s in supported_scales]
            
            if exact_scales:
                # Exact mode: only allow scales that are explicitly listed
                if scale_int not in supported_scales_int:
                    error_msg = f"Scale {scale} not supported for {workload_key}_{model_size}."
                    return False, ValidationErrorType.SCALE_NOT_SUPPORTED, error_msg, supported_scales
            else:
                # Default mode: allow exact matches and scales above maximum (with warning)
                if supported_scales_int:  # Only validate if scales are defined
                    min_scale = min(supported_scales_int)
                    max_tested_scale = max(supported_scales_int)
                    
                    if scale_int < min_scale:
                        error_msg = f"Scale {scale} is below minimum supported scale for {workload_key}_{model_size}."
                        return False, ValidationErrorType.SCALE_NOT_SUPPORTED, error_msg, [f"minimum: {min_scale}"] + supported_scales
                    elif scale_int in supported_scales_int:
                        # Exact match - valid
                        pass
                    elif scale_int > max_tested_scale:
                        logger.warning(f"Scale {scale} for {workload_key}_{model_size} exceeds maximum tested scale of {max_tested_scale}. This configuration has not been validated.")
                    else:
                        error_msg = f"Scale {scale} not supported for {workload_key}_{model_size}."
                        return False, ValidationErrorType.SCALE_NOT_SUPPORTED, error_msg, supported_scales
                        
        except (ValueError, TypeError):
            error_msg = f"Invalid scale format '{scale}' for {workload_key}_{model_size}. Scale must be a number."
            return False, ValidationErrorType.SCALE_NOT_SUPPORTED, error_msg, supported_scales

    # If we reached here, validation passed
    logger.debug(f"Validation successful for {workload_key}_{model_size} (dtype: {dtype}, scale: {scale}).")
    return True, None, "", []

def valid_workload(workloads, workload_key, model_size, dtype=None, scale=None, cluster_gpu_type=None, cluster_config=None):
    """Validate if a given workload, model_size, dtype, and scale are supported based on loaded metadata."""
    is_valid, error_type, error_msg, suggestions = validate_workload_with_details(
        workloads, workload_key, model_size, dtype, scale, cluster_gpu_type, cluster_config
    )
    
    if not is_valid:
        logger.debug(f"Validation failed: {error_msg}")
        if suggestions:
            logger.debug(f"Available options: {suggestions}")
    
    return is_valid

def format_validation_error(workload_key, model_size, dtype, scale, cluster_gpu_type, error_type, error_msg, suggestions):
    """Format a user-friendly validation error message with suggestions."""
    lines = [f"Error: {error_msg}"]
    
    if suggestions:
        if error_type == ValidationErrorType.WORKLOAD_NOT_FOUND:
            lines.append(f"Available workloads: {', '.join(suggestions[:10])}")
            if len(suggestions) > 10:
                lines.append("  (use 'llmb-run list' to see all installed workloads)")
        elif error_type == ValidationErrorType.WORKLOAD_NOT_INSTALLED:
            lines.append(f"Installed workloads: {', '.join(suggestions)}")
        elif error_type == ValidationErrorType.GPU_TYPE_NOT_SUPPORTED:
            lines.append(f"Supported GPU types: {', '.join(suggestions)}")
        elif error_type == ValidationErrorType.MODEL_SIZE_NOT_SUPPORTED:
            lines.append(f"Available model sizes: {', '.join(suggestions)}")
        elif error_type == ValidationErrorType.DTYPE_NOT_SUPPORTED:
            lines.append(f"Supported data types: {', '.join(suggestions)}")
        elif error_type == ValidationErrorType.SCALE_NOT_SUPPORTED:
            lines.append(f"Supported scales: {', '.join(map(str, suggestions))}")
    
    return "\n".join(lines)

def model_optimizations(model_overrides):
    """Convert parameter overrides into OPITMIZATION_NAME, CODE for recipe."""
    optimizations = []
    for key, value in model_overrides.items():
        if key in nemo_model_params:
            optimizations.append(f"{nemo_model_params[key]}={value}")
        else:
            logger.warning(f"Unknown model parameter: {key}={value}")
    
    return ' '.join(optimizations)

def get_slurm_env_vars(config):
    """Convert slurm config section to SLURM_ environment variables."""
    slurm_env = {}
    slurm_config = config.get('slurm', {})
    
    if slurm_config.get('account'):
        slurm_env['SLURM_ACCOUNT'] = str(slurm_config['account'])
    if slurm_config.get('gpu_partition'):
        slurm_env['SLURM_PARTITION'] = str(slurm_config['gpu_partition'])
    if slurm_config.get('gpu_gres'):
        slurm_env['SLURM_GPUS_PER_NODE'] = str(slurm_config['gpu_gres'])
    
    return slurm_env

def get_workload_config(config, workload_key):
    """Get workload-specific configuration from cluster config."""
    workload_configs = config.get('workloads', {}).get('config', {})
    return workload_configs.get(workload_key, {})

def get_gpu_type(config, workloads, task):
    """Determine GPU type for a task from cluster config only."""
    # GPU type is determined only from cluster config for homogeneous environments
    return config.get('launcher', {}).get('gpu_type', 'h100')

def get_launcher_type(workloads, task):
    """Determine launcher type from workload metadata."""
    metadata = workloads[task.workload_key]['metadata']
    return metadata.get('run', {}).get('launcher_type', 'sbatch')

def launch_sbatch(config, task, workloads):
    """Launch a task using the legacy sbatch method."""
    max_scale = 2304  # TODO: Make this dynamic
    gpus_per_node = 8  # TODO: Make this dynamic
    
    if task.scale > max_scale:
        logger.warning(f"Scale {task.scale} exceeds maximum allowed scale of {max_scale}. Skipping job.")
        return None
    
    job = {
        "workload": task.workload_key,
        "LLMB_INSTALL": config['launcher']['llmb_install'],
        "dir": workloads[task.workload_key]["dir"],
        "MODEL_SIZE": task.model_size,
        "DTYPE": task.dtype,
        "NODES": task.scale // gpus_per_node,
        "enable_profile": task.profile,
    }
    
    cmd = f"sbatch --parsable --output={config['cwd']}/slurm-%j.out -N {job['NODES']} -J {job['workload']}_{job['MODEL_SIZE']}_{job['DTYPE']}_{job['NODES']} launch.sh"
    
    env = os.environ.copy()
    
    # Set basic job environment variables
    env["LLMB_INSTALL"] = job["LLMB_INSTALL"]
    env["MODEL_SIZE"] = job["MODEL_SIZE"]
    env["DTYPE"] = job["DTYPE"]
    env["JOB_TOTAL_GPUS"] = str(task.scale)
    env["GPU_TYPE"] = get_gpu_type(config, workloads, task)
    
    # Legacy STAGE_PATH for backward compatibility
    env["STAGE_PATH"] = job["LLMB_INSTALL"]
    
    # Add SLURM environment variables
    slurm_env = get_slurm_env_vars(config)
    env.update(slurm_env)
    
    # Reduce step count when doing profiling
    if task.profile:
        env['ENABLE_PROFILE'] = 'true'
    
    # Handle environment variables from config
    if 'environment' in config:
        env_vars = {k: str(v) for k, v in config['environment'].items()}
        env.update(env_vars)
    
    # Convert all task override values to strings
    task_env = {k: str(v) for k, v in task.env_overrides.items()}
    env.update(task_env)
    
    # Handle model parameter overrides
    if task.model_overrides:
        env['OPTIMIZATION_NAME'] = 'cc'  # Custom Config
        env['OPTIMIZATION_CODE'] = model_optimizations(task.model_overrides)
    
    try:
        logger.debug(f"Command: {cmd}")
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            check=True,
            text=True,
            env=env,
            cwd=job['dir']
        )
        job_id = result.stdout.strip()
        logger.info(format_task_output(task, prefix="SUBMITTED: ", suffix=f"jobid={job_id}"))
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"error={e.stderr.strip()}"))
        return None

def launch_nemo2(config, task, workloads):
    """Launch a task using the Nemo2 method with venv activation."""
    workload_config = get_workload_config(config, task.workload_key)
    venv_path = workload_config.get('venv_path')
    venv_type = workload_config.get('venv_type')
    
    if not venv_path:
        logger.error(f"venv_path not found in config for workload {task.workload_key}")
        return None
    
    if not venv_type:
        logger.error(f"venv_type not found in config for workload {task.workload_key}")
        return None
    
    try:
        # Get venv environment with the correct type
        env = get_venv_environment(venv_path, venv_type)
        
        # Set basic job environment variables
        env["LLMB_INSTALL"] = config['launcher']['llmb_install']
        env["MODEL_SIZE"] = task.model_size
        env["DTYPE"] = task.dtype
        env["JOB_TOTAL_GPUS"] = str(task.scale)
        env["GPU_TYPE"] = get_gpu_type(config, workloads, task)
        
        # Add SLURM environment variables
        slurm_env = get_slurm_env_vars(config)
        env.update(slurm_env)
        
        # Reduce step count when doing profiling
        if task.profile:
            env['ENABLE_PROFILE'] = 'true'
        
        # Handle environment variables from config
        if 'environment' in config:
            env_vars = {k: str(v) for k, v in config['environment'].items()}
            env.update(env_vars)
        
        # Handle workload-specific environment variables
        workload_env = workload_config.get('environment', {})
        workload_env_str = {k: str(v) for k, v in workload_env.items()}
        env.update(workload_env_str)
        
        # Convert all task override values to strings
        task_env = {k: str(v) for k, v in task.env_overrides.items()}
        env.update(task_env)
        
        # Run launch.sh script in the workload directory
        workload_dir = workloads[task.workload_key]["dir"]
        launch_script = "launch.sh"
        
        logger.debug(f"Running {launch_script} in {workload_dir} with venv {venv_path} (type: {venv_type})")
        result = subprocess.run(
            [f"./{launch_script}"],
            env=env,
            cwd=workload_dir,
            check=True,
            text=True
        )
        
        logger.info(format_task_output(task, prefix="LAUNCHED: ", suffix="completed"))
        return "local"  # Return a placeholder since there's no job ID
        
    except ValueError as e:
        logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"venv error: {e}"))
        return None
    except subprocess.CalledProcessError as e:
        logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"launch error: {e}"))
        return None

def format_task_output(task, prefix="", suffix=""):
    """Format task details in a consistent way with aligned columns."""
    # Fixed width fields for alignment
    workload_field = f"{task.workload_key}_{task.model_size}"
    dtype_field = f"dtype={task.dtype}"
    scale_field = f"scale={task.scale}"
    profile_field = f"profile={task.profile}"
    
    # Build the base output with aligned fields
    output = f"{prefix}{workload_field:<30} {dtype_field:<12} {scale_field:<12} {profile_field:<15}"
    
    # Add optional fields if they exist
    if task.env_overrides:
        output += f" env={task.env_overrides}"
    if task.model_overrides:
        output += f" params={task.model_overrides}"
    if suffix:
        output += f" {suffix}"
    return output

def print_tasks(task_list):
    """Print task details in a consistent format."""
    for task in task_list:
        logger.info(format_task_output(task, prefix="Task: "))

def run_tests(config, task_list, workloads):
    """Run tests using the appropriate launcher for each task."""
    for task in task_list:
        launcher_type = get_launcher_type(workloads, task)
        
        if launcher_type == 'nemo':
            job_id = launch_nemo2(config, task, workloads)
        elif launcher_type == 'sbatch':
            job_id = launch_sbatch(config, task, workloads)
        else:
            logger.error(f"Unknown launcher_type '{launcher_type}' for workload {task.workload_key}")
            continue

def get_slurm_job_status(jobid: int):
    cmd = f"sacct -X --format=State --noheader -j {jobid}"
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
        job_status = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running sacct: {e.stderr}")
        job_status = None
    return job_status

def print_avail_workloads(workloads, cluster_gpu_type=None, verbose=False, cluster_config=None):
    """Print available workloads with their supported configurations."""
    if not workloads:
        logger.info("No workloads found.")
        return
    
    installed_workloads = cluster_config.get('workloads', {}).get('installed', [])
    
    filtered_workloads = {k: v for k, v in workloads.items() if k in installed_workloads}
    
    if not filtered_workloads:
        logger.info("No installed workloads found.")
        return
    
    logger.info("Available workloads:")
    logger.info("=" * 50)
    
    for workload_key in sorted(filtered_workloads.keys()):
        metadata = filtered_workloads[workload_key]['metadata']
        gpu_configs = metadata.get('run', {}).get('gpu_configs', {})
        
        # Filter by cluster GPU type if specified
        if cluster_gpu_type and cluster_gpu_type in gpu_configs:
            relevant_configs = {cluster_gpu_type: gpu_configs[cluster_gpu_type]}
        else:
            relevant_configs = gpu_configs
        
        if not relevant_configs:
            continue
            
        logger.info(f"\n{workload_key}:")
        
        # Collect all model sizes, dtypes, and scales across GPU types
        all_model_sizes = set()
        model_details = {}
        
        for gpu_type, config in relevant_configs.items():
            model_configs = config.get('model_configs', [])
            for model_config in model_configs:
                model_size = model_config.get('model_size')
                if model_size:
                    all_model_sizes.add(model_size)
                    if model_size not in model_details:
                        model_details[model_size] = {
                            'dtypes': set(),
                            'scales': set(),
                            'gpu_types': set()
                        }
                    
                    # Ensure dtypes is always a list, even if it's a single string
                    dtypes_value = model_config.get('dtypes', [])
                    if isinstance(dtypes_value, str):
                        dtypes_value = [dtypes_value]
                    model_details[model_size]['dtypes'].update(dtypes_value)
                    model_details[model_size]['scales'].update(map(str, model_config.get('scales', [])))
                    model_details[model_size]['gpu_types'].add(gpu_type)
        
        if verbose:
            for model_size in sorted(all_model_sizes):
                details = model_details[model_size]
                logger.info(f"  {model_size}:")
                logger.info(f"    Data types: {', '.join(sorted(details['dtypes']))}")
                logger.info(f"    Scales: {', '.join(sorted(details['scales'], key=int))}")
                if len(details['gpu_types']) > 1 or not cluster_gpu_type:
                    logger.info(f"    GPU types: {', '.join(sorted(details['gpu_types']))}")
        else:
            model_sizes = sorted(all_model_sizes)
            logger.info(f"  Model sizes: {', '.join(model_sizes)}")
    
    if not verbose:
        logger.info(f"\nUse --verbose to see detailed configuration for each workload.")
    logger.info("")

def show_workload_details(workloads, workload_key, cluster_gpu_type=None, cluster_config=None):
    """Show detailed information about a specific workload."""
    if workload_key not in workloads:
        logger.error(f"Workload '{workload_key}' not found.")
        return
    
    installed_workloads = cluster_config.get('workloads', {}).get('installed', [])
    if installed_workloads and workload_key not in installed_workloads:
        logger.error(f"Workload '{workload_key}' is not installed on this cluster.")
        logger.info(f"Installed workloads: {', '.join(installed_workloads)}")
        return
    
    metadata = workloads[workload_key]['metadata']
    gpu_configs = metadata.get('run', {}).get('gpu_configs', {})
    
    logger.info(f"Workload: {workload_key}")
    logger.info("=" * (len(workload_key) + 10))
    
    # Filter by cluster GPU type if specified
    if cluster_gpu_type and cluster_gpu_type in gpu_configs:
        relevant_configs = {cluster_gpu_type: gpu_configs[cluster_gpu_type]}
    else:
        relevant_configs = gpu_configs
    
    for gpu_type, config in relevant_configs.items():
        logger.info(f"\nGPU Type: {gpu_type}")
        model_configs = config.get('model_configs', [])
        
        for model_config in model_configs:
            model_size = model_config.get('model_size', 'Unknown')
            dtypes = model_config.get('dtypes', [])
            # Ensure dtypes is always a list, even if it's a single string
            if isinstance(dtypes, str):
                dtypes = [dtypes]
            scales = model_config.get('scales', [])
            exact_scales = model_config.get('exact_scales', False)
            
            logger.info(f"  Model Size: {model_size}")
            logger.info(f"    Data types: {', '.join(dtypes) if dtypes else 'None specified'}")
            
            if scales:
                scale_info = f"    Scales: {', '.join(map(str, scales))}"
                if exact_scales:
                    scale_info += " (exact matches only)"
                else:
                    scale_info += f" (or higher, max tested: {max(scales)})"
                logger.info(scale_info)
            else:
                logger.info("    Scales: None specified")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='llmb-run: Tool for launching multiple or single LLM benchmarking workloads.'
    )

    subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode of operation')
    
    # List Mode Subparser
    list_parser = subparsers.add_parser('list', help='List available workloads and their configurations.')
    list_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed configuration for each workload.'
    )
    list_parser.add_argument(
        '-w', '--workload',
        type=str,
        help='Show detailed information for a specific workload.'
    )
    
    # Bulk Mode Subparser
    bulk_parser = subparsers.add_parser('bulk', help='Submit multiple jobs from a specification file.')
    bulk_parser.add_argument(
        '-d', '--dryrun',
        action='store_true',
        help='List all jobs to be submitted without actually submitting them.'
    )
    bulk_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output including debug information.'
    )
    bulk_parser.add_argument(
        'input_file',
        type=str,
        help='Path to the workload specification file (simple .txt or advanced .yaml).'
    )
    
    # Single Job Subparser
    single_parser = subparsers.add_parser('single', help='Submit a single job with specified parameters.')
    single_parser.add_argument(
        '-w', '--workload',
        type=str,
        required=True,
        help='Name of the workload (e.g., "pretraining_nemotron").'
    )
    single_parser.add_argument(
        '-s', '--model_size',
        type=str,
        required=True,
        help='Size of the model (e.g., 7b, 13b).'
    )
    single_parser.add_argument(
        '--dtype',
        type=str,
        required=True,
        help='Data type (e.g., fp16, bf16).'
    )
    single_parser.add_argument(
        '--scale',
        type=int,
        required=True,
        help='Scale parameter indicating the number of GPUs.'
    )
    # TODO: Time limit feature not yet implemented
    # Temporarily removed time_limit argument until feature is implemented
    
    single_parser.add_argument(
        '-p', '--profile',
        action='store_true',
        help='Enable Profiling for job.'
    )
    single_parser.add_argument(
        '-d', '--dryrun',
        action='store_true',
        help='List the job to be submitted without actually submitting it.'
    )
    single_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output including debug information.'
    )

    return parser.parse_args()

def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)

def main():
    """Main entry point for the llmb-run CLI."""
    args = parse_arguments()
    setup_logging(args.verbose)
    cluster_config = get_cluster_config()
    cluster_config['cwd'] = pathlib.Path.cwd() # Output slurm logs to current working directory.

    # Available workloads
    workloads = get_workloads(cluster_config)

    if args.mode == 'bulk':
        # Use the wrapper to auto-select the parser based on file extension.
        tasks_parsed = get_tasks_wrapper(workloads, args.input_file, cluster_config)
        if args.input_file.endswith(('.yaml', '.yml')):
            task_list = flatten_yaml_tasks(tasks_parsed)
        else:
            task_list = gen_tasks(tasks_parsed)

        if args.dryrun:
            print_tasks(task_list)
            logger.info("Dry run enabled. Jobs will not be submitted.")
        else:
            run_tests(cluster_config, task_list, workloads)
    elif args.mode == 'single':
        # Single Job Mode
        workload_input = args.workload
        model_size = args.model_size
        dtype = args.dtype
        scale = args.scale
        
        # Get cluster GPU type for validation
        cluster_gpu_type = cluster_config.get('launcher', {}).get('gpu_type')
        
        # Validate workload, model_size, dtype, and scale against metadata
        is_valid, error_type, error_msg, suggestions = validate_workload_with_details(
            workloads, workload_input, model_size, dtype=dtype, scale=scale, 
            cluster_gpu_type=cluster_gpu_type, cluster_config=cluster_config
        )
        if not is_valid:
            user_error = format_validation_error(
                workload_input, model_size, dtype, scale, cluster_gpu_type, error_type, error_msg, suggestions
            )
            logger.error(user_error)
            exit(1)
        
        # Create a single WorkloadTask
        single_task = WorkloadTask(
            workload_key=workload_input,
            model_size=model_size,
            dtype=dtype,
            scale=scale,
            profile=args.profile
        )

        if args.dryrun:
            print_tasks([single_task])
            logger.info("Dry run enabled. Job will not be submitted.")
        else:
            run_tests(cluster_config, [single_task], workloads)
    elif args.mode == 'list':
        # List Mode
        cluster_gpu_type = cluster_config.get('launcher', {}).get('gpu_type')
        workload = args.workload
        if workload:
            show_workload_details(workloads, workload, cluster_gpu_type, cluster_config)
        else:
            print_avail_workloads(workloads, cluster_gpu_type, args.verbose, cluster_config)

if __name__ == '__main__':
    main()
