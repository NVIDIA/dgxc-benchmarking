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

"""Task management for workload execution."""

import ast
import itertools
import logging
import re
from dataclasses import dataclass, field

import yaml

from llmb_run.metadata_utils import normalize_model_dtype_config
from llmb_run.workload_validator import (
    format_validation_error,
    validate_workload_with_details,
)

logger = logging.getLogger('llmb_run.task_manager')


@dataclass
class WorkloadTask:
    workload_key: str  # Full workload key (e.g., 'pretraining_nemotron')
    model_size: str
    dtype: str
    scale: int
    profile: bool = False
    env_overrides: dict = field(default_factory=dict)
    model_overrides: dict = field(default_factory=dict)


def merge_dicts(default, override):
    """Helper to merge two dictionaries (override takes precedence)."""
    result = default.copy()
    result.update(override)
    return result


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
                            workload_key,
                            model_size,
                            None,
                            None,
                            cluster_config.get('launcher', {}).get('gpu_type') if cluster_config else None,
                            error_type,
                            error_msg,
                            suggestions,
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
                        raise ValueError(f"Invalid task format: {e}") from e

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
                    for name, value in zip(param_names, param_combination, strict=False):
                        current_params[name] = value

                    # Generate tasks for this combination
                    for dtype, scale in itertools.product(dtypes, scales):
                        # Add regular performance runs
                        for _r in range(repeats):
                            tasks_list.append((workload, model_size, dtype, scale, profile, task_env, current_params))
                        # Add one profiling run if requested
                        if add_profile:
                            tasks_list.append((workload, model_size, dtype, scale, True, task_env, current_params))
            else:
                # No sweeps needed, proceed as before
                current_params = default_params.copy()
                current_params.update(regular_params)
                for dtype, scale in itertools.product(dtypes, scales):
                    # Add regular performance runs
                    for _r in range(repeats):
                        tasks_list.append((workload, model_size, dtype, scale, profile, task_env, current_params))
                    # Add one profiling run if requested
                    if add_profile:
                        tasks_list.append((workload, model_size, dtype, scale, True, task_env, current_params))

        workload_tasks.setdefault(workload, {})[model_size] = tasks_list
    return workload_tasks


def get_tasks_wrapper(workloads, input_file, cluster_config=None):
    """Dispatcher for task parsing based on file extension."""
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
                    for _r in range(repeats):
                        # If optional params exist, the first is profile.
                        profile = params[0] if params else False
                        task_list.append(WorkloadTask(workload_key_from_task, model_size, dt, scale, profile))
    return task_list


def flatten_yaml_tasks(advanced_tasks):
    """Flatten tasks from YAML advanced format to WorkloadTask objects."""
    task_list = []
    for workload in advanced_tasks:
        for model_size in advanced_tasks[workload]:
            for t in advanced_tasks[workload][model_size]:
                # t is a tuple: (workload, model_size, dtype, scale, profile, env_overrides, model_overrides)
                w, m, dt, scale, profile, env_overrides, model_overrides = t
                task_list.append(
                    WorkloadTask(w, m, dt, scale, profile, env_overrides=env_overrides, model_overrides=model_overrides)
                )
    return task_list


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


def generate_submit_all_tasks(
    workloads,
    cluster_config,
    max_scale,
    repeats=1,
    profile=False,
    min_scale=False,
    dtype_filter=None,
    workload_filter=None,
    specific_scales=None,
):
    """Generate tasks for all installed workloads up to max_scale.

    Args:
        workloads: Dictionary of available workloads from get_workloads()
        cluster_config: Cluster configuration dictionary
        max_scale: Maximum scale (number of GPUs) to test up to, or None for metadata scales.
        repeats: Number of repeats for each configuration (default: 1)
        profile: Whether to enable profiling for all tasks (default: False)
        min_scale: If True, only run minimum scale per metadata (default: False)
        dtype_filter: List of dtypes to filter by, or None for all (default: None)
        workload_filter: List of workloads to filter by, or None for all (default: None)
        specific_scales: List of specific scales to run, or None to use max_scale/min_scale logic (default: None)

    Returns:
        list: List of WorkloadTask objects for all valid configurations
    """
    # Get configuration details
    installed_workloads = cluster_config.get('workloads', {}).get('installed', [])
    cluster_gpu_type = cluster_config.get('launcher', {}).get('gpu_type')

    if not cluster_gpu_type:
        logger.error("No GPU type specified in cluster configuration")
        return []

    max_scale_str = max_scale if max_scale is not None else "metadata scales"
    logger.info(
        f"Discovering tasks for installed workloads (max_scale: {max_scale_str}, repeats: {repeats}, profile: {profile}, min_scale: {min_scale})"
    )
    if dtype_filter:
        logger.info(f"Filtering dtypes: {dtype_filter}")
    if workload_filter:
        logger.info(f"Filtering workloads: {workload_filter}")

    task_list = []
    filtered_workloads = {}

    # Filter workloads by installation status and type
    allowed_types = ['pretrain', 'finetune']
    for workload_key, workload_data in workloads.items():
        if workload_key not in installed_workloads:
            continue

        workload_type = workload_data.get('workload_type', '')
        if workload_type not in allowed_types:
            logger.debug(f"Skipping {workload_key}: workload_type '{workload_type}' not in {allowed_types}")
            continue

        # Apply workload filter if specified
        if workload_filter:
            # Check if workload_key matches any filter (either exact match or filter starts with workload_key)
            workload_matches = False
            for filter_item in workload_filter:
                # Exact match or filter starts with workload (e.g., pretrain_nemotron matches pretrain_nemotron_340b)
                if workload_key == filter_item or filter_item.startswith(workload_key + '_'):
                    workload_matches = True
                    break
            if not workload_matches:
                logger.debug(f"Skipping {workload_key}: not in workload filter {workload_filter}")
                continue

        filtered_workloads[workload_key] = workload_data

    if not filtered_workloads:
        logger.info("No installed pretrain/finetuning workloads found")
        return []

    logger.info(f"Found {len(filtered_workloads)} eligible workloads: {', '.join(filtered_workloads.keys())}")

    # Generate tasks for each eligible workload
    for workload_key, workload_data in filtered_workloads.items():
        _generate_workload_tasks(
            workload_key,
            workload_data,
            cluster_gpu_type,
            max_scale,
            repeats,
            profile,
            task_list,
            min_scale,
            dtype_filter,
            workload_filter,
            specific_scales,
        )

    logger.info(f"Generated {len(task_list)} tasks across {len(filtered_workloads)} workloads")
    return task_list


def _generate_workload_tasks(
    workload_key,
    workload_data,
    cluster_gpu_type,
    max_scale,
    repeats,
    profile,
    task_list,
    min_scale=False,
    dtype_filter=None,
    workload_filter=None,
    specific_scales=None,
):
    """Generate tasks for a single workload and add them to task_list.

    Args:
        workload_key: The workload identifier
        workload_data: Workload metadata and configuration
        cluster_gpu_type: GPU type of the cluster
        max_scale: Maximum scale to test up to, or None for metadata scales.
        repeats: Number of repeats per configuration
        profile: Whether to enable profiling
        task_list: List to append generated tasks to
        min_scale: If True, only run minimum scale per metadata (default: False)
        dtype_filter: List of dtypes to filter by, or None for all (default: None)
        workload_filter: List of workload filters, may include workload_modelsize (default: None)
        specific_scales: List of specific scales to run, or None to use max_scale/min_scale logic (default: None)
    """
    metadata = workload_data['metadata']
    gpu_configs = metadata.get('run', {}).get('gpu_configs', {})

    # Check if workload supports the cluster's GPU type
    if cluster_gpu_type not in gpu_configs:
        logger.warning(f"Skipping {workload_key}: no configuration for GPU type '{cluster_gpu_type}'")
        return

    gpu_config = gpu_configs[cluster_gpu_type]
    model_configs = gpu_config.get('model_configs', [])

    # Generate tasks for each model configuration
    for model_config in model_configs:
        model_size = model_config.get('model_size')
        if not model_size:
            continue

        # Apply workload_modelsize filter if specified
        if workload_filter:
            workload_modelsize = f"{workload_key}_{model_size}"
            model_matches = False
            for filter_item in workload_filter:
                if filter_item == workload_key or filter_item == workload_modelsize:
                    model_matches = True
                    break
            if not model_matches:
                logger.debug(f"Skipping {workload_modelsize}: not in workload filter {workload_filter}")
                continue

        # Normalize dtypes to a mapping of dtype -> {scales, exact_scales}
        dtype_map = normalize_model_dtype_config(model_config)

        if not dtype_map:
            logger.warning(f"Skipping {workload_key}_{model_size}: no dtypes defined")
            continue

        # Create tasks for permutations per dtype respecting per-dtype scales
        for dtype, cfg in dtype_map.items():
            # Apply dtype filter if specified
            if dtype_filter and dtype not in dtype_filter:
                logger.debug(f"Skipping {workload_key}_{model_size} dtype={dtype}: not in dtype filter {dtype_filter}")
                continue

            dtype_scales = cfg.get('scales', [])
            exact_scales = cfg.get('exact_scales', model_config.get('exact_scales', False))

            if not dtype_scales:
                logger.warning(f"Skipping {workload_key}_{model_size} dtype={dtype}: no scales defined")
                continue

            if specific_scales is not None:
                # Use specific scales, but only those supported by the workload
                scales_to_test = []
                for requested_scale in specific_scales:
                    if exact_scales:
                        # For exact scales, only include scales that are explicitly supported
                        if requested_scale in dtype_scales:
                            scales_to_test.append(requested_scale)
                        else:
                            logger.debug(
                                f"Skipping scale {requested_scale} for {workload_key}_{model_size} dtype={dtype}: not in supported exact scales {dtype_scales}"
                            )
                    else:
                        # For non-exact scales, follow the same logic as max_scale validation
                        if dtype_scales:  # Only validate if scales are defined
                            min_supported_scale = min(dtype_scales)
                            max_tested_scale = max(dtype_scales)

                            if requested_scale < min_supported_scale:
                                logger.debug(
                                    f"Skipping scale {requested_scale} for {workload_key}_{model_size} dtype={dtype}: below minimum supported scale {min_supported_scale}"
                                )
                            elif requested_scale in dtype_scales or requested_scale > max_tested_scale:
                                # Either exact match or above max tested (will get warning in validation)
                                scales_to_test.append(requested_scale)
                            else:
                                logger.debug(
                                    f"Skipping scale {requested_scale} for {workload_key}_{model_size} dtype={dtype}: not supported"
                                )
                        else:
                            # No scale restrictions defined, accept the requested scale
                            scales_to_test.append(requested_scale)
            elif min_scale:
                # If min_scale flag is set, only use the minimum scale (optionally constrained by max_scale)
                if max_scale is not None:
                    min_valid_scale = (
                        min(scale for scale in dtype_scales if scale <= max_scale)
                        if any(scale <= max_scale for scale in dtype_scales)
                        else None
                    )
                    if min_valid_scale is None:
                        logger.debug(
                            f"No valid min scale for {workload_key}_{model_size} dtype={dtype} within max_scale={max_scale}"
                        )
                        continue
                    scales_to_test = [min_valid_scale]
                else:
                    # No max_scale limit, just use the minimum scale from metadata
                    scales_to_test = [min(dtype_scales)]
            else:
                scales_to_test = _generate_scales_up_to_max(dtype_scales, max_scale, exact_scales)

            if not scales_to_test:
                max_scale_str = max_scale if max_scale is not None else "metadata scales"
                logger.debug(
                    f"No valid scales for {workload_key}_{model_size} dtype={dtype} within max_scale={max_scale_str}"
                )
                continue

            for scale in scales_to_test:
                for _ in range(repeats):
                    task_list.append(
                        WorkloadTask(
                            workload_key=workload_key,
                            model_size=model_size,
                            dtype=dtype,
                            scale=scale,
                            profile=profile,
                        )
                    )


def _generate_scales_up_to_max(metadata_scales, max_scale, exact_scales):
    """Generate list of scales to test up to max_scale.

    Args:
        metadata_scales: List of scales from metadata file
        max_scale: Maximum scale (number of GPUs) to test, or None for metadata scales.
        exact_scales: If True, only use scales from metadata (up to max)

    Returns:
        list: Sorted list of scales to test
    """
    if not metadata_scales:
        return []

    metadata_scales_int = sorted([int(s) for s in metadata_scales])

    if exact_scales:
        # Only use scales from metadata (optionally up to max)
        if max_scale is not None:
            return [s for s in metadata_scales_int if s <= max_scale]
        else:
            return metadata_scales_int

    # Use all metadata scales up to max, plus power-of-2 scales beyond max metadata scale
    if max_scale is not None:
        scales_to_test = [s for s in metadata_scales_int if s <= max_scale]

        # If max_scale is greater than the highest metadata scale, add power-of-2 scales
        max_metadata_scale = max(metadata_scales_int)
        if max_scale > max_metadata_scale:
            # Find next power of 2 after max_metadata_scale
            next_power = 1
            while next_power <= max_metadata_scale:
                next_power *= 2

            # Add power-of-2 scales up to max_scale
            while next_power <= max_scale:
                scales_to_test.append(next_power)
                next_power *= 2
    else:
        # No max limit, just use metadata scales
        scales_to_test = metadata_scales_int

    return sorted(set(scales_to_test))
