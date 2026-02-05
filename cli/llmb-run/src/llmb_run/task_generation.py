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

"""Unified task generation logic for llmb-run."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llmb_run.metadata_utils import normalize_model_dtype_config
from llmb_run.task_loader import (
    flatten_yaml_tasks,
    gen_tasks,
    get_tasks_wrapper,
)
from llmb_run.tasks import WorkloadTask

logger = logging.getLogger('llmb_run.task_generation')


class ValidationError(Exception):
    """Custom exception for validation errors during task generation."""

    pass


@dataclass
class TaskGenerationRequest:
    """Encapsulates all task generation parameters."""

    workloads: Dict[str, Any]
    cluster_config: Dict[str, Any]

    # Specification methods
    workload: Optional[str] = None  # Single or comma-separated
    model_size: Optional[str] = None  # Only for explicit mode
    dtype: Optional[str] = None  # Single or comma-separated
    scale: Optional[str] = None  # Single or comma-separated
    max_scale: Optional[int] = None  # For discovery
    min_scale: bool = False  # For discovery
    exact_scales: bool = False  # For discovery - prevent power-of-2 expansion
    file_path: Optional[str] = None  # For file mode

    # Modifiers
    repeats: int = 1
    profile: bool = False
    extra_slurm_params: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        """Validate parameter combinations."""
        # File mode restrictions
        if self.file_path:
            if any([self.workload, self.model_size, self.max_scale, self.min_scale]):
                raise ValidationError(
                    "Cannot mix --file with workload specifications\n"
                    "Either use: llmb-run submit -f my-tests.yaml\n"
                    "Or specify: llmb-run submit -w X -s Y -d Z --scale N"
                )

        # Model size restrictions
        if self.model_size:
            if not self.workload:
                raise ValidationError("--model-size requires --workload")

            # If model_size provided, only one workload allowed
            if ',' in (self.workload or ''):
                raise ValidationError(
                    "Global '--model-size' / '-s' cannot be used with multiple workloads.\n\n"
                    "To target specific model sizes, append them to the workload name.\n"
                    "Workloads without a suffix will run ALL available sizes.\n\n"
                    "Example:\n"
                    "  llmb-run submit -w pretrain_llama3.1_70b,pretrain_nemotron-h"
                )

        # Scale mutual exclusivity
        if self.scale and (self.max_scale or self.min_scale):
            raise ValidationError(
                "Cannot use --scale with --max-scale or --min-scale\n"
                "Either specify exact scale(s): --scale 128,256,512\n"
                "Or use discovery: --max-scale 512"
            )

        # Scale requirement: Always required except in file mode
        if not self.file_path:
            if not any([self.scale, self.max_scale, self.min_scale]):
                raise ValidationError(
                    "Must specify scale parameter\n"
                    "Either specify exact scale(s): --scale 128,256,512\n"
                    "Or use discovery: --max-scale 512 or --min-scale"
                )


def generate_tasks(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate tasks from unified request specification."""
    try:
        request.validate()
    except ValidationError as e:
        # Re-raise as ValueError to be compatible with typical main.py handling or catch explicitly
        raise ValueError(str(e)) from e

    if request.file_path:
        return _generate_from_file(request)

    if request.model_size:
        # Has explicit model size
        # Always use discovery/targeted mode logic which supports metadata validation,
        # implicit dtypes, and various scale specifications (exact, max, min).
        return _generate_explicit_workload_with_scale_discovery(request)
    else:
        # Discovery mode: workload names include size
        return _generate_discovery_tasks(request)


def parse_comma_list(value: Optional[str]) -> List[str]:
    """Parse comma-separated list, handling spaces."""
    if not value:
        return []
    # Strip spaces and filter empty strings
    return [item.strip() for item in value.split(',') if item.strip()]


def _generate_explicit_workload_with_scale_discovery(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate tasks for explicit workload with scale discovery.

    Example: llmb-run submit -w pretrain_nemotron4 -s 340b -d fp8 --max-scale 512
    Generates: nemotron4_340b at all supported scales up to 512
    """
    workload_key = request.workload
    model_size = request.model_size

    # Filter by the specific workload_modelsize combo
    # generate_submit_all_tasks supports filtering by "workload_key" or "workload_key_model_size"
    workload_filter = [f"{workload_key}_{model_size}"]

    dtype_filter = parse_comma_list(request.dtype) if request.dtype else None

    # Handle specific scales if provided
    specific_scales = parse_comma_list(request.scale)
    if specific_scales:
        specific_scales = [int(s) for s in specific_scales]
    else:
        specific_scales = None

    return generate_submit_all_tasks(
        request.workloads,
        request.cluster_config,
        request.max_scale,
        request.repeats,
        request.profile,
        min_scale=request.min_scale,
        exact_scales=request.exact_scales,
        dtype_filter=dtype_filter,
        workload_filter=workload_filter,
        specific_scales=specific_scales,
        extra_slurm_params=request.extra_slurm_params,
    )


def _generate_discovery_tasks(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate tasks using discovery mode (workload names include size)."""
    # Reuse existing generate_submit_all_tasks logic
    workload_filter = parse_comma_list(request.workload) if request.workload else None
    dtype_filter = parse_comma_list(request.dtype) if request.dtype else None

    # If request.scale is provided (specific scales), we pass it.
    # Discovery mode supports --scale 128,256 too (applied to filtered workloads).
    specific_scales = parse_comma_list(request.scale)
    if specific_scales:
        specific_scales = [int(s) for s in specific_scales]
    else:
        specific_scales = None

    return generate_submit_all_tasks(
        request.workloads,
        request.cluster_config,
        request.max_scale,
        request.repeats,
        request.profile,
        min_scale=request.min_scale,
        exact_scales=request.exact_scales,
        dtype_filter=dtype_filter,
        workload_filter=workload_filter,
        specific_scales=specific_scales,
        extra_slurm_params=request.extra_slurm_params,
    )


def _generate_from_file(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate tasks from file specification."""
    # Reuse existing get_tasks_wrapper logic
    tasks_parsed = get_tasks_wrapper(request.workloads, request.file_path, request.cluster_config)

    if request.file_path.endswith(('.yaml', '.yml')):
        tasks = flatten_yaml_tasks(tasks_parsed)
    else:
        tasks = gen_tasks(tasks_parsed)

    # Propagate extra_slurm_params to all generated tasks
    if request.extra_slurm_params:
        for task in tasks:
            task.extra_slurm_params = request.extra_slurm_params

    return tasks


def generate_submit_all_tasks(
    workloads,
    cluster_config,
    max_scale,
    repeats=1,
    profile=False,
    min_scale=False,
    exact_scales=False,
    dtype_filter=None,
    workload_filter=None,
    specific_scales=None,
    extra_slurm_params: Optional[Dict[str, Any]] = None,
):
    """Generate tasks for all installed workloads up to max_scale.

    By default (Discovery Mode), only 'pretrain' and 'finetune' workloads are included.
    Other types (e.g., 'inference', 'microbenchmark') can be included by explicitly
    requesting them via `workload_filter`.

    Args:
        workloads: Dictionary of available workloads from get_workloads()
        cluster_config: Cluster configuration dictionary
        max_scale: Maximum scale (number of GPUs) to test up to, or None for metadata scales.
        repeats: Number of repeats for each configuration (default: 1)
        profile: Whether to enable profiling for all tasks (default: False)
        min_scale: If True, only run minimum scale per metadata (default: False)
        exact_scales: If True, only use scales from metadata (no power-of-2 expansion) (default: False)
        dtype_filter: List of dtypes to filter by, or None for all (default: None)
        workload_filter: List of workloads to filter by, or None for all (default: None)
        specific_scales: List of specific scales to run, or None to use max_scale/min_scale logic (default: None)
        extra_slurm_params: Optional dictionary of extra Slurm parameters to apply to jobs.

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
    logger.debug(
        f"Discovering tasks for installed workloads (max_scale: {max_scale_str}, repeats: {repeats}, profile: {profile}, min_scale: {min_scale})"
    )
    if dtype_filter:
        logger.debug(f"Filtering dtypes: {dtype_filter}")
    if workload_filter:
        logger.debug(f"Filtering workloads: {workload_filter}")

    task_list = []
    filtered_workloads = {}

    # Filter workloads by installation status and type
    allowed_types = ['pretrain', 'finetune']
    for workload_key, workload_data in workloads.items():
        if workload_key not in installed_workloads:
            continue

        workload_type = workload_data.get('workload_type', '')
        if not workload_filter and workload_type not in allowed_types:
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

    logger.debug(f"Found {len(filtered_workloads)} eligible workloads: {', '.join(filtered_workloads.keys())}")

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
            runtime_exact_scales=exact_scales,
            dtype_filter=dtype_filter,
            workload_filter=workload_filter,
            specific_scales=specific_scales,
            extra_slurm_params=extra_slurm_params,
        )

    logger.debug(f"Generated {len(task_list)} tasks across {len(filtered_workloads)} workloads")
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
    runtime_exact_scales=False,
    dtype_filter=None,
    workload_filter=None,
    specific_scales=None,
    extra_slurm_params: Optional[Dict[str, Any]] = None,
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
        runtime_exact_scales: If True, only use scales from metadata (no power-of-2 expansion) (default: False)
        dtype_filter: List of dtypes to filter by, or None for all (default: None)
        workload_filter: List of workload filters, may include workload_modelsize (default: None)
        specific_scales: List of specific scales to run, or None to use max_scale/min_scale logic (default: None)
        extra_slurm_params: Optional dictionary of extra Slurm parameters to apply to jobs.
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
            metadata_exact_scales = cfg.get('exact_scales', model_config.get('exact_scales', False))
            # Logical OR: if runtime flag is set OR metadata says exact, then use exact scales
            effective_exact_scales = runtime_exact_scales or metadata_exact_scales

            if not dtype_scales:
                logger.warning(f"Skipping {workload_key}_{model_size} dtype={dtype}: no scales defined")
                continue

            if specific_scales is not None:
                # Use specific scales, but only those supported by the workload
                scales_to_test = []
                for requested_scale in specific_scales:
                    if effective_exact_scales:
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
                scales_to_test = _generate_scales_up_to_max(dtype_scales, max_scale, effective_exact_scales)

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
                            extra_slurm_params=extra_slurm_params or {},
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
