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

import logging
import os
import sys
from typing import Annotated, Optional

import typer

from llmb_run.config_manager import get_cluster_config
from llmb_run.exemplar import generate_exemplar_tasks
from llmb_run.job_launcher import run_tests
from llmb_run.metadata_utils import parse_workload_name
from llmb_run.task_generation import TaskGenerationRequest, ValidationError, generate_tasks
from llmb_run.tasks import (
    format_task_output,
)
from llmb_run.workload_validator import (
    format_validation_error,
    get_workloads,
    print_avail_workloads,
    validate_workload_with_details,
)


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


# Exit codes
EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 1
EXIT_SYSTEM_ERROR = 2


# Create the Typer app
app = typer.Typer(
    help='llmb-run: Tool for launching multiple or single LLM benchmarking workloads.',
    no_args_is_help=True,
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


class AppContext:
    """Context object to share configuration across commands."""

    def __init__(self):
        self.cluster_config = None
        self.workloads = None
        self.verbose = False


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: Annotated[
        bool, typer.Option('-v', '--verbose', help='Enable verbose output including debug information.')
    ] = False,
):
    """
    Main callback that runs before any command.
    Loads cluster configuration and workloads once, and configures logging.
    """
    # Check for SLURM environment
    if 'SLURM_JOB_ID' in os.environ:
        logger.error(
            "🚫: `llmb-run` does not currently support running within a SLURM allocation. Please run this script directly from a login node outside of a SLURM job."
        )
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    # Set up logging
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)

    # Create context object
    app_ctx = AppContext()
    app_ctx.verbose = verbose
    ctx.obj = app_ctx

    # Load configuration
    try:
        app_ctx.cluster_config = get_cluster_config()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e

    # Load workloads
    try:
        app_ctx.workloads = get_workloads(app_ctx.cluster_config)
    except Exception as e:
        logger.error(f"Failed to load workloads: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e


def validate_bulk_tasks(task_list, workloads, cluster_config):
    """Validate all tasks in a bulk job and return validated tasks with error summary.

    Returns:
        tuple: (validated_tasks, validation_summary)
            where validation_summary is a dict with error counts and unique error types
    """
    cluster_gpu_type = cluster_config.get('launcher', {}).get('gpu_type')
    validated_tasks = []
    error_summary = {}

    for task in task_list:
        is_valid, error_type, error_msg, suggestions = validate_workload_with_details(
            workloads,
            task.workload_key,
            task.model_size,
            dtype=task.dtype,
            scale=task.scale,
            cluster_gpu_type=cluster_gpu_type,
            cluster_config=cluster_config,
        )
        if is_valid:
            validated_tasks.append(task)
        else:
            # Group errors by type and message for cleaner reporting
            error_key = (error_type, error_msg, tuple(str(s) for s in suggestions))
            if error_key not in error_summary:
                error_summary[error_key] = {
                    'count': 0,
                    'error_type': error_type,
                    'error_msg': error_msg,
                    'suggestions': suggestions,
                    'example_task': task,
                }
            error_summary[error_key]['count'] += 1

    return validated_tasks, error_summary


def _handle_no_tasks_error(app_ctx: AppContext, request: TaskGenerationRequest):
    """Provide helpful error when no tasks are generated."""
    logger.error("No matching job configurations found.")

    # YAML file case: specific hint, skip list output
    if request.file_path and request.file_path.endswith(('.yaml', '.yml')):
        logger.error("For YAML files, ensure you have at least one entry under 'tasks:'")
        logger.error("  Example: tasks:")
        logger.error("             - dtypes: fp8")
        logger.error("               scales: [128, 256]")
    else:
        # Show what IS available - print_avail_workloads() handles the header
        logger.info("")  # Empty line for spacing

        # Build workload filter: either specific workloads or None for all
        workload_filter = None
        if request.workload:
            # Parse comma-separated workload list and strip model size suffixes.
            # Users can specify workloads as "pretrain_foo_7b" in discovery mode,
            # but print_avail_workloads expects base workload keys like "pretrain_foo".
            parsed = [parse_workload_name(w.strip())[0] for w in request.workload.split(',') if w.strip()]
            workload_filter = parsed if parsed else None

        # Call once with the filter (single or multiple workloads, or None for all)
        print_avail_workloads(
            app_ctx.workloads,
            app_ctx.cluster_config,
            cluster_gpu_type=app_ctx.cluster_config.get('launcher', {}).get('gpu_type'),
            verbose=True,
            workload_filter=workload_filter,
        )


def report_validation_results(validated_tasks, error_summary, task_list, cluster_config, mode_name="job"):
    """Report validation results in a consistent format across different modes.

    Args:
        validated_tasks: List of valid tasks
        error_summary: Dictionary of validation errors
        task_list: Original list of all tasks
        cluster_config: Cluster configuration
        mode_name: Name of the mode for error messages (e.g., "bulk", "submit-all")
    """
    cluster_gpu_type = cluster_config.get('launcher', {}).get('gpu_type')

    if error_summary:
        total_errors = sum(err['count'] for err in error_summary.values())
        logger.error(f"Validation failed for {total_errors} out of {len(task_list)} tasks:")

        for error_info in error_summary.values():
            count = error_info['count']
            example_task = error_info['example_task']
            error_type = error_info['error_type']
            error_msg = error_info['error_msg']
            suggestions = error_info['suggestions']

            # Use existing format_validation_error for consistent formatting
            formatted_error = format_validation_error(
                example_task.workload_key,
                example_task.model_size,
                example_task.dtype,
                example_task.scale,
                cluster_gpu_type,
                error_type,
                error_msg,
                suggestions,
            )

            # Add count prefix with example
            prefix = f"  ❌ {count}x {example_task.workload_key}_{example_task.model_size} (dtype={example_task.dtype})"

            # Split the formatted error and add prefix to first line, indent others
            error_lines = formatted_error.split('\n')
            logger.error(f"{prefix}: {error_lines[0]}")
            for line in error_lines[1:]:
                logger.error(f"     {line}")

        if not validated_tasks:
            logger.error(f"❌ No valid tasks found. Aborting {mode_name} submission.")
            raise typer.Exit(code=EXIT_VALIDATION_ERROR)
        else:
            logger.warning(f"⚠️  Proceeding with {len(validated_tasks)} valid tasks out of {len(task_list)} total.")
    else:
        logger.debug(f"✅ All {len(task_list)} tasks validated successfully.")


def _submit_impl(ctx: typer.Context, request: TaskGenerationRequest, dryrun: bool, mode_name: str = "submit"):
    """Shared implementation for all submission commands."""
    app_ctx: AppContext = ctx.obj

    try:
        # Generate tasks
        task_list = generate_tasks(request)

        # Sort tasks by scale in descending order (largest first)
        task_list.sort(key=lambda task: task.scale, reverse=True)

        if not task_list:
            _handle_no_tasks_error(app_ctx, request)
            raise typer.Exit(code=EXIT_VALIDATION_ERROR)

        # Validate tasks
        validated_tasks, error_summary = validate_bulk_tasks(task_list, app_ctx.workloads, app_ctx.cluster_config)

        # Report results
        report_validation_results(validated_tasks, error_summary, task_list, app_ctx.cluster_config, mode_name)

        # Print the concrete jobs we’re about to submit (kept concise; launcher output follows).
        logger.info(f"Jobs ({len(validated_tasks)}):")
        for task in validated_tasks:
            logger.info(format_task_output(task, prefix="  - "))

        if dryrun:
            logger.info("Dry run enabled. Jobs will not be submitted.")
        else:
            run_tests(app_ctx.cluster_config, validated_tasks, app_ctx.workloads)

    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Submission error: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e


@app.command()
def submit(
    ctx: typer.Context,
    workload: Annotated[
        Optional[str],
        typer.Option('-w', '--workload', help='Workload name (single or comma-separated for discovery).'),
    ] = None,
    model_size: Annotated[
        Optional[str],
        typer.Option(
            '-s',
            '--model-size',
            help='Size of the model (e.g., 7b, 13b). Requires explicit single workload via -w.',
        ),
    ] = None,
    dtype: Annotated[
        Optional[str],
        typer.Option('-d', '--dtype', help='Data type (e.g., fp16, bf16). Comma-separated list allowed.'),
    ] = None,
    scale: Annotated[
        Optional[str],
        typer.Option(
            '--scale',
            help='Scale parameter (number of GPUs). Comma-separated list allowed (e.g., "8,16"). Mutually exclusive with --max-scale.',
        ),
    ] = None,
    max_scale: Annotated[
        Optional[int], typer.Option('--max-scale', help='Maximum scale to test up to (discovery/mixed mode).')
    ] = None,
    min_scale: Annotated[
        bool,
        typer.Option('--min-scale', help='Only run the minimum supported scale (discovery/mixed mode).'),
    ] = False,
    exact_scales: Annotated[
        bool,
        typer.Option(
            '--exact-scales', help='Only use scales from metadata (no power-of-2 expansion beyond metadata max).'
        ),
    ] = False,
    file_path: Annotated[
        Optional[str], typer.Option('-f', '--file', help='Path to workload specification file (.txt or .yaml).')
    ] = None,
    repeats: Annotated[int, typer.Option('-r', '--repeats', help='Number of repeats for each test configuration.')] = 1,
    profile: Annotated[bool, typer.Option('-p', '--profile', help='Enable Profiling for jobs.')] = False,
    dryrun: Annotated[
        bool,
        typer.Option('--dry-run', help='List jobs without submitting them.'),
    ] = False,
    nice: Annotated[
        Optional[int], typer.Option('--nice', help='Lower the priority of the job using Slurm --nice feature.')
    ] = None,
):
    """
    Submit jobs using a unified interface. Supports explicit, discovery, and file-based modes.
    """
    app_ctx: AppContext = ctx.obj

    extra_slurm_params = {}
    if nice is not None:
        extra_slurm_params['nice'] = nice

    request = TaskGenerationRequest(
        workloads=app_ctx.workloads,
        cluster_config=app_ctx.cluster_config,
        workload=workload,
        model_size=model_size,
        dtype=dtype,
        scale=scale,
        max_scale=max_scale,
        min_scale=min_scale,
        exact_scales=exact_scales,
        file_path=file_path,
        repeats=repeats,
        profile=profile,
        extra_slurm_params=extra_slurm_params,
    )

    _submit_impl(ctx, request, dryrun, mode_name="submit")


@app.command(name="list")
def list_workloads(
    ctx: typer.Context,
    workload: Annotated[
        Optional[str], typer.Option('-w', '--workload', help='Show detailed information for a specific workload.')
    ] = None,
):
    """
    List available workloads and their configurations.
    """
    app_ctx: AppContext = ctx.obj
    cluster_gpu_type = app_ctx.cluster_config.get('launcher', {}).get('gpu_type')

    # Always use print_avail_workloads, with workload_filter if specified
    result = print_avail_workloads(
        app_ctx.workloads,
        app_ctx.cluster_config,
        cluster_gpu_type=cluster_gpu_type,
        verbose=True,
        workload_filter=workload,
    )

    if workload and not result:
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)


@app.command()
def single(
    ctx: typer.Context,
    workload: Annotated[
        str, typer.Option('-w', '--workload', help='Name of the workload (e.g., "pretraining_nemotron").')
    ],
    model_size: Annotated[str, typer.Option('-s', '--model-size', help='Size of the model (e.g., 7b, 13b).')],
    dtype: Annotated[str, typer.Option('--dtype', help='Data type (e.g., fp16, bf16).')],
    scale: Annotated[str, typer.Option('--scale', help='Scale parameter indicating the number of GPUs.')],
    profile: Annotated[bool, typer.Option('-p', '--profile', help='Enable Profiling for job.')] = False,
    dryrun: Annotated[
        bool, typer.Option('-d', '--dryrun', help='List the job to be submitted without actually submitting it.')
    ] = False,
    force: Annotated[
        bool, typer.Option('-f', '--force', help='Skip workload validation (deprecated, validation always runs).')
    ] = False,
):
    """
    (DEPRECATED) Submit a single job. Use 'llmb-run submit' instead.
    """
    logger.warning("⚠️  'single' command is deprecated. Please use 'llmb-run submit' instead.")
    logger.warning("   Use:")
    logger.warning(f"     ↳ llmb-run submit -w {workload} -s {model_size} --dtype {dtype} --scale {scale}")

    app_ctx: AppContext = ctx.obj

    # Note: 'force' is ignored in this redirect as _submit_impl enforces bulk validation.
    # However, validate_bulk_tasks validates against metadata, so it behaves similarly to standard validation.

    request = TaskGenerationRequest(
        workloads=app_ctx.workloads,
        cluster_config=app_ctx.cluster_config,
        workload=workload,
        model_size=model_size,
        dtype=dtype,
        scale=scale,  # Passed as string, TaskGenerationRequest handles parsing
        profile=profile,
    )

    _submit_impl(ctx, request, dryrun, mode_name="single")


@app.command()
def bulk(
    ctx: typer.Context,
    input_file: Annotated[
        str, typer.Argument(help='Path to the workload specification file (simple .txt or advanced .yaml).')
    ],
    dryrun: Annotated[
        bool, typer.Option('-d', '--dryrun', help='List all jobs to be submitted without actually submitting them.')
    ] = False,
):
    """
    (DEPRECATED) Submit multiple jobs from a specification file. Use 'llmb-run submit -f' instead.
    """
    logger.warning("⚠️  'bulk' command is deprecated. Please use 'llmb-run submit -f <file>' instead.")
    logger.warning("   Use:")
    logger.warning(f"     ↳ llmb-run submit -f {input_file}{' --dry-run' if dryrun else ''}")

    app_ctx: AppContext = ctx.obj

    request = TaskGenerationRequest(
        workloads=app_ctx.workloads,
        cluster_config=app_ctx.cluster_config,
        file_path=input_file,
    )

    _submit_impl(ctx, request, dryrun, mode_name="bulk")


@app.command()
def submit_all(
    ctx: typer.Context,
    max_scale: Annotated[
        Optional[int], typer.Option('--max-scale', help='Maximum scale (number of GPUs) to test up to.')
    ] = None,
    min_scale: Annotated[
        bool,
        typer.Option(
            '--min-scale', help='When set, only run the minimum scale per the metadata for all installed workloads.'
        ),
    ] = False,
    scales: Annotated[
        Optional[str],
        typer.Option(
            '--scales',
            help='Comma-separated list of specific scales to run (e.g., "8,16,32" or "16"). Mutually exclusive with --min-scale and --max-scale.',
        ),
    ] = None,
    dtype: Annotated[
        Optional[str],
        typer.Option(
            '--dtype',
            help='Comma separated list of dtypes to run. If unset, run all available dtypes per metadata for a workload.',
        ),
    ] = None,
    workloads: Annotated[
        Optional[str],
        typer.Option(
            '-w',
            '--workloads',
            help='Comma separated list of workloads to run. Reduces scope to only the specified workloads.',
        ),
    ] = None,
    repeats: Annotated[int, typer.Option('--repeats', help='Number of repeats for each test configuration.')] = 1,
    profile: Annotated[bool, typer.Option('-p', '--profile', help='Enable profiling for all jobs.')] = False,
    dryrun: Annotated[
        bool, typer.Option('-d', '--dryrun', help='List all jobs to be submitted without actually submitting them.')
    ] = False,
    nice: Annotated[
        Optional[int], typer.Option('--nice', help='Lower the priority of the job using Slurm --nice feature.')
    ] = None,
):
    """
    (DEPRECATED) Submit jobs for all installed recipes. Use 'llmb-run submit' instead.
    """
    logger.warning("⚠️  'submit-all' command is deprecated. Please use 'llmb-run submit' instead.")
    logger.warning("   Use:")
    submit_all_cmd = "llmb-run submit"
    if workloads:
        submit_all_cmd += f" -w {workloads}"
    if dtype:
        submit_all_cmd += f" --dtype {dtype}"
    if scales:
        submit_all_cmd += f" --scale {scales}"
    if max_scale is not None:
        submit_all_cmd += f" --max-scale {max_scale}"
    if min_scale:
        submit_all_cmd += " --min-scale"
    if repeats != 1:
        submit_all_cmd += f" -r {repeats}"
    if profile:
        submit_all_cmd += " -p"
    if nice is not None:
        submit_all_cmd += f" --nice {nice}"
    if dryrun:
        submit_all_cmd += " --dry-run"
    logger.warning(f"     ↳ {submit_all_cmd}")

    app_ctx: AppContext = ctx.obj

    extra_slurm_params = {}
    if nice is not None:
        extra_slurm_params['nice'] = nice

    request = TaskGenerationRequest(
        workloads=app_ctx.workloads,
        cluster_config=app_ctx.cluster_config,
        workload=workloads,
        dtype=dtype,
        scale=scales,
        max_scale=max_scale,
        min_scale=min_scale,
        repeats=repeats,
        profile=profile,
        extra_slurm_params=extra_slurm_params,
    )

    _submit_impl(ctx, request, dryrun, mode_name="submit-all")


@app.command()
def exemplar(
    ctx: typer.Context,
    dry_run: Annotated[
        bool,
        typer.Option('--dry-run', help='Print jobs without submitting them.'),
    ] = False,
    repeats: Annotated[
        Optional[int],
        typer.Option(
            '-r',
            '--repeats',
            help='Number of repeats for each test configuration. If not provided, uses value from exemplar.yaml config.repeats (default: 3).',
        ),
    ] = None,
):
    """
    Submit exemplar certification jobs from $LLMB_INSTALL/llmb_repo/exemplar.yaml.

    Runs workloads listed in exemplar.yaml for your cluster's GPU type.
    All workloads must be installed.

    Defaults: scale=512, profile=true, repeats=3 (override with -r)
    """
    app_ctx: AppContext = ctx.obj

    try:
        # Generate exemplar tasks (includes preflight checks and strict install gating)
        # Pass CLI repeats (None if not provided); generate_exemplar_tasks will use YAML value if None
        task_list = generate_exemplar_tasks(app_ctx.workloads, app_ctx.cluster_config, repeats=repeats)

        if not task_list:
            logger.error("No exemplar tasks generated. Check your configuration and installed workloads.")
            raise typer.Exit(code=EXIT_VALIDATION_ERROR)

        # Validate tasks
        validated_tasks, error_summary = validate_bulk_tasks(task_list, app_ctx.workloads, app_ctx.cluster_config)

        # Report results
        report_validation_results(
            validated_tasks, error_summary, task_list, app_ctx.cluster_config, mode_name="exemplar"
        )

        # Print the concrete jobs we're about to submit
        logger.info(f"Exemplar Certification Jobs ({len(validated_tasks)}):")
        for task in validated_tasks:
            logger.info(format_task_output(task, prefix="  - "))

        if dry_run:
            logger.info("Dry run enabled. Jobs will not be submitted.")
        else:
            run_tests(app_ctx.cluster_config, validated_tasks, app_ctx.workloads)

    except ValidationError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Exemplar submission error: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e


def cli():
    """Main entry point for the llmb-run CLI."""
    app()


if __name__ == '__main__':
    cli()
