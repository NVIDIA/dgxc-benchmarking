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
import logging
import os
import sys

from llmb_run.config_manager import get_cluster_config
from llmb_run.job_launcher import run_tests
from llmb_run.task_manager import (
    WorkloadTask,
    flatten_yaml_tasks,
    gen_tasks,
    generate_submit_all_tasks,
    get_tasks_wrapper,
    print_tasks,
)
from llmb_run.workload_validator import (
    format_validation_error,
    get_workloads,
    print_avail_workloads,
    show_workload_details,
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='llmb-run: Tool for launching multiple or single LLM benchmarking workloads.'
    )

    subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode of operation')

    # List Mode Subparser
    list_parser = subparsers.add_parser('list', help='List available workloads and their configurations.')
    list_parser.add_argument('-w', '--workload', type=str, help='Show detailed information for a specific workload.')

    # Bulk Mode Subparser
    bulk_parser = subparsers.add_parser('bulk', help='Submit multiple jobs from a specification file.')
    bulk_parser.add_argument(
        '-d', '--dryrun', action='store_true', help='List all jobs to be submitted without actually submitting them.'
    )
    bulk_parser.add_argument(
        '-v', '--verbose', action='store_true', help='Enable verbose output including debug information.'
    )
    bulk_parser.add_argument(
        'input_file', type=str, help='Path to the workload specification file (simple .txt or advanced .yaml).'
    )

    # Submit All Subparser
    submit_all_parser = subparsers.add_parser('submit-all', help='Submit jobs for all installed recipes.')
    submit_all_parser.add_argument('--max-scale', type=int, help='Maximum scale (number of GPUs) to test up to.')
    submit_all_parser.add_argument(
        '--min-scale',
        action='store_true',
        help='When set, only run the minimum scale per the metadata for all installed workloads.',
    )
    submit_all_parser.add_argument(
        '--scales',
        type=str,
        help='Comma-separated list of specific scales to run (e.g., "8,16,32" or "16"). Mutually exclusive with --min-scale and --max-scale.',
    )
    submit_all_parser.add_argument(
        '--dtype',
        type=str,
        help='Comma separated list of dtypes to run. If unset, run all available dtypes per metadata for a workload.',
    )
    submit_all_parser.add_argument(
        '--workloads',
        '-w',
        type=str,
        help='Comma separated list of workloads to run. Reduces scope to only the specified workloads.',
    )
    submit_all_parser.add_argument(
        '--repeats', type=int, default=1, help='Number of repeats for each test configuration (default: 1).'
    )
    submit_all_parser.add_argument('-p', '--profile', action='store_true', help='Enable profiling for all jobs.')
    submit_all_parser.add_argument(
        '-d', '--dryrun', action='store_true', help='List all jobs to be submitted without actually submitting them.'
    )
    submit_all_parser.add_argument(
        '-v', '--verbose', action='store_true', help='Enable verbose output including debug information.'
    )

    # Single Job Subparser
    single_parser = subparsers.add_parser('single', help='Submit a single job with specified parameters.')
    single_parser.add_argument(
        '-w', '--workload', type=str, required=True, help='Name of the workload (e.g., "pretraining_nemotron").'
    )
    single_parser.add_argument('-s', '--model_size', type=str, required=True, help='Size of the model (e.g., 7b, 13b).')
    single_parser.add_argument('--dtype', type=str, required=True, help='Data type (e.g., fp16, bf16).')
    single_parser.add_argument(
        '--scale', type=int, required=True, help='Scale parameter indicating the number of GPUs.'
    )
    single_parser.add_argument('-p', '--profile', action='store_true', help='Enable Profiling for job.')
    single_parser.add_argument(
        '-d', '--dryrun', action='store_true', help='List the job to be submitted without actually submitting it.'
    )
    single_parser.add_argument(
        '-v', '--verbose', action='store_true', help='Enable verbose output including debug information.'
    )
    single_parser.add_argument(
        '-f', '--force', action='store_true', help='Skip workload validation (for debugging purposes).'
    )

    return parser.parse_args()


def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)


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
            logger.error(f"{prefix}: {error_lines[0].replace('Error: ', '')}")
            for line in error_lines[1:]:
                logger.error(f"     {line}")

        if not validated_tasks:
            logger.error(f"❌ No valid tasks found. Aborting {mode_name} submission.")
            raise SystemExit(1)
        else:
            logger.warning(f"⚠️  Proceeding with {len(validated_tasks)} valid tasks out of {len(task_list)} total.")
    else:
        logger.debug(f"✅ All {len(task_list)} tasks validated successfully.")


def main():
    """Main entry point for the llmb-run CLI."""
    if 'SLURM_JOB_ID' in os.environ:
        logger.error(
            "🚫: `llmb-run` does not currently support running within a SLURM allocation. Please run this script directly from a login node outside of a SLURM job."
        )
        raise SystemExit(1)
    args = parse_arguments()
    # Preserve compatibility: not all subcommands define --verbose
    setup_logging(getattr(args, 'verbose', False))

    try:
        cluster_config = get_cluster_config()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit(1) from e

    # Available workloads
    try:
        workloads = get_workloads(cluster_config)
    except Exception as e:
        logger.error(f"Failed to load workloads: {e}")
        raise SystemExit(1) from e

    if args.mode == 'bulk':
        try:
            # Use the wrapper to auto-select the parser based on file extension.
            tasks_parsed = get_tasks_wrapper(workloads, args.input_file, cluster_config)
            if args.input_file.endswith(('.yaml', '.yml')):
                task_list = flatten_yaml_tasks(tasks_parsed)
            else:
                task_list = gen_tasks(tasks_parsed)

            # Validate all tasks and get summary
            validated_tasks, error_summary = validate_bulk_tasks(task_list, workloads, cluster_config)

            # Report validation results
            report_validation_results(validated_tasks, error_summary, task_list, cluster_config, "bulk")

            if args.dryrun:
                print_tasks(validated_tasks)
                logger.info("Dry run enabled. Jobs will not be submitted.")
            else:
                run_tests(cluster_config, validated_tasks, workloads)
        except Exception as e:
            logger.error(f"Bulk mode error: {e}")
            raise SystemExit(1) from e

    elif args.mode == 'single':
        # Single Job Mode
        workload_input = args.workload
        model_size = args.model_size
        dtype = args.dtype
        scale = args.scale

        # Get cluster GPU type for validation
        cluster_gpu_type = cluster_config.get('launcher', {}).get('gpu_type')

        # Validate workload, model_size, dtype, and scale against metadata (unless --force is used)
        if not args.force:
            is_valid, error_type, error_msg, suggestions = validate_workload_with_details(
                workloads,
                workload_input,
                model_size,
                dtype=dtype,
                scale=scale,
                cluster_gpu_type=cluster_gpu_type,
                cluster_config=cluster_config,
            )
            if not is_valid:
                user_error = format_validation_error(
                    workload_input, model_size, dtype, scale, cluster_gpu_type, error_type, error_msg, suggestions
                )
                logger.error(user_error)
                raise SystemExit(1)
        else:
            logger.warning("Workload validation skipped due to --force flag. Use with caution.")

        # Create a single WorkloadTask
        single_task = WorkloadTask(
            workload_key=workload_input, model_size=model_size, dtype=dtype, scale=scale, profile=args.profile
        )

        if args.dryrun:
            print_tasks([single_task])
            logger.info("Dry run enabled. Job will not be submitted.")
        else:
            try:
                run_tests(cluster_config, [single_task], workloads)
            except Exception as e:
                logger.error(f"Single job launch error: {e}")
                raise SystemExit(1) from e

    elif args.mode == 'submit-all':
        # Submit All Mode
        try:
            # Validate mutual exclusivity and required arguments
            if args.scales is not None:
                # --scales is mutually exclusive with --max-scale and --min-scale
                if args.max_scale or args.min_scale:
                    logger.error(
                        "--scales is mutually exclusive with --max-scale and --min-scale. Please specify only --scales or the other options."
                    )
                    raise SystemExit(1)
            else:
                # Original validation: either max-scale or min-scale (or both) must be provided when not using --scales
                if not args.max_scale and not args.min_scale:
                    logger.error(
                        "Either --max-scale or --min-scale (or both) must be provided when not using --scales."
                    )
                    raise SystemExit(1)

            # Parse filtering options
            min_scale = args.min_scale
            max_scale = args.max_scale if args.max_scale else None  # None means no max limit when using min-scale only

            # Parse specific scales if provided
            specific_scales = None
            if args.scales:
                try:
                    specific_scales = [int(s.strip()) for s in args.scales.split(',')]
                    if not specific_scales:
                        logger.error("--scales cannot be empty.")
                        raise SystemExit(1)
                    # Validate that all scales are positive integers
                    if any(scale <= 0 for scale in specific_scales):
                        logger.error("All scales must be positive integers.")
                        raise SystemExit(1) from None
                except ValueError as e:
                    logger.error("--scales must be a comma-separated list of integers (e.g., '8,16,32' or '16').")
                    raise SystemExit(1) from e

            dtype_filter = [d.strip() for d in args.dtype.split(',')] if args.dtype else None
            workload_filter = [w.strip() for w in args.workloads.split(',')] if args.workloads else None

            # Generate tasks for all installed workloads
            task_list = generate_submit_all_tasks(
                workloads,
                cluster_config,
                max_scale,
                args.repeats,
                args.profile,
                min_scale=min_scale,
                dtype_filter=dtype_filter,
                workload_filter=workload_filter,
                specific_scales=specific_scales,
            )

            if not task_list:
                logger.error("No tasks generated. Check that workloads are installed and compatible with your cluster.")
                raise SystemExit(1)

            # Validate all tasks using existing bulk validation logic
            validated_tasks, error_summary = validate_bulk_tasks(task_list, workloads, cluster_config)

            # Report validation results
            report_validation_results(validated_tasks, error_summary, task_list, cluster_config, "submit-all")

            if args.dryrun:
                print_tasks(validated_tasks)
                logger.info("Dry run enabled. Jobs will not be submitted.")
            else:
                run_tests(cluster_config, validated_tasks, workloads)
        except Exception as e:
            logger.error(f"Submit-all mode error: {e}")
            raise SystemExit(1) from e

    elif args.mode == 'list':
        # List Mode
        cluster_gpu_type = cluster_config.get('launcher', {}).get('gpu_type')
        workload = args.workload
        if workload:
            show_workload_details(workloads, workload, cluster_gpu_type, cluster_config)
        else:
            print_avail_workloads(workloads, cluster_gpu_type, True, cluster_config)


if __name__ == '__main__':
    main()
