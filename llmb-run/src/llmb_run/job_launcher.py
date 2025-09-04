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

"""Job launching functionality for different launcher types."""

import importlib
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime

import yaml
from rich.console import Console

from llmb_run.config_manager import get_slurm_env_vars, get_workload_config
from llmb_run.constants import (
    GPU_TYPE_TO_NUM_GPUS,
    NEMO_MODEL_PARAMS,
    SLURM_OUTPUT_PATTERN,
)
from llmb_run.slurm_utils import SlurmJob, get_cluster_name
from llmb_run.task_manager import format_task_output

logger = logging.getLogger('llmb_run.job_launcher')
console = Console()


# internal mode
_internal_modules_available = False
try:
    internal_module = importlib.import_module("llmb_run.internal")
    _internal_modules_available = True
except ModuleNotFoundError:
    # Internal extensions unavailable – safe to ignore.
    pass

# Pre-compiled regex patterns for filtering Nemo2 status instructions
NEMO2_PYTHON_API_PATTERN = re.compile(
    r'# The experiment was run with the following tasks.*?'
    r'experiment\.cancel\([^)]*\)[^\n]*',
    re.DOTALL
)

NEMO2_CLI_PATTERN = re.compile(
    r'# You can inspect this experiment at a later point in time using the CLI as well:.*?'
    r'nemo experiment cancel [^\n]*',
    re.DOTALL
)

NEMO2_CLEANUP_NEWLINES_PATTERN = re.compile(r'\n\s*\n\s*\n+')


def filter_nemo2_status_instructions(output: str) -> str:
    """
    Filter out Nemo2 experiment status instructions from launcher output.
    
    Removes the sections that tell users how to check job status using
    experiment.status(), experiment.logs(), etc. and 'nemo experiment' CLI commands.
    Each section is filtered independently to handle ordering changes.
    
    Args:
        output: The raw output from the Nemo2 launcher
        
    Returns:
        Filtered output with status instructions removed
    """
    filtered_output = output
    
    # Remove Python API section using pre-compiled pattern
    filtered_output = NEMO2_PYTHON_API_PATTERN.sub('', filtered_output)
    
    # Remove CLI section using pre-compiled pattern
    filtered_output = NEMO2_CLI_PATTERN.sub('', filtered_output)
    
    # Clean up multiple consecutive newlines using pre-compiled pattern
    filtered_output = NEMO2_CLEANUP_NEWLINES_PATTERN.sub('\n\n', filtered_output)
    
    return filtered_output.strip()


def parse_nemo2_output(output: str) -> tuple[str, str, str]:
    """Parse Nemo launcher output to extract job ID and local directory.
    
    Args:
        output: Complete output from the Nemo launcher
        
    Returns:
        tuple: (job_id, local_directory, error_message) 
               job_id and local_directory are None if not found or error occurred
               error_message is None if no error, otherwise contains error description
    """
    job_id = None
    local_dir = None
    error_msg = None
    
    # Find all Task headers to validate we have the expected structure
    task_headers = re.findall(r'^Task (\d+):', output, re.MULTILINE)
    
    if not task_headers:
        error_msg = "No Task headers found in output"
        return None, None, error_msg
    
    if len(task_headers) > 1:
        error_msg = f"Multiple tasks found ({len(task_headers)} tasks: {task_headers}). Expected only Task 0."
        return None, None, error_msg
      
    # Extract the Task 0 section for parsing
    task_0_match = re.search(r'Task 0:.*?(?=\n\n|\nTask \d+:|\n#|$)', output, re.DOTALL)
    if not task_0_match:
        error_msg = "Could not extract Task 0 section from output"
        return None, None, error_msg
    
    task_0_section = task_0_match.group(0)
    
    # Look for job ID pattern within Task 0 section: "- Job id: 3530909"
    job_id_match = re.search(r'- Job id:\s+(\d+)', task_0_section)
    if job_id_match:
        job_id = job_id_match.group(1)
    
    # Look for local directory pattern within Task 0 section: "- Local Directory: /path/to/directory"
    local_dir_match = re.search(r'- Local Directory:\s+(.+)', task_0_section)
    if local_dir_match:
        local_dir = local_dir_match.group(1).strip()
    
    # Validate that if we found a job_id, we should also have a local_dir
    if job_id and not local_dir:
        error_msg = f"Found job ID {job_id} but no local directory - this should not happen"
        return None, None, error_msg
    
    return job_id, local_dir, None

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

class JobLauncher(ABC):
    """Abstract base class for job launchers."""
    
    def __init__(self, config, workloads):
        self.config = config
        self.workloads = workloads
    
    @abstractmethod
    def launch(self, task) -> SlurmJob:
        """Launch a single task.
        
        Args:
            task: WorkloadTask to launch
            
        Returns:
            SlurmJob: SlurmJob object containing, job_workdir may be None in some cases.
        """
        pass
    
    def get_gpu_type(self, task):
        """Determine GPU type for a task from cluster config only."""
        return self.config.get('launcher', {}).get('gpu_type')


    def model_optimizations(self, model_overrides):
        """Convert parameter overrides into OPTIMIZATION_NAME, CODE for recipe."""
        optimizations = []
        for key, value in model_overrides.items():
            if key in NEMO_MODEL_PARAMS:
                optimizations.append(f"{NEMO_MODEL_PARAMS[key]}={value}")
            else:
                logger.warning(f"Unknown model parameter: {key}={value}")
        
        return ' '.join(optimizations)


class SbatchLauncher(JobLauncher):
    """Launcher for SLURM sbatch jobs.
    
        TODO: Needs work if used for new sbatch jobs."""
    
    def launch(self, task):
        """Launch a task using the legacy sbatch method."""
        job = {
            "workload": task.workload_key,
            "LLMB_INSTALL": self.config['launcher']['llmb_install'],
            "dir": self.workloads[task.workload_key]["dir"],
            "MODEL_SIZE": task.model_size,
            "DTYPE": task.dtype
        }
        
        # We need to pass node counts and ntasks-per-node to sbatch jobs.
        if os.environ.get('GPUS_PER_NODE'):
            gpus_per_node = int(os.environ.get('GPUS_PER_NODE'))
        else:
            gpu_type = self.get_gpu_type(task)
            if gpu_type not in GPU_TYPE_TO_NUM_GPUS:
                raise ValueError(f"Invalid GPU type specified: '{gpu_type}'. Valid types in 'llmb-run' modules are: {', '.join(GPU_TYPE_TO_NUM_GPUS.keys())}")
            gpus_per_node = GPU_TYPE_TO_NUM_GPUS[gpu_type]
        # Ensure we have at least one node for jobs that use less than all GPUs on a node.
        num_nodes = (task.scale + gpus_per_node - 1) // gpus_per_node
        num_nodes = str(num_nodes)
        # If the job uses less than all GPUs on a node, set ntasks-per-node to the number of GPUs used.
        ntasks_per_node = str(gpus_per_node) if task.scale >= gpus_per_node else str(task.scale)
        
        cmd = [
            "sbatch",
            "--parsable",
            f"--output={self.config['cwd']}/{SLURM_OUTPUT_PATTERN}",
            "-J",
            f"{job['workload']}_{job['MODEL_SIZE']}_{job['DTYPE']}",
            "-N",
            num_nodes,
            f"--ntasks-per-node={ntasks_per_node}",
            "launch.sh"
        ]
        
        env = os.environ.copy()
        
        # Set basic job environment variables
        env["LLMB_INSTALL"] = job["LLMB_INSTALL"]
        env["MODEL_SIZE"] = job["MODEL_SIZE"]
        env["DTYPE"] = job["DTYPE"]
        env["JOB_TOTAL_GPUS"] = str(task.scale)
        env["GPU_TYPE"] = self.get_gpu_type(task)
                
        # Add SLURM environment variables
        slurm_env = get_slurm_env_vars(self.config)
        env.update(slurm_env)
        
        if task.profile:
            env['ENABLE_PROFILE'] = 'true'
        
        # Handle environment variables from config
        if 'environment' in self.config:
            env_vars = {k: str(v) for k, v in self.config['environment'].items()}
            env.update(env_vars)
        
        # Convert all task override values to strings
        task_env = {k: str(v) for k, v in task.env_overrides.items()}
        env.update(task_env)
        
        # Handle model parameter overrides
        if task.model_overrides:
            env['OPTIMIZATION_NAME'] = 'cc'  # Custom Config
            env['OPTIMIZATION_CODE'] = self.model_optimizations(task.model_overrides)
        
        try:
            logger.debug(f"Command: {cmd}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True,
                env=env,
                cwd=job['dir']
            )
            job_id = result.stdout.strip()
            logger.info(format_task_output(task, prefix="SUBMITTED: ", suffix=f"jobid={job_id}"))
            
            # Create llmb-config.yaml file in the current directory
            create_llmb_config(task, job_id, None, self.config, self.workloads)
            
            # TODO: This job directory is not correct.
            return SlurmJob(job_id=job_id, job_workdir=None)
        except subprocess.CalledProcessError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"error={e.stderr.strip()}"))
            return SlurmJob(job_id=None, job_workdir=None)


class Nemo2Launcher(JobLauncher):
    """Launcher for Nemo2 jobs with virtual environment support."""
    
    def launch(self, task):
        """Launch a task using the Nemo2 method with venv activation."""
        workload_config = get_workload_config(self.config, task.workload_key)
        venv_path = workload_config.get('venv_path')
        venv_type = workload_config.get('venv_type')
        
        if not venv_path:
            logger.error(f"venv_path not found in config for workload {task.workload_key}")
            return SlurmJob(job_id=None, job_workdir=None)
        
        if not venv_type:
            logger.error(f"venv_type not found in config for workload {task.workload_key}")
            return SlurmJob(job_id=None, job_workdir=None)
        
        try:
            # Get venv environment with the correct type
            env = get_venv_environment(venv_path, venv_type)
            
            # Set basic job environment variables
            env["LLMB_INSTALL"] = self.config['launcher']['llmb_install']
            env["MODEL_SIZE"] = task.model_size
            env["DTYPE"] = task.dtype
            env["JOB_TOTAL_GPUS"] = str(task.scale)
            env["GPU_TYPE"] = self.get_gpu_type(task)
            
            # Add SLURM environment variables
            slurm_env = get_slurm_env_vars(self.config)
            env.update(slurm_env)
            
            if task.profile:
                env['ENABLE_PROFILE'] = 'true'
            
            # Handle environment variables from config
            if 'environment' in self.config:
                env_vars = {k: str(v) for k, v in self.config['environment'].items()}
                env.update(env_vars)
            
            # Handle workload-specific environment variables
            workload_env = workload_config.get('environment', {})
            workload_env_str = {k: str(v) for k, v in workload_env.items()}
            env.update(workload_env_str)
            
            # Convert all task override values to strings
            task_env = {k: str(v) for k, v in task.env_overrides.items()}
            env.update(task_env)
            
            # Run launch.sh script in the workload directory
            workload_dir = self.workloads[task.workload_key]["dir"]
            launch_script = "launch.sh"
            
            logger.debug(f"Running {launch_script} in {workload_dir} with venv {venv_path} (type: {venv_type})")
            
            # Create spinner with task info
            spinner_text = f"Launching {task.workload_key} (model_size={task.model_size}, dtype={task.dtype}, scale={task.scale})"
            
            job_id = None
            local_dir = None
            parse_error = None
            
            with console.status(spinner_text, spinner="bouncingBall"):
                result = subprocess.run(
                    [f"./{launch_script}"],
                    env=env,
                    cwd=workload_dir,
                    capture_output=True,
                    text=True
                )
                
                # Parse output for job ID and local directory
                job_id, local_dir, parse_error = parse_nemo2_output(result.stdout)
            
            # Display the filtered output after spinner completes
            console.print("\n[bold green]Launch Output:[/bold green]")
            filtered_output = filter_nemo2_status_instructions(result.stdout)
            console.print(filtered_output)
            
            if result.returncode == 0:
                if parse_error:
                    console.print(f"\n[bold red]Parse Error:[/bold red] {parse_error}")
                    logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"output parsing failed: {parse_error}"))
                    return SlurmJob(job_id=None, job_workdir=None)
                else:
                    logger.info(format_task_output(task, prefix="LAUNCHED: ", suffix=f"jobid={job_id}"))
                    logger.info(f"JobID: {job_id}, Workdir: {local_dir}")
                    
                    # Create llmb-config.yaml file in the experiment directory
                    create_llmb_config(task, job_id, local_dir, self.config, self.workloads)
                    
                    return SlurmJob(job_id=job_id, job_workdir=local_dir)
            else:
                console.print("\n[bold red]Launch Error:[/bold red]")
                console.print(result.stderr)
                logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"launch failed with return code {result.returncode}"))
                return SlurmJob(job_id=None, job_workdir=None)
            
        except ValueError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"venv error: {e}"))
            return SlurmJob(job_id=None, job_workdir=None)
        except subprocess.CalledProcessError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"launch error: {e}"))
            return SlurmJob(job_id=None, job_workdir=None)
        except Exception as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"unexpected error: {e}"))
            return SlurmJob(job_id=None, job_workdir=None)


def create_launcher(launcher_type, config, workloads):
    """Factory function to create the appropriate launcher."""
    if launcher_type == 'sbatch':
        return SbatchLauncher(config, workloads)
    elif launcher_type == 'nemo':
        return Nemo2Launcher(config, workloads)
    else:
        raise ValueError(f"Unknown launcher type: {launcher_type}")


def run_tests(config, task_list, workloads):
    """Run tests using the appropriate launcher for each task."""
    launchers = {}
    
    for task in task_list:
        # Get launcher type from workload metadata
        metadata = workloads[task.workload_key]['metadata']
        launcher_type = metadata.get('run', {}).get('launcher_type')
        if not launcher_type:
            logger.error(f"Missing required 'run.launcher_type' field in metadata for workload {task.workload_key}")
            continue
        
        # Create launcher if not already created
        if launcher_type not in launchers:
            try:
                launchers[launcher_type] = create_launcher(launcher_type, config, workloads)
            except ValueError as e:
                logger.error(f"Failed to create launcher for {task.workload_key}: {e}")
                continue
        
        # Launch the task
        slurm_job = launchers[launcher_type].launch(task)

        # Run post-processing pipeline (resparse, upload, workload inspector)
        if slurm_job.job_id and slurm_job.job_workdir:
            try:
                pipeline = importlib.import_module("llmb_run.internal.post_processing_pipeline")
                
                results = pipeline.run_post_processing_pipeline(
                    config=config,
                    launcher_type=launcher_type,
                    slurm_job=slurm_job,
                    task=task,
                    workload_metadata=workloads[task.workload_key].get('metadata')
                )
                
                # Log any errors
                if results.has_errors:
                    for error in results.errors:
                        logger.warning(f"Post-processing pipeline error: {error}")
                        
            except ModuleNotFoundError:
                # Internal extensions unavailable – safe to ignore.
                pass
            except Exception as e:
                logger.warning(f"Post-processing pipeline failed: {e}")

def create_llmb_config(task, job_id, workdir, config, workloads):
    """Create llmb-config.yaml file in the experiment's folder.
    
    Args:
        task: WorkloadTask object containing job parameters
        job_id: Job ID returned from the launcher
        workdir: Working directory path (may be None for some launchers)
        config: Cluster configuration
        workloads: Workloads dictionary
        
    Returns:
        str: Path to the created config file, or None if creation failed
    """
    try:
        # Determine the config file location
        if workdir:
            # For Nemo2 launcher, use the workdir returned by the launcher
            config_dir = workdir
        else:
            # For other launchers, use the installed workload directory
            llmb_install = config['launcher']['llmb_install']
            workload_key = task.workload_key
            config_dir = os.path.join(llmb_install, 'workloads', workload_key)
        
        # Create the config file path
        config_file_path = os.path.join(config_dir, f'llmb-config_{job_id}.yaml')
        
        # Ensure the directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Get workload metadata 
        workload_info = workloads[task.workload_key]
        metadata = workload_info['metadata']
        
        # Build the configuration dictionary
        # calculate fw_version
        image_path = metadata.get('container', {}).get('images', [])
        fw_version = image_path[0].split(':')[-1] if image_path else 'unknown'

        # Cluster name mandatory for internal use
        cluster_name = config.get('launcher', {}).get('cluster_name')
        if not cluster_name:
            cluster_name = get_cluster_name()
            if cluster_name:
                logger.warning(
                    f"\n{'='*80}\n"
                    f"⚠️  WARNING: cluster_name not configured!\n"
                    f"   Using '{cluster_name}' detected from SLURM configuration.\n"
                    f"   To fix this, add the following to your cluster_config.yaml:\n"
                    f"   \n"
                    f"   launcher.cluster_name: {cluster_name}\n"
                    f"{'='*80}"
                )
            if _internal_modules_available and not cluster_name:
                logger.error("Cluster name not configured, and unable to auto-detect. Required for post processing jobs.")
                raise ValueError("Cluster name not configured, and unable to auto-detect. Required for post processing jobs.")

        llmb_config = {
            'job_info': {
                'job_id': job_id,
                'launch_time': datetime.now().isoformat(),
            },
            'workload_info': {
                'framework': metadata.get('general', {}).get('framework', 'unknown'),
                'gsw_version': metadata.get('general', {}).get('gsw_version', 'unknown'), 
                'fw_version': fw_version,
                'workload_type': workload_info.get('workload_type', ''),
                'synthetic_dataset': workload_info.get('synthetic_dataset', True),
            },
            'model_info': {
                'model_name': workload_info.get('workload', ''),
                'model_size': task.model_size,
                'dtype': task.dtype,
                'scale': task.scale,
                'gpu_type': config.get('launcher', {}).get('gpu_type', 'unknown')
            },
            'cluster_info': {
                'cluster_name': cluster_name,
                'gpus_per_node': config.get('slurm', {}).get('gpu_gres', ''),
                'llmb_install': config.get('launcher', {}).get('llmb_install', ''),
                'llmb_repo': config.get('launcher', {}).get('llmb_repo', ''),
                'slurm_account': config.get('slurm', {}).get('account', ''),
                'slurm_gpu_partition': config.get('slurm', {}).get('gpu_partition', '')
            },
            'container_info': {
                'images': image_path,
            },
            'job_config': {
                'profile_enabled': task.profile,
                'env_overrides': task.env_overrides,
                'model_overrides': task.model_overrides
            }
        }
        
        # Write the config file
        with open(config_file_path, 'w') as f:
            yaml.dump(llmb_config, f, default_flow_style=False, sort_keys=False)
        
        logger.debug(f"Created llmb-config.yaml at: {config_file_path}")
        return config_file_path
        
    except Exception as e:
        logger.warning(f"Failed to create llmb-config.yaml: {e}")
        return None 