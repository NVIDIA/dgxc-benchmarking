# llmb-run

## Overview

A lightweight tool for automating submission of single jobs and batches of workloads.

## Quick Start

### Installation

The recommended way to install llmb-run is using the automated installer script:

```bash
# Run the installer script
$LLMB_REPO/install.sh
```

This script will:
1. Install required dependencies
2. Install llmb-run as a Python package
3. Launch the interactive installer to:
   - Configure your SLURM cluster settings
   - Select GPU type (h100, gb200, etc.)
   - Choose and install workloads
   - Create your `cluster_config.yaml`

### First Steps

After installation completes, you must change to your installation directory before using llmb-run:

```bash
# Change to your installation directory
cd $LLMB_INSTALL

# Verify installation and list available workloads
llmb-run list

# Run your first job (example)
llmb-run single -w pretrain_nemotron4 -s 340b --dtype fp8 --scale 256
```

**Note**: llmb-run requires access to `cluster_config.yaml` which is located in your installation directory. Always run llmb-run commands from this directory.

### Alternative Installation Methods

If you need to install llmb-run without the automated installer, see [Alternative Installation Methods](#alternative-installation-methods) below.

## Configuration

The `cluster_config.yaml` file contains several main sections:

### launcher
Configuration for the launcher system:
- `llmb_repo`: Path to the LLM benchmarking collection repository
- `llmb_install`: Base installation directory for workloads and data
- `gpu_type`: GPU type for your cluster (`h100`, `gb200`, etc.)

### environment
Environment variables that will be appended to every job:
- Common settings include `HF_TOKEN` and `RUN_CONF_*` settings

### slurm
Slurm-specific configuration:
- `account`: Slurm account name
- `gpu_partition`: GPU partition name
- `gpu_gres`: Only set if GRES is required for your cluster
- `cpu_partition`: CPU partition name (optional)

### workloads
Workload configuration:
- `installed`: List of workloads installed on this cluster
- `config`: Workload-specific configuration (typically managed by installer)

**Note**: The script validates workloads against the `installed` list and GPU type compatibility. Only workloads that support your cluster's GPU type and are in the installed list will be available.

## Commands

llmb-run supports four main commands: `list`, `single`, `bulk`, and `submit-all`.

### List Command

The list command helps you discover available workloads and their configurations.

#### Basic Usage
```bash
llmb-run list
```

#### Options
- `-w, --workload <name>`: Show detailed information for a specific workload

#### Examples

1. List all installed workloads:
```bash
llmb-run list
```

2. Show details for a specific workload:
```bash
llmb-run list -w pretrain_llama3.1
```

### Single Job

The single job submission mode allows you to submit individual workload jobs with specific configurations.

#### Basic Usage
```bash
llmb-run single -w <workload> -s <model_size> --dtype <fp8/bf16> --scale <num_gpus>
```

#### Required Flags
- `-w, --workload`: Name of the workload to run (e.g., pretrain_llama3.1, pretrain_nemotron4)
- `-s, --model_size`: Size of the model (e.g., 405b, 340b, 314b)
- `--dtype`: Data type for the model (supported values: fp8, bf16)
- `--scale`: Number of GPUs to use for the job

#### Optional Flags
- `-p, --profile`: Enable profiling for the job. When enabled:
  - Sets `ENABLE_PROFILE=true`
  - Useful for performance analysis and debugging
- `-d, --dryrun`: Preview mode that lists the job parameters without submitting
- `-v, --verbose`: Enable verbose output including debug information


#### Examples

1. Basic job submission:
```bash
llmb-run single -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256
```

2. Job with profiling enabled:
```bash
# Enable profiling with GPU metrics
ENABLE_GPU_METRICS=true llmb-run single -w pretrain_nemotron4 -s 340b --dtype fp8 --scale 256 -p

# Enable profiling without GPU metrics (default)
llmb-run single -w pretrain_nemotron4 -s 340b --dtype fp8 --scale 256 -p
```

3. Preview job parameters (dry run):
```bash
llmb-run single -w pretrain_grok1 -s 314b --dtype fp8 --scale 256 -d
```

4. Verbose output with profiling:
```bash
llmb-run single -w pretrain_grok1 -s 314b --dtype bf16 --scale 128 -p -v
```

**Note**: The script validates the workload and model size against available configurations and your cluster's GPU type. If the combination is invalid, it will show suggestions and error out before submission.

### Bulk Submission

There are two methods for bulk submission: basic and advanced. The basic method is intended for quickly submitting a variety of workloads that require minimal customization. The advanced method uses YAML format and provides more flexibility for complex configurations.

Create a job script using one of the formats described below.

```bash
llmb-run bulk <job_script>
```
- `-d, --dryrun`: List all jobs to be submitted without actually submitting them
- `-v, --verbose`: Enable verbose output including debug information

It is recommended that you run with `-d` first to ensure your tasks match what you intended.

### Job Script Basic

For simple configurations, you can use a basic format:
```
<workloadname>_<model size>:
(<precision>, [<list of scales in num gpus>], repeats, enable_profiling)
([<precisions can be list>], ...)
```
`enable_profiling` is not a required parameter and is assumed False if missing.

Example:
```
pretrain_nemotron4_340b:
('bf16', [128, 256, 512], 3)
('fp8', [128], 1)
```
This will run Nemotron 340b bf16 at 128, 256 and 512 GPUs 3x each. And fp8 128 GPUs 1x.

You can also mix and match:
```
pretrain_nemotron4_340b:
(['bf16','fp8'], [128, 256, 512], 3)
('fp8', [1024], 1, True)
```
For more details and examples of the basic format, see the [Bulk_Examples.md](Bulk_Examples.md) file.

### Job Script Advanced (YAML)

The advanced YAML format provides more flexibility and control over job configurations. The structure is as follows:

```yaml
<workloadname>_<model_size>:
  defaults:
    env:
      ENV_VAR: "value"
  dtypes: ['fp8', 'bf16']  # Top level dtypes (optional)
  scales: [128, 256]       # Top level scales (optional)
  repeats: 3               # Top level repeats (optional)
  tasks:
    - dtypes: <precision>  # can also be a list ['fp8', 'bf16']
      scales: [128, 256]
      repeats: 3
      profile: true        # Example of a profiling task
      overrides:
        env:
          ENV_VAR: "different value"
```

#### Key Components:

- `defaults`: Global settings that apply to all tasks under a workload
  - `env`: Environment variables to be set for all jobs

- `tasks`: Individual job configurations
  - `dtypes`: Data types to use (fp8, bf16)
  - `scales`: Number of GPUs to use
  - `repeats`: Number of times to run each configuration
  - `profile`: Set to true to enable profiling for this task (default: false)
  - `overrides`: Task-specific overrides for `env`.

For comprehensive examples of both simple and complex configurations, see the [Bulk_Examples.md](Bulk_Examples.md) file.

### Submit All Jobs

The `submit-all` command automatically discovers all installed pretrain and finetune workloads and generates jobs based on their metadata, up to a specified maximum scale.

#### Basic Usage
```bash
llmb-run submit-all --max-scale <num_gpus>
```

#### Required Flags
- `--max-scale`: Maximum scale (number of GPUs) to test up to. The tool will generate jobs for all supported scales up to this limit, automatically extending with power-of-2 scales if not explicitly defined in metadata (unless `exact_scales: true` is set).

#### Optional Flags
- `--repeats <num>`: Number of repeats for each test configuration (default: 1).
- `-p, --profile`: Enable profiling for all generated jobs.
- `-d, --dryrun`: Preview mode that lists all jobs to be submitted without actually submitting them.
- `-v, --verbose`: Enable verbose output including debug information.

#### Examples

1. Preview all jobs up to 256 GPUs:
```bash
llmb-run submit-all --max-scale 256 --dryrun
```

2. Submit all jobs up to 512 GPUs with 3 repeats each:
```bash
llmb-run submit-all --max-scale 512 --repeats 3
```

3. Submit all jobs with profiling enabled, up to 1024 GPUs:
```bash
llmb-run submit-all --max-scale 1024 --profile
```

## Job Configuration Files

When you launch a job using `llmb-run`, a `llmb-config_<JOBID>.yaml` file is automatically created in the experiment's folder. This file contains comprehensive information about the job configuration and can be useful for:

- **Job tracking**: Keep a record of all job parameters and settings
- **Reproducibility**: Recreate the exact same job configuration later
- **Debugging**: Understand what parameters were used for a specific run
- **Analysis**: Extract job metadata for performance analysis

### Config File Location

- **Nemo2 launcher**: The config file is created in the experiment's working directory (returned by the launcher)
- **Sbatch launcher**: The config file is created in the current working directory

### Config File Structure

The `llmb-config_<JOBID>.yaml` file contains the following sections:

```yaml
job_info:
  job_id: "3530909"                    # SLURM job ID
  launch_time: "2025-01-15T10:30:45"  # ISO timestamp of job launch

workload_info:
  framework: "nemo2"                   # Framework used (nemo2, maxtext, etc.)
  gsw_version: "25.07"                 # GSW version
  fw_version: "25.04.00"               # Framework version from container image
  workload_type: "pretrain"            # Type of workload (pretrain, finetune, etc.)
  synthetic_dataset: true              # Whether synthetic dataset is used

model_info:
  model_name: "nemotron4"               # Model name
  model_size: "340b"                   # Model size
  dtype: "fp8"                         # Data type (fp8, bf16)
  scale: 256                           # Number of GPUs
  gpu_type: "h100"                     # GPU type

cluster_info:
  cluster_name: "cluster1"             # Cluster name
  gpus_per_node: "8"                   # GPUs per node configuration
  llmb_install: "/path/to/install"     # LLMB installation path
  llmb_repo: "/path/to/repo"           # Repository path
  slurm_account: "account_name"        # SLURM account
  slurm_gpu_partition: "partition"     # SLURM partition

container_info:
  images:                              # Container images used
    - "nvcr.io#nvidia/nemo:25.04.00"

job_config:
  profile_enabled: true                # Whether profiling was enabled
  env_overrides:                       # Environment variable overrides
    DEBUG: "true"
  model_overrides:                     # Model parameter overrides
    seq_len: 8192
```

See [example_llmb_config.yaml](example_llmb_config.yaml) for a complete example.

## Troubleshooting

### Common Issues and Solutions

1. **Invalid Workload/Model Size**
   ```
   ERROR: Invalid Workload / Model Size: workload_name_model_size
   ```
   - Ensure the workload and model size combination exists and is compatible with your GPU type
   - Use `llmb-run list` to see available workloads
   - Use `llmb-run list -w <workload_name>` for detailed workload information

2. **Workload Not Installed**
   ```
   ERROR: Workload 'workload_name' is not installed on this cluster.
   ```
   - Check your `cluster_config.yaml` file's `workloads.installed` list
   - Ensure the workload is properly installed and listed

3. **GPU Type Not Supported**
   ```
   ERROR: GPU type 'h100' not supported for workload 'workload_name'.
   ```
   - Check if the workload supports your cluster's GPU type
   - Use `llmb-run list -w <workload_name>` to see supported GPU types

4. **Missing Configuration**
   ```
   FileNotFoundError: cluster_config.yaml not found
   ```
   - Solution: Create a `cluster_config.yaml` file in your working directory
   - See the Configuration section for the required format

5. **Job Submission Fails**
   - Check your Slurm account and partition settings in `cluster_config.yaml`
   - If your system does not support GRES, make sure `SBATCH_GPUS_PER_NODE` is not in your environment section
   - Use `-v` flag for verbose output to see detailed error messages

## Alternative Installation Methods

These methods require additional setup and are recommended only for advanced users:

### Option 1: Install using uv (Recommended for Manual Install)

`uv` is a fast Python package manager that can install tools in isolated environments.

```bash
# Install from the project directory (assuming $LLMB_REPO is your repository root)
uv tool install $LLMB_REPO/cli/llmb-run

# Or from git
# uv tool install git+https://github.com/NVIDIA/dgxc-benchmarking#subdirectory=cli/llmb-run
```

### Option 2: Install as a Package (pip)
```bash
# Install from the project directory
cd llmb-run
pip install .

# Note: You must:
# 1. Create cluster_config.yaml manually (see Configuration section)
# 2. Always run llmb-run from the directory containing cluster_config.yaml
```

### Option 3: Direct Execution
```bash
# Make the script executable
chmod +x llmb-run

# Run directly (must be in directory with cluster_config.yaml)
./llmb-run --help
```
### Option 4: Python Module
```bash
# Run as a Python module (must be in directory with cluster_config.yaml)
llmb-run --help
```
**Note**: These alternative methods require you to:
1. Create your own `cluster_config.yaml`
2. Install workloads manually
3. Set up any required virtual environments
4. Download container images
5. Always run llmb-run from the directory containing cluster_config.yaml

For most users, we recommend using the automated installer script described in Quick Start.


## Environment Variables Reference

The following environment variables are recognized to control behavior:

| Variable | Purpose | Input |
|---|---|---|
| `LLMB_SKIP_PP` | Disable post-processing job submission | `1`, `true`, or `yes` to disable |

## Development

This project uses `uv` for dependency management and `tox` for multi-environment testing.

### Environment Setup

1. **Install uv**: [Follow official instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. **Sync environment**: Creates a virtualenv and installs dependencies from `uv.lock`.
   ```bash
   uv sync
   ```

### Managing Dependencies

- **Add a dependency**: `uv add <package>`
- **Add a dev dependency**: `uv add --dev <package>`
- **Update lockfile**: Run this after modifying `pyproject.toml` (including version bumps) or dependencies.
   ```bash
   uv lock
   ```

### Running Tests

- **Quick (Current Python)**:
  ```bash
  uv run pytest
  ```
- **Full Matrix (Multiple Python versions)**:
  ```bash
  # Requires tox and tox-uv
  uv tool install tox --with tox-uv
  tox
  ```
