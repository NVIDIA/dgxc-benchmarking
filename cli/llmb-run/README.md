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
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256
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
- `HF_TOKEN`: Hugging Face token (required for some models)
- Common settings include `RUN_CONF_*` settings

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

llmb-run's primary interface is the `submit` command, which handles all job submission modes. The `list` command is also available for discovery.

### CLI Structure (Global vs Command Options)

`llmb-run` has **global options** that must appear **before** the command name, and **command options** that appear after the command name.

```bash
llmb-run [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]
```

Tip: use `llmb-run -h` to see global options, and `llmb-run <command> -h` (e.g. `llmb-run submit -h`) to see command-specific options.

**Global options (apply to all commands):**
- `-v, --verbose`: Enable verbose output including debug information.

**Examples:**
```bash
# Correct: global option BEFORE the command
llmb-run -v submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256

# Incorrect: global option AFTER the command (this will not work)
llmb-run submit -v -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256
```

### Submit Command

The `submit` command is a unified interface for all job submissions. It supports three main workflows:

#### Choose a Submit Workflow

Pick the workflow that matches how you want to run:

- **Explicit (single job)**: You provide `--workload`, `--model_size`, `--dtype`, and `--scale`.
  - Pattern: `llmb-run submit -w <workload> -s <model_size> --dtype <dtype> --scale <scale>`
- **Auto-discovery (submit all / many)**: You provide discovery constraints and llmb-run generates jobs from installed workload metadata.
  - Pattern: `llmb-run submit --max-scale <num_gpus>`
  - Example: `llmb-run submit --max-scale 512` (submits eligible installed workloads up to 512 GPUs; see the section below for additional limiting flags)
- **File-based (batch; special cases)**: You provide an input file and llmb-run submits the jobs listed in it.
  - Pattern: `llmb-run submit -f <file_path>`

#### 1. Single Job Submission (Explicit)
Submit a single workload with specific parameters.

```bash
llmb-run submit -w <workload> -s <model_size> --dtype <dtype> --scale <scale>
```

**Required Flags:**
- `-w, --workload`: Name of the workload (e.g., `pretrain_llama3.1`)
- `-s, --model_size`: Model size (e.g., `405b`, `70b`).
- `--dtype`: Data type (e.g., `fp8`, `bf16`).
- `--scale`: Number of GPUs. Accepts a single value or a comma-separated list.

**Examples:**
```bash
# Run a single configuration
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256

# Run multiple scales for the same workload
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128,256,512
```

#### 2. File-Based Submission (Batch)
Submit multiple jobs defined in a file. This replaces the old `bulk` command.

```bash
llmb-run submit -f <file_path>
```

**Supported Formats:**
- **Simple (.txt)**: For basic configurations.
- **Advanced (.yaml)**: For complex configurations with overrides and environment variables.

See [Bulk_Examples.md](Bulk_Examples.md) for detailed file format specifications and examples.

**Example:**
```bash
llmb-run submit -f my_experiment.yaml
```

#### 3. Auto-Discovery (Submit All)
Automatically discover and submit jobs for installed workloads based on metadata. This replaces the old `submit-all` command.

```bash
llmb-run submit --max-scale <num_gpus>
```

**Flags:**
- `--max-scale`: Run all workloads up to this scale.
- `--min-scale`: Run only the minimum supported scale for each workload.
- `--exact-scales`: Only use scales explicitly listed in workload metadata (no power-of-2 expansion beyond metadata max).
- `-w, --workload`: Limit discovery to specific workloads (comma-separated).
- `--scale`: specific scales to run (comma-separated).

**Examples:**
```bash
# Run all installed workloads up to 512 GPUs
llmb-run submit --max-scale 512

# Run up to 512 GPUs but only at metadata-supported scales (avoid scale expansion)
llmb-run submit --max-scale 512 --exact-scales

# Run specific scales for all workloads
llmb-run submit --scale 128,256
```

#### Submit Options (All Submit Modes)
These flags apply to all `llmb-run submit` modes (explicit, file-based, and auto-discovery):
- `-r, --repeats <N>`: Repeat each job N times (default: 1).
- `-p, --profile`: Enable profiling for all submitted jobs.
- `--dry-run`: Print the jobs that would be submitted without running them.

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

### Exemplar Command (Cloud Certification)

The exemplar command runs the cloud certification workload suite.

#### Basic Usage
```bash
llmb-run exemplar
```

#### Options
- `--dry-run`: Preview all jobs without submitting
- `-r, --repeats INTEGER`: Number of times to run each job (default: 3).
- Profiling: Always enabled for the exemplar suite (no flag required).


#### Behavior
- Runs all eligible `pretrain` workloads at **scale 512**, with profiling enabled.
- Eligibility:
  - Workload type is `pretrain`.
  - The workload supports the cluster's GPU type (from `cluster_config.yaml`).
  - The per‑dtype configuration explicitly lists `scale: 512` (implicit ranges are not used).
  - The workload is listed under `workloads.installed` in `cluster_config.yaml`.
- Enforces strict validation (install gating): if any workload that meets eligibility is not installed, the command fails.
- Runs 3 profiled repetitions per job by default (required for certification). You can override the repeat count for debugging via `-r/--repeats`.


#### Troubleshooting Missing Workloads
If `llmb-run exemplar` fails due to missing workloads, do **not** use `llmb-install express`.
Instead, verify your installed workloads and add missing ones:

```bash
cd $LLMB_INSTALL
llmb-install
# Select the missing workloads from the menu
```

### Job Configuration Files

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
  model_name: "llama3.1"               # Model name
  model_size: "405b"                   # Model size
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
    - "nvcr.io#nvidia/nemo:25.11.01"

job_config:
  profile_enabled: true                # Whether profiling was enabled
  env_overrides:                       # Environment variable overrides
    DEBUG: "true"
  model_overrides:                     # Model parameter overrides
    seq_len: 8192
```

See [example_llmb_config.yaml](example_llmb_config.yaml) for a complete example.

### Deprecated Commands

The following commands are deprecated and will be removed in a future release. Please migrate to `llmb-run submit`.

- `single`: Replaced by `llmb-run submit`
- `bulk`: Replaced by `llmb-run submit -f <file>`
- `submit-all`: Replaced by `llmb-run submit` (with discovery flags like `--max-scale`)

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
   - Re-run with verbose output to see detailed error messages, e.g. `llmb-run -v submit ...`

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
./llmb-run submit --help
```
### Option 4: Python Module
```bash
# Run as a Python module (must be in directory with cluster_config.yaml)
llmb-run submit --help
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

## Exit Codes

`llmb-run` uses the following exit codes for automation support:

- **0**: Success. The operation completed successfully (e.g., jobs submitted, list displayed).
- **1**: Validation Error. Invalid arguments, configuration errors (missing `cluster_config.yaml`), or validation failures. These are issues that typically require user intervention to fix.
- **2**: System Error. Unexpected failures during job submission, SLURM environment issues, or other infrastructure-related failures beyond immediate user control.
