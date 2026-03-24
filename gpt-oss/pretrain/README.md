# Overview

This recipe contains information and scripts to produce performance results for the GPT OSS 120B training workload. The scripts help perform environment setup and launch benchmark jobs. Configurations use weak scaling methodology (global batch size scales proportionally with GPU count).

## GB300

| GPT OSS Model Size |  GPUs  | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ------------------ | :----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 120b               | 64-512 |   BF16   |  4096  |   36   | False |  1  |  1  |  1  | 64  |  1  | GPUs/64 | NA  |  4  | GPUs\*20 |  5  | False |

## GB200

| GPT OSS Model Size |  GPUs  | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ------------------ | :----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 120b               | 64-512 |   BF16   |  4096  |   36   | False |  1  |  1  |  1  | 64  |  1  | GPUs/64 | NA  |  4  | GPUs\*20 |  5  | False |

## B300

| GPT OSS Model Size |  GPUs  | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP   | VP  | MBS |   GBS    | GA  |  CG   |
| ------------------ | :----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :----: | :-: | :-: | :------: | :-: | :---: |
| 120b               | 64-512 |   BF16   |  4096  |   36   | False |  1  |  1  |  1  |  8  |  1  | GPUs/8 | NA  |  4  | GPUs\*20 |  5  | False |

## B200

| GPT OSS Model Size |  GPUs  | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ------------------ | :----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 120b               | 64-512 |   BF16   |  4096  |   36   | False |  1  |  1  |  1  | 64  |  1  | GPUs/64 | NA  |  4  | GPUs\*20 |  5  | False |

## H100

| GPT OSS Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ------------------ | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 120b               | 64-1024 |   BF16   |  4096  |   36   | False |  1  |  4  |  1  |  8  |  1  | GPUs/32 | NA  |  1  | GPUs\*20 | 80  | False |

# Performance Measurement and Analysis

Performance for GPT OSS training is measured by the achieved GPU FLOPS via the `TFLOPS_per_GPU` metric, which indicates computational throughput efficiency. Additionally, training step timing (milliseconds per iteration) is captured and logged for every training step in the main training log file [see Output Locations](#output-locations).

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing_mbridge.sh` script to analyze iterations 35-44 and calculate mean and standard deviation for reliable performance metrics for both TFLOPS per GPU and timing measurements.

### Running the parse_train_timing_mbridge.sh script

To analyze training timing from your experiment results, run the script from the workload directory. In an installed environment, recipe files are available under `$LLMB_INSTALL/llmb_repo` (a copy created by the installer).

```bash
# Basic usage - parses results in the directory named 'experiments' in the current folder
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh

# Specify a different experiments directory
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh /path/to/experiments

# Output in CSV format
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh --format=csv

# Output in JSON format
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh --format=json

# Show full filenames instead of shortened versions
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh --full-names
```

Example output:

```shell
Elapsed Time (ms) and MODEL_TFLOPS/GPU Analysis (iterations 35-44)
================================================================================
Experiment                                                                                   Status Time Mean (ms) Time Std (ms) MODEL_TFLOPS_per_GPU Mean MODEL_TFLOPS_per_GPU Std
------------------------------------------------------------------------------------------ -------- ------------- ------------ ------------------- ------------------
pretrain_gpt_oss_120b_bf16_gpus64_tp1_pp1_cp1_vpNone_ep64_etp1_mbs4_gbs1280_1697683         Success      5197.940        7.013              428.11               0.59
```

To obtain throughput as a tokens per second measurement, follow this formula:

```shell
(throughput in tokens per second) = (sequence length) * (global batch size) / training_step_timing
```

To calculate time to train estimate:

```shell
(time to train in days) = (total tokens) / (throughput in tokens per second) / (number of seconds in a day)
```

To calculate the model flops utilization (MFU):

```shell
MFU = (achieved TFLOPS_per_GPU) / (peak GPU FLOPS)
```

**Peak theoretical throughput across GPUs and Data Types (in TFLOPS)**

For peak theoretical throughput values used in MFU calculations, see the [Peak Theoretical Throughput](../../README.md#peak-theoretical-throughput) section in the main README.

# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via `export HF_TOKEN=<your token>`.

Requires Python 3.12.x, or conda.

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_gpt_oss`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see [Output Locations](#output-locations) for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run submit -w pretrain_gpt_oss --scale 64
```

### Additional SLURM Parameters

Use a SLURM reservation:

```bash
ADDITIONAL_SLURM_PARAMS="reservation=my_reservation" llmb-run submit -w pretrain_gpt_oss --scale 128
```

Run on specific nodes:

```bash
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002" llmb-run submit -w pretrain_gpt_oss --scale 128
```

Exclude specific nodes:

```bash
ADDITIONAL_SLURM_PARAMS="exclude=node003,node004" llmb-run submit -w pretrain_gpt_oss --scale 128
```

Combine multiple parameters (semicolon-separated):

```bash
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002;reservation=my_reservation;exclusive" llmb-run submit -w pretrain_gpt_oss --scale 128
```

For more details on llmb-run usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:

- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/gpt-oss/pretrain/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use (e.g. 64, 128, 256, 512)
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs
  - `b300` - NVIDIA B300 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**

- `ADDITIONAL_SLURM_PARAMS`: Extra `sbatch` flags (e.g. `--nodelist`, `--reservation`), semicolon-separated
  - Example: `"nodelist=node001,node002;reservation=my_reservation;exclusive"`

### Example Commands

Train GPT OSS 120B with BF16 precision on 128 GB200 GPUs:

```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 ./launch.sh
```

# Output Locations

All benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```
experiments/
├── <experiment_name>/
│   └── <experiment_name>_<timestamp>/
│       ├── <experiment_name>/
│       │   ├── log-<experiment_name>.out      # Main training log with performance data
│       │   ├── sbatch_<experiment_name>.out   # Batch script output  
│       │   └── nsys_profile/                  # Profiling output (when enabled)
│       │       └── *.nsys-rep files
│       └── [batch scripts and other files]
```

The `<experiment_name>` typically follows these patterns:

- `pretrain_gpt_oss_120b_bf16_<config>`

**Key files:**

- `log-<experiment_name>.out` - Contains training step timing and performance metrics analyzed by `parse_train_timing_mbridge.sh`
- `nsys_profile/` - Contains profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`

# Profiling

Profiling is supported with Nsight Systems.

## Run Nsight Profiling

To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` or use the `-p` flag when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Default Profiling Settings:

- **MPI Ranks:** all ranks
- **Job Steps:** 45-50
- **Output Location:** Profiling output saved alongside training results (see Output Locations)
- **Filename format:** `profile_${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURM_LOCALID}.nsys-rep`

**Example command:**

```shell
llmb-run submit -w pretrain_gpt_oss --scale 64 -p
```

### Customizing profiling behavior:

- Specify job steps to profile:
  - `RUN_CONF_PROFILE_START_STEP`: start profiling on this job step.
    Default: 45
  - `RUN_CONF_PROFILE_STOP_STEP`: stop profiling on this job step.
    Default: 50
- Enable GPU metrics collection:
  - `ENABLE_GPU_METRICS`: Enable GPU metrics collection during Nsight profiling (default: false)
  * When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
  * Provides additional GPU utilization, memory usage, and compute efficiency data
  * May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**

```shell
ENABLE_GPU_METRICS=true llmb-run submit -w pretrain_gpt_oss --scale 64 -p
```

### Viewing results

In order to view the profile traces (\*.nsys-rep files) interactively:

- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the job run.

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

<!-- NCCL trace support removed. Documentation section deleted intentionally. -->
