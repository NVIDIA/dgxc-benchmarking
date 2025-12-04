# Overview

This recipe contains information and scripts to produce performance results for the Nemotron-H pre-training workloads. The scripts help perform environment setup and launch benchmark jobs.

The recipes listed below progressively increase GPU count, with configurations weak-scaled to match.

## GB300

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| FP8 | 32 | 8192 | 118 | 2 | 1 | 1 | 1 | 16 | 1 | 1 | 96 | 6 |
| FP8 | 64 | 8192 | 118 | 2 | 1 | 1 | 1 | 32 | 1 | 1 | 192 | 6 |
| FP8 | 128 | 8192 | 118 | 2 | 1 | 1 | 1 | 64 | 1 | 1 | 384 | 6 |
| FP8 | 256 | 8192 | 118 | 2 | 1 | 1 | 1 | 128 | 1 | 1 | 768 | 6 |
| FP8 | 512 | 8192 | 118 | 2 | 1 | 1 | 1 | 256 | 1 | 1 | 1536 | 6 |

## B200 and GB200

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| FP8 | 32 | 8192 | 118 | 4 | 1 | 1 | 1 | 8 | 1 | 2 | 96 | 6 |
| FP8 | 64 | 8192 | 118 | 4 | 1 | 1 | 1 | 16 | 1 | 2 | 192 | 6 |
| FP8 | 128 | 8192 | 118 | 4 | 1 | 1 | 1 | 32 | 1 | 2 | 384 | 6 |
| FP8 | 256 | 8192 | 118 | 4 | 1 | 1 | 1 | 64 | 1 | 2 | 768 | 6 |
| FP8 | 512 | 8192 | 118 | 4 | 1 | 1 | 1 | 128 | 1 | 2 | 1536 | 6 |

## H100

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| FP8 | 64 | 8192 | 118 | 8 | 1 | 1 | 1 | 8 | 1 | 1 | 192 | 24 |
| FP8 | 128 | 8192 | 118 | 8 | 1 | 1 | 1 | 16 | 1 | 1 | 384 | 24 |
| FP8 | 256 | 8192 | 118 | 8 | 1 | 1 | 1 | 32 | 1 | 1 | 768 | 24 |
| FP8 | 512 | 8192 | 118 | 8 | 1 | 1 | 1 | 64 | 1 | 1 | 1536 | 24 |
| FP8 | 1024 | 8192 | 118 | 8 | 1 | 1 | 1 | 128 | 1 | 1 | 3072 | 24 |

# Performance Measurement and Analysis

Performance for Nemotron-H training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations).

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing.sh` script to analyze iterations 35-44 and calculate mean and standard deviation for reliable performance metrics.

### Running the parse_train_timing.sh script

To analyze training timing from your experiment results, run the script from the workload directory. Note, that `LLMB_REPO` is the directory containing the clone of the recipe repository.

```bash
# Basic usage - parses results in the directory named 'experiments' in the current folder
$LLMB_REPO/common/parse_train_timing.sh

# Specify a different experiments directory
$LLMB_REPO/common/parse_train_timing.sh /path/to/experiments

# Output in CSV format
$LLMB_REPO/common/parse_train_timing.sh --format=csv

# Output in JSON format
$LLMB_REPO/common/parse_train_timing.sh --format=json

# Show full filenames instead of shortened versions
$LLMB_REPO/common/parse_train_timing.sh --full-names
```

```shell
# Run the parse_train_timing script to analyze all experiments
common/parse_train_timing.sh $LLMB_WORKLOAD/experiments

To analyze training timing from your experiment results, run the script from the workload directory. Note, that `LLMB_REPO` is the directory containing the clone of the recipe repository.

```bash
# Basic usage - parses results in the directory named 'experiments' in the current folder
$LLMB_REPO/common/parse_train_timing.sh

# Specify a different experiments directory
$LLMB_REPO/common/parse_train_timing.sh /path/to/experiments

# Output in CSV format
$LLMB_REPO/common/parse_train_timing.sh --format=csv

# Output in JSON format
$LLMB_REPO/common/parse_train_timing.sh --format=json

# Show full filenames instead of shortened versions
$LLMB_REPO/common/parse_train_timing.sh --full-names
```

Example output:
```shell
Train Step Timing and TFLOPS Analysis (iterations 35-44)
================================================================================
Experiment                                                                                   Status Time Mean (s) Time Std (s) TFLOPS_per_GPU Mean TFLOPS_per_GPU Std
------------------------------------------------------------------------------------------ -------- ------------- ------------ ------------------- ------------------
pretrain_nemotronh_56b_fp8_gpus128_tp4_pp1_cp1_vpNone_mbs2_gbs384_1063626                   Success         5.310        0.007             1961.50               2.80
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(throughput in tokens per second) = (sequence length) * (global batch size) / training_step_timing
```
E.g. 8192 * 384 / 5.310 = 592415

To calculate time to train estimate:
```shell
(time to train in days) = (total tokens) / (throughput in tokens per second) / (number of seconds in a day)
```
E.g. 1e12 / 592415 / 86400 = 19.54 days


To calculate the model flops utilization (MFU):
```shell
MFU = (achieved TFLOPS_per_GPU) / (peak GPU FLOPS)
```

The peak theoretical throughput for GB200 FP8 is **4.9** PFLOPS.

E.g. Nemotron-H 56B FP8 on 128x GB200 GPUs
```shell
peak FLOPS for H100 FP8 = 4.9 PFLOPS
achieved TFLOPS_per_GPU = 1961.50 TFLOPS

MFU =  1961.50e+12 / 4.9e+15 = 40.03%
```

**Peak theoretical throughput across GPUs and Data Types (in TFLOPS)**

| Data Type | GB300 | GB200 | B200 | H100 |
| --------  | :---: | :---: | :---:| :---:|
| FP8       | 4900  | 4900  | 4500 | 1979 |  


# Prerequisites

Requires Python 3.12.x, or conda.

## Request Access

No special access required to run this benchmark.

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**
- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_nemotron-h`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).



# Prepare Dataset
Since Nemotron-H training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder ([see Output Locations](#output-locations) for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run single -w pretrain_nemotron-h -s 56b --dtype fp8 --scale 128

# Example with different scale
llmb-run single -w pretrain_nemotron-h -s 56b --dtype fp8 --scale 1024
```

For more details on llmb-run usage, see the [llmb-run documentation](../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/nemotron-h/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [MODEL_SIZE=<size>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**
- `JOB_TOTAL_GPUS`: Number of GPUs to use
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**
- `DTYPE`: Precision format (fixed: `fp8`)
  - `fp8` - FP8 precision (only supported precision)
- `MODEL_SIZE`: Model variant (fixed: `56b`)
  - `56b` - 56 billion parameter model (only supported size)

- `ADDITIONAL_SLURM_PARAMS`: Additional SLURM parameters (optional)
  - Format: Semicolon-separated key=value pairs (use semicolons when values contain commas)
  - Example: `"nodelist=node001,node002;constraint=gpu"`

**Note:** This workload only supports:
- FP8 precision
- 56B model size

### Example Commands

Train Nemotron-H 56B with FP8 precision on 128 GB300 GPUs:
```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=GB300 ./launch.sh
```

Train Nemotron-H 56B with FP8 precision on 128 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=GB200 ./launch.sh
```

Train Nemotron-H 56B with FP8 precision on 128 B200 GPUs:
```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=B200 ./launch.sh
```

Train Nemotron-H 56B with FP8 precision on 128 H100 GPUs:
```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=H100 ./launch.sh
```

### SLURM Node Specification Examples

Train on specific nodes:
```shell
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002" JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 ./launch.sh
```

Train with node constraints:
```shell
ADDITIONAL_SLURM_PARAMS="constraint=gpu&memory;exclusive" JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 ./launch.sh
```

Train using a SLURM reservation:
```shell
ADDITIONAL_SLURM_PARAMS="reservation=my_reservation" JOB_TOTAL_GPUS=512 GPU_TYPE=h100 ./launch.sh
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

The `<experiment_name>` typically follows the pattern: `pretrain_nemotron-h_56b_<dtype>_<scale>_<config>`

**Key files:**
- `log-<experiment_name>.out` - Contains training step timing and performance metrics analyzed by `parse_train_timing.sh`
- `nsys_profile/` - Contains profiling traces when `ENABLE_PROFILE=true`

# Run Nsight Profiling
To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Profiling job details:
* **MPI Ranks:** all ranks
* **Job Steps:** 45-50
* **Output Location:** Profiling output saved alongside training results ([see Output Locations](#output-locations))
* **Filename format:** `profile_{SLURM_JOBID}_{SLURM_NODEID}_{SLURM_PROCID}.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 ./launch.sh
```

### Customizing profiling behavior:
* Specify job steps to profile:
  * `PROFILE_START_STEP`: start profiling on this job step.
	  - Default: 45
  * `PROFILE_STOP_STEP`: stop profiling on this job step.
    - Default: 50
* Enable GPU metrics collection:
  * `ENABLE_GPU_METRICS`: Enable GPU metrics collection during NSight profiling (default: false)
  - When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
  - Provides additional GPU utilization, memory usage, and compute efficiency data
  - May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**
```shell
ENABLE_PROFILE=true ENABLE_GPU_METRICS=true JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 ./launch.sh
```

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.
