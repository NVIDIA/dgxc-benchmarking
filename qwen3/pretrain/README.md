# Overview

This recipe contains information and scripts to produce performance results for the Qwen3 pre-training workload. The scripts help perform environment setup and launch benchmark jobs.

**Supported Model Sizes:**
- `235b` - Qwen3-235B (235 billion parameters, 22 billion active)
- `30b` - Qwen3-30B (30 billion parameters, 3 billion active)

Weak scaling methodology is used in the configurations below.

## GB300

#### Qwen3 30B


| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  | 
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| BF16/FP8 | 8 | 4096 | 48 | 1 | 1 | 1 | 8 | 8 | 1 | 8 | 512 | 8 |
| BF16/FP8 | 16 | 4096 | 48 | 1 | 1 | 1 | 8 | 16 | 1 | 8 | 1024 | 8 |
| BF16/FP8 | 32 | 4096 | 48 | 1 | 1 | 1 | 8 | 32 | 1 | 8 | 2048 | 8 |
| BF16/FP8 | 64 | 4096 | 48 | 1 | 1 | 1 | 8 | 64 | 1 | 8 | 4096 | 8 |

#### Qwen3 235B


| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  | 
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| BF16/FP8 | 64 | 4096 | 94 | 1 | 1 | 1 | 64 | 64 | 1 | 2 | 1024 | 8 | 
| BF16/FP8 | 128 | 4096 | 94 | 1 | 1 | 1 | 64 | 128 | 1 | 2 | 2048 | 8 | 
| BF16/FP8 | 256 | 4096 | 94 | 1 | 1 | 1 | 64 | 256 | 1 | 2 | 4096 | 8 |
| BF16/FP8 | 512 | 4096 | 94 | 1 | 1 | 1 | 64 | 512 | 1 | 2 | 8192 | 8 |  


## GB200

#### Qwen3 30B


| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| BF16/FP8 | 8 | 4096 | 48 | 1 | 1 | 1 | 8 | 8 | 1 | 4 | 512 | 16 | 
| BF16/FP8 | 16 | 4096 | 48 | 1 | 1 | 1 | 8 | 16 | 1 | 4 | 1024 | 16 | 
| BF16/FP8 | 32 | 4096 | 48 | 1 | 1 | 1 | 8 | 32 | 1 | 4 | 2048 | 16 | 
| BF16/FP8 | 64 | 4096 | 48 | 1 | 1 | 1 | 8 | 64 | 1 | 4 | 4096 | 16 | 

#### Qwen3 235B


| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| BF16/FP8 | 64 | 4096 | 94 | 1 | 8 | 1 | 8 | 8 | 1 | 1 | 1024 | 128 | 
| BF16/FP8 | 128 | 4096 | 94 | 1 | 8 | 1 | 8 | 16 | 1 | 1 | 2048 | 128 | 
| BF16/FP8 | 256 | 4096 | 94 | 1 | 8 | 1 | 8 | 32 | 1 | 1 | 4096 | 128 | 
| BF16/FP8 | 512 | 4096 | 94 | 1 | 8 | 1 | 8 | 64 | 1 | 1 | 8192 | 128 | 

## B200

#### Qwen3 30B


| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| BF16/FP8 | 8 | 4096 | 48 | 1 | 1 | 1 | 8 | 8 | 1 | 1 | 512 | 64 | 
| BF16/FP8 | 16 | 4096 | 48 | 1 | 1 | 1 | 8 | 16 | 1 | 1 | 1024 | 64 |
| BF16/FP8 | 32 | 4096 | 48 | 1 | 1 | 1 | 8 | 32 | 1 | 1 | 2048 | 64 |
| BF16/FP8 | 64 | 4096 | 48 | 1 | 1 | 1 | 8 | 64 | 1 | 1 | 4096 | 64 |

#### Qwen3 235B


| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| BF16/FP8 | 64 | 4096 | 94 | 1 | 8 | 1 | 8 | 8 | 4 | 1 | 1024 | 128 | 
| BF16/FP8 | 128 | 4096 | 94 | 1 | 8 | 1 | 8 | 16 | 4 | 1 | 2048 | 128 | 
| BF16/FP8 | 256 | 4096 | 94 | 1 | 8 | 1 | 8 | 32 | 4 | 1 | 4096 | 128 | 
| BF16/FP8 | 512 | 4096 | 94 | 1 | 8 | 1 | 8 | 64 | 4 | 1 | 8192 | 128 | 

## H100

#### Qwen3 30B


| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| BF16/FP8 | 16 | 4096 | 48 | 1 | 2 | 1 | 8 | 8 | 12 | 1 | 512 | 64 | 
| BF16/FP8 | 32 | 4096 | 48 | 1 | 2 | 1 | 8 | 16 | 12 | 1 | 1024 | 64 | 
| BF16/FP8 | 64 | 4096 | 48 | 1 | 2 | 1 | 8 | 32 | 12 | 1 | 2048 | 64 | 
| BF16/FP8 | 128 | 4096 | 48 | 1 | 2 | 1 | 8 | 64 | 12 | 1 | 4096 | 64 | 


#### Qwen3 235B


| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| BF16/FP8 | 256 | 4096 | 94 | 2 | 8 | 1 | 32 | 16 | 4 | 1 | 2048 | 128 | 
| BF16/FP8 | 512 | 4096 | 94 | 2 | 8 | 1 | 32 | 32 | 4 | 1 | 4096 | 128 | 

# Performance Measurement and Analysis

Performance for Qwen3 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations).

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing_mbridge.sh` script to analyze iterations 35-44 and calculate mean and standard deviation for reliable performance metrics. We also get the achieved GPU FLOPS via `TFLOPS_per_GPU` metric.

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
Elapsed Time (ms) and TFLOPS/GPU Analysis (iterations 35-44)
================================================================================
Experiment                                                                                   Status Time Mean (ms) Time Std (ms) TFLOPS_per_GPU Mean TFLOPS_per_GPU Std
------------------------------------------------------------------------------------------ -------- ------------- ------------ ------------------- ------------------
pretrain_qwen3_235b_a22b_bf16_gpus256_tp2_pp8_cp1_vp4_ep32_mbs1_gbs2048_4006524             Success     25532.270       13.911              190.00               0.11
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(throughput in tokens per second) = (sequence length) * (global batch size) / training_step_timing
```

E.g. 

To calculate time to train estimate:
```shell
(time to train in days) = (total tokens) / (throughput in tokens per second) / (number of seconds in a day)
```
E.g. 


To calculate the model flops utilization (MFU):
```shell
MFU = (achieved TFLOPS_per_GPU) / (peak GPU FLOPS)
```

**Peak theoretical throughput across GPUs and Data Types (in TFLOPS)**

| Data Type | GB300 | GB200 | B200 | H100 |
| --------  | :---: | :---: | :---:| :---:|
| BF16      | 2450  | 2450  | 2250 | 989  |
| FP8       | 4900  | 4900  | 4500 | 1979 |


# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via ```export HF_TOKEN=<your token>```.

Requires Python 3.12.x, or conda.

## Request Access

No special access is required to run this benchmark.

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
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_qwen3`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).



# Prepare Dataset
Since Qwen3 training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run (235b model)
llmb-run submit -w pretrain_qwen3 -s 235b --dtype bf16 --scale 64

# Run a benchmark with llmb-run (30b model)
llmb-run submit -w pretrain_qwen3 -s 30b --dtype bf16 --scale 8

# Example with different scale
llmb-run submit -w pretrain_qwen3 -s 235b --dtype bf16 --scale 256

# Example with FP8-CS precision (default)
llmb-run submit -w pretrain_qwen3 -s 235b --dtype fp8 --scale 256

# Example with additional SLURM parameters
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002" llmb-run submit -w pretrain_qwen3 -s 235b --dtype bf16 --scale 256
```

For more details on llmb-run usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/qwen3/pretrain/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [MODEL_SIZE=<size>] [FP8_RECIPE=<type>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
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
- `DTYPE`: Precision format (default: `bf16`)
  - `bf16` - BFloat16 precision
  - `fp8`  - FP8 precision (supports CS/MX/SS)
- `FP8_RECIPE`: FP8 variant selector (default: `cs`)
  - **Note:** Available FP8_RECIPE options vary by GPU type. See the [Supported Configs](#overview).
  - `cs`- FP8-CS
  - `mx`- FP8-MX
- `MODEL_SIZE`: Model variant (default: `235b`)
  - `235b` - 235 billion parameter model (22 billion active)
  - `30b` - 30 billion parameter model (3 billion active)
- `ADDITIONAL_SLURM_PARAMS`: Additional SLURM parameters (optional)
  - Format: Semicolon-separated key=value pairs (use semicolons when values contain commas)
  - Example: `"nodelist=node001,node002;constraint=gpu"`

### Example Commands

Train Qwen3 235b with BF16 precision on 64 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=64 GPU_TYPE=gb200 MODEL_SIZE=235b ./launch.sh
```

Train Qwen3-30B with BF16 precision on 8 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=8 GPU_TYPE=gb200 MODEL_SIZE=30b ./launch.sh
```

### SLURM Node Specification Examples

Train on specific nodes:
```shell
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002" JOB_TOTAL_GPUS=64 GPU_TYPE=gb200 ./launch.sh
```

Train with node constraints:
```shell
ADDITIONAL_SLURM_PARAMS="constraint=gpu&memory;exclusive" JOB_TOTAL_GPUS=64 GPU_TYPE=gb200 ./launch.sh
```

Train using a SLURM reservation:
```shell
ADDITIONAL_SLURM_PARAMS="reservation=my_reservation" JOB_TOTAL_GPUS=256 GPU_TYPE=h100 ./launch.sh
```

# Output Locations

All benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```text
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

The `<experiment_name>` typically follows the pattern: `pretrain_qwen3_<model_size>_<dtype>_<scale>_<config>`

**Key files:**
- `log-<experiment_name>.out` - Contains training step timing and performance metrics analyzed by `parse_train_timing_mbridge.sh`
- `nsys_profile/` - Contains profiling traces when `ENABLE_PROFILE=true`

# Run Nsight Profiling
To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Profiling job details:
* **MPI Ranks:** all
* **Job Steps:** 45-50
* **Output Location:** Profiling output saved alongside training results (see Output Locations)
* **Filename format:** `profile_${SLURM_JOB_ID}_nodeId_rankId.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 ./launch.sh
```

### Customizing profiling behavior:
* Specify job steps to profile:
	* `PROFILE_START_STEP`: start profiling on this job step.
	- Default: 45
	* `PROFILE_STOP_STEP`: stop profiling on this job step.
	- Default: 50
* Enable GPU metrics collection:
	* `ENABLE_GPU_METRICS`: Enable GPU metrics collection during Nsight profiling (default: false)
	- When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
	- Provides additional GPU utilization, memory usage, and compute efficiency data
	- May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**
```shell
ENABLE_PROFILE=true ENABLE_GPU_METRICS=true JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 ./launch.sh
```

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.
