# Overview

This recipe contains information and scripts to produce performance results for the LLAMA4 Maverick pre-training workload. The scripts help perform environment setup and launch benchmark jobs. It supports both BF16 and FP8 precisions. 

The GB200 jobs listed below progressively increase GPU count, with configurations weak-scaled to match.

## GB200

|Model Size|Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  |ETP | MBS | GBS  | GA  |
|:---------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|:---:|
| 400b | BF16/FP8-MX  | 128    | 8192   | 48     | 1   | 2   | 1   | 64  | 64   | 12 | 1  | 1   | 1024  | 16 |
| 400b | BF16/FP8-MX  | 256    | 8192   | 48     | 1   | 2   | 1   | 64  | 128   | 12 | 1  | 1   | 2048  | 16 |
| 400b | BF16/FP8-MX  | 512    | 8192   | 48     | 1   | 2   | 1   | 64  | 256   | 12 | 1  | 1   | 4096  | 16 |

## H100

|Model Size|Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP | ETP  | MBS | GBS  | GA  |
|:---------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|:---:|
| 400b | BF16/FP8-DS  | 512    | 8192   | 48     | 4   | 1   | 1   | 128  | 128   | 1  | 4 | 1   | 1024  | 8 |
| 400b | BF16/FP8-DS  | 1024    | 8192   | 48     | 4   | 1   | 1   | 128  | 128   | 1  | 4 | 1   | 2048  | 8 |

# Performance Measurement and Analysis

Performance for LLAMA4 Maverick training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations). 

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing.sh` script to analyze iterations 35-44 and calculate mean and standard deviation for reliable performance metrics. We also get the achieved GPU FLOPS via `TFLOPS_per_GPU` metric.

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

Example output:
```shell
Train Step Timing and TFLOPS Analysis (iterations 35-44)
================================================================================
Experiment                                                                         Status Time Mean (s) Time Std (s) TFLOPS_per_GPU Mean TFLOPS_per_GPU Std
-------------------------------------------------------------------------------- -------- ------------- ------------ ------------------- ------------------
pretrain_llama4_e128_bf16_64nodes_tp1_pp12_cp1_vp12_ep64_etp1_1mbs_1024gbs_2784318  Success         7.359        0.008              527.78               1.11
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 1024 / 7.359 = 1139911

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 1139911 / 86400 = 10.15 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

 The model flops for LLAMA4 Maverick BF16  for GBS=1 is 8.92e14. Calculation shown [here](#notes).

E.g. LLAMA4 Maverick BF16 on 128x GB200 GPUs (GBS=1024)
```shell
peak FLOPS for GB200 BF16 = 2.45 PFLOPS
training step time = 7.781 s
model flops = 8.92e14

MFU = 1024 * 8.92e14 / 7.359 / 128 / 2.45e15 = 39.58%
```

# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via ```export HF_TOKEN=<your token>```.

Requires Python 3.12.x, or conda.

## Request Access

Access to Llama 4 Maverick must be requested through the [HuggingFace Llama 4](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct). The approval process is not automatic and could take a day or more.

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
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_llama4-maverick`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).


**Migration Note:**
If you previously used `STAGE_PATH`, replace it with `LLMB_INSTALL` (top-level). All output, logs, and checkpoints will be created under the workload's appropriate `LLMB_WORKLOAD` folder.

# Prepare Dataset
Since LLAMA4 Maverick training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run single -w pretrain_llama4-maverick -s 400b --dtype fp8 --scale 128

# Example with BF16 precision
llmb-run single -w pretrain_llama4-maverick -s 400b --dtype bf16 --scale 256
```

For more details on llmb-run usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/llama4/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [MODEL_SIZE=<size>] ./launch.sh
```

### Environment Variables

**Required:**
- `JOB_TOTAL_GPUS`: Number of GPUs to use (e.g., 128, 256, 512)
- `GPU_TYPE`: Type of GPU hardware
  - `gb200` - NVIDIA GB200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**
- `DTYPE`: Precision format (default: `bf16`)
  - `fp8` - FP8 precision
  - `bf16` - BFloat16 precision
- `MODEL_SIZE`: Model variant (fixed: `400b`)
  - `400b` - 400 billion parameter model (only supported size)

### Example Commands

Train LLAMA4 Maverick with BF16 precision on 128 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=128 DTYPE=bf16 GPU_TYPE=gb200 ./launch.sh
```

Train with FP8 precision on 256 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=256 DTYPE=fp8 GPU_TYPE=gb200 ./launch.sh
```

Train on 512 H100 GPUs with FP8 precision:
```shell
JOB_TOTAL_GPUS=512 DTYPE=fp8 GPU_TYPE=h100 ./launch.sh
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

The `<experiment_name>` typically follows the pattern: `pretrain_llama4-maverick_400b_<dtype>_<scale>_<config>`

**Key files:**
- `log-<experiment_name>.out` - Contains training step timing and performance metrics analyzed by `parse_train_timing.sh`
- `nsys_profile/` - Contains profiling traces when `ENABLE_PROFILE=true`

# Run Nsight Profiling

To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Profiling job details:
* **MPI Ranks:** 0-8
* **Job Steps:** 45-50
* **Output Location:** Profiling output saved alongside training results (see Output Locations)
* **Filename format:** `profile_${SLURM_JOB_ID}_nodeId_rankId.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true JOB_TOTAL_GPUS=128 DTYPE=bf16 GPU_TYPE=gb200 ./launch.sh
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
ENABLE_PROFILE=true ENABLE_GPU_METRICS=true JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 DTYPE=bf16 ./launch.sh
```

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

<!-- NCCL trace support removed. Documentation section deleted intentionally. -->

# Notes

```shell
Model Flops: 
12*GBS*(seq Length)*(num of layers)*(hidden size)*(hidden size)*((1+(number of query group)/(number of head)+0.5*(seq length)/hidden size)+((FFN Hidden Size)/(Hidden Size))*1.5*(Top K)+(Vocab Size)/2/(num of layers)/(hidden size))

Model Flops breakdown:
  attention flops = 12 * GBS * (seq length) * (number of layers) * (hidden size)^2 * (1 + (number of query group)/(number of head) + 0.5*(seq length)/hidden size) 


  mlp flops = 12 * GBS * (seq length) * (number of layers) * (hidden size)^2 * ((FFN Hidden Size)/(Hidden Size) * 1.5 * (Top K))

  embedding flops = 12 * GBS * (seq length) * (number of layers) * (hidden size)^2 * (Vocab Size / (2 * (number of layers) * (hidden size)))
  

llama4 maverick calculation:
  - Sequence length: 8192
  - Number of layers: 48
  - Hidden size: 5120
  - Vocab size: 202048
  - FFN Hidden Size: 16384
  - Number of heads: 40
  - Number of Experts: 128 
  - TopK: 1
  - Number of Query Groups: 8

    Attention flops = 12 * 8192 * 48 * (5120)^2 * (1 + (8/40) + 0.5*(8192/5120))
        = 247390116864000

    MLP flops = 12 * 8192 * 48 * (5120)^2 * ((16384/5120) * 1.5 * 1)
       = 593736279244800

    Embedding flops = 12 * 8192 * 48 * (5120)^2 * (202048 / (2 * 48 * 5120) = 50665495805952

    model flops = 8192 * (36238786560 + 40265318400 + 6219525120) = 678,702,806,466,560 
    247390116864000 + 593736279244800 + 50665495805952 = 891792891914752  = 8.92e14

```









