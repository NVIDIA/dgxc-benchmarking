# Overview

This recipe contains information and scripts to produce performance results for the Grok 1 training workload on GB200/B200/H100 platforms. The scripts help perform environment setup and launch benchmark jobs.
This variant of the workload is best-suited for clusters with GPUs below:

## GB200
* At least 128 GPUs with at least 80 GB memory each. Training of this 314-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* GB200 GPUs. This workload runs with FP8 and BF16 precision.
* Weak scaling methodology is used in configurations below.

The GB200 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.

| GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | ETP | DP  | VP  | MBS | GBS  | GA  |
|------|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 128  | 8192   | 64     | 4   | 1   | 1   | 8   | 4   | 32  | 1   | 1   | 256  | 8   |
| 256  | 8192   | 64     | 4   | 1   | 1   | 8   | 4   | 64  | 1   | 1   | 512  | 8   |
| 512  | 8192   | 64     | 4   | 1   | 1   | 8   | 4   | 128 | 1   | 1   | 1024 | 8   |

## B200
* At least 256 GPUs with at least 180 GB memory each. Training of this 314-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* B200 GPUs. This workload runs with FP8 and BF16 precision.
* Weak scaling methodology is used in configurations below.

The B200 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.

| GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | ETP | DP  | VP  | MBS | GBS  | GA  |
|------|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 256  | 8192   | 64     | 4   | 4   | 1   | 8   | 1   | 16   | 8   | 1   | 512   | 32 |
| 512  | 8192   | 64     | 4   | 4   | 1   | 8   | 1   | 32   | 8   | 1   | 1024  | 32 |
| 1024 | 8192   | 64     | 4   | 4   | 1   | 8   | 1   | 64  | 8   | 1   | 2048  | 32 |
 
## H100
* At least 512 GPUs with at least 80 GB memory each. Training of this 314-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* This workload runs with FP8 and BF16 precision.

| GPUs | SeqLen | Layers | TP | PP | CP | EP | ETP | DP | VP | MBS | GBS  | GA  |
|------|:------:|:------:|:--:|:--:|:--:|:--:|:---:|:--:|:--:|:---:|:----:|:---:|
| 512  | 8192   | 64     |  4 | 8  | 2  | 8  | 1   | 8  | 8  | 1   | 1024 | 128 |


# Performance Measurement and Analysis

Performance for Grok 1 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations).

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing.sh` script to analyze iterations 11-44 and calculate mean and standard deviation for reliable performance metrics. We also get the achieved GPU FLOPS via `TFLOPS_per_GPU` metric.

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

# Example output:
Train Step Timing and TFLOPS Analysis (iterations 11-44)
================================================================================
Experiment                                                                                   Status Time Mean (s) Time Std (s) TFLOPS_per_GPU Mean TFLOPS_per_GPU Std
------------------------------------------------------------------------------------------ -------- ------------- ------------ ------------------- ------------------
pretrain_grok1_314b_bf16_256_388660                                                         Success         7.382        0.040             1192.94               6.53
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 4096 / 24 = 1398101

To calculate time to train with 1T tokens estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 1398101 / 86400 = 8.28 days 


To calculate the model flops utilization (MFU). Calculation shown [here](#notes).
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) /peak GPU FLOPS)
```

For GB200 GPUs, peak theoretical throughput for FP8 is 4.9 PFLOPS and for BF16 is 2.45 PFLOPS.

The model flops for Grok 1 for GBS=1 per GPU is 4.27E+15

E.g. Grok 1 FP8 on 128x GB200 GPUs (GBS=256)
```shell
peak FLOPS for GB200 = 49 TFLOPS
training step time = 6.631
model flops = 4.27E+15

MFU = 256 * 4.27E+15 / 6.631 / 128 / 4.9E+15 = 26.3%
```

**Peak theoretical throughput across GPUs and Data Types (in TFLOPS)**

| Data Type | B200  | GB200 | H100 |
| --------  | :---: | :---: | :---:|
| BF16      | 2250  | 2450  | 989  |
| FP8       | 4500  | 4900  | 1979 |  



# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via ```export HF_TOKEN=<your token>```.

Requires Python 3.12.x, or conda.

## Request Access

Access to Llama 3 must be requested through the [HuggingFace Llama 3 page](https://huggingface.co/meta-llama/Meta-Llama-3-70B). The approval process is not automatic and could take a day or more. This access is required because the Grok pre-training script utilizes the Llama 3 tokenizer as a proxy.

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
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_grok1`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).




**Migration Note:**
If you previously used `STAGE_PATH`, replace it with `LLMB_INSTALL` (top-level). All output, logs, and checkpoints will be created under the workload's appropriate `LLMB_WORKLOAD` folder.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run single -w pretrain_grok1 -s 314b --dtype fp8 --scale 128

# Example with BF16 precision
llmb-run single -w pretrain_grok1 -s 314b --dtype bf16 --scale 256
```

For more details on llmb-run usage, see the [llmb-run documentation](../llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/grok1/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [MODEL_SIZE=<size>] ./launch.sh
```

### Environment Variables

**Required:**
- `JOB_TOTAL_GPUS`: Number of GPUs to use (e.g., 128, 256, 512)
- `GPU_TYPE`: Type of GPU hardware
  - `gb200` - NVIDIA GB200 GPUs 
  - `b200` - NVIDIA B200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**
- `DTYPE`: Precision format
  - `fp8` - FP8 precision
  - `bf16` - BFloat16 precision
- `MODEL_SIZE`: Model variant (fixed: `314b`)
  - `314b` - 314 billion parameter model (only supported size)


### Example Commands

Train Grok1 with FP8 precision on 128 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 DTYPE=fp8 ./launch.sh
```

Train with BF16 precision on 256 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 DTYPE=bf16 ./launch.sh
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

The `<experiment_name>` typically follows the pattern: `pretrain_grok1_314b_<dtype>_<scale>_<config>`

**Key files:**
- `log-<experiment_name>.out` - Contains training step timing and performance metrics analyzed by `parse_train_timing.sh`
- `nsys_profile/` - Contains profiling traces when `ENABLE_PROFILE=true`

# Profiling
We have two profiling methods supported: Nsight, and NCCL Trace.

**Note:** Profiling and NCCL Trace are currently mutually exclusive.

## Run Nsight Profiling

To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45 to 50 (starting at step 45 and ending before 50) will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Default Profiling Settings:
* **MPI Ranks:** all ranks
* **Job Steps:** 45-50
* **Output Location:** Profiling output saved alongside training results (see Output Locations)
* **Filename format:** `${MODEL}-${MODEL_SIZE}-${DTYPE}_${NUM_GPUS}g_${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURM_LOCALID}.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 DTYPE=fp8 ./launch.sh
```
### Customizing profiling behavior:
* Specify job steps to profile:
  * `RUN_CONF_PROFILE_START_STEP`: start profiling at this job step.
    Default: 45
  * `RUN_CONF_PROFILE_STOP_STEP`: stop profiling before this job step.
    Default: 50
* Enable GPU metrics collection:
  * `ENABLE_GPU_METRICS`: Enable GPU metrics collection during NSight profiling (default: false)
  - When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
  - Provides additional GPU utilization, memory usage, and compute efficiency data
  - May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**
```shell
ENABLE_PROFILE=true ENABLE_GPU_METRICS=true JOB_TOTAL_GPUS=256 GPU_TYPE=gb200  DTYPE=fp8 ./launch.sh
```


### Troubleshooting:

If you encounter issues, try the defaults `ENABLE_PROFILE=true` first as these should be broadly applicable to most systems.

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.


## Run NCCL Trace (For Debugging)

NCCL traces are a tool for understanding communication patterns within your benchmarking job. They provide detailed information on the types of NCCL calls being made (like AllReduce, Broadcast, etc.) and the size of the messages being exchanged.

**Important:** This feature is primarily intended for **troubleshooting and debugging purposes only**. It is not typically used during normal benchmark runs.

To collect NCCL Trace information, set the environment variable `ENABLE_NCCLTRACE=true` when submitting your job:

**Defaults for Tracing:**
*   **Duration:** Due to the large file sizes generated, tracing is limited to the first 5 steps of the job by default.
*   **Output Location:** NCCL trace information is included directly within the standard job log file (see Output Locations)

**Example command:**

```shell
ENABLE_NCCLTRACE=true JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 DTYPE=fp8 ./launch.sh
```

### Understanding NCCL Trace Results

Enabling NCCL tracing will generate a large volume of log messages labeled "NCCL Info". These messages provide details about individual communication operations. Be aware that these log files can be quite large, potentially exceeding 1GB.

Look for messages including:

```
"NCCL INFO AllReduce: opCount"
"NCCL INFO Broadcast: opCount"
"NCCL INFO AllGather: opCount"
"NCCL INFO ReduceScatter: opCount"
```

**Example Log Entry:**

```
[7] NCCL INFO AllReduce: opCount 2 sendbuff 0x7ffb4713c200 recvbuff 0x7ffb4713c200 count 1 datatype 1 op 0 root 0 comm 0x55556b100660 [nranks=128] stream 0x5555630c58b0
```

This example shows an `AllReduce` operation with details about the buffers, count, data type, and the participating ranks.

# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = 12 * (number of layers) * (hidden size)^2 * (1 + (number of query groups)/(number of attention heads) + (sequence length)/(hidden size)/2)
    mlp flops = 18 * (number of layers) * (FFN size) * (hidden size) * (top K)
    embedding flops = 6 * (vocab size) * (hidden size)

Grok1 314b calculation:
    sequence length = 8192
    attention flops = 12 * 64 * 6144^2 * (1 + 8/48 + 8192/6144/2) = 53,150,220,288
    mlp flops = 18 * 64 * 32768 * 6144 * 2 = 463,856,467,968
    embedding flops = 6 * 128256 * 6144 = 4,728,029,184

    model flops = 8192 * (53,150,220,288 + 463,856,467,968 + 4,728,029,184) = 4.27E15
```
**Note**:
Per-tensor delayed scaling recipe is used for FP8 training here.
