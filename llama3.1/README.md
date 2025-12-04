# Overview

This recipe contains information and scripts to produce performance results for the Llama3.1 8B, 70B, and 405B training workloads. The scripts help perform environment setup and launch benchmark jobs.
This variant of the workload is best-suited for clusters with GPUs below:

## GB300
* At least (8, 64, 128) GPUs for model sizes (8B, 70B, 405B) with at least 288 GB memory each.
* The GB300 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.
* This workload runs with FP8 precision.

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | RL | OL |
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|:--:|:--:|
  | 405b                | 128-1024 | FP8       | 8192   | 126    | True  |  2 | 1  | 1  | NA | NA  | GPUs/2  | NA |  1  | GPUs/2  | 1   | False | 0  | 80 |
  | 70b                 | 64-1024  | FP8       | 8192   | 80     | True  |  1 | 1  | 1  | NA | NA  | GPUs    | NA |  2  | GPUs*2  | 1   | False | 0  | 40 |
  | 8b                  | 8-128    | FP8 | 8192   | 32     | False |  1 | 1  | 1  | NA | NA  | GPUs    | NA |  2  | GPUs*16 | 8   | True  | 0  | 0  |

## GB200
* At least (8, 64, 128) GPUs for model sizes (8B, 70B, 405B) with at least 186 GB memory each.
* The GB200 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.
* This workload runs with FP8 precision.

  | Llama3.1 Model Size | GPUs    | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | RL | OL |
  |---------------------|:-------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|:--:|:--:|
  | 405b                | 128-512 | FP8       | 8192   | 126    | True  |  2 | 1  | 1  | NA | NA  | GPUs/2  | NA |  1  | GPUs/2  | 1   | False | 0  | 80 |
  | 70b                 | 64-512  | FP8       | 8192   | 80     | True  |  1 | 1  | 1  | NA | NA  | GPUs    | NA |  2  | GPUs*2  | 1   | False | 0  | 40 |
  | 8b                  | 8-128   | FP8 | 8192   | 32     | False |  1 | 1  | 1  | NA | NA  | GPUs    | NA |  2  | GPUs*16 | 8   | True  | 0  | 0  |


## B200
* At least (8, 64, 128) GPUs for model sizes (8B, 70B, 405B) with at least 180 GB memory each.
* The B200 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.
* This workload runs with FP8 precision.

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | RL | OL |
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|:--:|:--:|
  | 405b                | 128-1024 | FP8       | 8192   | 126    | False |  4 | 8  | 2  | NA | NA  | GPUs/64 | 8  |  1  | GPUs/2  | 32  | False | 0  | 0  |
  | 70b                 | 64-1024  | FP8       | 8192   | 80     | True  |  1 | 1  | 1  | NA | NA  | GPUs    | NA |  1  | GPUs*2  | 2   | False | 5  | 0  |
  | 8b                  | 8-128    | FP8 | 8192   | 32     | False |  1 | 1  | 1  | NA | NA  | GPUs    | NA |  2  | GPUs*16 | 8   | True  | 0  | 0  |


## H100
* At least (8, 64, 1024) GPUs for model sizes (8B, 70B, 405B) with at least 80 GB memory each.
* The H100 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.
* This workload runs with FP8 precision.

  | Llama3.1 Model Size | GPUs      | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP       | VP | MBS | GBS     | GA  | CG    | RL | OL | 
  |---------------------|:---------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:--------:|:--:|:---:|:-------:|:---:|:-----:|:--:|:--:|
  | 405b                | 1024 | FP8       | 8192   | 126    | False |  8 | 8  | 2  | NA | NA  | GPUs/128 | 8  |  1  | GPUs/2  | 64  | False | 0  | 0  |
  | 70b                 | 64-1024   | FP8       | 8192   | 80     | False |  4 | 8  | 1  | NA | NA  | GPUs/32  | 5  |  1  | GPUs*2  | 64  | False | 5  | 0  |
  | 8b                  | 8-128     | FP8       | 8192   | 32     | True  |  1 | 1  | 1  | NA | NA  | GPUs     | NA |  1  | GPUs*16 | 16  | True  | 0  | 0  |


# Performance Measurement and Analysis

Performance for Llama3.1 training is measured in milliseconds per iteration, or in other words milliseconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations).

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing_mbridge.sh` script to analyze iterations 35-44 and calculate mean and standard deviation for reliable performance metrics. We also get the achieved GPU FLOPS via the `TFLOPS_per_GPU` metric.

### Running the parse_train_timing_mbridge.sh script

To analyze training timing from your experiment results, run the script from the workload directory. Note, that `LLMB_REPO` is the directory containing the clone of the recipe repository.

```bash
# Basic usage - parses results in the directory named 'experiments' in the current folder
$LLMB_REPO/common/parse_train_timing_mbridge.sh

# Specify a different experiments directory
$LLMB_REPO/common/parse_train_timing_mbridge.sh /path/to/experiments

# Output in CSV format
$LLMB_REPO/common/parse_train_timing_mbridge.sh --format=csv

# Output in JSON format
$LLMB_REPO/common/parse_train_timing_mbridge.sh --format=json

# Show full filenames instead of shortened versions
$LLMB_REPO/common/parse_train_timing_mbridge.sh --full-names
```

Example output:
```shell
Elapsed Time (ms) and TFLOPS/GPU Analysis (iterations 35-44)
================================================================================
Experiment                                                                                   Status Time Mean (ms) Time Std (ms) TFLOPS_per_GPU Mean TFLOPS_per_GPU Std
------------------------------------------------------------------------------------------ -------- ------------- ------------ ------------------- ------------------
pretrain_llama31_405b_fp8_cs_gpus128_tp2_pp1_cp1_vpNone_ep1_mbs1_gbs64_658572               Success      5741.470       68.670             1636.80              20.89
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training step time in seconds) = (throughput in tokens per second)
```

E.g. 8192 * 64 / 5.74  = 91339

To calculate time to train with 1T tokens estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 91339 / 86400 = 126.72 days 


To calculate the model flops utilization (MFU). 
```shell
MFU = avg(TFLOPS_GPU) / (peak GPU FLOPS)
```

**Peak theoretical FP8 throughput across GPUs (in TFLOPS)**

|            | GB300 | GB200 | B200 | H100 |
|------------|:-----:|:-----:|:----:|:----:|
| Throughput | 4900  | 4900  | 4500 | 1979 |

E.g. Llama3.1 405b FP8 on 128x GB200 GPUs that has an average of 1636.8 TFLOPs per GPU for steps 34-44
```shell
peak FLOPS for GB200 = 4900 TFLOPS
avg(TFLOPS_GPU) = 1636.8
MFU =  1636.8 / 4900 = 33.40%
```


# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via ```export HF_TOKEN=<your token>```.

Requires Python 3.12.x, or conda.

## Request Access

Access to the Llama 3.1 models must be requested through [Meta's website](https://www.llama.com/llama-downloads/) then requested on the [HuggingFace Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-405B) page. The approval process is not automatic and could take a day or more.

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
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_llama3.1`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).



# Prepare Dataset
Since Llama3.1 training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see [Output Locations](#output-locations) for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run single -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128

#Example with Llama3.1 70B
llmb-run single -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 64

#Example with Llama3.1 8B at a higher scale
llmb-run single -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 16
```

For more details on llmb-run usage, see the [llmb-run documentation](../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/llama3.1/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**
- `JOB_TOTAL_GPUS`: Number of GPUs to use (e.g., 128, 256, 512)
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**
- `MODEL_SIZE`: Model variant (default: `405b`)
  - `405b` - 405 billion parameter model
  - `70b` - 70 billion parameter model
  - `8b` - 8 billion parameter model
- `ADDITIONAL_SLURM_PARAMS`: Additional SLURM parameters (optional)
  - Format: Semicolon-separated key=value pairs (use semicolons when values contain commas)
  - Example: `"nodelist=node001,node002;constraint=gpu"`

### Example Commands

Train Llama3.1 405B with FP8 precision on 128 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 ./launch.sh
```

Train with FP8 precision on 256 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 ./launch.sh
```

Train with FP8 precision on 1024 H100 GPUs:
```shell
JOB_TOTAL_GPUS=1024 GPU_TYPE=h100 ./launch.sh
```

Train with FP8 precision on 8 H100 GPUs with Llama3.1 8B:
```shell
MODEL_SIZE=8b JOB_TOTAL_GPUS=8 GPU_TYPE=h100 ./launch.sh
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

The `<experiment_name>` typically follows the pattern: `pretrain_llama3.1_<model_size>_fp8_cs_<config>`

**Key files:**
- `log-<experiment_name>.out` - Contains training step timing and performance metrics analyzed by `parse_train_timing_mbridge.sh`
- `nsys_profile/` - Contains profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`

# Profiling
Profiling is supported with Nsight Systems.

## Run Nsight Profiling

To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Default Profiling Settings:
* **MPI Ranks:** all ranks
* **Job Steps:** 45-50
* **Output Location:** Profiling output saved alongside training results (see Output Locations)
* **Filename format:** `profile_${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURM_LOCALID}.nsys-rep`

**Example command:**
```shell
llmb-run single -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 -p
```
### Customizing profiling behavior:
* Specify job steps to profile:
  * `RUN_CONF_PROFILE_START_STEP`: start profiling on this job step.
    Default: 45
  * `RUN_CONF_PROFILE_STOP_STEP`: stop profiling on this job step.
    Default: 50
* Enable GPU metrics collection:
  * `ENABLE_GPU_METRICS`: Enable GPU metrics collection during NSight profiling (default: false)
  - When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
  - Provides additional GPU utilization, memory usage, and compute efficiency data
  - May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**
```shell
ENABLE_GPU_METRICS=true llmb-run single -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 -p
```

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

<!-- NCCL trace support removed. Documentation section deleted intentionally. -->
