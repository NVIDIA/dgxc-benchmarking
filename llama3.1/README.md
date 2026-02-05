# Overview

This recipe contains information and scripts to produce performance results for the Llama3.1 8B, 70B, and 405B training workloads. The scripts help perform environment setup and launch benchmark jobs.

Weak scaling methodology is used in the configurations below.

This recipe supports the following precisions: FP8, BF16, NVFP4.

This variant of the workload supports the following precisions across the 4 gpu types: GB300, GB200, B200, H100:

| GPU Type | 8B Precisions | 70B Precisions | 405B Precisions |
|----------|---------------|----------------|-----------------|
| GB300    | FP8, NVFP4 | FP8, NVFP4 | FP8 |
| GB200    | FP8, NVFP4 | FP8, NVFP4 | FP8 |
| B200     | FP8, NVFP4 | FP8, NVFP4 | FP8, NVFP4 |
| H100     | BF16, FP8 | BF16, FP8 | BF16, FP8 |

This variant of the workload is best-suited for clusters with GPUs below:

## GB300
* At least (8, 64, 128) GPUs for model sizes (8B, 70B, 405B) with at least 288 GB memory each.
* The GB300 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    |
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|
  | 405b                | 128-512  | FP8       | 8192   | 126    | True  |  2 | 1  | 1  | 1  | 2   | GPUs/2  | NA |  1  | GPUs/2  | 1   | False |
  | 70b                 | 64-512   | FP8       | 8192   | 80     | False |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  2  | GPUs*2  | 1   | False |
  | 8b                  | 8-128    | FP8       | 8192   | 32     | False |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  4  | GPUs*16 | 4   | False |

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    |
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|
  | 70b                 | 64-512   | NVFP4       | 8192   | 80     | False |  1 | 4  | 1  | 1  | 1   | GPUs/4  | 5  |  1  | GPUs*2  | 8   | False |
  | 8b                  | 8-128    | NVFP4       | 8192   | 32     | False |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  4  | GPUs*16 | 4   | False |


## GB200
* At least (8, 64, 128) GPUs for model sizes (8B, 70B, 405B) with at least 186 GB memory each.
* The GB200 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | 
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|
  | 405b                | 128-512  | FP8       | 8192   | 126    | True  |  2 | 1  | 1  | 1  | 2   | GPUs/2  | NA |  2  | GPUs/2  | 1   | False | 
  | 70b                 | 64-512   | FP8       | 8192   | 80     | True  |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  2  | GPUs*2  | 1   | False | 
  | 8b                  | 8-128    | FP8       | 8192   | 32     | False |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  2  | GPUs*16 | 8   | False |

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | 
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|
  | 70b                 | 64-512   |  NVFP4    | 8192   | 80     | False |  2 | 4  | 1  | 1  | 2   | GPUs/8  | 5  |  1  | GPUs*2  | 16  | False | 
  | 8b                  | 8-128    |  NVFP4    | 8192   | 32     | False |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  4  | GPUs*16 | 4   | False |
  

## B200
* At least (8, 64, 128) GPUs for model sizes (8B, 70B, 405B) with at least 180 GB memory each.
* The B200 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | 
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|
  | 405b                | 128-1024 | FP8       | 8192   | 126    | False |  4 | 8  | 2  | 1  | 4   | GPUs/64 | 8  |  1  | GPUs/2  | 32  | False | 
  | 70b                 | 64-1024  | FP8       | 8192   | 80     | True  |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  1  | GPUs*2  | 2   | False | 
  | 8b                  | 8-128    | FP8       | 8192   | 32     | False |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  2  | GPUs*16 | 8   | False |

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | 
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|
  | 405b                | 128-1024 |  NVFP4    | 8192   | 126    | False |  4 | 8  | 2  | 1  |  4  | GPUs/64 | 4  |  1  | GPUs/2  | 32  | False |
  | 70b                 | 64-1024  |  NVFP4    | 8192   | 80     | False |  2 | 4  | 1  | 1  | 2   | GPUs/8  | 5  |  1  | GPUs*2  | 16  | False | 
  | 8b                  | 8-128    |  NVFP4    | 8192   | 32     | False |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  4  | GPUs*16 | 4   | False |


## H100
* At least (8, 64, 1024) GPUs for model sizes (8B, 70B, 405B) with at least 80 GB memory each.
* The H100 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.


| Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | 
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|
  | 405b                | 1024     | BF16      | 8192   | 126    | False |  8 | 8  | 2  | 1  | 2   | GPUs/32 | 5  |  1  | GPUs*2  | 64  | False |
  | 70b                 | 64-1024  | BF16      | 8192   | 80     | False |  4 | 4  | 2  | 1  | 1   | GPUs/32 | 5  |  1  | GPUs*2  | 64  | False | 
  | 8b                  | 8-128    | BF16      | 8192   | 32     | False |  1 | 1  | 2  | 1  | 1   | GPUs/2  | NA |  1  | GPUs*16 | 32  | False | 

  | Llama3.1 Model Size | GPUs     | Datatype  | SeqLen | Layers | FSDP  | TP | PP | CP | EP | ETP | DP      | VP | MBS | GBS     | GA  | CG    | 
  |---------------------|:--------:|:---------:|:------:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:-------:|:--:|:---:|:-------:|:---:|:-----:|
  | 405b                | 128-1024 | FP8       | 8192   | 126    | False |  4 | 8  | 2  | 1  | 2   | GPUs/16 | 8  |  1  | GPUs*4  | 64  | False | 
  | 70b                 | 64-1024  | FP8       | 8192   | 80     | True  |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  1  | GPUs*2  | 2   | False | 
  | 8b                  | 8-128    | FP8       | 8192   | 32     | False |  1 | 1  | 1  | 1  | 1   | GPUs    | NA |  2  | GPUs*16 | 8   | False |

# Performance Measurement and Analysis

Performance for Llama3.1 training is measured in milliseconds per iteration, or in other words milliseconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations).

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing_mbridge.sh` script to analyze iterations 35-44 and calculate mean and standard deviation for reliable performance metrics. We also get the achieved GPU FLOPS via the `TFLOPS_per_GPU` metric.

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
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128

#Example with Llama3.1 70B
llmb-run submit -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 64

#Example with Llama3.1 8B at a higher scale
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 16
```

For more details on llmb-run usage, see the [llmb-run documentation](../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/llama3.1/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
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
- `DTYPE`: Precision to run (default: `fp8`)
  - Supported values depend on `GPU_TYPE` and `MODEL_SIZE`. See the tables at the top of this README (and `metadata.yaml`) for supported combinations.
  - Common values: `fp8`, `bf16`, `nvfp4`
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

Train Llama3.1 70B with NVFP4 precision on 64 GB200 GPUs:
```shell
MODEL_SIZE=70b DTYPE=nvfp4 JOB_TOTAL_GPUS=64 GPU_TYPE=gb200 ./launch.sh
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

The `<experiment_name>` typically follows these patterns:
- For Llama3 8B/70B: `pretrain_llama3_<model_size>_<dtype>_<config>`
- For Llama3.1 405B: `pretrain_llama3.1_405b_<dtype>_<config>`

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
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 -p
```
### Customizing profiling behavior:
* Specify job steps to profile:
  * `RUN_CONF_PROFILE_START_STEP`: start profiling on this job step.
    Default: 45
  * `RUN_CONF_PROFILE_STOP_STEP`: stop profiling on this job step.
    Default: 50
* Enable GPU metrics collection:
  * `ENABLE_GPU_METRICS`: Enable GPU metrics collection during Nsight profiling (default: false)
  - When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
  - Provides additional GPU utilization, memory usage, and compute efficiency data
  - May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**
```shell
ENABLE_GPU_METRICS=true llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 -p
```

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

# Run With Checkpoints

Checkpoint save and load can be enabled for this workload in order to measure the impact of storage on checkpointing operations. The additional collected metrics are: time to save a checkpoint and time to load a checkpoint. 

## Save Checkpoint

Save checkpoint feature works for llama3.1 8b, 70b and 405b sizes with either FP8, BF16, or NVFP4 precisions. Make sure your file system has sufficient disk space to accommodate checkpoint sizes below:

| Model | Checkpoint Size | Minimum Tested  Scale GB300 | Minimum Tested Scale GB200 | Minimum Tested Scale B200 | Minimum Tested Scale H100
| :---: | :---:   |  :---: | :---: | :---: | :---: |
| 8b    | ~105 GB | 8      | 8     | 8     |   8   |
| 70b   | ~747 GB | 64     | 64    | 64    |   64  |
| 405b  | ~5.5 TB | 128    | 128   | 128   |   1024|

### How to enable
To save the checkpoints after pretraining llama3.1 model for `max_steps`, you need to set environment variable `ENABLE_CHECKPOINT=true`. At the end of the pretraining the checkpoints will be saved in the  `${LLMB_WORKLOAD}/experiments` folder. There is an option to specify the folder where you want to save the checkpoints. This can be enabled by setting environment variable `CHECKPOINT_DIR=/path/to/checkpoints`. 

```shell
experiment_name = pretrain_llama3_${MODEL_SIZE}_${DTYPE}_gpus${JOB_TOTAL_GPUS}_tp${tp}_pp${pp}_cp${cp}_vp${vp}_ep${ep}_mbs${mbs}_gbs${gbs}
timestamp = date '+%s'
Example directory where checkpoints are saved is ${LLMB_WORKLOAD}/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/
```
Command to run llama3.1 with checkpoint save enabled
```shell
ENABLE_CHECKPOINT=true DTYPE=<precision> MODEL_SIZE=<size> JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> ./launch.sh
```

### How to validate
- Check `${LLMB_WORKLOAD}/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/iter_0000050` folder that it contains *.distcp files
- Check job output log-*.out file (see Training section for reference) for entries like
  ```shell
    successfully saved checkpoint from iteration      50 to /nemo_run/code/nemo_experiments/default/checkpoints [ t 1/1, p 1/1 ] (min, max) time across ranks (ms): save-checkpoint ................................: (24895.07, 24895.13)
  ```

## Load Checkpoint

Load checkpoint feature works successfully at the following scales:

| Model | Checkpoint Size | Minimum Tested  Scale GB300 | Minimum Tested Scale GB200 | Minimum Tested Scale B200 | Minimum Tested Scale H100
| :---: | :---:   |  :---: | :---: | :---: | :---: |
| 8b    | ~105 GB | 8      | 8     | 8     |   8   |
| 70b   | ~747 GB | 64     | 64    | 64    |   64  |
| 405b  | ~5.5 TB | 128    | 128   | 128   |   1024|

**Note**:
- Running load checkpointing feature at other scales may run into CUDA OOM errors. 

### How to enable
To resume training from saved checkpoints, you need to set `LOAD_CHECKPOINT_PATH=<path_to_checkpoint_directory>` environment variable. Make sure the checkpoint files are under the `${LLMB_WORKLOAD}/experiments` directory and `LOAD_CHECKPOINT_PATH` variable is set to: `iter_0000050` directory containing distributed checkpoint files with extension `*.distcp`.

E.g., if the checkpoint was saved under `experiments/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128_1766132151/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/code/nemo_experiments/default/checkpoints/iter_0000050/*` then set the environment variable to a directory one level higher: 

`LOAD_CHECKPOINT_PATH=${LLMB_WORKLOAD}/experiments/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128_1766132151/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/code/nemo_experiments/default/checkpoints/iter_0000050`

The scripts will restore configuration from the checkpoint and resume training process. Training will run for 1 step after checkpoint has been loaded.

```shell
LOAD_CHECKPOINT_PATH=<your_path_to_checkpoint_directory> JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> DTYPE=<precision> MODEL_SIZE=<size> ./launch.sh
```

### How to validate

To validate that checkpoint was loaded successfully look for the entry like below in the main job log-*.out file and make sure there is only 1 iteration of training (see Training section for reference):

```shell
checkpoint:
...
...
...
  load:.../.../experiments/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128_1766132151/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/code/nemo_experiments/default/checkpoints/iter_0000050
...
...
...
[2025-12-19 10:49:15] iteration        1/       1 | consumed samples:          128 | elapsed time per iteration (ms): 20289.1 | learning rate: 3.000000E-05 | global batch size:   128 | lm loss: 1.258695E+01 | loss scale: 1.0 | grad norm: 10.275 | number of skipped iterations:   0 | number of nan iterations:   0 |Number of parameters in transformer layers in billions:  6.98

Number of parameters in embedding layers in billions: 1.05
Total number of parameters in billions: 8.03
Number of parameters in most loaded shard in billions: 8.0305
Theoretical memory footprints: weight and optimizer=57438.81 MB
[Rank 0] (after 1 iterations) memory (GB) | mem-allocated-gigabytes: 51.072 | mem-active-gigabytes: 51.072 | mem-inactive-gigabytes: 10.971 | mem-reserved-gigabytes: 216.89 | mem-max-allocated-gigabytes: 194.41 | mem-max-active-gigabytes: 195.46 | mem-max-inactive-gigabytes: 14.369 | mem-max-reserved-gigabytes: 216.89 | mem-alloc-retires: 0 | mem-allocated-count: 571
Deleting CUDA graphs
[after training is done] datetime: 2025-12-19 10:49:16
```


<!-- NCCL trace support removed. Documentation section deleted intentionally. -->

