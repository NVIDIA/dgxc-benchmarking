# Overview

This recipe contains information and scripts to produce performance results for the Nemotron 4 15b pre-training workloads. The scripts help perform environment setup and launch benchmark jobs.

## H100

15 billion parameter variant (FP8/BF16)

- At least 16 GPUs with at least 80GB memory each.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 15b   | BF16/FP8 | 16   | 4096  | 32     | 2   | 1   | 1   | NA  | 8   | NA  | 2   | 64   | 4 |
| 15b   | BF16/FP8 | 32   | 4096  | 32     | 2   | 1   | 1   | NA  | 16   | NA  | 2   | 128   | 4 |
| 15b   | BF16/FP8 | 64   | 4096  | 32     | 2   | 1   | 1   | NA  | 32   | NA  | 2   | 256   | 4 |
| 15b   | BF16/FP8 | 128   | 4096  | 32     | 2   | 1   | 1   | NA  | 64   | NA  | 2   | 512   | 4 |
| 15b   | BF16/FP8 | 256   | 4096  | 32     | 2   | 1   | 1   | NA  | 128   | NA  | 2   | 1024   | 4 |
| 15b   | BF16/FP8 | 512   | 4096  | 32     | 2   | 1   | 1   | NA  | 256  | NA  | 2   | 2048   | 4 |
| 15b   | BF16/FP8 | 1024   | 4096  | 32     | 2   | 1   | 1   | NA  | 512   | NA  | 2   | 4096   | 4 |
| 15b   | BF16/FP8 | 2048   | 4096  | 32     | 2   | 1   | 1   | NA  | 1024   | NA  | 2   | 8192   | 4 |

## GB200


15 billion parameter variant (FP8/BF16)

- At least 16 GPUs with at least 80GB memory each.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 15b   | BF16/FP8 | 16   | 4096  | 32     | 1   | 1   | 1   | NA  | 16   | NA  | 2   | 64   | 2 |
| 15b   | BF16/FP8 | 32   | 4096  | 32     | 1   | 1   | 1   | NA  | 32   | NA  | 2   | 128   | 2 |
| 15b   | BF16/FP8 | 64   | 4096  | 32     | 1   | 1   | 1   | NA  | 64   | NA  | 2   | 256   | 2 |
| 15b   | BF16/FP8 | 128   | 4096  | 32     | 1   | 1   | 1   | NA  |128   | NA  | 2   | 512   | 2 |
| 15b   | BF16/FP8 | 256   | 4096  | 32     | 1   | 1   | 1   | NA  |256   | NA  | 2   | 1024   | 2 |
| 15b   | BF16/FP8 | 512   | 4096  | 32     | 1   | 1   | 1   | NA  | 512   | NA  | 2   | 2048   | 2 |

## GB300


15 billion parameter variant (FP8/BF16)

- At least 16 GPUs with at least 80GB memory each.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 15b   | BF16/FP8 | 16   | 4096  | 32     | 1   | 1   | 1   | NA  | 16   | NA  | 2   | 64   | 2 |
| 15b   | BF16/FP8 | 32   | 4096  | 32     | 1   | 1   | 1   | NA  | 32   | NA  | 2   | 128   | 2 |
| 15b   | BF16/FP8 | 64   | 4096  | 32     | 1   | 1   | 1   | NA  | 64   | NA  | 2   | 256   | 2 |
| 15b   | BF16/FP8 | 128   | 4096  | 32     | 1   | 1   | 1   | NA  |128   | NA  | 2   | 512   | 2 |
| 15b   | BF16/FP8 | 256   | 4096  | 32     | 1   | 1   | 1   | NA  |256   | NA  | 2   | 1024   | 2 |
| 15b   | BF16/FP8 | 512   | 4096  | 32     | 1   | 1   | 1   | NA  | 512   | NA  | 2   | 2048   | 2 |

## B200

15 billion parameter variant (FP8/BF16)

- At least 16 GPUs with at least 180GB memory each.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 15b   | BF16/FP8 | 16   | 4096  | 32     | 1   | 1   | 1   | NA  | 16   | NA  | 2   | 64   | 2 |
| 15b   | BF16/FP8 | 32   | 4096  | 32     | 1   | 1   | 1   | NA  | 32   | NA  | 2   | 128   | 2 |
| 15b   | BF16/FP8 | 64   | 4096  | 32     | 1   | 1   | 1   | NA  | 64   | NA  | 2   | 256   | 2 |
| 15b   | BF16/FP8 | 128   | 4096  | 32     | 1   | 1   | 1   | NA  | 128   | NA  | 2   | 512   | 2 |
| 15b   | BF16/FP8 | 256   | 4096  | 32     | 1   | 1   | 1   | NA  | 256   | NA  | 2   | 1024   | 2 |
| 15b   | BF16/FP8 | 512   | 4096  | 32     | 1   | 1   | 1   | NA  | 512   | NA  | 2   | 2048   | 2 |
| 15b   | BF16/FP8 | 1024   | 4096  | 32     | 1   | 1   | 1   | NA  | 1024   | NA  | 2   | 4096   | 2 |

# Performance Measurement and Analysis

Performance for Nemotron4 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations).

This recipe supports two cluster types: Slurm and Run:ai. We expect cluster type to have a negligible impact to performance. Application performance on a Run:ai cluster matches performance on a Slurm cluster.

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing.sh` script to analyze iterations 35-44 and calculate mean and standard deviation for reliable performance metrics. We also get the achieved GPU FLOPS via `TFLOPS_per_GPU` metric.

### Running the parse_train_timing.sh script

To analyze training timing from your experiment results, run the script from the workload directory. In an installed environment, recipe files are available under `$LLMB_INSTALL/llmb_repo` (a copy created by the installer).

```bash
# Basic usage - parses results in the directory named 'experiments' in the current folder
$LLMB_INSTALL/llmb_repo/common/parse_train_timing.sh

# Specify a different experiments directory
$LLMB_INSTALL/llmb_repo/common/parse_train_timing.sh /path/to/experiments

# Output in CSV format
$LLMB_INSTALL/llmb_repo/common/parse_train_timing.sh --format=csv

# Output in JSON format
$LLMB_INSTALL/llmb_repo/common/parse_train_timing.sh --format=json

# Show full filenames instead of shortened versions
$LLMB_INSTALL/llmb_repo/common/parse_train_timing.sh --full-names
```

Example output:
```shell
Train Step Timing and TFLOPS Analysis (iterations 35-44)
================================================================================
Experiment                                                                                  Status Time Mean (s) Time Std (s) TFLOPS_per_GPU Mean TFLOPS_per_GPU Std
------------------------------------------------------------------------------------------ -------- ------------- ------------ ------------------- ------------------
pretrain_nemotron4_340b_fp8_gpus128_tp8_pp4_cp1_vp12_mbs1_gbs512_389757                     Success        21.542        0.010             1600.74               0.78
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 4096 * 256 / 2.693 = 389371

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 389371 / 86400 = 29.7 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```
The model flops for Nemotron4 15b for GBS=1 is 3.85e14. Calculation shown [here](#mfu-formula).

E.g. NeMotron4 15b BF16 on 64x H100 GPUs (GBS=256)
```shell
peak FLOPS for H100 BF16 = 989 TFLOPS
training step time = 2.693 s
model flops = 3.85e14

MFU = 256 * 3.85e14 / 2.693 / 64 / 989e+12 = 57.82%
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

Use the **installer** referenced in the [main README](../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_nemotron4-15b`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).


## Run:ai

Only the Nemotron4 15b model has been tested on Run:ai clusters. The LLMB installer is not compatible with Run:ai environments, so you must use the setup.sh script for environment preparation instead.

### Run:ai Environment Setup

**Caution:** This section is only for users installing Run:ai. Most users should use the common installer as described above.

#### Set the environment variables
```shell
# Set the path where all artifacts will be downloaded
export LLMB_INSTALL=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)
```

**Important:** `LLMB_INSTALL` used in this step must be used when running the workload.

#### Prepare python virtual environment

The workload relies on python virtual environment in order ensure there are no conflicts between required dependencies and user's packages. We require Python 3.12.x for the workload to work.

There are multiple choices available to set up virtual environment: 
* conda
* python venv

##### Conda 

To install and activate conda virtual environment
```shell
# pick INSTALL_PATH with sufficient disk space
INSTALL_PATH=~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $INSTALL_PATH/miniconda.sh
bash $INSTALL_PATH/miniconda.sh -b -p $INSTALL_PATH/miniconda3
$INSTALL_PATH/miniconda3/bin/conda init
source ~/.bashrc

conda create -n nemo2-nemotron python=3.12
conda activate nemo2-nemotron
```

When you are finished running this benchmark you can deactivate the environment, run this command
```shell
conda deactivate
```

##### Python venv

To install and activate python venv 
```shell
python3 -m venv $LLMB_INSTALL/venvs/<venv_name>
source $LLMB_INSTALL/venvs/<venv_name>/bin/activate
```

When you are finished running this benchmark you can deactivate the environment, run this command
```shell
deactivate
```

#### Setup script

Create a install directory by running the attached setup.sh. 

```shell
# activate virtual python environment setup previously
export CLUSTER_TYPE=runai
./setup.sh
```

**Note**: output log from running `setup.sh` script may include an error about tritonclient dependency. The error can be ignored as it doesn't affect benchmark functionality. 
It would look like this:
`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tritonclient 2.51.0 requires urllib3>=2.0.7, but you have urllib3 1.26.20 which is incompatible.`

### Install Run:ai CLI tooling


- Open https://nvidia.run.ai/ web page and login to your Run:ai cluster
- Install Run:ai (CLI tool) by clicking top right "? | Researcher Command Line Interface" and following instructions
- Obtain latest kubeconfig file (e.g., my_runaicluster.kubeconfig) from cluster administrator 
- `cp my_runaicluster.kubeconfig ~/.kube/config`
- Login to your cluster via Run:ai CLI:
  - `runai login` if you have login with browser available OR
  - `runai login remote-browser` otherwise where you can authenticate from any machine and copy/paste authentication code
- `runai kubeconfig set`
- Set your current cluster via Run:ai CLI tool:
  - `runai cluster` 
  - Pick your cluster and press Enter


**Required environment variables:**

Make sure that the environment variables below have been set before submitting recipe job on Run:AI cluster.
- `RUN_CONF_MOUNTS` - PVCs that will be mounted into your environment. This should be in the form `name:path:k8s-claimName`. Confirm that the path to PVC exists in Run:ai UI under "Assets | Data Sources"
  e.g., `export RUN_CONF_MOUNTS="workspace:/nemo2-workspace:nemo2-workspace-project-hprdj"`
- `BASE_URL`- NVIDIA Run:ai API url to use for experiment. Should look like `https://<base-url>/api/v1`. 
  e.g., `export BASE_URL=https://nvidia.run.ai/api/v1`
- `APP_ID` - name of NVIDIA Run:ai Application. The APP_ID value should be listed in Run:ai web UI under "Access | Applications"
  e.g., `export APP_ID=llmb-qa`
- `APP_SECRET` - NVIDIA Run:ai Application secret. Make sure the value is correct before launching the job. Contact your Run:ai cluster admin to provide you with latest application secret
- `PROJECT_NAME` - NVIDIA Run:ai Project to run the experiment in
  e.g., `llmb-testing`
- `PVC_DIR` - path to nemo-run dir in the mounted PVC. Should look like `/mount/nemo-run`
  e.g., `export PVC_DIR="/nemo2-workspace/nemo-run"`

These parameters must be set to launch your job on Run:ai. They can be set by exporting the environment variables.


# Prepare Dataset
Since Nemotron4 training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended, SLURM clusters only)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run submit -w pretrain_nemotron4-15b -s 15b --dtype fp8 --scale 128
```

For more details on llmb-run usage, see the [llmb-run documentation](../cli/llmb-run/README.md).

## Direct Method (SLURM or Run:ai)

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `${LLMB_WORKLOAD}/experiments/` folder.


**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/nemotron4-15b/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [MODEL_SIZE=<size>] [CLUSTER_TYPE=<type>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
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
- `DTYPE`: Precision format (default: `fp8`)
  - `fp8` - FP8 precision
  - `bf16` - BFloat16 precision
- `MODEL_SIZE`: Model variant (default: `15b`)
  - `15b` - 15 billion parameter model
- `CLUSTER_TYPE`: Cluster management system (default: `slurm`)
  - `slurm` - Slurm workload manager 
  - `runai` - Run:AI platform
- `ADDITIONAL_SLURM_PARAMS`: Additional SLURM parameters (optional)
  - Format: Semicolon-separated key=value pairs (use semicolons when values contain commas)
  - Example: `"nodelist=node001,node002;constraint=gpu"`

### SLURM Example Commands

Train Nemotron4 with default settings (FP8) on 128 H100 GPUs:
```shell
JOB_TOTAL_GPUS=16 GPU_TYPE=h100 ./launch.sh
```

Train Nemotron4 15B with BF16 precision on 128 H100 GPUs:
```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=h100 DTYPE=bf16 ./launch.sh
```

### SLURM Node Specification Examples

Train Nemotron4 15B on specific nodes:
```shell
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002" JOB_TOTAL_GPUS=16 GPU_TYPE=h100 ./launch.sh
```

Train Nemotron4 15B excluding specific nodes:
```shell
ADDITIONAL_SLURM_PARAMS="exclude=node003,node004" JOB_TOTAL_GPUS=128 GPU_TYPE=h100 ./launch.sh
```

Train with multiple SLURM parameters:
```shell
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002;mem=64GB;constraint=gpu" JOB_TOTAL_GPUS=32 GPU_TYPE=h100 ./launch.sh
```

Train using a SLURM reservation:
```shell
ADDITIONAL_SLURM_PARAMS="reservation=my_reservation" JOB_TOTAL_GPUS=64 GPU_TYPE=h100 ./launch.sh
```

Train with reservation and additional constraints:
```shell
ADDITIONAL_SLURM_PARAMS="reservation=gpu_reservation;constraint=gpu" JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 ./launch.sh
```


### Run:ai Example Commands

Train Nemotron4 with default settings (FP8) on 16 H100 GPUs on Run:ai cluster:
```shell
CLUSTER_TYPE=runai JOB_TOTAL_GPUS=16 GPU_TYPE=h100 MODEL_SIZE=15b ./launch.sh
```

Train Nemotron4 with BF16 precision on 64 H100 GPUs on Run:ai cluster:
```shell
CLUSTER_TYPE=runai JOB_TOTAL_GPUS=64 GPU_TYPE=h100 MODEL_SIZE=15b DTYPE=bf16 ./launch.sh
```

Once the job has been successfully submitted you can monitor its progress via Run:ai Web UI under "Workload Manager" section.


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

The `<experiment_name>` typically follows the pattern: `pretrain_nemotron4_<size>_<dtype>_<scale>_<config>`

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
ENABLE_PROFILE=true JOB_TOTAL_GPUS=128 DTYPE=fp8 GPU_TYPE=gb200 ./launch.sh
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
ENABLE_PROFILE=true ENABLE_GPU_METRICS=true JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 DTYPE=fp8 ./launch.sh
```

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

# Run With Checkpoints

Checkpoint save and load can be enabled for this workload in order to measure the impact of storage on checkpointing operations. The additional collected metrics are: time to save a checkpoint and time to load a checkpoint. 

## Save Checkpoint

Save checkpoint feature works for both Nemotron4 15b size with either FP8 or BF16 precision. Make sure your file system has sufficient disk space to accomodate checkpoint sizes below:

| Model | Checkpoint Size | Minimum Tested  Scale H100 | Minimum Tested Scale GB200 | Minimum Tested Scale B200 |
| :---: | :---: |  :---: | :---: | :---: |
| 15b | ~204 GB | 16 | 8 | 8 |

### How to enable
To save the checkpoints after pretraining nemotron4 model for `max_steps`, you need to set environment variable `ENABLE_CHECKPOINT=true`. At the end of the pretraining the checkpoints will be saved in the  `${LLMB_WORKLOAD}/experiments` folder.

```shell
experiment_name = pretrain_nemotron4_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}
timestamp = date '+%s'
Example directory where checkpoints are saved is ${LLMB_WORKLOAD}/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/
```
Command to run nemotron4 with checkpoint save enabled
```shell
ENABLE_CHECKPOINT=true DTYPE=<precision> MODEL_SIZE=<size> JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> ./launch.sh
```

### How to validate
- Check `${LLMB_WORKLOAD}/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/*/weights` folder that it contains *.distcp files
- Check job output log-*.out file (see Training section for reference) for entries like
  ```
  [NeMo I 2025-04-11 14:48:45 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 50 : Start time: 1744408121.151s : Save duration: 4.389s
  ```

## Load Checkpoint

Load checkpoint feature works successfully at the following scales:

| Model | Minimum Tested Scale H100 |  Minimum Tested Scale GB200 |  Minimum Tested Scale B200 |
| :---: | :---: | :---: | :---: |
| 15b | 64 | 8 | 8 |

**Note**:
- Running load checkpointing feature at other scales may run into CUDA OOM errors. 

### How to enable
To resume training from saved checkpoints, you need to set `LOAD_CHECKPOINT_PATH=<path_to_checkpoint_directory>` environment variable. Make sure the checkpoint files are under the `${LLMB_WORKLOAD}/experiments` directory and `LOAD_CHECKPOINT_PATH` variable is set to parent folder of the `weights` directory containing distributed checkpoint files with extension `*.distcp`.

E.g., if the checkpoint was saved under `${LLMB_WORKLOAD}/experiments/pretrain_nemotron4_15b_fp8_64/pretrain_nemotron4_15b_fp8_64_<timestamp>/pretrain_nemotron4_15b_fp8_64/code/nemo_experiments/default/checkpoints/*/weights` then set the environment variable to a directory one level higher: 

`LOAD_CHECKPOINT_PATH=${LLMB_WORKLOAD}/experiments/pretrain_nemotron4_15b_fp8_64/pretrain_nemotron4_15b_fp8_64_<timestamp>/pretrain_nemotron4_15b_fp8_64/code/nemo_experiments/default/checkpoints/default--None=0.0000-epoch=0-consumed_samples=12800.0`

The scripts will restore configuration from the checkpoint and resume training process. Training will run for 1 step after checkpoint has been loaded.

```shell
LOAD_CHECKPOINT_PATH=<your_path_to_checkpoint_directory> JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> DTYPE=<precision> MODEL_SIZE=<size> ./launch.sh
```

### How to validate

To validate that checkpoint was loaded successfully look for the entry like below in the main job log-*.out file (see Training section for reference):

```
[NeMo I 2025-04-11 14:46:18 nemo_logging:393] Global Checkpoint Load : Rank : 0 : Start time : 1744407969.270s : Time spent in load_checkpoint: 9.712s
```

# FAQ

## Failure detected by watchdog

For GB200 you may see the following error message
```shell
[rank368]:[E808 04:21:41.160918398 ProcessGroupNCCL.cpp:655] [Rank 368] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=5, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600001 milliseconds before timing out.
[rank368]:[E808 04:21:41.161005534 ProcessGroupNCCL.cpp:2299] [PG ID 0 PG GUID 0(default_pg) Rank 368]  failure detected by watchdog at work sequence id: 5 PG status: last enqueued work: 5, last completed work: 4
[rank368]:[E808 04:21:41.161011710 ProcessGroupNCCL.cpp:693] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[rank368]:[E808 04:21:41.161045406 ProcessGroupNCCL.cpp:2147] [PG ID 0 PG GUID 0(default_pg) Rank 368] First PG on this rank to signal dumping.
```

To fix, try running with TP_COMM_OVERLAP disabled like so:  
```bash
TP_COMM_OVERLAP=False llmb-run submit -w pretrain_nemotron4-15b -s 15b --dtype bf16 --scale 16
```

# MFU formula

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = 12 * (number of layers) * (hidden size)^2 * (1 + (number of query groups) / (number of heads) + (sequence length) / (hidden size))
    mlp flops = 12 * (number of layers) * (hidden size) * (ffn hidden size)
    embedding flops = 6 * (vocab size) * (hidden size) 

Nemotron4 15b calculation:
    sequence length = 4096
    number of layers = 32
    hidden size = 6144
    ffn hidden size = 24576
    number of heads = 48
    number of query groups = 8
    vocab size = 256000 
    attention flops = 12 * 32 * 6144^2 + 12 * 32 * 6144^2 * 8 / 48 + 12 * 32 * 6144 * 4096 = 14,495,514,624 + 2,415,919,104 + 9,663,676,416 = 26,575,110,144
    mlp flops = 12 * 32 * 6144 * 24576 = 57,982,058,496
    embedding flops = 6 * 256000 * 6144 = 9,437,184,000

    model flops = 4096 * (26,575,110,144 + 57,982,058,496 + 9,437,184,000) = 4096 * 93,994,352,640 = 3.85e14

```
**Note**:
Per-tensor delayed scaling recipe is used for FP8 training here.
