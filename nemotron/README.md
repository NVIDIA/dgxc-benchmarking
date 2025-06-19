



# Overview

This recipe contains information and scripts to produce performance results for the Nemotron 4 pre-training workloads. The scripts help perform environment setup and launch benchmark jobs.

# H100s specifications

15 billion parameter variant (FP8/BF16)

- At least 16 GPUs with at least 80GB memory each.

340 billion parameter variant (FP8/BF16)

- At least 128 GPUs with at least 80GB memory each.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 15b   | BF16/FP8  | 16   | 4096  | 32     | 2   | 1   | 1   | NA  | 8   | NA  | 2   | 64   | 4 |


| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 340b | BF16/FP8  | 256  | 4096   | 96     | 8   | 8   | 1   | NA  | 4   | 12  | 1   | 64   | 16 |

# GB200 specifications

340 billion parameter variant (FP8/BF16)

- At least 128 GPUs with at least 80GB memory each.
- The GB200 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | VP  | MBS | GBS | DP   | GA  |
|------|:--------:|:------:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 340b  | BF16/FP8 | 128  | 4096   | 96     | 8   | 4   | 1   | 12  | 1   | 32   |4    | 8   |
| 340b  | BF16/FP8 | 256  | 4096   | 96     | 8   | 4   | 1   | 12  | 1   | 64   |8    | 8   |
| 340b  | BF16/FP8 | 512  | 4096   | 96     | 8   | 4   | 1   | 12  | 1   | 128  |16   | 8   |

- Strong Scaling, where GBS remains constant across increasing GPU counts and reducing gradient accumulation. 

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | VP  | MBS | GBS | DP  | GA   |
|------|:--------:|:------:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 340b  | BF16/FP8 | 128  | 4096   | 96     | 8   | 4   | 1   | 12  | 1   | 512 | 4   | 128  |
| 340b  | BF16/FP8 | 256  | 4096   | 96     | 8   | 4   | 1   | 12  | 1   | 512 | 8   | 64   |
| 340b  | BF16/FP8 | 512  | 4096   | 96     | 8   | 4   | 1   | 12  | 1   | 512 | 16  | 32   |



# Expected Performance

Performance for Nemotron4 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations).

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing.sh` script to analyze iterations 11-44 and calculate mean and standard deviation for reliable performance metrics. We also get the achieved GPU FLOPS via `TFLOPS_per_GPU` metric.

```shell
Train Step Timing and TFLOPS Analysis (iterations 11-44)
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
The model flops for Nemotron4 15b for GBS=1 is 3.85e14. Calculation shown [here](#notes).

E.g. NeMotron4 15b BF16 on 64x H100 GPUs (GBS=256)
```shell
peak FLOPS for H100 BF16 = 989 TFLOPS
training step time = 2.693 s
model flops = 3.85e14

MFU = 256 * 3.85e14 / 2.693 / 64 / 989e+12 = 57.82%
```

# H100 Performance

The peak theoretical throughput for H100 FP8 is **1979** TFLOPS and for H100 BF16 is **989** TFLOPS.


| Nemotron4 15b BF16 H100 | 16x H100 GPUs  |
|---|:---:|
Training step time (seconds per step)|2.746|
Throughput in tokens per second |95463 |
Model flops utilization|56.7% |
Time to train 1T tokens in days|121 |

| Nemotron4 15b FP8 H100 | 16x H100 GPUs |
|---|:---:|
Training step time (seconds per step)| 1.988|
Throughput in tokens per second |131863 |
Model flops utilization| 39.1%|
Time to train 1T tokens in days|88 |

| Nemotron4 340b BF16 H100 | 256x H100 GPUs |
|---|:---:|
Training step time (seconds per step)|4.606|
Throughput in tokens per second |56913|
Model flops utilization|47.3%|
Time to train 1T tokens in days|203|

| Nemotron4 340b FP8 H100 | 256x H100 GPUs |
|---|:---:|
Training step time (seconds per step)|3.107|
Throughput in tokens per second |84372|
Model flops utilization|35%|
Time to train 1T tokens in days|137|


# GB200 Performance

The peak theoretical throughput for GB200 FP8 is **4.9** PFLOPS and for GB200 BF16 is **2.45** PFLOPS.

| Nemotron 340b BF16 GB200 | 128x GB200 GPUs (GBS=32) | 256x GB200 GPUs (GBS=64) | 512x GB200 GPUs (GBS=128) |
|---|:---:|:---:|:---:|
| Training step time (seconds per step) | 2.189 | 2.19 | 2.229 |
| Throughput in tokens per second       | 59877 |119700| 235212|  
| Model flops utilization               | 40.2% | 40.2%| 39.4% |  
| Time to train 1T tokens in days       | 193   | 96   | 49    |  

| Nemotron 340b FP8 GB200 | 128x GB200 GPUs (GBS=32) | 256x GB200 GPUs (GBS=64)  | 512x GB200 GPUs (GBS=128) |
|---|:---:|:---:|:---:|
| Training step time (seconds per step) | 1.483  | 1.517  | 1.549 |  
| Throughput in tokens per second       | 88383  | 172804 | 338468|
| Model flops utilization               | 29.7%  | 28.9%  | 28.4% | 
| Time to train 1T tokens in days       | 130    | 67     | 34    |  


| Nemotron 340b BF16 GB200 Strong Scaling | 128x GB200 GPUs (GBS=512) Strong Scaling | 256x GB200 GPUs (GBS=512) Strong Scaling | 512x GB200 GPUs (GBS=512) Strong Scaling |
|---|:---:|:---:|:---:|
| Training step time (seconds per step) |32.66 | 16.58 | 8.468 |
| Throughput in tokens per second       |64211 | 126486|247656 |
| Model flops utilization               |43%   | 42.4% | 41.5% |
| Time to train 1T tokens in days       |180   | 92    | 47    |


| Nemotron 340b FP8 GB200 Strong Scaling | 128x GB200 GPUs (GBS=512) Strong Scaling | 256x GB200 GPUs (GBS=512) Strong Scaling | 512x GB200 GPUs (GBS=512) Strong Scaling |
|---|:---:|:---:|:---:|
| Training step time (seconds per step) |21.83  | 11.21 | 5.668 |
| Throughput in tokens per second       |96067  | 187078| 369998|
| Model flops utilization               |32.2%  | 31.4% | 31%   |
| Time to train 1T tokens in days       |120    | 62    | 31    |

# Prerequisites

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

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../README.md):

Note, that a new directory layout and key variables are now used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretraining_nemotron`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

If you are an advanced user and need to perform a manual environment setup (e.g., for debugging or custom environments), see the [Advanced/Manual Environment Setup](#advancedmanual-environment-setup) section at the end of this file.

**Migration Note:**
If you previously used `STAGE_PATH`, replace it with `LLMB_INSTALL` (top-level) and `LLMB_WORKLOAD` (workload-specific). All output, logs, and checkpoints are now under `LLMB_WORKLOAD`.

# Prepare Dataset
Since Nemotron4 training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run single -w pretraining_nemotron -s 15b --dtype fp8 --scale 128

# Example with 340B model
llmb-run single -w pretraining_nemotron -s 340b --dtype bf16 --scale 256
```

For more details on llmb-run usage, see the [llmb-run documentation](../llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `${LLMB_WORKLOAD}/experiments/` folder.


**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/nemotron/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> [DTYPE=<precision>] [MODEL_SIZE=<size>] [GPU_TYPE=<type>] [STRONG_SCALING=<bool>] ./launch.sh
```

### Environment Variables

**Required:**
- `JOB_TOTAL_GPUS`: Number of GPUs to use

**Optional:**
- `DTYPE`: Precision format (default: `fp8`)
  - `fp8` - FP8 precision
  - `bf16` - BFloat16 precision
- `MODEL_SIZE`: Model variant (default: `15b`)
  - `15b` - 15 billion parameter model
  - `340b` - 340 billion parameter model
- `GPU_TYPE`: Type of GPU hardware (default: `gb200`)
  - `gb200` - NVIDIA GB200 GPUs
  - `h100` - NVIDIA H100 GPUs
- `STRONG_SCALING`: Scaling behavior (default: `false`)
  - `true` - Keep global batch size constant when scaling GPUs
  - `false` - Scale global batch size with GPU count

### Example Commands

Train Nemotron4 with default settings (15B, FP8, GB200, Slurm) on 128 GPUs:
```shell
# activate virtual environment
JOB_TOTAL_GPUS=128 ./launch.sh
```

Train Nemotron4 15B with BF16 precision on 128 H100 GPUs using Slurm:
```shell
JOB_TOTAL_GPUS=128 DTYPE=bf16 GPU_TYPE=h100 ./launch.sh
```

Train Nemotron4 340B with FP8 precision on 256 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=256 MODEL_SIZE=340b ./launch.sh
```

Train Nemotron4 340B, FP8 precision with strong scaling enabled on 256 GB200 GPUs:
```shell
JOB_TOTAL_GPUS=256 MODEL_SIZE=340b STRONG_SCALING=true ./launch.sh
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

The `<experiment_name>` typically follows the pattern: `pretraining_nemotron_<size>_<dtype>_<scale>_<config>`

**Key files:**
- `log-<experiment_name>.out` - Contains training step timing and performance metrics analyzed by `parse_train_timing.sh`
- `nsys_profile/` - Contains profiling traces when `ENABLE_PROFILE=true`

# Run Nsight Profiling
To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 46-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Profiling job details:
* **MPI Ranks:** 0-8
* **Job Steps:** 46-50
* **Output Location:** Profiling output saved alongside training results (see Output Locations)
* **Filename format:** `profile_${SLURM_JOB_ID}_nodeId_rankId.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true JOB_TOTAL_GPUS=128 DTYPE=fp8 MODEL_SIZE=340b GPU_TYPE=gb200 ./launch.sh
```

### Customizing profiling behavior:
* Specify job steps to profile:
	* `PROFILE_START_STEP`: start profiling on this job step.
	- Default: 46
	* `PROFILE_STOP_STEP`: stop profiling on this job step.
	- Default: 50

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

Save checkpoint feature works for both Nemotron4 15b and 340b sizes with either FP8 or BF16 precision. Make sure your file system has sufficient disk space to accomodate checkpoint sizes below:

| Model | Checkpoint Size | Supported Scales |
| :---: | :---: |  :---: |
| 15b | ~204 GB | 16-2048 |
| 340b | ~4.4 TB | 512-2048 |

### How to enable
To save the checkpoints after pretraining nemotron4 model for `max_steps`, you need to set environment variable `ENABLE_CHECKPOINT=true`. At the end of the pretraining the checkpoints will be saved in the  `${LLMB_WORKLOAD}/experiments` folder.

```shell
experiment_name = pretrain_nemotron4_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}
timestamp = date '+%s'
Example directory where checkpoints are saved is ${LLMB_WORKLOAD}/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/
```
Command to run nemotron4 with checkpoint save enabled
```shell
ENABLE_CHECKPOINT=true DTYPE=<fp8,bf16> MODEL_SIZE=<15b/340b> JOB_TOTAL_GPUS=<16,..,2048> GPU_TYPE=<gb200,h100> STRONG_SCALING=<true/false> ./launch.sh
```

### How to validate
- Check `${LLMB_WORKLOAD}/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/*/weights` folder that it contains *.distcp files
- Check job output log-*.out file (see Training section for reference) for entries like
  ```
  [NeMo I 2025-04-11 14:48:45 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 50 : Start time: 1744408121.151s : Save duration: 4.389s
  ```

## Load Checkpoint

Load checkpoint feature works successfully at the following scales:

| Model | Minimum Tested Scale |
| :---: | :---: |
| 15b | 64 |
| 340b | 512 |

**Note**:
- Running load checkpointing feature at other scales may run into CUDA OOM errors. 

### How to enable
To resume training from saved checkpoints, you need to set `LOAD_CHECKPOINT_PATH=<path_to_checkpoint_directory>` environment variable. Make sure the checkpoint files are under the `${LLMB_WORKLOAD}/experiments` directory and `LOAD_CHECKPOINT_PATH` variable is set to parent folder of the `weights` directory containing distributed checkpoint files with extension `*.distcp`.

E.g., if the checkpoint was saved under `${LLMB_WORKLOAD}/experiments/pretrain_nemotron4_15b_fp8_64/pretrain_nemotron4_15b_fp8_64_<timestamp>/pretrain_nemotron4_15b_fp8_64/code/nemo_experiments/default/checkpoints/*/weights` then set the environment variable to a directory one level higher: 

`LOAD_CHECKPOINT_PATH=${LLMB_WORKLOAD}/experiments/pretrain_nemotron4_15b_fp8_64/pretrain_nemotron4_15b_fp8_64_<timestamp>/pretrain_nemotron4_15b_fp8_64/code/nemo_experiments/default/checkpoints/default--None=0.0000-epoch=0-consumed_samples=12800.0`

The scripts will restore configuration from the checkpoint and resume training process. Training will run for 1 step after checkpoint has been loaded.

```shell
LOAD_CHECKPOINT_PATH=<your_path_to_checkpoint_directory> DTYPE=<fp8,bf16> MODEL_SIZE=<15b/340b> JOB_TOTAL_GPUS=<16,..,2048> GPUS_PER_NODE=<2,8> STRONG_SCALING=<true/false> ./launch.sh
```

### How to validate

To validate that checkpoint was loaded successfully look for the entry like below in the main job log-*.out file (see Training section for reference):

```
[NeMo I 2025-04-11 14:46:18 nemo_logging:393] Global Checkpoint Load : Rank : 0 : Start time : 1744407969.270s : Time spent in load_checkpoint: 9.712s
```

# Notes

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

Nemotron4 340b calculation:
    sequence length = 4096
    number of layers = 96
    hidden size = 18432
    vocab size = 256000 
    ffn hidden size = 73728
    number of heads = 96
    number of query groups = 8
    attention flops = 12 * 96 * 18432^2 + 12 * 96 * 18432^2 * 8 / 96 + 12 * 96 * 18432 * 4096 = 391,378,894,848 + 32,614,907,904 + 86,973,087,744 = 510,966,890,496
    mlp flops = 12 * 96 * 18432 * 73728 = 1,565,515,579,392
    embedding flops = 6 * 256000 * 18432 = 28,311,552,000

    model flops = 4096 * (510,966,890,496 + 1,565,515,579,392 + 28,311,552,000) = 8.62124e15

```


# Advanced/Manual Environment Setup

> **Caution:** This section is for advanced users who need to manually set up the environment. Most users should use the common installer as described above.

### Set the environment variables
```shell
# Set the path where all artifacts will be downloaded
export LLMB_INSTALL=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)
```

**Important:** `LLMB_INSTALL` used in this step must be used when running the workload.

### Prepare python virtual environment

The workload relies on python virtual environment in order ensure there are no conflicts between required dependencies and user's packages. We require Python 3.12.x for the workload to work.

There are multiple choices available to set up virtual environment: 
* conda
* python venv

#### Conda 

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

#### Python venv

To install and activate python venv 
```shell
python3 -m venv $LLMB_INSTALL/venv/<venv_name>
source $LLMB_INSTALL/venv/<venv_name>/bin/activate
```

When you are finished running this benchmark you can deactivate the environment, run this command
```shell
deactivate
```

### Setup script

Create a install directory by running the attached setup.sh.

***Important***

The setup script must be run while you are in your virtual environment. 
This script clones the ***NeMo*** repository, installs ***megatron-core*** and ***nemo run***, and installs required packages to the python environment to enable NeMo-Run launcher functionality. 

Be sure to **deactivate** your virtual environment before importing the container image to be used to run the workload. 

**SLURM:**

```shell
# activate virtual python environment setup previously
export CLUSTER_TYPE=slurm
# virtual environment must be active
./setup.sh
# deactivate virtual environment
srun --account ${SBATCH_ACCOUNT} --partition ${SBATCH_PARTITION} bash -c "enroot import --output ${LLMB_INSTALL}/images/nvidia+nemo+25.04.00.sqsh docker://nvcr.io#nvidia/nemo:25.04.00"
```
***Important***
The above srun command converts the docker image to a ```.sqsh``` file under the $LLMB_INSTALL/images folder. Your virtual env must be deactivated before running this command. 


**Note**: output log from running `setup.sh` script may include an error about tritonclient dependency. The error can be ignored as it doesn't affect benchmark functionality. 
It would look like this:
`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tritonclient 2.51.0 requires urllib3>=2.0.7, but you have urllib3 1.26.20 which is incompatible.`
