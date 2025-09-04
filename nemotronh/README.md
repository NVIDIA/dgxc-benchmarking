# Overview

This recipe contains information and scripts to produce performance results for the Nemotron-H pre-training workloads. The scripts help perform environment setup and launch benchmark jobs.

The H100 recipes listed below progressively increase GPU count, with configurations weak-scaled to match.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 56b   | FP8   | 32   | 8192  | 118   | 8     | 1     | 1     | NA    | 4     | NA    | 1     | 96    | 24    |
| 56b   | FP8   | 64   | 8192  | 118   | 8     | 1     | 1     | NA    | 8     | NA    | 1     | 192   | 24    |
| 56b   | FP8   | 128  | 8192  | 118   | 8     | 1     | 1     | NA    | 16    | NA    | 1     | 384   | 24    |
| 56b   | FP8   | 256  | 8192  | 118   | 8     | 1     | 1     | NA    | 32    | NA    | 1     | 768   | 24    |
| 56b   | FP8   | 512  | 8192  | 118   | 8     | 1     | 1     | NA    | 64    | NA    | 1     | 1536  | 24    |
| 56b   | FP8   | 1024 | 8192  | 118   | 8     | 1     | 1     | NA    | 128   | NA    | 1     | 3072  | 24    |
| 56b   | FP8   | 2048 | 8192  | 118   | 8     | 1     | 1     | NA    | 256   | NA    | 1     | 6144  | 24    |

# Expected Performance

Performance for Nemotron-H training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in the main training log file [see Output Locations](#output-locations).

Since the early training steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing.sh` script to analyze iterations 11-44 and calculate mean and standard deviation for reliable performance metrics.

**Note:** TFLOPS_per_GPU not currently supported for this model, values returned by the results parser are **invalid**.

```shell
# Run the parse_train_timing script to analyze all experiments
common/parse_train_timing.sh $LLMB_WORKLOAD/experiments

# Example output:
Train Step Timing Analysis (iterations 11-44)
==================================================================
Experiment                                                                         Status Time Mean (s) Time Std (s) TFLOPS_per_GPU Mean TFLOPS_per_GPU Std
-------------------------------------------------------------------------------- -------- ------------- ------------ ------------------- ------------------
pretrain_nemotronh_56b_fp8_gpus128_tp8_pp1_cp1_vpNone_mbs1_gbs768                 Success        15.858        1.262              532.39             472.74
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 768 / 15.892  = 395888

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 395895 / 86400 = 29.24 days


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is **1979** TFLOPS.

The model flops for Nemotron-H 56b for GBS=1 is 2.816e15. Calculation shown [here](#notes).

E.g. Nemotron-H 56b FP8 on 256x H100 GPUs (GBS=768)
```shell
peak FLOPS for H100 FP8 = 1979 TFLOPS
training step time = 15.892 s
model flops = 2.816e15

MFU = 768 * 2.816e15 / 15.892 / 256 / 1979e+12 = 26.86%
```

| Nemotron-H 56b FP8 | 32x H100 GPUs | 64x H100 GPUs | 128x H100 GPUs | 256x H100 GPUs  |  512x H100 GPUs  | 1024x H100 GPUs  | 2048x H100 GPUs  |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
Training step time (seconds per step)|15.119|15.058|15.496|15.892|15.586|15.678|15.723|
Throughput in tokens per second |52016|104454|203003|395888|807321|1605168|3201148|
Model flops utilization|28.23%|28.35%|27.55%|26.86%|27.39%|27.23%|27.15%|
Time to train 1T tokens in days|222.51|110.81|57.01|29.24|14.34|7.21|3.62|

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

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../README.md):

Note, that a new directory layout and key variables are now used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretraining_nemotronh`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

If you are an advanced user and need to perform a manual environment setup (e.g., for debugging or custom environments), see the [Advanced/Manual Environment Setup](#advancedmanual-environment-setup) section at the end of this file.

**Migration Note:**
If you previously used `STAGE_PATH`, replace it with `LLMB_INSTALL` (top-level) and `LLMB_WORKLOAD` (workload-specific). All output, logs, and checkpoints are now under `LLMB_WORKLOAD`.

# Prepare Dataset
Since Nemotron-H training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run single -w pretrain_nemotronh -s 56b --dtype fp8 --scale 256

# Example with different scale
llmb-run single -w pretrain_nemotronh -s 56b --dtype fp8 --scale 1024
```

For more details on llmb-run usage, see the [llmb-run documentation](../llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/nemotronh/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> [DTYPE=<precision>] [MODEL_SIZE=<size>] [GPU_TYPE=<type>] ./launch.sh
```

### Environment Variables

**Required:**
- `JOB_TOTAL_GPUS`: Number of GPUs to use

**Optional:**
- `DTYPE`: Precision format (fixed: `fp8`)
  - `fp8` - FP8 precision (only supported precision)
- `MODEL_SIZE`: Model variant (fixed: `56b`)
  - `56b` - 56 billion parameter model (only supported size)
- `GPU_TYPE`: Type of GPU hardware (fixed: `h100`)
  - `h100` - NVIDIA H100 GPUs (only supported type)

**Note:** This workload only supports:
- FP8 precision
- 56B model size  
- H100 GPUs

### Example Commands

Train Nemotron-H 56B with FP8 precision on 256 H100 GPUs:
```shell
# activate virtual environment
JOB_TOTAL_GPUS=256 ./launch.sh
```

Train on 1024 H100 GPUs:
```shell
JOB_TOTAL_GPUS=1024 ./launch.sh
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

The `<experiment_name>` typically follows the pattern: `pretraining_nemotronh_56b_<dtype>_<scale>_<config>`

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
* **Filename format:** `profile_{SLURM_JOBID}_{SLURM_NODEID}_{SLURM_PROCID}.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true JOB_TOTAL_GPUS=256 ./launch.sh
```
### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

## Run NCCL Trace (For Debugging)

NCCL traces can be a powerful tool for understanding communication patterns within your benchmarking job. They provide detailed information on the types of NCCL calls being made (like AllReduce, Broadcast, etc.) and the size of the messages being exchanged.

**Important:** This feature is primarily intended for **troubleshooting and debugging purposes only**. It is not typically used during normal benchmark runs.

To collect NCCL Trace information, set the environment variable `ENABLE_NCCLTRACE=true` when submitting your job:

**Defaults for Tracing:**
*   **Duration:** Due to the large file sizes generated, tracing is limited to the first 5 steps of the job by default.
*   **Output Location:** NCCL trace information is included directly within the standard job log file (see Output Locations)

**Example command:**

```shell
ENABLE_NCCLTRACE=true JOB_TOTAL_GPUS=256 ./launch.sh
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
mlp_block_flops = (2 * seq_len * hidden_size * intermediate_size) * num_mlp_layers

attn_block_flops = (
		2 * seq_len * hidden_size * num_attention_heads * attention_head_dim +
		2 * seq_len * hidden_size * num_key_value_heads * attention_head_dim +
		2 * seq_len * seq_len * hidden_size
	) * num_attn_layers

mamba_block_flops = (
		seq_len * hidden_size * (mamba_num_heads * mamba_head_dim * 3 + n_groups * ssm_state_size * 2 + mamba_num_heads) +
		seq_len * hidden_size * expand * chunk_size +
		2 * seq_len * hidden_size * expand * ssm_state_size +
		seq_len * chunk_size * ssm_state_size * n_groups
	) * num_mamba_layers

lm_head_flops = seq_len * hidden_size * vocab_size

total_flops = (mlp_block_flops + attn_block_flops + mamba_block_flops + lm_head_flops) * 6

Nemotron-H 56b calculation:
	seq_len = 8192
	hidden_size = 8192
	intermediate_size = 32768
	num_mlp_layers = 54
	num_mamba_layers = 54
	num_attn_layers = 10
	num_attention_heads = 64
	num_key_value_heads = 8
	attention_head_dim = 128
	mamba_num_heads = 256
	mamba_head_dim = 64
	expand = 2
	chunk_size = 256
	ssm_state_size = 256
	n_groups = 8
	vocab_size = 131072

	mlp_block_flops = (2 * 8192 * 8192 * 32768) * 54 = 237,494,511,599,616
	attn_block_flops = (2 * 8192 * 8192 * 64 * 128 + 2 * 8192 * 8192 * 8 * 128 + 2 * 8192 * 8192 * 8192) * 10 = 23,364,622,090,240
	mamba_block_flops = (8192 * 8192 * (256 * 64 * 3 + 8 * 256 * 2 + 256) + 8192 * 8192 * 2 * 256 + 2 * 8192 * 8192 * 2 * 256 + 8192 * 256 * 256 * 8) * 54 = 199,690,209,460,224
	lm_head_flops = 8192 * 8192 * 131072 = 8,796,093,022,208
	total_flops = (237,494,511,599,616 + 23,364,622,090,240 + 199,690,209,460,224 + 8,796,093,022,208) * 6 = 2.816e15

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

conda create -n nemo2-nemotronh python=3.12
conda activate nemo2-nemotronh
```

When you are finished running this benchmark you can deactivate the environment, run this command
```shell
conda deactivate
```

#### Python venv

To install and activate python venv 
```shell
python3 -m venv $LLMB_INSTALL/venvs/<venv_name>
source $LLMB_INSTALL/venvs/<venv_name>/bin/activate
```

When you are finished running this benchmark you can deactivate the environment, run this command
```shell
deactivate
```

### Setup script

Create a install directory by running the attached setup.sh. The script converts the docker image to a ```.sqsh``` file under the $LLMB_INSTALL/images folder and installs required packages to the python environment to enable NeMo-Run launcher functionality.

**Important:** Make sure the previous step has been completed and python virtual environment is active. Run the setup script using the following command.

**SLURM:**

```shell
# activate virtual python environment setup previously
./setup.sh
```
To fetch the image ensure your virtual environment has been **deactivated**, then run:

```shell
srun --account ${SBATCH_ACCOUNT} --partition ${SBATCH_PARTITION} bash -c "enroot import --output ${LLMB_INSTALL}/images/nvidia+nemo+25.04.01.sqsh docker://nvcr.io#nvidia/nemo:25.04.01"
```
**Note**: output log from running `setup.sh` script may include an error about tritonclient dependency. The error can be ignored as it doesn't affect benchmark functionality. 
It would look like this:
`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tritonclient 2.51.0 requires urllib3>=2.0.7, but you have urllib3 1.26.20 which is incompatible.`