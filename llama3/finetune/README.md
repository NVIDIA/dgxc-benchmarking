# Overview

This recipe contains information and scripts to produce performance results for the LLAMA3 70B finetuning (LoRa) workload. The scripts help perform environment setup and launch benchmark jobs. It supports both BF16 and FP8 precisions.

## GB300

| Model Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :--------: | :-------: | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|    70b     | BF16/FP8  |  8   |  4096  |   80   |  1  |  1  |  1  |  8  | N/A |  1  | 32  |  4  |
|    70b     | BF16/FP8  |  16  |  4096  |   80   |  1  |  1  |  1  | 16  | N/A |  1  | 64  |  4  |

## GB200

| Model Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :--------: | :-------: | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|    70b     |   BF16    |  8   |  2048  |   80   |  1  |  1  |  1  |  8  | N/A |  1  | 64  |  8  |
|    70b     |    FP8    |  8   |  4096  |   80   |  1  |  2  |  1  |  4  | N/A |  1  | 32  |  8  |
|    70b     |   BF16    |  16  |  2048  |   80   |  1  |  1  |  1  | 16  | N/A |  1  | 128 |  8  |
|    70b     |    FP8    |  16  |  4096  |   80   |  1  |  2  |  1  |  8  | N/A |  1  | 64  |  8  |

## H100

| Model Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :--------: | :-------: | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|    70b     |   BF16    |  8   |  4096  |   80   |  1  |  4  |  1  |  2  | 20  |  1  | 32  | 16  |
|    70b     |    FP8    |  8   |  4096  |   80   |  2  |  4  |  1  |  1  | 20  |  1  | 32  | 32  |
|    70b     |   BF16    |  16  |  4096  |   80   |  1  |  4  |  1  |  4  | 20  |  1  | 64  | 16  |
|    70b     |    FP8    |  16  |  4096  |   80   |  2  |  4  |  1  |  2  | 20  |  1  | 64  | 32  |

# Performance Measurement and Analysis

Performance for LLAMA3 70B LoRa finetuning is measured by the achieved GPU FLOPS via the `TFLOPS_per_GPU` metric, which indicates computational throughput efficiency. Additionally, finetuning step timing (seconds per iteration) is captured and logged for every finetuning step in the main finetuning log file [see Output Locations](#output-locations).

Since the early finetuning steps typically take much longer time (with input prefetch, activation memory allocation, and JIT compilation), we use the `parse_train_timing_mbridge.sh` script to analyze iterations 35-44 and calculate mean and standard deviation for reliable performance metrics for both TFLOPS per GPU and timing measurements.

> **Note:** The `MODEL_TFLOP/s/GPU` value reported by Megatron-Bridge in the training log is incorrect for LoRA finetuning in this release. Use `parse_train_timing_mbridge.sh` to obtain accurate TFLOPS per GPU, which computes the correct value using the LoRA-specific FLOPs formula accounting for the FLOPs breakdown across frozen and trainable parameters.

### Running the parse_train_timing_mbridge.sh script

To analyze training timing from your experiment results, run the script from the workload directory. In an installed environment, recipe files are available under `$LLMB_INSTALL/llmb_repo` (a copy created by the installer).

```bash
# Basic usage - parses results in the directory named 'experiments' in the current folder
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh

# Specify a different experiments directory
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh /path/to/experiments


# Show full filenames instead of shortened versions
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh --full-names
```

Example output:

```shell
Elapsed Time (ms) (iterations 35-44)
================================================================================
Experiment                                                                                   Status Time Mean (ms) Time Std (ms) MODEL_TFLOPS_per_GPU Mean MODEL_TFLOPS_per_GPU Std
------------------------------------------------------------------------------------------ -------- ------------- ------------ ------------------- ------------------
lora_llama3_70b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_etpNone_mbs1_gbs32_1097062                Success      3314.210        6.861             1382.35               2.86
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

## Request Access

Access to Llama 3 70B must be requested through the [HuggingFace Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct). The approval process is not automatic and could take a day or more.

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

Note, that a new directory layout and key variables are now used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/finetune_llama3`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

## Prepare Checkpoint

The recommended method to prepare the model checkpoint is by using the **installer** referenced in the [main README](../../README.md).

When the installer runs the setup for this recipe, it will run an interactive `srun` job to import the Llama-3 70B Finetuning checkpoint. Please note the following:

- This step runs interactively as part of the installer and may take some time to complete.
- It requires GPU resources. Ensure your SLURM partition has available GPUs.
- To verify that the checkpoint has been downloaded successfully, refer to the [Storage Requirements and Verification](#storage-requirements-and-verification) section.

If you encounter issues with the automated download via the installer, you can use the manual instructions provided in the "[Manual Checkpoint Download (Troubleshooting Only)](#manual-checkpoint-download-troubleshooting-only)" section at the end of this document. This method is intended for troubleshooting purposes only.

## Prepare Dataset

The LLAMA3 70B LoRa finetuning uses the SQUAD dataset. The dataset is generated automatically on the first run via `setup_experiment.py` and will be placed under `$LLMB_WORKLOAD/checkpoint_and_dataset/datasets`. No manual dataset preparation is required.

# Run Finetuning

Once the environment has been prepared, it is time to train a model. The finetuning runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run submit -w finetune_llama3 --dtype fp8 --scale 8

# Example with BF16 precision
llmb-run submit -w finetune_llama3 --dtype bf16 --scale 8
```

### Additional SLURM Parameters

Use a SLURM reservation:

```bash
ADDITIONAL_SLURM_PARAMS="reservation=my_reservation" llmb-run submit -w finetune_llama3 --dtype fp8 --scale 8
```

Run on specific nodes:

```bash
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002" llmb-run submit -w finetune_llama3 --dtype bf16 --scale 8
```

Exclude specific nodes:

```bash
ADDITIONAL_SLURM_PARAMS="exclude=node003,node004" llmb-run submit -w finetune_llama3 --dtype bf16 --scale 8
```

Combine multiple parameters (semicolon-separated):

```bash
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002;reservation=my_reservation;exclusive" llmb-run submit -w finetune_llama3 --dtype bf16 --scale 8
```

For more details on llmb-run usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run finetuning directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:

- Ensure your virtual environment is activated before running the finetuning commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_INSTALL/llmb_repo/llama3/finetune`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [MODEL_SIZE=<size>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use (e.g., 8)
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**

- `DTYPE`: Precision format (default: `bf16`)
  - `fp8` - FP8 precision
  - `bf16` - BFloat16 precision
- `MODEL_SIZE`: Model variant (fixed: `70b`)
  - `70b` - 70 billion parameter model (only supported size)
- `ADDITIONAL_SLURM_PARAMS`: Extra `sbatch` flags (e.g. `--nodelist`, `--reservation`), semicolon-separated
  - Example: `"nodelist=node001,node002;reservation=my_reservation;exclusive"`

### Example Commands

Train LLAMA3 70B LoRa with BF16 precision on 8 GB300 GPUs:

```shell
JOB_TOTAL_GPUS=8 DTYPE=bf16 GPU_TYPE=gb300 ./launch.sh
```

Train with FP8 precision on 8 GB300 GPUs:

```shell
JOB_TOTAL_GPUS=8 DTYPE=fp8 GPU_TYPE=gb300 ./launch.sh
```

Train on 8 H100 GPUs with FP8 precision:

```shell
JOB_TOTAL_GPUS=8 DTYPE=fp8 GPU_TYPE=h100 ./launch.sh
```

# Output Locations

All benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```text
experiments/
├── <experiment_name>/
│   └── <experiment_name>_<timestamp>/
│       ├── <experiment_name>/
│       │   ├── log-<experiment_name>.out      # Main finetuning log with performance data
│       │   ├── sbatch_<experiment_name>.out   # Batch script output  
│       │   └── nsys_profile/                  # Profiling output (when enabled)
│       │       └── *.nsys-rep files
│       └── [batch scripts and other files]
```

The `<experiment_name>` typically follows the pattern: `lora_llama3_70b_<dtype>_<scale>_<config>`

**Key files:**

- `log-<experiment_name>.out` - Contains training step timing and performance metrics analyzed by `parse_train_timing.sh`
- `nsys_profile/` - Contains profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`

# Profiling

Profiling is supported with Nsight Systems.

## Run Nsight Profiling

To enable profiling with Nsight Systems, use the `-p` flag with `llmb-run` or set `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Profiling job details:

- **MPI Ranks:** 0-8
- **Job Steps:** 45-50
- **Output Location:** Profiling output saved alongside finetuning results (see Output Locations)
- **Filename format:** `profile_${SLURM_JOB_ID}_nodeId_rankId.nsys-rep`

**Example command:**

```shell
llmb-run submit -w finetune_llama3 --dtype bf16 --scale 8 -p
```

### Customizing profiling behavior:

- Specify job steps to profile:
  - `PROFILE_START_STEP`: start profiling on this job step.
  * Default: 45
  - `PROFILE_STOP_STEP`: stop profiling on this job step.
  * Default: 50

### Viewing results

In order to view the profile traces (\*.nsys-rep files) interactively:

- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run.

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

<!-- NCCL trace support removed. Documentation section deleted intentionally. -->

# Storage Requirements and Verification

**Note:** The LLAMA3 70B LoRa checkpoint is approximately **747GB** in size, which requires significant storage space. Ensure your file system has sufficient space (at least 1TB recommended) to accommodate the checkpoint, dataset, and temporary files.

To verify that the checkpoint has been correctly downloaded and is available, check for the following directory structure within `${LLMB_WORKLOAD}/checkpoint_and_dataset/` after the installer's setup tasks have completed:

```text
checkpoint_and_dataset/
├── datasets/          # Dataset files (populated on first run)
├── hub/              # HuggingFace hub cache
├── llama3_70b/        # Model checkpoints (~747GB)
└── xet/
```

The `datasets/` folder will be populated with the SQUAD dataset automatically on the first run via `setup_experiment.py`. After the first run, the folder structure will look like:

```text
datasets/
├── packed/
├── squad/
│   └── plain_text/
│       └── 0.0.0/
│           └── 7b6d24c440a36b6815f21b70d25016731768db1f/
│               ├── cache-aedd28d6a9c0b57f.arrow
│               ├── cache-ee746ac879d304bb.arrow
│               ├── dataset_info.json
│               ├── squad-train.arrow
│               └── squad-validation.arrow
├── test.jsonl
├── test.jsonl.idx.info
├── test.jsonl.idx.npy
├── training.jsonl
├── training.jsonl.idx.info
├── training.jsonl.idx.npy
├── validation.jsonl
├── validation.jsonl.idx.info
└── validation.jsonl.idx.npy
```

Within the `llama3_70b/` folder, you should be able to find the downloaded checkpoint with the following structure:

```text
llama3_70b/
├── context/
└── weights/
    ├── __0_0.distcp    # 66GB
    ├── __0_1.distcp    # 66GB
    ├── common.pt
    └── metadata.json
```

# Manual Checkpoint Download (Troubleshooting Only)

These instructions are provided for troubleshooting purposes, in case the automated checkpoint import via the installer encounters issues.

## Activate Virtual Environment

Before running the `download_ckpt_dataset.sh` script, you must activate the virtual environment associated with the `finetune_llama3` workload. This environment contains the necessary dependencies for Megatron-Bridge to run the download process.

To find the correct virtual environment path, consult the `cluster_config.yaml` file located in your `${LLMB_INSTALL}` directory. Look under the `workloads.config.finetune_llama3.venv_path` key.

Once you have the `venv_path`, activate it using one of the following commands, depending on whether it's a `venv` or `conda` environment:

- **For `venv` environments:**
  ```bash
  source <venv_path>/bin/activate
  ```
- **For `conda` environments:**
  ```bash
  conda activate <venv_path>
  ```

## Important Environment Variables

The `download_ckpt_dataset.sh` script relies on several environment variables that are typically set by the installer. When running this script manually, you must ensure these variables are set in your environment:

- `HF_TOKEN`: Your HuggingFace access token. (e.g., `export HF_TOKEN=<your token>`)
- `GPU_TYPE`: The type of GPU hardware you are using (`gb300`, `gb200` or `h100`). (e.g., `export GPU_TYPE=h100`)
- `LLMB_INSTALL`: The top-level installation directory for all benchmarking artifacts. (e.g., `export LLMB_INSTALL=/path/to/llm_benchmarking_install`)
- `LLMB_WORKLOAD`: The workload-specific directory. This is usually derived from `LLMB_INSTALL`. (e.g., `export LLMB_WORKLOAD=${LLMB_INSTALL}/workloads/finetune_llama3`)
- `SBATCH_ACCOUNT`: Your Slurm account. (e.g., `export SBATCH_ACCOUNT=your_slurm_account`)
- `SBATCH_PARTITION`: The Slurm partition (queue) to use. (e.g., `export SBATCH_PARTITION=your_partition`)
- `SBATCH_GPUS_PER_NODE`: If your Slurm partition requires GRES set to at least one. (e.g., `export SBATCH_GPUS_PER_NODE=1`)
- `TIME_LIMIT`: The default is 55 minutes. Increase this value if you experience timeouts, particularly if your network speed is slower. Use the `HH:MM:SS` format (e.g., `export TIME_LIMIT=01:30:00` for 1 hour and 30 minutes).

## Download Commands

Once these environment variables are set, you can use the following command to manually download the checkpoint:

```shell
./download_ckpt_dataset.sh
```

**Note:** This command may take some time to complete. Refer to the [Storage Requirements and Verification](#storage-requirements-and-verification) section to confirm successful download.
