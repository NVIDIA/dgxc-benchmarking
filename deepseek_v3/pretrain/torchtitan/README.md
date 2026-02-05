# Overview

This recipe contains information and scripts to produce performance results for the DeepSeek-V3 pre-training workload using the **TorchTitan** framework. The scripts help perform environment setup and launch benchmark jobs.

TorchTitan is a proof-of-concept for Large-scale LLM training using native PyTorch. This implementation leverages TorchTitan's distributed training capabilities with FSDP (Fully Sharded Data Parallel), tensor parallelism, pipeline parallelism, and expert parallelism for efficient training of the DeepSeek-V3 671B parameter model.

## Supported GPU Configurations

This recipe supports **H100**, **B200**, and **GB200** GPUs. The tables below show the **default benchmark configurations**; all values can be overridden via environment variables (see [Run Training](#run-training)).

Only BF16 precision is supported by this recipe.

## GB200

| Size | Precision | GPUs | SeqLen | Steps | DP | TP | EP | PP | MBS | GBS | GA | Dataset |
|------|:---------:|:----:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:---:|:--:|:-------:|
| 671B | BF16      | 256  | 4096   | 200   | 32 | 1  | 32 | 8  | 16  | 512 | 1  | C4      |

## B200

| Size | Precision | GPUs | SeqLen | Steps | DP | TP | EP | PP | MBS | GBS | GA | Dataset |
|------|:---------:|:----:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:---:|:--:|:-------:|
| 671B | BF16      | 256  | 4096   | 200   | 32 | 1  | 32 | 8  | 16  | 512 | 1  | C4      |

## H100

| Size | Precision | GPUs | SeqLen | Steps | DP | TP | EP | PP | MBS | GBS  | GA | Dataset |
|------|:---------:|:----:|:------:|:-----:|:--:|:--:|:--:|:--:|:---:|:----:|:--:|:-------:|
| 671B | BF16      | 512  | 4096   | 200   | 64 | 1  | 32 | 8  | 16  | 1024 | 1  | C4      |

# Prerequisites

## HuggingFace Account

A HuggingFace account is required to download the tokenizer and dataset. You will need to:
1. [Create a HuggingFace access token](https://huggingface.co/settings/tokens)
2. Add the generated token to your environment:
```bash
export HF_TOKEN=<your token>
```

## Python Requirements

Requires Python 3.12.x, or conda.

## Request Access

No special access is required to run this benchmark. The DeepSeek-V3.1-Base tokenizer is publicly available on HuggingFace.

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**
- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../../../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_deepseek-v3-torchtitan`.
- `TORCHTITAN_HOME`: TorchTitan installation directory, e.g. `${LLMB_WORKLOAD}/torchtitan`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see [Output Locations](#output-locations) below).

## Installation Steps

The installer will automatically:
1. Pull and convert the PyTorch container image (nvidia/pytorch:25.10-py3)
2. Clone the TorchTitan repository (commit: f1a96b34ff4c752b246a3e381976b7d74387bee6)
3. Install TorchTitan into the container (`install_torchtitan_to_container.sh`)
4. Download the DeepSeek-V3.1-Base tokenizer from HuggingFace (`download_hf_assets.sh`)
5. Download the C4 dataset from HuggingFace (`download_dataset.sh`)
6. Apply the DeepSeek-V3 fix patch (`apply_fix.sh`)

**Note**: The tokenizer and dataset downloads are performed automatically as part of the setup tasks defined in `metadata.yaml`.

# Prepare Dataset

The C4 dataset is automatically downloaded during the environment setup process. The download script fetches the English subset of the C4 dataset from HuggingFace and stores it in `$LLMB_INSTALL/datasets/c4`.

If you need to manually download or re-download the dataset, you can run:

```bash
cd $LLMB_WORKLOAD
sbatch download_dataset.sh
```

# Run Training

Once the environment has been prepared, it is time to train the model. The training runs for 200 steps by default (configurable). Log files and results are stored under `${LLMB_WORKLOAD}/experiments/` in per-job folders (see [Output Locations](#output-locations) for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run DeepSeek-V3 671B BF16 (scale = number of GPUs)
llmb-run submit -w pretrain_deepseek-v3-torchtitan -s 671b --dtype bf16 --scale 256
llmb-run submit -w pretrain_deepseek-v3-torchtitan -s 671b --dtype bf16 --scale 512
```

For more details on llmb-run usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

## Direct Method

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/deepseek_v3/pretrain/torchtitan/`

### Environment variables

**Required:**

- `GPU_TYPE`: Type of GPU hardware
  - `h100` - NVIDIA H100 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `gb200` - NVIDIA GB200 GPUs

- `JOB_TOTAL_GPUS`: Total number of GPUs to use for training

- `LLMB_INSTALL`: Path to the installation directory for all workloads

**Optional:**

- `GPUS_PER_NODE`: Number of GPUs per node (default: 8 for H100/B200, 4 for GB200)
- `DATA_PARALLEL_SHARD_DEGREE`: Data parallel sharding degree (default: 64 for H100, 32 for B200, 32 for GB200)
- `EXPERT_PARALLEL_DEGREE`: Expert parallel degree for MoE (default: 32)
- `PIPELINE_PARALLEL_DEGREE`: Pipeline parallel degree (default: 8)
- `DATASET_PATH`: Path to the dataset (default: `$LLMB_INSTALL/datasets/c4`)
- `SEQ_LEN`: Sequence length (default: 4096)
- `TRAINING_STEPS`: Number of training steps (default: 200)
- `LOCAL_BATCH_SIZE`: Local batch size per GPU (default: 16)
- `LOG_RANK`: Rank to log from (default: 448 for H100/B200, 224 for GB200)
- `RUN_CONF_IMAGE`: Override container image path
- `RUN_CONF_MOUNTS`: Additional container mounts
- `ADDITIONAL_SLURM_PARAMS`: Additional SLURM parameters (optional)
  - Format: Semicolon-separated parameters supporting both `key=value` pairs and standalone flags
  - Use semicolons as delimiters (especially when values contain commas or ampersands)
  - Examples:
    - Key=value pairs: `"nodelist=node001,node002;constraint=gpu&memory"`
    - Standalone flags: `"exclusive"`
    - Mixed: `"constraint=gpu&memory;exclusive"`

## Running the Launch Script

### Command Template

```bash
GPU_TYPE=<type> JOB_TOTAL_GPUS=<number> sbatch launch.sh
```

### Example Commands

Train on H100 GPUs (minimum configuration):
```bash
GPU_TYPE=h100 JOB_TOTAL_GPUS=512 sbatch launch.sh
```

Train on B200 GPUs:
```bash
GPU_TYPE=b200 JOB_TOTAL_GPUS=256 sbatch launch.sh
```

Train on GB200 GPUs:
```bash
GPU_TYPE=gb200 JOB_TOTAL_GPUS=256 sbatch launch.sh
```

Train with custom training steps:
```bash
GPU_TYPE=h100 JOB_TOTAL_GPUS=1024 TRAINING_STEPS=5000 sbatch launch.sh
```

Train with custom parallelism settings:
```bash
GPU_TYPE=h100 JOB_TOTAL_GPUS=512 \
  DATA_PARALLEL_SHARD_DEGREE=32 \
  EXPERT_PARALLEL_DEGREE=16 \
  PIPELINE_PARALLEL_DEGREE=4 \
  sbatch launch.sh
```

### SLURM Node Specification Examples

Train on specific nodes:
```bash
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002" GPU_TYPE=h100 JOB_TOTAL_GPUS=512 sbatch launch.sh
```

Train with node constraints:
```bash
ADDITIONAL_SLURM_PARAMS="constraint=gpu&memory;exclusive" GPU_TYPE=b200 JOB_TOTAL_GPUS=256 sbatch launch.sh
```

Train using a SLURM reservation:
```bash
ADDITIONAL_SLURM_PARAMS="reservation=my_reservation" GPU_TYPE=gb200 JOB_TOTAL_GPUS=256 sbatch launch.sh
```

## Configuration Files

The training uses a TOML configuration file located at:
```
$LLMB_INSTALL/llmb_repo/deepseek_v3/pretrain/torchtitan/deepseek_v3_671b.toml
```

This file contains:
- Model architecture specifications (DeepSeek-V3 671B)
- Optimizer settings (AdamW with lr=2.2e-4)
- Learning rate scheduler configuration (warmup_steps=100, decay_ratio=0.8, cosine decay)
- Activation checkpointing settings (full mode enabled)
- Compilation options (model and loss compilation enabled)
- Float8 quantization options (disabled by default)
- Profiling settings (disabled by default)
- Metrics logging settings (log_freq=10)

Command-line arguments passed to the launch script will override the settings in the TOML file.

# Output Locations

All job outputs are organized in a **two-level directory structure** under `$LLMB_WORKLOAD/experiments/`:

```text
$LLMB_WORKLOAD/experiments/<workload>_<size>_<dtype>_gpus<number>/
└── <unix_timestamp>/
    ├── llmb-config_<SLURM_JOB_ID>.yaml       # Job configuration (created by llmb-run)
    ├── slurm-<SLURM_JOB_ID>.out              # Main Slurm job output
    ├── log-torchtitan_*.out                  # Training stdout (per-rank logs)
    ├── log-torchtitan_*.err                  # Training stderr
    └── outputs/                              # Training outputs and dumps
        └── profile_trace/                    # Profiling traces (if enabled)
```

**Note:** The `<unix_timestamp>` subdirectory name is the Unix epoch timestamp (in seconds) when the job was launched.

**Example:** For a 671B BF16 model run on 512 GPUs, outputs are stored in:
```
$LLMB_WORKLOAD/experiments/pretrain_deepseek-v3-torchtitan_671b_bf16_gpus512/1769818909/
```
where `1769818909` is the Unix timestamp of the job launch time.

**Key files:**
- `llmb-config_*.yaml` - Job configuration including model, scale, and cluster info
- `slurm-*.out` - Slurm job outputs (main job, parsing, uploader)
- `log-torchtitan_*.out` - Training step timing and performance metrics
- `log-torchtitan_*.err` - Training error messages and warnings

Additional outputs (if enabled in the TOML config):
- `outputs/` - Training outputs, dumps, and profiling traces
- `outputs/tb/` - TensorBoard logs (if enabled)
- `outputs/checkpoint/` - Model checkpoints (if enabled)

# Performance Measurement and Analysis

Performance for DeepSeek-V3 training is measured by seconds per iteration (training step time) and TFLOPS per GPU. These metrics are logged for every training step in the main training log file.

## Analyzing Training Performance

To extract performance metrics from the training logs:

```bash
# Navigate to the experiments directory and find your job folder
cd $LLMB_WORKLOAD/experiments
ls -lt  # List experiment configurations

# Navigate to a specific experiment configuration (e.g., 671B BF16 on 512 GPUs)
cd pretrain_deepseek-v3-torchtitan_671b_bf16_gpus512/

# List runs by Unix timestamp (most recent first)
ls -lt

# Navigate to a specific run directory (using the Unix timestamp)
cd <unix_timestamp>/

# View the training log
tail -f log-torchtitan_*.out

# Extract timing information (after warmup)
grep "step:" log-torchtitan_*.out | tail -20
```

Look for log entries containing:
- `step:` - Training step number
- `loss:` - Training loss value
- `tps:` - Tokens per second
- `tflops:` - TFLOPS per GPU
- `mfu:` - Model FLOPs Utilization

### Extracting TPS (tokens/sec)

The training log includes per-step tokens/sec in lines like:
```
tps: 299
```

To print the most recent TPS values:

```bash
# From within a specific run directory
grep -h "tps:" log-torchtitan_*.out | tail -20
```

## Calculating Throughput

To calculate throughput in tokens per second:

```
throughput (tokens/sec) = (sequence length) × (global batch size) / (training step time in seconds)
```

Where:
- Sequence length = 4096 (default)
- Global batch size = (local batch size) × (gradient accumulation steps) × (number of GPUs) / (data parallel shard degree)

Example for H100 with 512 GPUs:
```
global_batch_size = 16 × 1 × 512 / 64 = 128  (where GA=1)
throughput = 4096 × 128 / (step_time_seconds)
```

## Model FLOPs Utilization (MFU)

Model FLOPs Utilization indicates how efficiently the model is using the available compute:

```
MFU = (achieved TFLOPS per GPU) / (peak theoretical TFLOPS)
```

**Peak theoretical throughput across GPUs and Data Types (in TFLOPS)**

| Data Type | GB200 | B200 | H100 |
| --------  | :---: | :---:| :---:|
| BF16      | 2450  | 2250 | 989  |
| FP8       | 4900  | 4500 | 1979 | 


# Troubleshooting

## Common Issues

### Out of Memory (OOM)

If you encounter OOM errors:
1. Reduce `LOCAL_BATCH_SIZE`
2. Increase parallelism degrees (especially pipeline parallel)
3. Enable full activation checkpointing (already enabled by default)

### NCCL Timeout

If you see NCCL timeout errors:
1. Increase `[comm] init_timeout_seconds` in the TOML config (default: 1200 seconds)
2. Check network connectivity between nodes
3. Verify Slurm allocation includes all requested GPUs

### Container Mount Issues

If the container cannot access files:
1. Verify `LLMB_INSTALL` and `LLMB_WORKLOAD` paths are accessible
2. Add additional mounts via `RUN_CONF_MOUNTS` if needed
3. Check file permissions

### GPU Type Not Supported

The error "Torchtitan recipes only supports h100, b200 and gb200 GPU types" means:
- You're trying to use a GPU type not supported by this recipe
- Currently supported: h100, b200, gb200

# Advanced Configuration

## Custom Dataset

To use a different dataset:
1. Place your dataset in `$LLMB_INSTALL/datasets/<dataset_name>`
2. Set `DATASET_PATH=$LLMB_INSTALL/datasets/<dataset_name>` when launching
3. Update the TOML config if needed to specify the dataset format

## Profiling

There are two ways to enable PyTorch/TorchTitan profiling:

### Option 1: Using the launch flag or environment variable

Set `ENABLE_PROFILE=true` when launching (or use the `-p` flag). The launch script will pass the TorchTitan override `--profiling.enable_profiling` and write traces to the `outputs/` directory within your run folder.

Example:

```bash
# Using the -p flag
llmb-run submit -w pretrain_deepseek-v3-torchtitan -s 671b --dtype bf16 --scale 256 -p

# Or using the environment variable
ENABLE_PROFILE=true llmb-run submit -w pretrain_deepseek-v3-torchtitan -s 671b --dtype bf16 --scale 256
```

To view the generated traces, inspect the `outputs/` directory within your run folder.


### Option 2: Modifying the TOML configuration file

Edit the TOML configuration file (`deepseek_v3_671b.toml`, see [Configuration Files](#configuration-files)) and set:

```toml
[profiling]
enable_profiling = true
save_traces_folder = "profile_trace"  # customize as needed
profile_freq = 10
```

With this method, traces will be saved to the `outputs/<save_traces_folder>/` directory within your run folder.


## TensorBoard and Weights & Biases

To enable TensorBoard logging, add the following to the TOML configuration file (`deepseek_v3_671b.toml`, see [Configuration Files](#configuration-files)):

```toml
[metrics]
enable_tensorboard = true
save_tb_folder = "tb"
```

To view the generated logs, inspect the `outputs/tb/` directory within your run folder.

To enable Weights & Biases logging:

```toml
[metrics]
enable_wandb = true
```

> **Note:** For W&B, ensure you have authenticated via `wandb login` or set the `WANDB_API_KEY` environment variable.

# Additional Resources

- [TorchTitan GitHub Repository](https://github.com/pytorch/torchtitan)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan/tree/main/docs)
- [DeepSeek-V3 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

# Known Limitations

1. **Dataset**: Only C4 dataset is configured by default. Custom datasets require manual configuration.
2. **Checkpointing**: Model checkpointing is disabled by default for benchmarking purposes.
3. **Monitoring**: TensorBoard and Weights & Biases integrations are disabled by default.
4. **Flex Attention**: Disabled in the current configuration (uses standard causal attention).

For production training runs, you may want to enable checkpointing and monitoring in the TOML configuration file.
