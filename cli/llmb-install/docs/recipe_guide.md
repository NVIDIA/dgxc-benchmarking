# Recipe Development Guide

This guide explains how to create and configure workload recipes using the `metadata.yaml` file. It covers all available configuration options and patterns for defining workloads.

## Overview

Each workload recipe requires a `metadata.yaml` file that defines:
- **General Information**: Workload identification and framework
- **Container Images**: Runtime environment containers
- **Repositories**: Git repositories for dependencies
- **Downloads**: Offline assets (tokenizers, models, datasets)
- **Setup**: Virtual environment and dependency installation
- **Tools**: Workload-specific tool versions (e.g., nsys)
- **Run Configuration**: GPU configs, model sizes, and test scales

## Metadata Structure

A complete `metadata.yaml` follows this structure:

```yaml
general:
  # Workload identification
  
container:
  # Container images
  
repositories:  # Optional
  # Git repositories

downloads:  # Optional
  # Offline assets (tokenizers, models, datasets)
  
tools:  # Optional
  # Tool versions
  
setup:  # Optional
  # Dependencies and setup tasks
  
run:
  # Launch configuration and GPU configs
```

## General Section (Required)

Identifies the workload at a high level:

```yaml
general:
  workload: nemotron4              # workload model name
  workload_type: pretrain          # Type of workload
  framework: nemo2                 # Framework used
  model: nemotron4                 # Optional: Override model name in llmb-config
```

### Fields

- **`workload`** (string, required): Name of the workload, must match the directory name
- **`workload_type`** (enum, required): One of:
  - `pretrain` - Pre-training workloads
  - `inference` - Inference workloads
  - `finetune` - Fine-tuning workloads
- **`framework`** (string, required): Framework name (e.g., `nemo2`, `maxtext`, `megatron`)
- **`model`** (string, optional): Model name to use in `llmb-config_jobid.yaml` for `model_info.model_name`. If not specified, defaults to the `workload` value. Useful when multiple workload directories share the same base model (e.g., `llama3.1` and `llama3.3` both use `model: llama3`)

**Note**: Version information is managed centrally in `release.yaml` at the repository root and does not need to be specified in individual recipe metadata files.

## Container Section (Required)

Defines the OCI container images that provide the runtime environment.

### Simple Format (Same Container for All GPUs)

```yaml
container:
  images: 
    - 'nvcr.io#nvidia/nemo:25.07.01'
```

### Multiple Images

```yaml
container:
  images:
    - 'nvcr.io#nvidia/nemo:25.07.01'
    - 'nvcr.io#nvidia/pytorch:24.12-py3'
```

### Custom Image Names

Override the automatically generated filename:

```yaml
container:
  images:
    - url: 'nvcr.io#nvidia/nemo:25.07.01'
      name: 'my-custom-name.sqsh'
```

### GPU-Conditional Images

Use different containers for different GPU types:

```yaml
container:
  images:
    by_gpu:
      h100: 'nvcr.io#nvidia/nemo:25.01'
      gb200: 'nvcr.io#nvidia/nemo:25.05'
      default: 'nvcr.io#nvidia/nemo:25.07.01'  # Fallback for other GPUs
```

**Note**: Image URLs use `#` instead of `/` between registry and image path.

## Repositories Section (Optional)

Defines Git repositories to clone during setup. These can be used as dependencies or referenced in the setup.

### Simple Format

```yaml
repositories:
  nemo:
    url: "https://github.com/NVIDIA/NeMo.git"
    commit: "763ffa8b00a2fca9f7a204e14111ed190de7d947"  # Full 40-char SHA
  megatron_core:
    url: "https://github.com/NVIDIA/Megatron-LM.git"
    commit: "ac198fc0d60a8c748597e01ca4c6887d3a7bcf3d"
```

### GPU-Conditional Repositories

```yaml
repositories:
  by_gpu:
    h100:
      nemo:
        url: "https://github.com/NVIDIA/NeMo.git"
        commit: "abc123..."
    gb200:
      nemo:
        url: "https://github.com/NVIDIA/NeMo.git"
        commit: "def456..."
    default:
      nemo:
        url: "https://github.com/NVIDIA/NeMo.git"
        commit: "789abc..."
```

**Important**: Commit must be the full 40-character SHA hash, not a short hash or tag.

## Downloads Section (Optional)

Specifies offline assets to download during installation. This section is used to ensure models and tokenizers are available in air-gapped or offline environments.

### HuggingFace Assets

The recommended way to specify HuggingFace assets is using the `huggingface` list. This allows you to specify both tokenizers and model configurations.

```yaml
downloads:
  huggingface:
    - repo_id: Qwen/Qwen3-30B-A3B
      assets: [tokenizer, config]   # Optional: defaults to both if omitted
```

#### Fields

- **`repo_id`** (string, required): The HuggingFace repository ID.
- **`assets`** (list of enums, optional): Which assets to download. Allowed values: `tokenizer`, `config`. 
  - If omitted, it defaults to **both** `[tokenizer, config]`.

#### Behavior and Rules

- **No Weights**: This section does **NOT** download model weights (SafeTensors/Pickle). It only downloads metadata, tokenizers, and configuration files.
- **Download vs. Verify**: Downloads run first, then a separate verification step checks that required assets load offline (`local_files_only=True`). This is an internal implementation split (two functions), not a separate lifecycle phase.
- **Caching**: Assets are cached in `$LLMB_INSTALL/.cache/huggingface` and made available to workloads via the `HF_HOME` environment variable.

### Legacy: hf_tokenizers

The `hf_tokenizers` key is supported for backward compatibility but is restricted to tokenizers only. It does **not** download model configurations.

```yaml
downloads:
  hf_tokenizers:
    - 'meta-llama/Meta-Llama-3-70B'
```

> [!IMPORTANT]
> **Exclusivity Rule**: You cannot use both `hf_tokenizers` and `huggingface` within the same `metadata.yaml` file. Mixing them will result in a validation error.

### Migration Guidance

Existing recipes using `hf_tokenizers` should eventually migrate to the `huggingface` structure. Note that `hf_tokenizers` only downloads the tokenizer, while the new `huggingface` key defaults to both tokenizer and config.

**Legacy (Tokenizer only):**
```yaml
downloads:
  hf_tokenizers:
    - 'nvidia/Nemotron-4-340B-Base'
```

**Migrated (Tokenizer only):**
```yaml
downloads:
  huggingface:
    - repo_id: nvidia/Nemotron-4-340B-Base
      assets: [tokenizer]
```

### Examples

#### 1. Default (Tokenizer + Config)
Omit the `assets` field to download both.
```yaml
downloads:
  huggingface:
    - repo_id: Qwen/Qwen3-30B-A3B
```

#### 2. Tokenizer-only (Nemotron Pattern)
```yaml
downloads:
  huggingface:
    - repo_id: nvidia/Nemotron-4-340B-Base
      assets: [tokenizer]
```

#### 3. Config-only (Rare)
```yaml
downloads:
  huggingface:
    - repo_id: meta-llama/Llama-3.1-405B
      assets: [config]
```

**Note**: Accessing private or gated models requires the `HF_TOKEN` environment variable to be set during the installation phase.

## Tools Section (Optional)

Specifies workload-specific tool versions (currently supports `nsys` for profiling).

**Only use this section when you need a specific tool version.** If your container's tools work fine, omit this section.

### Simple Format (All GPUs Same Version)

```yaml
tools:
  nsys: "2025.5.1.121-3638078"
```

### GPU-Conditional Tools

Use different versions for different GPU types:

```yaml
tools:
  nsys:
    by_gpu:
      h100: "2025.1.1.118-3638078"
      gb200: "2025.5.1.121-3638078"
      default: "2025.4.1.172-3634357"  # Optional: fallback version
```

### Partial GPU Coverage

Only specify versions for GPUs that need custom tools (others use container version):

```yaml
tools:
  nsys:
    by_gpu:
      h100: "2025.1.1.118-3638078"
      gb200: "2025.5.1.121-3638078"
      # b200 and other GPUs will use container nsys (no download)
```

**Resolution Logic**:
1. If GPU explicitly listed in `by_gpu` → use that version
2. Else if `default` key exists → use default version
3. Else → use container version (no download)

For more details, see [tools.md](tools.md).

## Setup Section (Optional)

Defines virtual environment creation, dependencies, and setup tasks.

### Basic Setup with Dependencies

```yaml
setup:
  venv_req: true  # Create a Python virtual environment
  dependencies:
    pip:
      - package: nemo
        repo_key: nemo
        install_target: '.[nlp]'
      - 'scipy<1.13.0'
      - 'bitsandbytes==0.46.0'
      - package: megatron-core
        repo_key: megatron_core
```

### Dependencies Reference

#### Pip Dependencies

Simple string format (PyPI package):
```yaml
dependencies:
  pip:
    - 'numpy==1.24.0'
    - 'torch>=2.0'
```

Repository-based package:
```yaml
dependencies:
  pip:
    - package: nemo           # Package name
      repo_key: nemo          # References key in repositories section
      install_target: '.[nlp]'  # Optional: extras or specific target
      editable: true          # Optional: install in editable mode (-e)
```

#### Git Dependencies

```yaml
dependencies:
  git:
    my_package:
      repo_key: my_repo       # References key in repositories section
      install_method:
        type: clone           # 'clone' or 'script'
        path: 'subdir'        # Optional: subdirectory within repo
```

### Setup Tasks

Run custom commands during setup:

```yaml
setup:
  venv_req: true
  tasks:
    - name: "Download dataset"
      cmd: "python download_data.py --output $DATASET_DIR"
      job_type: local        # 'local', 'nemo2', 'srun', or 'sbatch'
      requires_gpus: false   # Optional: whether task needs GPUs
      env:                   # Optional: environment variables
        DATASET_DIR: "/data"
```

**Task Types**:
- `local`: Run on current node
- `nemo2`: Run with nemo2 launcher
- `srun`: Run via SLURM srun
- `sbatch`: Submit as SLURM batch job

### Legacy Setup Script

> **⚠️ DEPRECATED:** The `setup_script` functionality is deprecated and will be removed in a future release. Please migrate to the `tasks` feature above for all setup operations.

For backward compatibility only:

```yaml
setup:
  setup_script: "setup.sh"  # Path to setup script (DEPRECATED - use tasks instead)
  venv_req: true
```

## Run Section (Required)

Defines how the workload is launched and what configurations to test.

### Basic Structure

```yaml
run:
  launcher_type: 'nemo'      # Launcher type
  launch_script: 'launch.sh' # Launch script path
  gpu_configs:               # Per-GPU configurations
    h100:
      model_configs:
        - model_size: '405b'
          dtypes: ['fp8', 'bf16']
          scales: [512, 1024, 2048]
```

### Launcher Types

- **`nemo`**: NeMo launcher (nemo2 workloads)
- **`megatron_bridge`**: Megatron bridge launcher
- **`sbatch`**: Direct SLURM sbatch submission

### GPU Configs

Define test configurations for each GPU type:

```yaml
gpu_configs:
  h100:
    model_configs:
      - model_size: '15b'
        dtypes: ['fp8', 'bf16']
        scales: [16, 32, 64, 128]
  b200:
    model_configs:
      - model_size: '15b'
        dtypes: ['fp8']
        scales: [32, 64, 128, 256]
```

**Supported GPU Types**: `h100`, `b200`, `gb200`, `gb300`

### Model Configs

Each model config specifies:

#### Simple Format

```yaml
model_configs:
  - model_size: '340b'
    dtypes: ['fp8', 'bf16']
    scales: [128, 256, 512, 1024]
    exact_scales: false  # Optional: allow power-of-2 extension
```

#### Per-Dtype Scales

Define different scales for different dtypes:

```yaml
model_configs:
  - model_size: '405b'
    dtypes:
      fp8: [128, 256, 512]      # Short form
      bf16:                     # Long form with exact_scales
        scales: [256, 512]
        exact_scales: true
```

**Fields**:
- **`model_size`** (string, required): Model size identifier (e.g., `'7b'`, `'405b'`)
- **`dtypes`** (required): Precision types to test. Can be:
  - Single dtype: `'fp8'`
  - List: `['fp8', 'bf16']`
  - Mapping: `fp8: [128, 256]` or `fp8: {scales: [128, 256], exact_scales: true}`
- **`scales`** (list, optional): GPU counts to test (legacy, used when dtypes is not a mapping)
- **`exact_scales`** (bool, optional): If `false` (default), scales are extended to max with power-of-2 values

**Supported dtypes**: `fp8`, `bf16`, `nvfp4`

## GPU-Conditional Configuration Pattern

Many sections support GPU-specific overrides using the `by_gpu` pattern.

### General Pattern

```yaml
section_name:
  by_gpu:
    h100: <value_for_h100>
    b200: <value_for_b200>
    gb200: <value_for_gb200>
    gb300: <value_for_gb300>
    default: <fallback_value>  # Optional
```

### Resolution Logic

1. Check if GPU type explicitly listed → use that value
2. Else if `default` key exists → use default value
3. Else → use top-level value or system default

### Sections Supporting by_gpu

- **`container.images`**: Different containers per GPU
- **`repositories`**: Different repository versions per GPU
- **`tools`**: Different tool versions per GPU

## Complete Example

Here's a complete `metadata.yaml` example:

```yaml
general:
  workload: nemotron4
  workload_type: pretrain
  framework: nemo2

container:
  images: 
    - 'nvcr.io#nvidia/nemo:25.07.01'

repositories:
  nemo:
    url: "https://github.com/NVIDIA/NeMo.git"
    commit: "763ffa8b00a2fca9f7a204e14111ed190de7d947"
  megatron_core:
    url: "https://github.com/NVIDIA/Megatron-LM.git"
    commit: "ac198fc0d60a8c748597e01ca4c6887d3a7bcf3d"
  nemo_run:
    url: "https://github.com/NVIDIA/NeMo-Run.git"
    commit: "04f900a9c1cde79ce6beca6a175b4c62b99d7982"

downloads:
  huggingface:
    - repo_id: 'nvidia/Nemotron-4-340B-Base'
      assets: [tokenizer]

tools:
  nsys:
    by_gpu:
      h100: "2025.5.1.121-3638078"
      gb200: "2025.5.1.121-3638078"
      default: "2025.4.1.172-3634357"

setup:
  venv_req: true
  dependencies:
    pip:
      - package: nemo
        repo_key: nemo
        install_target: '.[nlp]'
      - 'scipy<1.13.0'
      - 'bitsandbytes==0.46.0'
      - package: megatron-core
        repo_key: megatron_core
      - package: nemo_run
        repo_key: nemo_run

run:
  launcher_type: 'nemo'
  launch_script: 'launch.sh'
  gpu_configs:
    h100:
      model_configs:
        - model_size: '15b'
          dtypes: ['fp8', 'bf16']
          scales: [16, 32, 64, 128, 256, 512, 1024, 2048]
        - model_size: '340b'
          dtypes: ['fp8', 'bf16']
          scales: [256, 512, 1024, 2048]
    b200:
      model_configs:
        - model_size: '15b'
          dtypes: ['fp8', 'bf16']
          scales: [16, 32, 64, 128, 256, 512, 1024]
        - model_size: '340b'
          dtypes: ['fp8', 'bf16']
          scales: [128, 256, 512, 1024]
```

## Validation

Validate your metadata file:

```bash
python -m yamale -s .gitlab/ci/metadata_schema.yaml <workload>/metadata.yaml
```

The schema validates:
- Required vs optional fields
- Field types (string, int, bool, list, etc.)
- Enum values (GPU types, dtypes, launcher types)
- Format requirements (commit SHA length, version patterns)

## Best Practices

### 1. Use GPU-Conditional Config Sparingly

Only use `by_gpu` when configurations truly differ by GPU type. Simple deployments should use the same config across GPUs.

### 2. Pin Dependencies Explicitly

```yaml
# Good
dependencies:
  pip:
    - 'scipy==1.12.0'
    - 'numpy>=1.24,<2.0'

# Avoid
dependencies:
  pip:
    - 'scipy'  # No version = unpredictable behavior
```

### 3. Use Full Commit Hashes

Always use full 40-character SHA hashes for repository commits:

```yaml
repositories:
  nemo:
    url: "https://github.com/NVIDIA/NeMo.git"
    commit: "763ffa8b00a2fca9f7a204e14111ed190de7d947"  # Good
    # commit: "763ffa8"  # BAD: short hash will fail validation
```

### 4. Document Scale Choices

Include comments explaining why certain scales are chosen:

```yaml
scales: [128, 256, 512, 1024]  # Tested scales for memory-optimal configs
exact_scales: true  # Don't extend - these are the only supported scales
```

### 5. Test After Schema Changes

Always validate and test install after modifying metadata:

```bash
# Validate schema
python -m yamale -s .gitlab/ci/metadata_schema.yaml workload/metadata.yaml

# Test installation
llmb-install express /tmp/test-install --workloads your_workload
```

## Common Patterns

### Pattern: Multi-Model Workload

```yaml
run:
  gpu_configs:
    h100:
      model_configs:
        - model_size: '7b'
          dtypes: ['fp8', 'bf16']
          scales: [8, 16, 32]
        - model_size: '70b'
          dtypes: ['fp8', 'bf16']
          scales: [64, 128, 256]
        - model_size: '405b'
          dtypes: ['fp8']
          scales: [512, 1024, 2048]
```

### Pattern: Inference Workload with Setup Task

```yaml
setup:
  venv_req: true
  tasks:
    - name: "Download model weights"
      cmd: "python download_weights.py --model $MODEL_NAME"
      job_type: local
      requires_gpus: false
      env:
        MODEL_NAME: "llama-3.1-405b"
        HF_TOKEN: "$HF_TOKEN"  # References environment variable
  dependencies:
    pip:
      - 'transformers>=4.35'
      - 'accelerate>=0.24'
```

### Pattern: GPU-Specific Container and Tools

```yaml
container:
  images:
    by_gpu:
      h100: ['nvcr.io#nvidia/nemo:25.01']
      gb200: ['nvcr.io#nvidia/nemo:25.05-gb']
      default: ['nvcr.io#nvidia/nemo:25.07.01']

tools:
  nsys:
    by_gpu:
      gb200: "2025.6.0.125-3638078"  # GB200 needs newer nsys
      # Other GPUs use container nsys
```

## Troubleshooting

### Invalid Schema Errors

**Error**: `workload_type: 'training' is not valid under any of the given enum values`

**Solution**: Use valid enum values. Check the schema for allowed values:
- workload_type: `pretrain`, `inference`, `finetune`
- GPU types: `h100`, `b200`, `gb200`, `gb300`
- dtypes: `fp8`, `bf16`, `nvfp4`

### Repository Commit Issues

**Error**: `commit: '763ffa8' is not valid - must be 40 characters`

**Solution**: Use full commit hash:
```bash
# Get full hash
git rev-parse HEAD
# Or from GitHub: click commit, copy full SHA from URL or UI
```

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'megatron'`

**Solution**: Ensure package is in dependencies and repo_key references correct repository:

```yaml
repositories:
  megatron_core:
    url: "https://github.com/NVIDIA/Megatron-LM.git"
    commit: "..."

setup:
  dependencies:
    pip:
      - package: megatron-core
        repo_key: megatron_core  # Must match repository key
```

## Additional Resources

- **[Tools Configuration Guide](tools.md)**: Detailed tool version configuration
- **[Main README](../README.md)**: Installation and usage guide
- **[Headless Installation](headless-installation.md)**: Automated deployment guide

## Schema Reference

The complete schema is defined in `.gitlab/ci/metadata_schema.yaml`. Key enums and types:

### Enums
- **GPU Types**: `h100`, `b200`, `gb200`, `gb300`, `default` (for by_gpu only)
- **Workload Types**: `pretrain`, `inference`, `finetune`, `tools`
- **Dtypes**: `fp8`, `bf16`, `nvfp4`
- **Launcher Types**: `nemo`, `megatron_bridge`, `sbatch`
- **Job Types**: `local`, `nemo2`, `srun`, `sbatch`

### Format Patterns
- **commit**: Full 40-character SHA hash
- **image URLs**: Use `#` instead of `/` (e.g., `nvcr.io#nvidia/nemo:25.07.01`)

