# Running DGXC Benchmarks on AWS with EFA

This guide explains how to adapt the DGXC Performance Recipes for AWS GPU instances that use **Elastic Fabric Adapter (EFA)** instead of InfiniBand for inter-node communication. It covers building an EFA-optimized container, applying required code patches, configuring NCCL environment variables, and verifying correct EFA operation.

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Build the EFA-Upgraded Container](#3-build-the-efa-upgraded-container)
4. [Convert Docker Image to Sqsh](#4-convert-docker-image-to-sqsh)
5. [Install the Benchmark Framework](#5-install-the-benchmark-framework)
6. [Required Code Changes](#6-required-code-changes)
7. [NCCL and EFA Environment Variables](#7-nccl-and-efa-environment-variables)
8. [Verify EFA Is Working](#8-verify-efa-is-working)
9. [Running Benchmarks](#9-running-benchmarks)
10. [Cluster Configuration Examples](#10-cluster-configuration-examples)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Overview

The DGXC Performance Recipes are designed for the NVIDIA DGX Cloud Reference Architecture, which uses InfiniBand for inter-node GPU communication. AWS GPU instances (P5en with H200, P6-B200 with B200) use **Elastic Fabric Adapter (EFA)** instead. While the stock NeMo containers include a basic EFA/libfabric stack, there is a **critical NCCL plugin naming mismatch** that causes NCCL to silently fall back to TCP sockets, resulting in significant performance degradation.

### The NCCL Plugin Naming Bug

NCCL 2.29+ uses `shinit_v2`, which sets `NCCL_NET_PLUGIN=aws-ofi`. This tells NCCL to search for a shared library named `libnccl-net-aws-ofi.so`. However, the AWS EFA installer packages the plugin as `libnccl-net-ofi.so` (without the `aws-` prefix).

**If NCCL cannot find the plugin by the expected name, it silently falls back to TCP sockets.** There is no error message -- only degraded performance. The fix is to create symlinks in the container image (see [Section 3](#3-build-the-efa-upgraded-container)).

### What This Guide Covers

- Building a container with an upgraded EFA stack, NCCL, and the critical symlink fix
- Code patches for EFA environment variables, NCCL tuning, and memory optimization
- Verification procedures to confirm EFA is operational
- Troubleshooting for common failure modes

### What This Guide Does Not Cover

- Benchmark performance numbers or tuning sweep results
- Inference workloads (SGLang, TRT-LLM, Dynamo)
- CloudWatch metrics publishing or monitoring
- GPU kernel profiling

---

## 2. Prerequisites

### Clone the Repository

```bash
git clone https://github.com/NVIDIA/dgxc-benchmarking.git
cd dgxc-benchmarking
```

Set a variable pointing to the repository root -- this is referenced throughout the guide:

```bash
export LLMB_REPO=$(pwd)
```

### AWS Instance Types

| Instance Type | GPU | Memory per GPU | GPUs per Node | Networking |
|---|---|---|---|---|
| p5en.48xlarge | NVIDIA H200 SXM | 141 GB HBM3e | 8 | EFA |
| ml.p6-b200.48xlarge | NVIDIA B200 | 183 GB HBM3e | 8 | EFA |

### Software Requirements

| Component | Minimum Version | Notes |
|---|---|---|
| Slurm | 24.11+ | Must be in PATH (`export PATH=/opt/slurm/bin:$PATH`) |
| Enroot | 3.4.1+ | Container runtime for Slurm |
| Pyxis | 0.20+ | Slurm plugin for Enroot |
| Docker | 29.x | Needed on head node for building container images |
| EFA driver | 2.17+ | Host-level EFA support |
| NVIDIA driver | 570+ | GPU driver |
| CUDA | 12.8+ | Host CUDA toolkit |
| Shared filesystem | FSx for Lustre | Mounted at a shared path (e.g., `/fsx`) |

### Additional Prerequisites

- A **HuggingFace token** (`HF_TOKEN`) with access to gated model repos (e.g., Meta-Llama)
- EFA verified on a **compute node** (not the head/login node): `/opt/amazon/efa/bin/fi_info -p efa` should list available EFA devices. The `fi_info` binary is not in the default `PATH`; use the full path shown here

---

## 3. Build the EFA-Upgraded Container

The stock NeMo container ships with an older EFA/libfabric stack. The Dockerfile provided in this guide upgrades the EFA components and fixes the NCCL plugin naming issue.

### What the Dockerfile Upgrades

| Component | Stock Container | Upgraded |
|---|---|---|
| EFA installer | Container default | 1.47.0 |
| Libfabric | Container default | Latest from EFA installer |
| aws-ofi-nccl | Container default (manual build) | Latest from EFA installer (DEB package) |
| rdma-core | Container default | Latest from EFA installer |
| NCCL | 2.28.x | 2.29.3 (matches host) |
| GDRCopy | Not present | v2.5.1 |

### Key Steps in the Dockerfile

1. **Install the EFA Installer** -- Provides libfabric, rdma-core, and the aws-ofi-nccl plugin
2. **Remove duplicate aws-ofi-nccl** -- The base image has a manually-compiled copy at `/opt/amazon/aws-ofi-nccl/lib/`; the EFA installer places its version at `/opt/amazon/ofi-nccl/lib/`
3. **Create critical symlinks** -- Maps `libnccl-net-ofi.so` to `libnccl-net-aws-ofi.so` (and the tuner equivalent) so NCCL can discover the plugin
4. **Upgrade NCCL** -- Ensures container NCCL version matches the host
5. **Install GDRCopy** -- Enables GPU-direct RDMA memory copies
6. **Fix library paths** -- Updates `ld.so.conf.d` and rebuilds the linker cache
7. **Dump build environment to `/etc/environment`** -- Pyxis/Enroot sources `/etc/environment` at container startup to set environment variables. The Dockerfile writes the full container environment to this file so that all NVIDIA, CUDA, PyTorch, and EFA variables are available at runtime. Without this, critical variables like `PATH` (missing `/opt/venv/bin`) and `TORCH_CUDA_ARCH_LIST` would be unset, causing import errors and CUDA arch detection failures

### Build the Image

```bash
# From the repository root
docker build -t nemo-efa-upgraded:25.11.01 \
  -f alternative_recipes/aws/container/nemo-efa-upgraded.Dockerfile .
```

The Dockerfile is located at [`alternative_recipes/aws/container/nemo-efa-upgraded.Dockerfile`](container/nemo-efa-upgraded.Dockerfile).

> **Note:** The build takes approximately 15-20 minutes depending on network speed.

---

## 4. Convert Docker Image to Sqsh

Enroot/Pyxis requires `.sqsh` (SquashFS) container images. Use `enroot import` to convert the Docker image built in the previous step.

### Configure Enroot Paths

Enroot needs scratch space for caching and temporary files. The defaults may point to directories that are too small or don't exist. Configure them to use a shared or local scratch filesystem:

```bash
export ENROOT_CACHE_PATH=/scratch/enroot/cache/
export ENROOT_DATA_PATH=/scratch/enroot/data/
export ENROOT_TEMP_PATH=/scratch/enroot/tmp/
```

> **Note:** The exact paths depend on your cluster configuration. Use any filesystem with at least 80 GB of free space. Common choices include `/scratch`, `/tmp` (if large enough), or a subdirectory on your shared filesystem.

### Import the Image

```bash
# Create the output directory
mkdir -p /fsx/$USER/dgxc/images

# Import the Docker image as a sqsh file
enroot import -o /fsx/$USER/dgxc/images/nvidia+nemo+25.11.01-efa-nccl29.sqsh \
  dockerd://nemo-efa-upgraded:25.11.01
```

This reads the image from the local Docker daemon, extracts the filesystem layers, and creates a SquashFS archive. It takes approximately 5-10 minutes for a ~36 GB sqsh file.

> **Note:** The `dockerd://` scheme imports from the local Docker daemon. If you see `Unable to find image locally`, verify the image was built successfully with `docker images | grep nemo-efa-upgraded`.

---

## 5. Install the Benchmark Framework

Follow the main repository [README](../../README.md) to run the installer. This sets up the CLI tools, downloads the base container image, and prepares the workload directories.

```bash
cd $LLMB_REPO
./install.sh
```

The installer will prompt you to configure your cluster and select workloads. Choose `pretrain_llama3.1` as the workload.

After the installer completes, set `LLMB_INSTALL` to the installation directory you chose during setup:

```bash
export LLMB_INSTALL=/fsx/$USER/dgxc    # adjust to your chosen install path
```

### Post-Install Fixes

**Check `gpu_gres` and `cpu_gres`:** The installer may set these to `null` in `cluster_config.yaml`. If so, fix them manually:

```bash
# Check the current values
grep gres $LLMB_INSTALL/cluster_config.yaml

# If they show 'null', fix them (8 GPUs per node for P5en and P6-B200)
sed -i 's/gpu_gres: null/gpu_gres: 8/' $LLMB_INSTALL/cluster_config.yaml
sed -i 's/cpu_gres: null/cpu_gres: 8/' $LLMB_INSTALL/cluster_config.yaml
```

**Move the EFA container image:** If you built the sqsh file to a temporary location in Section 4, move it into the install directory:

```bash
# Only needed if the sqsh file is not already in $LLMB_INSTALL/images/
mv /fsx/$USER/dgxc/images/nvidia+nemo+25.11.01-efa-nccl29.sqsh \
   $LLMB_INSTALL/images/
```

---

## 6. Required Code Changes

Four patches are provided to adapt the benchmark framework for AWS EFA. Apply them after running the installer.

### Summary of Changes

| Patch | Target File (relative to workload) | Purpose |
|---|---|---|
| `executors.py.patch` | `Megatron-Bridge/scripts/performance/utils/executors.py` | Add EFA env vars, NCCL tuning flags, `expandable_segments:True` |
| `launch.sh.patch` | `llmb_repo/llama3.1/launch.sh` | Point to the EFA-upgraded container image |
| `perf_plugins.py.patch` | `Megatron-Bridge/scripts/performance/perf_plugins.py` | Preserve cuDNN fused RMSNorm for H100 70B BF16 |
| `llama3_llm_pretrain.py.patch` | `Megatron-Bridge/scripts/performance/configs/llama3/llama3_llm_pretrain.py` | Fix `os.fork()` OOM on 70B models (`num_workers=0`) |

### Applying the Patches

```bash
# Apply patches to Megatron-Bridge (which is a git repository)
cd $LLMB_INSTALL/workloads/pretrain_llama3.1/Megatron-Bridge
git apply $LLMB_REPO/alternative_recipes/aws/patches/executors.py.patch
git apply $LLMB_REPO/alternative_recipes/aws/patches/perf_plugins.py.patch
git apply $LLMB_REPO/alternative_recipes/aws/patches/llama3_llm_pretrain.py.patch

# Apply launch script change
# Note: $LLMB_INSTALL is NOT a git repository, so use patch instead of git apply.
cd $LLMB_INSTALL
patch -p1 < $LLMB_REPO/alternative_recipes/aws/patches/launch.sh.patch
```

### Patch Details

#### executors.py -- EFA and NCCL Configuration

Adds the following environment variables to the Slurm executor:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` -- Critical memory optimization for BF16 workloads
- `FI_PROVIDER=efa` -- Forces the EFA libfabric provider
- `NCCL_SOCKET_IFNAME=^docker,lo,veth` -- Excludes virtual network interfaces from NCCL socket communication
- NCCL tuning flags: `NCCL_BUFFSIZE`, `NCCL_P2P_NET_CHUNKSIZE`, `NCCL_TUNER_PLUGIN`

See [Section 7](#7-nccl-and-efa-environment-variables) for the full environment variable reference.

#### launch.sh -- Container Image Reference

Changes the container image path from the stock NeMo sqsh to the EFA-upgraded version:

```
nvidia+nemo+25.11.01.sqsh  -->  nvidia+nemo+25.11.01-efa-nccl29.sqsh
```

#### perf_plugins.py -- cuDNN Fused RMSNorm

By default, `perf_plugins.py` deletes the `NVTE_NORM_FWD_USE_CUDNN` and `NVTE_NORM_BWD_USE_CUDNN` environment variables for the H100 70B BF16 configuration. This patch adds an exception to preserve them, enabling cuDNN fused RMSNorm which provides a small but consistent improvement on H200 GPUs.

> **Note for B200:** cuDNN fused RMSNorm showed negative results on B200 (SM100 architecture). This patch only applies to the H100/H200 code path.

#### llama3_llm_pretrain.py -- os.fork() OOM Fix

When training 70B models, the PyTorch DataLoader attempts to fork worker processes after the model is already loaded into GPU memory. On instances with high GPU memory utilization, this causes out-of-memory errors. The fix sets:

```python
cfg.dataset.num_workers = 0
cfg.dataset.pin_memory = False
```

---

## 7. NCCL and EFA Environment Variables

The following environment variables are set in `executors.py` after applying the patch. They configure EFA networking and tune NCCL for AWS GPU instances.

### EFA Networking

| Variable | Value | Purpose |
|---|---|---|
| `FI_PROVIDER` | `efa` | Forces NCCL to use the EFA libfabric provider instead of auto-detecting |
| `NCCL_SOCKET_IFNAME` | `^docker,lo,veth` | Excludes Docker bridge, loopback, and veth interfaces from NCCL socket communication |

### NCCL Tuning

| Variable | Value | Purpose |
|---|---|---|
| `NCCL_BUFFSIZE` | `8388608` (8 MB) | NCCL communication buffer size. 8 MB provides a marginal improvement over the default 4 MB |
| `NCCL_P2P_NET_CHUNKSIZE` | `8388608` (8 MB) | Base chunk size for point-to-point network transfers. **Note:** For 70B and 405B models, `perf_plugins.py` overrides this to `2097152` (2 MB) via `nccl_pp_comm_chunksize` |
| `NCCL_TUNER_PLUGIN` | `/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so` | Loads the aws-ofi-nccl tuner plugin for algorithm/protocol selection |
| `NCCL_NVLS_ENABLE` | `0` | Disables NVLink SHARP. Testing showed no performance benefit on AWS instances, and it increases GPU memory reservation |

### Memory and Compute Optimization

| Variable | Value | Purpose |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Enables PyTorch's expandable memory segments allocator. **Critical for BF16 workloads** -- without this, memory fragmentation causes severe throughput degradation |
| `NVTE_NORM_FWD_USE_CUDNN` | `1` | Enables cuDNN fused RMSNorm in the forward pass (Transformer Engine). Small improvement on H200 |
| `NVTE_NORM_BWD_USE_CUDNN` | `1` | Enables cuDNN fused RMSNorm in the backward pass (Transformer Engine). Small improvement on H200 |

### NCCL_P2P_NET_CHUNKSIZE Override Behavior

The DGXC benchmark framework sets `NCCL_P2P_NET_CHUNKSIZE` in two places:

1. **`executors.py`** -- Sets the base value as an environment variable (applies to all models)
2. **`perf_plugins.py`** -- Overrides with `nccl_pp_comm_chunksize = 2097152` specifically for 70B and 405B models

For 70B/405B models, `perf_plugins.py` always wins because it runs after the executor configuration. To change the chunksize for these models, modify it in `perf_plugins.py`.

---

## 8. Verify EFA Is Working

This is the most important validation step. Without EFA, NCCL falls back to TCP sockets with no error message.

### Enable NCCL Debug Logging

Temporarily add these to your executor environment variables or Slurm submission:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

### Good -- EFA Is Active

Look for these patterns in the job output:

```
NCCL INFO Loaded net plugin Libfabric (v9)
NCCL INFO Using network Libfabric
NCCL INFO Selected provider is efa (found 16 nics)
NCCL INFO Using transport protocol RDMA
```

### Bad -- Fallen Back to TCP Sockets

These patterns indicate NCCL could not find the EFA plugin:

```
NCCL INFO Could not find: aws-ofi libnccl-net-aws-ofi.so
NCCL INFO Failed to initialize NET plugin IB
NCCL INFO Using network Socket
NCCL INFO GPU Direct RDMA Disabled for HCA 0
```

### Diagnostic Script

Run this inside the container to verify the EFA stack installation:

```bash
srun --partition=<PARTITION> --account=<ACCOUNT> --gpus-per-node=8 -N 1 \
  --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.11.01-efa-nccl29.sqsh \
  --container-mounts=/fsx:/fsx \
  bash -c '
    echo "=== Plugin symlinks ==="
    ls -la /opt/amazon/ofi-nccl/lib/libnccl-net-*
    ls -la /opt/amazon/ofi-nccl/lib/libnccl-*tuner*
    echo ""
    echo "=== Libfabric version ==="
    fi_info --version
    echo ""
    echo "=== EFA provider ==="
    fi_info -p efa 2>&1 | head -5
    echo ""
    echo "=== NCCL version ==="
    python3 -c "import torch; print(torch.cuda.nccl.version())"
  '
```

---

## 9. Running Benchmarks

After building the container and applying the patches, use the standard `llmb-run` CLI to submit benchmark jobs.

### Example: Pretrain Llama 3.1 8B (64 GPUs)

```bash
cd $LLMB_INSTALL
export HF_TOKEN="<YOUR_HF_TOKEN>"

llmb-run submit -w pretrain_llama3.1 -s 8b -d bf16 --scale 64
```

### Example: Pretrain Llama 3.1 70B (64 GPUs)

```bash
llmb-run submit -w pretrain_llama3.1 -s 70b -d bf16 --scale 64
```

### Example: Pretrain Llama 3.1 70B FP8 (64 GPUs)

```bash
llmb-run submit -w pretrain_llama3.1 -s 70b -d fp8 --scale 64
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch job logs
tail -f $LLMB_INSTALL/workloads/pretrain_llama3.1/experiments/*/slurm-*.out

# Check completed job
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,ExitCode
```

### Verify EFA in the First Run

On your first run, enable NCCL debug logging (see [Section 8](#8-verify-efa-is-working)) to confirm EFA is active. Once verified, disable debug logging for subsequent runs to reduce log noise.

---

## 10. Cluster Configuration Examples

After running the installer (`./install.sh`), a `cluster_config.yaml` file is created in `$LLMB_INSTALL`. Below are example configurations for AWS GPU instances.

### H200 (P5en) -- ParallelCluster

```yaml
launcher:
  gpu_type: h100          # No H200-specific profile exists; h100 is compatible
  node_architecture: x86_64
install:
  venv_type: venv
  method: slurm
  node_architecture: x86_64
environment:
  HF_TOKEN: <YOUR_HF_TOKEN>
slurm:
  account: <YOUR_SLURM_ACCOUNT>
  gpu_partition: <YOUR_GPU_PARTITION>
  gpu_gres: 8
  cpu_partition: <YOUR_CPU_PARTITION>
  cpu_gres: 8
```

> **Note:** Use `gpu_type: h100` for H200 instances. The H100 parallelism profiles are compatible with H200 hardware. There are no H200-specific profiles in the framework.

### B200 (P6-B200) -- SageMaker HyperPod

```yaml
launcher:
  gpu_type: b200
  node_architecture: x86_64
install:
  venv_type: uv
  method: slurm
  node_architecture: x86_64
environment:
  HF_TOKEN: <YOUR_HF_TOKEN>
slurm:
  account: <YOUR_SLURM_ACCOUNT>
  gpu_partition: <YOUR_GPU_PARTITION>
  gpu_gres: 8
  cpu_partition: <YOUR_CPU_PARTITION>
  cpu_gres: 8
```

> **Key difference:** B200 uses `gpu_type: b200`, which selects B200-specific parallelism presets. These presets use Megatron FSDP (fully sharded data parallel) instead of the TP/PP parallelism used for H100/H200.

---

## 11. Troubleshooting

### Performance is significantly lower than expected

**Symptom:** Throughput is ~20-25% lower than expected.

**Cause:** NCCL is using TCP sockets instead of EFA/RDMA. This happens silently when the NCCL plugin symlinks are missing.

**Fix:** Verify the container has the correct symlinks (Section 3, step 3). Enable NCCL debug logging and check for `Using network Socket` vs `Using network Libfabric` ([Section 8](#8-verify-efa-is-working)).

### `enroot import dockerd://` fails with "Unable to find image locally"

**Cause:** Enroot cannot access the Docker daemon's image store, or the image was not built successfully.

**Fix:** First verify the image exists with `docker images | grep nemo-efa-upgraded`. If the image is listed but enroot still cannot find it, there are two common causes:

1. **Missing enroot paths:** Ensure the `ENROOT_CACHE_PATH`, `ENROOT_DATA_PATH`, and `ENROOT_TEMP_PATH` environment variables are set to paths with sufficient disk space (see [Section 4](#4-convert-docker-image-to-sqsh)).

2. **Docker buildx attestation manifests:** On Docker installations using the containerd image store, `docker build` may create attestation manifests (provenance and SBOM) that produce a manifest list which the classic Docker API cannot resolve. You can confirm this if `docker images` lists the image but `docker image inspect <image>` returns "No such image". To fix, rebuild with attestations disabled:

   ```bash
   docker build --provenance=false --sbom=false \
     -t nemo-efa-upgraded:25.11.01 \
     -f alternative_recipes/aws/container/nemo-efa-upgraded.Dockerfile .
   ```

### `os.fork()` causes OOM on 70B models

**Cause:** The PyTorch DataLoader forks worker processes after the model is loaded into GPU memory, exceeding available memory.

**Fix:** Apply the `llama3_llm_pretrain.py.patch` which sets `num_workers=0` and `pin_memory=False`.

### `numactl: command not found`

**Cause:** The NeMo executor injects `numactl` pre-commands, but `numactl` is not installed on the compute nodes.

**Fix:** Install `numactl` on the compute nodes, or remove/comment out the numactl lines from the executor in `executors.py`.

### `CUDA error: dependency created on uncaptured work in another stream`

**Cause:** CUDA graphs are enabled together with TP communication overlap userbuffers at TP=4 (H100/H200 configuration). This combination is incompatible.

**Fix:** Do not enable CUDA graphs with TP=4 and userbuffers communication overlap. The default H100 70B BF16 config correctly has `cuda_graph_impl=None`. Only TP=2 configurations (B200) can use CUDA graphs, and even then, `expandable_segments:True` may conflict with graph capture.

### `NCCL_P2P_NET_CHUNKSIZE` changes have no effect on 70B

**Cause:** `perf_plugins.py` overrides the value set in `executors.py` for 70B and 405B models.

**Fix:** Modify the chunksize in `perf_plugins.py` (look for `nccl_pp_comm_chunksize`), not in `executors.py`.

### `ModuleNotFoundError: No module named 'nemo_run'` or `IndexError` in CUDA arch detection

**Cause:** The container's `/etc/environment` file is missing or incomplete. Pyxis/Enroot relies on this file to set environment variables like `PATH` and `TORCH_CUDA_ARCH_LIST` inside the container.

**Fix:** Ensure you are using the Dockerfile from this guide, which dumps the full build environment to `/etc/environment` (step 7 in [Section 3](#3-build-the-efa-upgraded-container)). If you built a custom Dockerfile, add the environment dump step before the final verification stage.

### `gpu_gres: null` in `cluster_config.yaml`

**Cause:** The installer sometimes fails to detect the number of GPUs per node and sets `gpu_gres` and `cpu_gres` to `null`.

**Fix:** Manually edit `$LLMB_INSTALL/cluster_config.yaml` and set both values to `8` (see [Section 5](#5-install-the-benchmark-framework)).
