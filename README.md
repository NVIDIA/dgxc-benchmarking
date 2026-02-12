# DGX Cloud Benchmarking - Performance Recipes

Performance Recipes are ready-to-use templates for evaluating performance of specific AI use cases across hardware and software combinations. These containerized recipes allow users to quickly set up and run standardized benchmarking methodology in their own environment, ensuring consistent and comparable results across platforms. 

These Performance Recipes support performance characterization
- across a variety of defined AI workloads, including pre-training, fine tuning, and inference. 
- across GPU-based infrastructure, whether running on-premises or with cloud service providers (CSPs). 

Each recipe maps to one workload and can be run at various cluster scales and precisions. These workloads are tested against the NVIDIA Reference Architecture and those results are provided as a baseline for comparison. These performance metrics are collected from production environments and are subject to real-world variability.


## Prerequisites

To use the Performance Recipes, make sure you have the following prerequisites installed on your cluster:

### General Prerequisites

*   Bash 4.2 or newer
*   [Git LFS](https://git-lfs.com/)
*   [NGC Registry Access](https://org.ngc.nvidia.com/setup)
*   NGC CLI 3.148.1 or newer (Optional, required for NIM Inference workloads)
*   Python 3.12.x
*   [CUDA](https://developer.nvidia.com/cuda-downloads): at least 12.3, recommended: 12.8 or newer
*   [NV Driver](https://www.nvidia.com/en-us/drivers/): at least 535.129.03, recommended 570.172.08 or newer
*   [OFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/): 5.9-0.5.6.0.127 or newer
*   [NCCL](https://developer.nvidia.com/nccl/nccl-download): 2.19.4 or newer

### Cluster-Specific Prerequisites

Depending on your cluster's job scheduler, ensure the following are met:

*   **Slurm Clusters**
    *   Version 22.x or newer
    *   `task/affinity` plugin required for process pinning
    *   [Enroot](https://github.com/NVIDIA/enroot/)
    *   [Pyxis](https://github.com/NVIDIA/pyxis)


## Quick Start Guide

> **Important:** Before proceeding with installation, please review the [Known Issues](#known-issues) section.

1. Clone the repository:
   ```bash
   git clone https://github.com/NVIDIA/dgxc-benchmarking.git
   cd dgxc-benchmarking
   ```
2. Obtain Model Access (if required):
   Some workloads require special access or HuggingFace tokens. Please refer to the [Model Access Requirements](#model-access-requirements) table to determine if your chosen workloads need additional approvals or a HuggingFace token (HF_TOKEN in installer). If a token is required, generate one from [HuggingFace settings](https://huggingface.co/settings/tokens) as applicable.
   
   **Note:** Approvals are not instantaneous, please plan accordingly.

3. (Optional) For NIM Inference workloads only:
   - Generate an NGC API key from the [NGC Registry](https://org.ngc.nvidia.com/setup)
   - Install and configure the NGC CLI:
    <details>
    
    <summary>x86</summary>
    
    ```bash
    curl -L https://ngc.nvidia.com/downloads/ngccli_linux.zip -o ngccli_linux.zip
    unzip -q ngccli_linux.zip -d $HOME/.local/bin
    rm ngccli_linux.zip
    export PATH=$HOME/.local/bin:$PATH
    ngc config set
    ```
    
    </details>
    
    <details>
    
    <summary>arm64</summary>
    
    ```bash
    curl -L https://ngc.nvidia.com/downloads/ngccli_arm64.zip -o ngccli_arm64.zip
    unzip -q ngccli_arm64.zip -d $HOME/.local/bin
    rm ngccli_arm64.zip
    export PATH=$HOME/.local/bin/ngc-cli:$PATH
    ngc config set
    ```
    </details>

4. Run the installer:
   
   **Important:** Installation may take several hours, influenced by selected recipes, internet speed, and your current node's resources. Consider using a tool like `tmux` or `screen`.

   ```bash
   # The installer will add packages to your current Python environment.
   # We recommend to activate a virtual environment with python 3.12.x before running installer command below.
   # The installer has been tested with venv and conda environments; other solutions may work but are not officially supported.
   
   ./install.sh
   ```
   The installer will:
   - Install required Python packages in your current environment
   - Set up your benchmarking environment
   - Configure SLURM settings
   - Let you select workloads to install
   - Prepare all necessary dependencies

   > **Note:** For detailed installation options, workload-specific virtual environments, and troubleshooting, see the [Installer README](cli/llmb-install/README.md).

5. Run a benchmark:
   ```bash
   # Navigate to your installed workload directory
   cd $LLMB_INSTALL
   
   # Example: Run Nemotron4 340B pretrain on 256 GPUs with FP8 precision
   llmb-run single -w pretrain_nemotron4 -s 340b --dtype fp8 --scale 256
   ```

### Directory Layout and Key Variables

After running the installer, the following directory structure is created:

- `LLMB_REPO`: Directory containing the clone of the recipe repository.
- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_nemotron4`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

**Example structure:**
```
$LLMB_INSTALL/
  ├── images/
  ├── datasets/
  ├── venvs/
  └── workloads/
        └── pretrain_nemotron4/   # <- $LLMB_WORKLOAD
              ├── NeMo/
              ├── ...
              └── experiments/
```

`LLMB_REPO`, `LLMB_INSTALL`, and `LLMB_WORKLOAD` are shorthand terms for directory locations; `LLMB_INSTALL` is the only environment variable that needs to be set by the user.



## Workload Resources Overview

Each workload resource includes:
* **Configuration details**: Comprehensive software and hardware setup information.
* **Performance scripts**: Predefined scripts to generate and analyze performance results.

The overview page for each workload highlights target performance metrics for the specified configuration, focusing on speed measurements such as the time taken per training step and the number of tokens processed per second.

## Available Benchmarks

The following tables list each benchmark used to evaluate the model's performance, along with their specific configurations.

**Note:** The "Scale (# of GPUs)" column indicates the minimum supported scale and the maximum scale tested for each workload. The recipes may function at larger scales (unless otherwise noted in workload specific README), although they have not been explicitly validated beyond the listed maximum.

### GB300 Workloads

| Type | Framework | Model | Container Version | Model Size | Scale (# of GPUs) | Precision | Model Access Required | Checkpointing | Cluster Type |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pretrain | Megatron-Bridge | [DeepSeek V3](deepseek_v3/pretrain/README.md) | 25.09.00 | 671B | 128-512 | FP8, BF16 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 405B | 128-1024 | FP8 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 70B | 64-1024 | FP8 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 8B | 8-128 | FP8 | Yes | No | Slurm |
| Pretrain | NeMo | [Nemotron4](nemotron4-15b/README.md) | 25.09.00 | 15B | 16-512 | FP8, BF16 | No | Yes | Slurm |
| Pretrain | NeMo | [Nemotron4](nemotron4-340b/README.md) | 25.09.00 | 340B | 128-512 | FP8, BF16 | No | Yes | Slurm |
| Pretrain | NeMo | [Grok1](grok1/README.md) | 25.09.00 | 314B | 128-512 | FP8, BF16 | Yes | No | Slurm |
| Pretrain | NeMo | [Nemotron-H](nemotron-h/README.md) | 25.09.00 | 56B | 32-512 | FP8 | No | No | Slurm |


### GB200 Workloads

| Type | Framework | Model | Container Version | Model Size | Scale (# of GPUs) | Precision | Model Access Required | Checkpointing | Cluster Type |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pretrain | NeMo | [Nemotron4](nemotron4-15b/README.md) | 25.09.00 | 15B | 16-512 | FP8, BF16 | No | Yes | Slurm |
| Pretrain | NeMo | [Nemotron4](nemotron4-340b/README.md) | 25.07.01 | 340B | 128-512 | FP8, BF16 | No | Yes | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 405B | 128-512 | FP8 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 70B | 64-512 | FP8 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 8B | 8-128 | FP8 | Yes | No | Slurm |
| Pretrain | NeMo | [DeepSeek V3](deepseek_v3/pretrain/README.md) | 25.09.00 | 671B | 128-512 | FP8, BF16 | Yes | No | Slurm |
| Pretrain | NeMo | [Grok1](grok1/README.md) | 25.09.00 | 314B | 128-512 | FP8, BF16 | Yes | No | Slurm |
| Pretrain | NeMo | [Nemotron-H](nemotron-h/README.md) | 25.09.00 | 56B | 32-512 | FP8 | No | No | Slurm |
| Inference | TRT-LLM | [DeepSeek R1](deepseek_r1/inference/trtllm/README.md) | 1.1.0rc5 | 671B | 4 | nvfp4 | No | No | Slurm |
| Inference | Dynamo | [DeepSeek R1](deepseek_r1/inference/dynamo/README.md) | 0.6.1 | 671B | 32 | NVFP4 | No | No | Slurm |
| Inference | SGLang | [DeepSeek R1](deepseek_r1/inference/sglang/README.md) | v0.5.3-cu129-gb200 | 671B | 4 | nvfp4 | No | No | Slurm |
| Inference | TRT-LLM | [Llama 3.3](llama3.3/inference/README.md) | 1.1.0rc5 | 70b | 1-4 | nvfp4 | Yes | No | Slurm |
| Inference | Dynamo + TRT-LLM | [GPT-OSS Inference](gpt-oss/inference/README.md) | 0.5.1-rc0.pre3 | 120B | 4+ | mxfp4 | No | No | Kubernetes |
| Microbenchmark | TRT-LLM | [GPT-OSS](microbenchmarks/cpu_overhead/README.md) | 1.1.0rc5 | 120B | 1-4 | mxfp4 | Yes | No | Slurm |


### B200 Workloads

| Type | Framework | Model | Container Version | Model Size | Scale (# of GPUs) | Precision | Model Access Required | Checkpointing | Cluster Type |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pretrain | NeMo | [Nemotron4](nemotron4-15b/README.md) | 25.09.00 | 15B | 16-1024 | FP8, BF16 | No | Yes | Slurm |
| Pretrain | NeMo | [Nemotron4](nemotron4-340b/README.md) | 25.07.01 | 340B | 128-1024 | FP8, BF16 | No | Yes | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 405B | 128-1024 | FP8 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 70B | 64-1024 | FP8 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 8B | 8-128 | FP8 | Yes | No | Slurm |
| Pretrain | NeMo | [DeepSeek V3](deepseek_v3/pretrain/README.md) | 25.09.00 | 671B | 128-1024 | FP8, BF16 | Yes | No | Slurm |
| Pretrain | NeMo | [Grok1](grok1/README.md) | 25.09.00 | 314B | 256-1024 | FP8, BF16 | Yes | No | Slurm |
| Pretrain | NeMo | [Nemotron-H](nemotron-h/README.md) | 25.09.00 | 56B | 32-512 | FP8 | No | No | Slurm |
| Inference | Dynamo | [DeepSeek R1](deepseek_r1/inference/dynamo/README.md) | 0.6.1 | 671B | 32 | NVFP4 | No | No | Slurm |
| Inference | SGLang | [DeepSeek R1](deepseek_r1/inference/sglang/README.md) | v0.5.3rc0-cu128-b200 | 671B | 8 | nvfp4 | No | No | Slurm |
| Microbenchmark | TRT-LLM | [GPT-OSS](microbenchmarks/cpu_overhead/README.md) | 1.1.0rc5 | 120B | 1-4 | mxfp4 | Yes | No | Slurm |

### H100 Workloads

Baseline performance metrics were collected using workloads on the NVIDIA DGX H100 Reference Architecture. For more information see [DGX H100 Systems](https://blogs.nvidia.com/blog/dgx-h100-systems-shipping/).

| Type | Framework | Model | FW Container Version | Model Size | Scale (# of GPUs) | Precision | Model Access Required | Checkpointing | Cluster Type |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pretrain | NeMo | [Nemotron4](nemotron4-15b/README.md) | 25.09.00 | 15B | 16-2048 | FP8, BF16 | No | Yes | Slurm, Run:ai |
| Pretrain | NeMo | [Nemotron4](nemotron4-340b/README.md) | 25.09.00 | 340B | 256-2048 | FP8, BF16 | No | Yes | Slurm, Run:ai |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 405B | 1024 | FP8 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 70B | 64-1024 | FP8 | Yes | No | Slurm |
| Pretrain | Megatron-Bridge | [Llama 3.1](llama3.1/README.md) | 25.09.00 | 8B | 8-128 | FP8 | Yes | No | Slurm |
| Pretrain | NeMo | [DeepSeek V3](deepseek_v3/pretrain/README.md) | 25.09.00 | 671B | 1024 | BF16 | Yes | No | Slurm |
| Pretrain | NeMo | [Grok1](grok1/README.md) | 25.09.00 | 314B | 512-2048 | FP8, BF16 | Yes | No | Slurm |
| Pretrain | NeMo | [Nemotron-H](nemotron-h/README.md) | 25.09.00 | 56B | 64-2048 | FP8 | No | No | Slurm |
| Inference | TRT-LLM | [DeepSeek R1](deepseek_r1/inference/trtllm/README.md) | 1.1.0rc5 | 671B | 16 | fp8 | No | No | Slurm |
| Inference | Dynamo | [DeepSeek R1](deepseek_r1/inference/dynamo/README.md) | 0.6.1 | 671B | 48 | FP8 | No | No | Slurm |
| Inference | TRT-LLM | [Llama 3.3](llama3.3/inference/README.md) | 1.1.0rc5 | 70b | 2 | fp8 | Yes | No | Slurm |
| Microbenchmark | TRT-LLM | [GPT-OSS](microbenchmarks/cpu_overhead/README.md) | 1.1.0rc5 | 120B | 1-4 | mxfp4 | Yes | No | Slurm |

### Deprecated

| Type | Framework | Model | FW Container Version | Model Size | Scale (# of GPUs) | Precision | Model Access Required | Checkpointing | Cluster Type | Last Version |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Finetuning | HF | Llama 2 | 24.02-py3 | 70B | 8-512 | FP8, BF16 | Yes | No | Slurm | 25.01.1 |
| Finetuning | HF | Mistral | 24.02-py3 | 7B | 8-256 | FP8, BF16 | Yes | No | Slurm | 25.01.1 |
| Pretrain | Jax | Llama 2 | jax:maxtext-2024-12-09 | 70B | 128-2048 | FP8, BF16 | No | No | Slurm | 25.01.1 |
| Pretrain | Jax | GPT3 | jax:pax-2024-03-04 | 175B | 128-2048 | FP8, BF16 | No | No | Slurm | 25.01.1 |
| Pretrain | Maxtext | Llama3 |  25.01 |70B | 128-2048 | FP8, BF16 | No | No | Slurm | 25.04.02 |
| Pretrain | NeMo | GPT3 | 24.12 | 175B | 128-2048 | FP8, BF16 | No | No | Slurm | 25.04.02 |
| Pretrain | NeMo | Llama4 Maverick | 25.07.01 | 400B | 512-2048 | FP8, BF16 | Yes | No | Slurm | 25.08 |
| Fine-Tuning (SFT, LORA) | NeMo | Llama 3 | 24.12 | 8B, 70B | 8-32 | FP8, BF16 | Yes | No | Slurm | 25.04.02 |
| Finetune | NeMo Maverick | Llama4 | 25.07.01 | 400B | 256 | FP8, BF16 | Yes | No | Slurm | 25.08 |
| Inference | NIM | Llama 3 | 1.0.3 | 70B | 4 | FP8 | Yes | No | Slurm | 25.05.04 |
| Inference | NIM, SGLang | DeepSeek R1 | 1.7.2 | 671B | 16 | fp8 | No | No | Slurm | 25.08 |
| Inference | NIM & NeMo Retriever (NVIDIA Enterprise RAG) | Llama 3.1 and 3.2 | instruct:1.3.3, rerank:1.3, embed:1.3.1 | 70b, 1b | 1-8 | n/a | Yes | No | Slurm | 25.08 |
| Inference | TRT-LLM | Llama 4 | 1.0.0rc1 | 17b | 8 | fp8 | Yes | No | Slurm | 25.08 |

## Model Access Requirements

Some models require a HuggingFace account and HF_TOKEN, others require repo specific approvals. Please review the table below for specific access requirements for each recipe. 

**Note:** approval processes are not immediate and may take some time.

| Recipe Type    | Recipe Name      | HF Token Required        | Additional Approval Required | Details/Link for Approval                                             |
| :------------- | :--------------- | :----------------------- | :--------------------------- | :-------------------------------------------------------------------- |
| Pretrain       | Llama 3.1        | Yes                      | Yes                          | [HuggingFace Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-405B) |
| Pretrain       | DeepSeek V3      | Yes                      | No                           | N/A                                                                   |
| Pretrain       | Grok1            | Yes                      | Yes                          | [HuggingFace Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-70B) |
| Pretrain       | Nemotron4        | Yes                      | No                           | N/A                                                                   |
| Inference      | Llama 3.3        | Yes                      | Yes                          | [HuggingFace Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| Inference      | DeepSeek R1      | Yes                      | No                           | N/A                                                                    |
| Microbenchmark | CPU overhead     | Yes                      | No                           | [HuggingFace GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) |

# Reference Infrastructure

The LLM Benchmarking Collection published baseline benchmark results using the following reference infrastructures, CSP-specific configurations, and software.

## GB300 Reference Architecture

Baseline performance metrics for GB300 workloads were collected using systems equipped with the NVIDIA GB300 Grace Blackwell Superchip. For more information see [NVIDIA Blackwell Platform](https://www.nvidia.com/en-us/data-center/gb300-nvl72/?nvid=nv-int-solr-857371-vt34).

* GB300 Grace Blackwell Superchip
  * CPU: 72 Arm Neoverse V2 cores with 4x 128b SVE2
    * 3.5 GHz (max boost)
    * Low-latency coherent interconnect between Grace CPU and B300 GPUs
    * RAM: 960 GiB LPDDR5X (2x 480 GiB) | 546 GB/s
    * Total Accessible Memory: 2 TiB
    * 64x PCIe Gen5 lanes
  * 2x B300 GPUs
    * 288GB HBM3e per GPU (576GB total per superchip)
    * Memory bandwidth 12 TB/s per GPU
* NVLink: NVLink 5th Generation
  * 1.8 TB/s per GPU bandwidth
* System Memory: Coherent memory architecture between Grace CPU and Blackwell GPUs

## GB200 Reference Architecture

Baseline performance metrics for GB200 workloads were collected using the NVIDIA DGX GB200 Reference Architecture. For more information see [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

* GB200 Grace Blackwell Superchip
  * CPU: 72 Arm Neoverse V2 cores with 4x 128b SVE2
    * 3.5 GHz (max boost)
    * Low-latency coherent interconnect between Grace CPU and B200 GPUs
    * RAM: 960 GiB LPDDR5X (2x 480 GiB) | 546 GB/s
    * Total Accessible Memory: 1.7 TiB
    * 64x PCIe Gen5 lanes
  * 2x Blackwell GPUs
    * Memory bandwidth 16 TB/s
* NVLink: NVLink 5th Generation
  * 1.8 TB/s per GPU bandwidth

## B200 Reference Architecture

Baseline performance metrics for B200 workloads were collected using systems equipped with NVIDIA B200 GPUs. For more information see [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).

* GPU: 8xB200 192GB HBM3e (1.5TB total)
  * TDP 1000W
  * Memory bandwidth 8 TB/s
* CPU: Intel Xeon Platinum 8570 x2
  * 40 cores per socket
  * 4 Ghz (max boost)
  * RAM: 1 TiB | 1.6 TB/s per socket
  * 48x PCIe Gen5 lanes
* NVLink: NVLink 5th Generation
  * 1.8 TB/s per GPU bandwidth
  * 18 Links per GPU
* InfiniBand:
  * Compute links: 8x 400 Gbit/s
* System Memory: 2TB

## H100 Reference Architecture

Baseline performance metrics for H100 workloads were collected using the NVIDIA DGX H100 Reference Architecture. For more information see [DGX H100 Systems](https://blogs.nvidia.com/blog/dgx-h100-systems-shipping/).

* GPU: 8xH100 80GB HBM3 (640GB total)
  * TDP 700W
  * Memory bandwidth 3.2 TB/s
* CPU: 2x Intel Sapphire Rapids, Intel(R) Xeon(R) Platinum 8480CL E5
  * 112 cores (56 cores per CPU)
  * 2.00 GHz (Base), 3.8 GHz (Max boost)
  * Numa nodes per socket = 1
  * PCIe Gen5
* NVLink: NVLink 4th Generation
  * 900 GB/s per GPU bandwidth
  * 18 Links per GPU
* InfiniBand:
  * Compute links: 8x 400 Gbit/s
  * Storage links: 2x 400 Gbit/s
* System Memory: 2TB
* Local Storage:
  * 2x 1.92TB NVMe M.2
  * 8x 3.84TB NVMe U.2


## CSP Specific Configurations
AI platforms may vary in implementation, such as differences in network fabric and virtualization implementations, and thus require different tuning. 
For optimal performance, users should leverage the correct implementation for their platform. The example platform-specific tuning is provided as a starting point. Further tuning may be necessary if instance type varies from the Reference Architecture. 

### AWS

For NeMo based images EFA support is already included starting with version 25.02 (nvcr.io/nvidia/nemo:25.02).

For other images or if you need to update Enable Elastic Fabric Adapter (EFA) follow the [step-by-step guide](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-efa.html#your-algorithms-training-efa-install). Use the [reference NCCL tests Dockerfile with EFA support](https://github.com/aws-samples/awsome-distributed-training/blob/main/micro-benchmarks/nccl-tests/nccl-tests.Dockerfile). 

### GCP
Ensure that all required pre-conditions for [GCP cluster deployment](https://cloud.google.com/ai-hypercomputer/docs/create/create-slurm-cluster) have been met. 

Configure Compute Fabric with TCP-X by ensuring the following environment variables are set and present for your environment. 

```shell
NCCL_LIB_DIR='/var/lib/tcpxo/lib64' source /var/lib/tcpxo/lib64/nccl-env-profile.sh; \
	  export NCCL_FASTRAK_CTRL_DEV=enp0s12; \
	  export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0; \
	  export NCCL_SOCKET_IFNAME=enp0s12; \
	  export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices; \
	  export NCCL_NET=FasTrak; \
	  ls /var/lib/tcpxo/lib64;"
```

**Important:** 
* The above example hasn't been tested with the latest TCP-X version. Check with your cluster admin for the most recent instructions.
* If additional files need to be mounted into running container, they should be placed under `$LLMB_WORKLOAD` folder as this location is already mounted. 

### Azure 
Requires two settings for optimal performance:
   1. **NCCL_TOPO_FILE**=`<path to topo file under $LLMB_WORKLOAD>`.
      * The VM topology files ensure that the correct CPUs, GPUs and NICs are bound together. Location of this file varies, it **must** be mounted into the container.
      * **Important:** Place NCCL Topology file under `$LLMB_WORKLOAD` folder as this location is already mounted into running container. 
   2. **NCCL_P2P_CHUNKSIZE**=2097152
      * Increasing message size for NCCL send/recv for optimal performance


Example Configuration for a training recipe:
```shell
export NCCL_TOPO_FILE=$LLMB_WORKLOAD/nvd5-topo.xml # Exact location varies by cluster
export NCCL_P2P_NET_CHUNKSIZE=2097152
```

# Release Notes

For the latest updates, improvements, and breaking changes, see the [CHANGELOG](CHANGELOG).


# FAQ

Contains synopsis and resolution for known issues

## 1. Training logs contain multiple userbuffers.cu messages

### Symptom
Large scale pre-training run logs contain message like below:

```
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 18 [2]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 18 [4]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 19 [2]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 19 [4]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 22 [2]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 22 [4]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 23 [2]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 23 [4]: expecting 1 got 0
```

### Solution
These usually mean that one of the GPUs is hanging. Possible resolutions: 
  * re-running the job on a different set of nodes
  * rebooting affected nodes.

## 2. Slurm job failed, need to find log files

### Symptom
A Slurm job failed during benchmark run. E.g., a nemotron benchmark job with ID=2041792 failed

```
sacct -j 2041792
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode
------------ ---------- ---------- ---------- ---------- ---------- --------
2041792        launch.sh     batch test              224     FAILED      1:0
2041792.bat+      batch            test              224     FAILED      1:0
2041792.ext+     extern            test              224  COMPLETED      0:0
2041792.0          bash            test              224     FAILED      1:0
```

### Solution

#### NeMo2 (e.g., Nemotron4)
You can find log files associated with this run under `$LLMB_WORKLOAD/experiments/pretrain_nemotron4_<size>_<dtype>_<scale>_<config>` folder. The folder will have subfolders that will contain `log-account.pretrain_nemotron4_<size>_<dtype>_<scale>_<config>.out` files with a root cause error message.

E.g., for the job failure above and assuming the nemotron 15b job ran on 16 GPUs, used version 25.05, and with precision bf16 the path will be under `$LLMB_WORKLOAD/experiments/pretrain_nemotron4_15b_bf16_gpus16_tp1_pp1_cp1_vp1_mbs2_gbs64/...`

Search for errors in the `log-account.pretrain_nemotron4_15b_bf16_gpus16_tp1_pp1_cp1_vp1_mbs2_gbs64_3358926_0.out` file. 

## 3. Unable to use venv required by benchmark

### Symptom

If a benchmark requires virtual python environment (venv) but `virtualenv` executable isn't available on the login node and/or login nodes cannot be updated by non-sudo users, you would see errors like below when trying to setup venv

```shell
bash-5.2$ virtualenv
bash: virtualenv: command not found
```

### Solution

There are alternative virtual environment options available like **conda**.

To install and activate conda virtual environment
```shell
# pick INSTALL_PATH with sufficient disk space
INSTALL_PATH=~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $INSTALL_PATH/miniconda.sh
bash $INSTALL_PATH/miniconda.sh -b -p $INSTALL_PATH/miniconda3
$INSTALL_PATH/miniconda3/bin/conda init
source ~/.bashrc
```

When you are finished running this benchmark you can deactivate the environment, run this command
```shell
conda deactivate
```

## 4.Tritonclient errors during installation

### Symptom

During installation of a recipe you may see an error about `tritonclient` dependency. It would look like this:

`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tritonclient 2.51.0 requires urllib3>=2.0.7, but you have urllib3 1.26.20 which is incompatible.`

### Solution

The error can be ignored as it doesn't affect benchmark functionality. 

# Known Issues

## 1. Multiple GPU Types on Single Cluster

### Issue
The `llmb-install` tool currently supports only one GPU type per installation. If your cluster contains multiple GPU types (e.g., H100 and B200), you cannot install workloads for both GPU types in a single installation.

### Workaround
Create separate installations for each GPU type:

1. Run the installer once for your first GPU type (e.g., H100):
   ```bash
   ./install.sh
   # Select H100 workloads and specify an installation directory
   ```

2. Run the installer again for your second GPU type (e.g., B200):
   ```bash
   ./install.sh
   # Select B200 workloads and specify a different installation directory
   ```

Each installation will have its own `LLMB_INSTALL` directory. Use the appropriate `LLMB_INSTALL` directory for running workloads for each GPU type.

## 2. Cleanup-phase errors after successful run

### Issue
Some workloads complete all timesteps but print errors during the cleanup phase. This previously caused the Slurm job to be marked as failed.

### Workaround
We now detect this case and convert the exit code so Slurm reports success when the run actually finished. Log files will still contain the cleanup errors. If the job completed all timesteps and Slurm shows COMPLETED, you can ignore cleanup errors in the logs. This will be fixed in a future release.

## 3. uv 0.9.29+ breaks all recipes that use nemo_run

### Issue
Nearly every recipe installs `nemo_run` and will fail with `uv` `0.9.29+` due to uv rejecting unknown fields in `pyproject.toml` files.

### Workaround
Run `./install.sh` from this release. It enforces `uv <=0.9.28`, which avoids the strict parser breakage.

# Support

Terminology used in these recipes is explained in the [Appendix](APPENDIX.md).

For questions or to provide feedback, please contact LLMBenchmarks@nvidia.com