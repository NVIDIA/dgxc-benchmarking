# LLM Benchmarking Collection

Evaluate performance and efficiency of deep learning models with the LLM Benchmarking Collection.

The LLM Benchmarking Collection provides a suite of tools to quantify the performance of large language models (LLMs) and fine-tuning workloads across GPU-based infrastructure, whether running on-premises or with cloud service providers (CSPs).

## Prerequisites

Before you use the LLM Benchmarking Collection, make sure you have installed the following packages on your cluster.
* Bash 4.2 or newer
* Slurm 22.x or newer
* [Enroot](https://github.com/NVIDIA/enroot/)
* [NGC Registry Access](https://org.ngc.nvidia.com/setup)
* Python 3.9.x or newer
* CUDA 12.3 or newer
* NV Driver: 535.129.03 or newer
* OFED: 5.9-0.5.6.0.127 or newer
* NCCL: 2.19.4 or newer


## Accessing and Using the Collection

Follow these steps to access and start using the LLM Benchmarking Collection:
1. **Sign in**: Sign in to the [NVIDIA NGC container registry](http://ngc.nvidia.com) and search for "DGX Cloud Benchmarking"
2. **Log in**: connect to your cluster’s login node.
3. **Download Recipe**: Download the recipe for your desired model family from NGC to your login node. Downloads can be found in the "File Browser" tab in each model.
4. **Set Up Benchmarking Recipe**: extract the recipe
5. **Run Setup Script**: Execute the setup.sh script to download all required assets, including the model container.
6. **(Optional) Generate Dataset**: If required, run generate_dataset.sh to create the dataset.
7. **Start Benchmarking**: follow the selected model’s README.md instructions to launch benchmarking workload.
8. **Evaluate Results**: follow the methodology described in the selected model’s README.md to gather performance metrics and compare to published baseline values.


## Workload Resources Overview

Each workload resource includes:
* **Configuration details**: Comprehensive software and hardware setup information.
* **Performance scripts**: Predefined scripts to generate and analyze performance results.

The overview page for each workload highlights target performance metrics for the specified configuration, focusing on speed measurements such as the time taken per training step and the number of tokens processed per second.

## Available Benchmarks

The following table lists each benchmark used to evaluate the model’s performance, along with their specific configurations.


| Workload      |    Type    | Description                 | Container Version | Dataset    | Max Scale (#GPUs) | DTYPE     |
| --------------- | :-----------: | :---------------------------- | :-----------------: | :----------- | :---------- | :---------- |
| Nemotron4     |  Training  | 15B and 340B benchmarks     |       24.09       | Synthetic  | 2048      | FP8, BF16 |
| Nemo Megatron |  Training  | 175B benchmarks             |       24.05       | Pile       | 2048      | FP8, BF16 |
| Llama 3.1     |  Training  | 8B, 70B and 405B benchmarks |       24.09       | Pile       | 2304      | FP8, BF16 |
| PaXML         |  Training  | 5B and 175B benchmarks      |     24.03.04     | Synthetic  | 2048      | FP8, BF16 |
| Maxtext       |  Training  | Llama2 70B benchmarks       |    2024.12.09    | Synthetic  | 2048      | FP8, BF16 |
| Grok1         |  Training  | Grok1 314B benchmarks       |       24.09       | Synthetic  | 2048      | FP8, BF16 |
| Llama 2       | Fine Tuning | Hugging Face 70B benchmarks |       24.02       | HF Llama2  | 512       | BF16      |
| Mistral       | Fine Tuning | Hugging Face 7B benchmarks  |       24.02       | HF Mistral | 256       | BF16      |

Baseline performance metrics were using workloads on the NVIDIA DGX H100 Reference Architecture. For more information see [DGX H100 Systems](https://blogs.nvidia.com/blog/dgx-h100-systems-shipping/).

# Reference Infrastructure

The LLM Benchmarking Collection published baseline benchmark results using the following infrastructure, CSP-specific configurations, and software.

* GPU: 8xH100 80GB HBM3 (640GB total)
  * TDP 700W
  * Memory bandwidth 3.2 TBs
* CPU: 2x Intel Sapphire Rapids, Intel(R) Xeon(R) Platinum 8480CL E5
  * 112 cores (56 cores per CPU)
  * 2.00 GHz (Base), 3.8 Ghz (Max boost)
  * Numa nodes per socket = 1
  * PCIe Gen5
* NVLink: NVLink 4th Generation
  * 900 GB/s (GPU to GPU NVLink bidirectional bandwidth)
  * 18 Links per GPU
* InfiniBand:
  * Compute links: 8x 400 Gbit/s
  * Storage links: 2x 400 Gbit/s
* System Memory: 2TB
* Local Storage:
  * 2x 1.92TB NVMe M.2
  * 8x 3.84TB NVMe U.2

## CSP Specific Configurations

The benchmarks were built on the NVIDIA Reference Architecture. To achieve optimal performance on each CSP, you must make the following changes.

* **AWS**: Enable Elastic Fabric Adapter (EFA) support by following the [step-by-step guide](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-efa.html#your-algorithms-training-efa-install). Use the [reference NCCL tests Dockerfile with EFA support](https://github.com/aws-samples/awsome-distributed-training/blob/main/micro-benchmarks/nccl-tests/nccl-tests.Dockerfile).
* **GCP**: Configure Compute Fabric with TCP-X by ensuring the following environment variables are set and present for your environment.

```shell
NCCL_LIB_DIR='/var/lib/tcpxo/lib64' source /var/lib/tcpxo/lib64/nccl-env-profile.sh; \
	  export NCCL_FASTRAK_CTRL_DEV=enp0s12; \
	  export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0; \
	  export NCCL_SOCKET_IFNAME=enp0s12; \
	  export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices; \
	  export NCCL_NET=FasTrak; \
	  ls /var/lib/tcpxo/lib64;"
```
* **Azure**: Requires four settings for optimal performance: two environment variables and two slurm parameters.
   1. **NCCL_TOPO_FILE**=`<path to topo file>`.
      * The VM topology files ensure that the correct CPUs, GPUs and NICs are bound together. Location of this file varies, it **must** be mounted into the container.
   2. **NCCL_P2P_CHUNKSIZE**=2097152
      * Increasing message size for NCCL send/recv for optimal performance
   3. **--container-env**=NCCL_TOPO_FILE,NCCL_P2P_CHUNKSIZE
      * Pyxis flag to override these variables if they exist in the container with the new settings above.
   4. **--cpu-bind=mask_cpu**:”fff,fff000,fff000000,fff000000000,fff000000000000,fff000000000000000,fff000000000000000000,fff000000000000000000000"
      * CPU pinning binds a specific, optimal set of CPUs per mpi process.


Example Configuration for Nemo Megatron Launcher:
```shell
export NCCL_TOPO_FILE=/opt/microsoft/nvd5-topo.xml # Exact location varies by cluster
export NCCL_P2P_NET_CHUNKSIZE=2097152
srun --container-image ${IMAGE} \
   --container-writable \
   --container-mounts ${NCCL_TOPO_FILE},${DATA_DIR}:/datasets/,${RESULT_DIR},$INDEX_MAPPING_DIR,${STAGE_PATH}/cfg:/cfg/ \
   --container-env=NCCL_TOPO_FILE,NCCL_P2P_NET_CHUNKSIZE \   
   --cpu-bind=mask_cpu:"fff,fff000,fff000000,fff000000000,fff000000000000,fff000000000000000,fff000000000000000000,fff000000000000000000000" \
   --no-container-mount-home
    <snip> ...
```


# Support

For questions or to provide feedback, please contact LLMBenchmarks@nvidia.com
