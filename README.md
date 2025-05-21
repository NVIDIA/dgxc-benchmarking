# DGX Cloud Benchmarking - Performance Recipes

Performance Recipes are ready-to-use templates for evaluating performance of specific AI use cases across hardware and software combinations. These containerized recipes allow users to quickly set up and run standardized benchmarking methodology in their own environment, ensuring consistent and comparable results across platforms. 

These Performance Recipes support performance characterization
- across a variety of defined AI workloads, including pre-training, fine tuning, and inference. 
- across GPU-based infrastructure, whether running on-premises or with cloud service providers (CSPs). 

Each recipe maps to one workload and can be run at various cluster scales and precisions. These workloads are tested against the NVIDIA Reference Architecture and those results are provided as a baseline for comparison. These performance metrics are collected from production environments and are subject to real-world variability.


## Prerequisites

At this time, Performance Recipes require Slurm as the cluster's job scheduler.
Before you use the Performance Recipes, make sure you have installed the following packages on your cluster.

* Bash 4.2 or newer
* Slurm 22.x or newer
* [Enroot](https://github.com/NVIDIA/enroot/)
* [NGC Registry Access](https://org.ngc.nvidia.com/setup)
* Python 3.10.12 or newer
* CUDA 12.3 or newer
* [NV Driver: 535.129.03 or newer](https://www.nvidia.com/en-us/drivers/)
* [OFED: 5.9-0.5.6.0.127 or newer](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/)
* [NCCL: 2.19.4 or newer](https://developer.nvidia.com/nccl/nccl-download)


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

| Benchmark | Framework | Container Version | Model | Model Size | Type | Max Scale (# of GPUs) | Precision | Model Access Required |
| :----| --------- | :---------------: | :---- | :--------: | :--- | :-------------------- | :-------- | :-------------------- |
| Nemotron | NeMo | 24.12, 24.09 | Nemotron4 | 15B, 340B | Training | 2048 | FP8, BF16 | No |
| Megatron | NeMo | 24.12 | GPT3      | 175B      | Training | 2048 | FP8, BF16 | No |
| Llama | NeMo | 24.12 | Llama 3.1 | 8B, 70B, 405B | Training | 2304 | FP8, BF16 | Yes |
| Maxtext | Maxtext | 25.01 | Llama3 | 70B | Training | 2048 | FP8, BF16 | No |
| Grok | NeMo | 24.12 | Grok1 | 314B | Training  | 2048 | FP8, BF16 | No |
| SFT | NeMo | 24.12 | Llama 3 | 8B, 70B | Supervised Fine-Tuning | 32 | FP8, BF16 | Yes |
| LORA | NeMo | 24.12 | Llama 3 | 8B, 70B | LORA Fine-Tuning | 32 | FP8, BF16 | Yes | 
| RAG Blueprint Pipeline | NIM | instruct:1.3.3, rerank:1.3, embed:1.3.1 | Llama 3.1 and 3.2 | 70b, 1b | Inference | n/a | n/a | Yes 
| NIM | NIM | 1.0.3 | Llama 3|  70B | Inference | 4 | FP8 | Yes |


Baseline performance metrics were using workloads on the NVIDIA DGX H100 Reference Architecture. For more information see [DGX H100 Systems](https://blogs.nvidia.com/blog/dgx-h100-systems-shipping/).

Note, the benchmarks are updated monthly. For older releases you can use Search feature in NGC Resources section.

E.g., here are the steps to locate benchmarks from release 24.11.1:
* Click on [Resources](https://catalog.ngc.nvidia.com/resources) tab
* Type "24.11.1 (DGXC Benchmarking)" in the "Search resources..." field
* You shall now see list of benchmarks with version 24.11.1


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
AI platforms may vary in implementation, such as differences in network fabric and virtualization implementations, and thus require different tuning. 
For optimal performance, users should leverage the correct implementation for their platform. The example platform-specific tuning is provided as a starting point. Further tuning may be necessary if instance type varies from the Reference Architecture. 

### AWS
Enable Elastic Fabric Adapter (EFA) support by following the [step-by-step guide](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-efa.html#your-algorithms-training-efa-install). Use the [reference NCCL tests Dockerfile with EFA support](https://github.com/aws-samples/awsome-distributed-training/blob/main/micro-benchmarks/nccl-tests/nccl-tests.Dockerfile). 

  Additionally CPU pinning has been found to improve performance. To enable set these three Slurm flags:
  1. --cpu-bind=verbose
     * Displays the pin mask used
  2. --cpus-per-task=12
     * How many cores to allocate with each MPI process
  3. --hint=nomultithread
     * Do not use any SMT cores. *Note:* these settings are an example based on a dual socket 48 core system. Please adjust for your specific environment.

  If you need to build a new docker image, use **setup.sh** script from corresponding benchmark folder to determine the exact base image. The setup.sh will include commands for importing docker image used by the workload: e.g., nemotron/setup.sh includes `"enroot import --output ${STAGE_PATH}/nvidia+nemo+24.12.sqsh docker://nvcr.io#nvidia/nemo:24.12"` which translates into `nvcr.io/nvidia/nemo:24.12` as the base image.

### GCP
Ensure that all required pre-conditions for [GCP cluster deployment](https://cloud.google.com/ai-hypercomputer/docs/create/create-slurm-cluster#before-you-begin) have been met. 

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
### Azure 
Requires four settings for optimal performance: two environment variables and two slurm parameters.
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

# Release Notes
## Performance Recipes collection version 25.02 

Adds support for the following workloads:
  * Pre-training: 
      * Maxtext Llama 3 70B
         * Precisions supported: FP8, BF16.
  * Fine tuning:
      * Supervised Fine-Tuning NeMo 24.12 Llama 3 8B, 70B. 
         * Precisions supported: FP8, BF16
      * LORA Fine-Tuning NeMo 24.12 Llama 3 8B, 70B FP8, BF16. 
         * Precisions supported: FP8, BF16
  * Inference
      * Llama 3 70B, NVIDIA NIM llama-3-70B-instruct version 1.0.3 
         * Precisions supported: FP8
      * Enterprise RAG Pipeline, NVIDIA AI Blueprint

The following workloads will no longer be maintained moving forward, starting in collection version 25.02: 
  * Pre-training 
      * Paxml GPT3 175B
      * Maxtext Llama 2 70B
  * Fine Tuning 
      * Hugging Face Llama 2 70B
      * Hugging Face Mistral 7x8B


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
A slurm job failed during benchmark run. E.g., a nemotron benchmark job with ID=2041792 failed

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
You can find log files associated with this run under `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/15b/$JOB_TOTAL_GPUS` folder. The folder will contain `log-nemo_nemotron4_*.out` and `log-nemo_nemotron4_*.err` files that will have a root cause error message.

E.g., for the job failure above and assuming the nemotron 15b job ran on 16 GPUs, used version 25.02, and with precision bf16 the path will be under `$STAGE_PATH/results/25.02/bf16/15b/16/log-nemo_nemotron4_15b_16_2041792.*`

Search for errors in the `log-nemo_nemotron4_15b_16_2041792.err` or `log-nemo_nemotron4_15b_16_2041792.out` files.

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

# Support

For questions or to provide feedback, please contact LLMBenchmarks@nvidia.com