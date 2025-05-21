# Background
The LLM Benchmarking Collection provides an easy path to reproduce the latest performance results for deep learning workloads.

The Resource for each workload contains software and hardware configuration information plus the scripts necessary to obtain performance results. On the overview page for each workload, you can also find target performance results for the given configuration. 

Currently, the LLM Benchmarking Collection focuses on measuring speed metrics like time per training step and tokens per second.

| Workload | Type | Description | Container Version | Dataset | Max Scale | DTYPE |
|---|:---:|:---|:---:|:---|:---|:---|
| Nemotron4 | Training | 15B and 340B benchmarks | 24.09 | Synthetic | 2048 | FP8, BF16 
| Nemo Megatron | Training | 175B benchmarks | 24.05 | Pile | 2048 | FP8, BF16 
| Llama 3.1 | Training | 8B, 70B and 405B benchmarks | 24.09 | Pile | 2304 | FP8, BF16 
| PaXML | Training | 5B and 175B benchmarks | 24.03.04 | Synthetic | 2048 | FP8, BF16 
| Maxtext | Training | Llama2 70B benchmarks | 2024.12.09 | Synthetic | 2048 | FP8, BF16 
| Grok1 | Training | Grok1 314B benchmarks | 24.09 | Synthetic | 2048 | FP8, BF16 
| Llama 2 | Fine Tuning | Hugging Face 70B benchmarks | 24.02 | HF Llama2 | 512 | BF16 
| Mistral | Fine Tuning | Hugging Face 7B benchmarks | 24.02 | HF Mistral | 256 | BF16 

Baseline performance was obtained by running these workloads on NVIDIA DGX H100 Reference Architecture - [EOS](https://blogs.nvidia.com/blog/eos/).

# Prerequisites

## Infrastructure Reference
* GPU: 8xH100 80GB HBM3  (640GB total)
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

The benchmarks are created on the NVIDIA Reference Architecture, the following changes are needed for optimal performance on each CSP

### AWS

Instructions on adding EFA support: 
* [Step-by-step guide](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-efa.html#your-algorithms-training-efa-install)
* [Reference PyTorch image with EFA support](https://github.com/aws-samples/awsome-distributed-training/blob/main/2.ami_and_containers/containers/pytorch/0.nvcr-pytorch-aws.dockerfile)

### GCP

GCP utilizes TCP-X for the compute fabric. Ensure the following variables are correct and present for your environment.

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

Since Nemo Megatron, Paxml, HF Llama2 and HF Mistral [_*only*_] are using older containers that do not have an Azure topology file, these workloads will fail to run. To solve this, apply the following steps:
* Set `NCCL_TOPO_FILE=<full path to topo>`
* Mount topology file into the container
* Set `--container-env=NCCL_TOPO_FILE` flag to override any container settings.

Example from Nemo Megatron Launcher:
```shell
export NCCL_TOPO_FILE=/opt/microsoft/topo.xml # Exact location varies by cluster
srun --container-image ${IMAGE} \
     --container-writable \
     --container-mounts ${NCCL_TOPO_FILE},${DATA_DIR}:/datasets/,${RESULT_DIR},$INDEX_MAPPING_DIR,${STAGE_PATH}/cfg:/cfg/ \
     --container-env=NCCL_TOPO_FILE \
     --no-container-mount-home
	 <snip> ...
```


## Software stack
* Bash 4.2 or newer
* Slurm 22.x or newer
* [Enroot](https://github.com/NVIDIA/enroot/)
* [NGC Registry Access](https://org.ngc.nvidia.com/setup)
* Python 3.9.x or newer
* CUDA 12.3 or newer
* NV Driver: 535.129.03
* OFED: 5.9-0.5.6.0.127
* NCCL: 2.19.4

# Support
For questions or to provide feedback, please contact LLMBenchmarks@nvidia.com
