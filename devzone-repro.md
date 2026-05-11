# NVIDIA Deep Learning Inference Performance Reproduction Guide

This repository provides instructions reproduce inference performance data from the the [NVIDIA Deep Learning Performance - AI Inference](https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference) page.

## Prerequisites

Before configuring the orchestrator, ensure you have downloaded the required model weights from Hugging Face:

- **DeepSeek-R1 (DSR1):** [DeepSeek-R1-0528-NVFP4-v2](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2)
- **DeepSeek-V4-Pro (DSv4-Pro):** [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- **gpt-oss-120b:** [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- **Qwen3.5-397B (NVFP4):** [Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)
- **Kimi-K2.5:** [Kimi-K2.5-NVFP4](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4)

## Environment Setup

Benchmarking is orchestrated using [srt-slurm](https://github.com/NVIDIA/srt-slurm), a command-line tool for distributed LLM inference benchmarks on SLURM clusters. (Support for benchmarking Kubernetes clusters coming soon.)

1. **Clone and Install:**

```bash
# Enter a directory on NFS, accessible by all nodes of your cluster.
git clone https://github.com/NVIDIA/srt-slurm.git
cd srt-slurm

# Initialize virtual environment and install dependencies (not shown)
uv venv
uv pip install -e .
```

2. **Initialize SLURM Workspace:**
   Execute the setup command below. You will be prompted to specify your SLURM account and partition.

```bash
#One-time setup (downloads NATS/ETCD, creates srtslurm.yaml)
make setup ARCH=aarch64  # or ARCH=x86_64
```

3. **Configure Model Paths:**
   The setup script will generate an `srtslurm.yaml` file. Edit this file to append your local model paths. The alias on the left must match the `model.path` value used in the recipe YAMLs:

```yaml
model_paths:
  dsr1: /path/to/local/dsr1
  dsv4-pro: /path/to/local/dsv4-pro
  qwen3.5-nvfp4: /path/to/local/qwen3.5-397b-nvfp4
  kimi-k25-nvfp4: /path/to/local/kimi-k2.5-nvfp4
```

Depending on your cluster configuration, you may need to specify additional arguments in srtslurm.yaml (SLURM account/partition, container image aliases, GPU-per-node defaults, etc.). See https://github.com/NVIDIA/srt-slurm/blob/main/srtslurm.yaml.example for the full reference.

## Running the Benchmarks

To execute a benchmark, apply the target configuration file using the `srtctl` CLI:

```bash
srtctl apply -f <path-to-config-file>
```

Available benchmarking configurations for published performance data are mapped below. Select the recipe that matches your target performance profile.

| Model            | 1K/1K                                                                                                                                  | 8K/1K                                                                                                                                  | 128K/8K                                                                                                                                                                |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DSR1**         | [GB300](https://github.com/NVIDIA/srt-slurm/blob/main/recipes/gb300-fp4/1k1k/max_tpt.yaml)                                             | [GB300](https://github.com/NVIDIA/srt-slurm/blob/main/recipes/gb300-fp4/8k1k/max_tpt.yaml)                                             | [GB300](https://github.com/NVIDIA/srt-slurm/blob/main/recipes/gb300-fp4/128k8k/maxthroughput-ctx3_pp4_gen1_dep8_batch32_eplb0_mtp0.yaml)                               |
| **DSv4-Pro**     | [GB300](https://github.com/NVIDIA/srt-slurm/blob/main/recipes/dsv4-pro/sglang/gb300-fp4/1k1k/agg/stp/agg-max-tpt-tep.yaml)              | _Coming soon_                                                                                                                          |                                                                                                                                                                        |
| **gpt-oss-120b** | [B200](https://github.com/NVIDIA/srt-slurm/blob/main/recipes/trtllm/gpt-oss-120b/b200-fp4/1k1k/agg-tp1.yaml)                            | [B200](https://github.com/NVIDIA/srt-slurm/blob/main/recipes/trtllm/gpt-oss-120b/b200-fp4/8k1k/agg-tp1.yaml)                            |                                                                                                                                                                        |
| **Kimi-K2.5**    | _Coming soon_                                                                                                                          | _Coming soon_                                                                                                                          |                                                                                                                                                                        |
| **Qwen3.5-397B** | [GB200](https://github.com/NVIDIA/srt-slurm/blob/main/recipes/qwen3.5/nvfp4/agg/stp_prefix_off/tp4.yaml)                                | _Coming soon_                                                                                                                          |                                                                                                                                                                        |

## Support

Terminology used in recipes is explained in the [Appendix](APPENDIX.md).

For questions or to provide feedback, please contact LLMBenchmarks@nvidia.com
