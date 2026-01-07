# DeepSeek V3 Pre-training

This directory contains performance recipes for DeepSeek V3 pre-training workloads using different frameworks.

## Available Frameworks

### TorchTitan
- **Path**: [torchtitan/](torchtitan/README.md)
- **Description**: Large-scale LLM training using native PyTorch with FSDP, tensor parallelism, pipeline parallelism, and expert parallelism
- **Model Size**: 671B parameters
- **Supported GPUs**: GB200, B200, H100
- **Precision**: BF16 (FP8 not currently supported in this variant)

### Megatron-Bridge
- **Path**: [megatron_bridge/](megatron_bridge/README.md)
- **Description**: Training using NeMo Megatron-Bridge framework
- **Model Size**: 671B parameters
- **Supported GPUs**: GB300, GB200, B200, H100
- **Precision**: FP8, BF16

## Quick Start

Please refer to the framework-specific README files for detailed setup and running instructions:

- [TorchTitan README](torchtitan/README.md)
- [Megatron-Bridge README](megatron_bridge/README.md)


For complete installation instructions, see the [main repository README](../../README.md).

