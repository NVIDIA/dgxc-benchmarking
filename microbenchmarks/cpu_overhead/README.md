# Overview
This recipe provides instructions for running microbenchmarks that measure CPU overhead on single node.

We consider two CPU overhead scenarios:
- **Pytorch kernel launch latency**: In this test, we measure how fast the CPU can push tiny kernels into the GPU. The higher the average execution time per benchmark, the worse the CPU overhead.
- **Tokenization throughput**: This test measures CPU-only performance through the tokenization step used in inference and RL. The higher the time taken to generate N tokens, the worse the CPU overhead.

The script uses TRT-LLM release containers to benchmark tokenization throughput for the [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) model on H100/B200/GB200 platforms. 

# Performance Measurement and Analysis 
Below, we list the test configuration and performance of the kernel launch and tokenization tests on GB200, B200 and H100.

## Pytorch kernel launch test

| GEMM M size | GB200 (usec) | B200 (usec) | H100 (usec) |
|:------------|:------------:|:-----------:|:-----------:|
| 4           |   11.3       | 6.4         | 6.8         |
| 8           |   13.0       | 6.8         | 6.8         |
| 16          |   13.0       | 6.8         | 6.8         |
| 32          |   17.7       | 9.2         | 9.1         |
| 64          |   17.7       | 9.2         | 9.1         |
| 128         |   13.0       | 6.8         | 6.8         |
| 256         |   13.0       | 6.8         | 6.8         |
| 512         |   17.7       | 9.2         | 9.1         |

## Tokenization throughput test

| Num prompts | GB200 (sec) | B200 (sec) | H100 (sec) |
|:------------|:-----------:|:----------:|:----------:|
| 100000      |   39        | 34         | 34         |

# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). You will need this token during the LLMB Installation when preparing your environment.

# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/microbenchmark_cpu_overhead`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**
- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run per use case (** Recommended **)

# Run both CPU overhead tests
llmb-run single -w microbenchmark_cpu_overhead -s 120b --dtype mxfp4 --scale 1
 
# Run the pytorch kernel launch test
USE_CASES="kernel_launch" llmb-run single -w microbenchmark_cpu_overhead -s 120b --dtype mxfp4 --scale 1

# Run the tokenization throughput test
USE_CASES="tokenization" llmb-run single -w microbenchmark_cpu_overhead -s 120b --dtype mxfp4 --scale 1
```


For more details on llmb-run usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

### Results/Log files 
Results for the workload are stored at `$LLMB_INSTALL/workloads/microbenchmark_cpu_overhead/experiments/cpu_overhead_tests`

You should expect to see separate logs for each use case:

```
├── <use_case>_overhead_%j.err  # Error logs
├── <use_case>_overhead_%j.out  # Benchmarking output
```

The `*.out` file provides key performance metrics:

```
Kernel launch test:

Benchmarking GEMM size: 4x4 ...
Average execution time for size:4 is 11.299694776535034 us
Benchmarking GEMM size: 8x8 ...
Average execution time for size:8 is 12.973302841186523 us
Benchmarking GEMM size: 16x16 ...
Average execution time for size:16 is 12.918899536132812 us
Benchmarking GEMM size: 32x32 ...
Average execution time for size:32 is 17.679811716079712 us
Benchmarking GEMM size: 64x64 ...
Average execution time for size:64 is 17.74930453300476 us
Benchmarking GEMM size: 128x128 ...
Average execution time for size:128 is 12.853708982467651 us
Benchmarking GEMM size: 256x256 ...
Average execution time for size:256 is 12.944073677062988 us
Benchmarking GEMM size: 512x512 ...
Average execution time for size:512 is 17.642942667007446 us
Results saved to gemm_benchmark_results.csv

Tokenization test:

Tokenization time: 39s
```
