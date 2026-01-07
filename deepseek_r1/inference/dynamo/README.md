# TensorRT-LLM Benchmark Scripts for DeepSeek R1 model

This directory contains scripts for running TensorRT-LLM with Dynamo on DeepSeek-R1 671B model. Inference is run using disaggregated mode of Dynamo with the SLURM job scheduler.

The **Maximum throughput** configuration is used. The system is configured to generate as many tokens per second as possible. This typically involves large batch sizes, long generation lengths, and aggressive request packing to fully utilize GPU compute. While this approach increases overall efficiency and hardware utilization, it also results in higher latency for individual requests. It's ideal for offline processing, batch jobs, or queued summarization tasks.

# Performance Measurement and Analysis 

Below, we list the inference configuration and benchmarking performance of the DeepSeek R1 model for the maximum throughput configuration on GB200, B200 and H100. The ISL (input sequence length) is 8150 and the OSL (output sequence length) is 1024.

With disaggregated mode, prefill and decode stages are separated into multiple jobs. The user can configure the number of servers as well as the parallelism configuration of each server (TP/PP/PP) independently for prefill and decode. This enables one to optimize TTFT (Time To First Token) and TPOT (Time Per Output Token) independently unlike aggregated inference where optimizing for both metrics is more difficult.

## Inference Configurations

For the prefill server, the `max_batch_size` is always 1 since there is enough parallelism to saturate the GPU with a large ISL (input sequence length). In fact, the ISL=8150 is large enough that chunked prefill (prefill split into multiple iterations) needs to be enabled to prevent OOM (Out of Memory) errors on the GPU. The `max_num_tokens` is set to 4096 (on GB200 / B200) and 2048 (on H100). The `concurrency` and `num_requests` is determined by the decode servers. The `num_requests` is set to `8 * concurrency` to minimize the overhead of the warmup and cooldown phases.

### GB200 Inference

| Job Type    |  Servers  |  GPUs (total)  |  max_batch_size  | concurrency   | max_num_tokens   | kv_cache_free_gpu_mem_fraction   | Quantization   | num_requests   |  TP |  PP |  EP | attn_dp_enabled   |
|:------------|:---------:|:--------------:|:----------------:|:-------------:|:----------------:|:--------------------------------:|:--------------:|:--------------:|:---:|:---:|:---:|:-----------------:|
| prefill     |  6        | 24             | 1                | -             | 4096             | 0.75                             |     NVFP4      | -              |  4  |   1 |  4  |    YES            |
| decode      |  1        | 8              | 256              | 2048          | 256              | 0.8                              |     NVFP4      | 16384          |  8  |   1 |  8  |    YES            |

### B200 Inference

| Job Type    |  Servers  |  GPUs (total)  |  max_batch_size  | concurrency   | max_num_tokens   | kv_cache_free_gpu_mem_fraction   | Quantization   | num_requests   |  TP |  PP |  EP | attn_dp_enabled   |
|:------------|:---------:|:--------------:|:----------------:|:-------------:|:----------------:|:--------------------------------:|:--------------:|:--------------:|:---:|:---:|:---:|:-----------------:|
| prefill     |  3        | 24             | 1                | -             | 4096             | 0.75                             |     NVFP4      | -              |  8  |   1 |  8  |    YES            |
| decode      |  1        | 8              | 256              | 2048          | 256              | 0.8                              |     NVFP4      | 16384          |  8  |   1 |  8  |    YES            |

### H100 Inference

| Job Type    |  Servers  |  GPUs (total)  |  max_batch_size  | concurrency   | max_num_tokens   | kv_cache_free_gpu_mem_fraction   | Quantization   | num_requests   |  TP |  PP |  EP | attn_dp_enabled   |
|:------------|:---------:|:--------------:|:----------------:|:-------------:|:----------------:|:--------------------------------:|:--------------:|:--------------:|:---:|:---:|:---:|:-----------------:|
| prefill     |  2        | 32             | 1                | -             | 2048             | 0.75                             |     FP8        | -              |  8  |   2 |  8  |    YES            |
| decode      |  1        | 16             | 128              | 1024          | 128              | 0.8                              |     FP8        | 8192           |  8  |   2 |  8  |    YES            |

**Note**
- kv_cache_free_gpu_mem_fraction: fraction of memory allocated to store the kv cache values after loading the model weights.
- attn_dp_enabled: This flag in the config.yml dictates whether `Data parallelism` is enabled or disabled for the attention layers.
- you can find more information on the trt-llm build parameters here (https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html)

More details about the inference terms can be found here [Appendix](../../../APPENDIX.md)

# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/inference_deepseek-r1-dynamo`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**
- `SBATCH_PARTITION` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_TIMELIMIT` - Time limit configured for the Dynamo benchmark to complete (1:00:00 by default).
- `SBATCH_GPUS_PER_NODE` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

# Running Benchmarks using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL
```

## GB200 and B200
```bash
llmb-run submit -w inference_deepseek-r1-dynamo -s 671b --dtype nvfp4 --scale 32
```

## H100
```bash
SBATCH_TIMELIMIT=2:00:00 llmb-run submit -w inference_deepseek-r1-dynamo -s 671b --dtype fp8 --scale 48
```

**Streaming:**
- You can toggle streaming on and off. When off, users receive the entire response (all output tokens) back at once instead of receiving output tokens as they are generated.
  - **By default, streaming is turned on in this workload**
  -  If turned off -- TTFT (Time to First Token) and TPOT (Time per Output Token) metrics are not applicable, since individual token delivery is bypassed.

```bash
# Example of turning streaming off on GB200 or B200
STREAMING=false llmb-run submit -w inference_deepseek-r1-dynamo -s 671b --dtype nvfp4 --scale 32
```

For more details on llmb-run usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

## Run Benchmarks (Direct Method)

The scripts below are set up to run inference using the max-throughput settings.

**Important**: 
- Ensure your virtual environment is activated before running the benchmark commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/deepseek_r1/inference/dynamo/`

## Commands

```shell
# GB200
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=gb200 ./launch.sh

# B200
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=b200 ./launch.sh

# H100
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=h100 ./launch.sh
```

## Exploring Pareto-Optimal Design Space

The script `launch.sh` uses several environment variables to control parallelism settings, and concurrency. The defaults are set for max-throughput, but a user can explore the design space of **Throughput per GPU** vs **Throughput per User** by tuning these parameters for various ISL/OSL combinations.
1. ISL - Input sequence length
2. OSL - Output sequence length
3. NCTX - Number of servers used for prefill computation.
4. CTX_WORLD_SIZE - Number of GPUs allocated for each prefill server. Total number of GPUs allocated for prefill is `NCTX * CTX_WORLD_SIZE`.
5. CTX_MAX_BATCH_SIZE - Maximum number of sequences processed in each prefill iteration. Set to `1` by default.
6. CTX_MAX_NUM_TOKENS - Maximum number of tokens for each prefill iteration. If chunked prefill is enabled, this can be less than `ISL`.
7. CTX_GPU_MEMORY_FRACTION - Fraction of GPU memory allocated for storing weights and KV cache in each prefill server.
8. NGEN - Number of servers used for decode computation.
9. GEN_WORLD_SIZE - Number of GPUs allocated for each decode server. Total number of GPUs allocated for decode is `NGEN * GEN_WORLD_SIZE`.
10. GEN_MAX_BATCH_SIZE - Maximum number of sequences processed in each decode iteration.
11. GEN_MAX_NUM_TOKENS - Maximum number of tokens for each decode iteration. This is generally set to be equal to `GEN_MAX_BATCH_SIZE` is the disaggregated setting.
12. GEN_GPU_MEMORY_FRACTION - Fraction of GPU memory allocated for storing weights and KV cache in each decode server.
13. GEN_EPLB_NUM_SLOTS - Number of slots allocated for expert-load-balancing (EPLB).
14. GEN_MTP_SIZE - Number of tokens predicted for a sequence from each decode iteration.
15. CONCURRENCY - Number of sequences that are decoded in parallel on each decode server. The total concurrency is `CONCURRENCY * NGEN`


## Result, Configuration, and Log Files
Result, configuration, and log files for the workload are stored at `$LLMB_INSTALL/workloads/inference_deepseek-r1-dynamo/experiments/isl<ISL>_osl<OSL>_<timestamp>`.

The `perf_summary.txt` file provides a summary of performance results in the following format.

```
...
=========================================================
PERFORMANCE OVERVIEW
=========================================================
Request Throughput (req/sec)                  : <value>
Output Token Throughput (tokens/sec)          : <value>
Output Token Throughput per User (tps/user)   : <value>
Output Token Throughput per GPU (tps/gpu)     : <value>
Total Token Throughput per GPU (tokens/sec)   : <value>
Average request latency (ms)                  : <value>
Average time to first token [TTFT] (ms)       : <value>
Average time per output token [TPOT] (ms)     : <value>
...
```

More detailed measurements for throughput and latency are present in the following directory: `deepseek-r1-dynamo-openai-chat-concurrency<concurrency>`.
- `profile_export_genai_perf.json` provides `min`, `max`, `p99`, `p95`, `p90`, `p75`, `p50`, `p25`, `p10`, `p5`, `p1` measurements for each throughput/latency statistic. It also provides information about GPU power usage, GPU temperature, and GPU memory temperature.
- `profile_export_genai_perf.csv` file provides a comma-separated version of the measurements in `profile_export_genai_perf.json`.

The configuration files are stored in the `configs` folder.
- `deployment.yaml` contains the number of prefill and decode servers, as well as the total number of GPUs used for inference.
- `prefill.yaml` is the configuration file for each prefill server. It contains the parallelism information, max batch size, max number of tokens, CUDA graph sizes and the configurations for the KV cache, and MoE (Mixture of Experts) implementation.
- `decode.yaml` contains the corresponding information for each decode server.
- `machines.txt` contains the list of machines on which the inference was run.

The logs for the various jobs spawned are stored in the `logs` folder.
- `output_prefill_<index>.log` contains the logs for each prefill server.
- `output_decode_<index>.log` contains the logs for each decode server.
- `output_server.log` contains the logs for the frontend servers.
- `bench.log` contains the logs for the load generator.
- `prefill.log` contains the concatenated logs from all prefill servers for all prefill iterations.
- `decode.log` contains the concatenated logs from all decode servers for all decode iterations.

## FAQ

- For each inference benchmarking run, the load generator (GenAI-Perf) first completes execution after it has received responses for all its requests. This triggers cancellations to all servers, including the frontend, prefill and decode servers. This is expected, and is a normal part of the benchmarking setup. Users should look for the `perf_summary.txt` file in the experiment directory to determine whether the benchmark ran successfully.

## Known Issues

- Workers may encounter out-of-memory (OOM) errors during inference, especially with larger configurations. Users can typically workaround this by lowering `CTX_MAX_NUM_TOKENS` if the OOM is on the prefill workers, or by lowering `GEN_MAX_BATCH_SIZE` / `GEN_MAX_NUM_TOKENS` if the OOM is on the decode workers. `CTX_MAX_BATCH_SIZE` is typically always set to 1. If these changes do not fix the OOM, users can further lower `CTX_GPU_MEMORY_FRACTION` for the prefill workers or `GEN_GPU_MEMORY_FRACTION` for the decode workers to allocate more memory for intermediate tensors.
