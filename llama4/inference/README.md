# Overview
This recipe provides instructions and scripts for benchmarking the performance of the Llama4 model with Nvidia TRT-LLM (TensorRT-LLM) benchmark suite.

The script uses TRT-LLM release containers to benchmark the [Llama-4-Maverick-17B-128E-Instruct-FP8](https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8) inference workload for maximum throughput and min latency scenario bencharking.

<div style="background-color: #ffeeba; padding: 10px; border-left: 6px solid #f0ad4e;">
  <strong>⚠️ WARNING:</strong> Currently this recipe only supports Llama-4-Maverick-17B-128E-Instruct-FP8 on 4x/8x GB200 GPUs.
</div> 


# Benchmarking setup

We consider two benchmarking scenarios:
- **Maximum throughput**: the system is configured to generate as many tokens per second as possible. This typically involves large batch sizes, long generation lengths, and aggressive request packing to fully utilize GPU compute. While this approach increases overall efficiency and hardware utilization, it also results in higher latency for individual requests. It's ideal for offline processing, batch jobs, or queued summarization tasks.
- **Minimum latency**: prioritizes fast response times for each individual request. This often involves smaller batch sizes (sometimes just one), shorter generation lengths, and minimal scheduling overhead. While this reduces GPU efficiency and overall throughput, it significantly lowers response time, making it suitable for real-time, interactive applications like chatbots or low-latency APIs.
 
# Performance Measurement and Analysis 

Llama4 model was benchmarked for maximum throughput performance. The goal is to reach the maximum number of output tokens per second per GPU.

## GB200

### Max throughput configurations

| Use Case      |   GPUs | ISL   | OSL   |   max_batch_size | concurrency   | max_num_tokens   | kv_cache_free_gpu_mem_fraction   | Quantization   | num_requests   |   TP |   PP |   EP | attn_dp_enabled   |
|:--------------|:------:|:-----:|:-----:|:----------------:|:-------------:|:----------------:|:--------------------------------:|:--------------:|:--------------:|:----:|:----:|:----:|:-----------------:|
| reasoning     |      8 | 1,000 | 1,000 |        896       | 4,096         | 8,192            |          0.9                     | FP8            | 16,384         |    8 |    1 |    8 |       YES         |
| chat          |      8 | 128   | 128   |        896       | 4,096         | 8,192            |          0.9                     | FP8            | 16,384         |    8 |    1 |    8 |       YES         |
| summarization |      8 | 8,000 | 512   |        896       | 1,024         | 22,528           |          0.85                    | FP8            | 4,096          |    8 |    1 |    8 |       YES         |
| generation    |      8 | 512   | 8,000 |        896       | 1,024         | 2,048            |          0.9                     | FP8            | 2,048          |    8 |    1 |    8 |       YES         |


### Min latency configurations

| Use Case      |   GPUs | ISL   | OSL   |   max_batch_size | concurrency   | max_num_tokens   | kv_cache_free_gpu_mem_fraction   | Quantization   | num_requests   |   TP |   PP |   EP | attn_dp_enabled   |
|:--------------|:------:|:-----:|:-----:|:----------------:|:-------------:|:----------------:|:--------------------------------:|:--------------:|:--------------:|:----:|:----:|:----:|:-----------------:|
| reasoning     |      4 | 1,000 | 1,000 |        1         | 1             | 3,072            |          0.75                    | FP8            | 20             |    4 |    1 |    1 |       NO          |
| chat          |      4 | 128   | 128   |        1         | 1             | 8,192            |          0.75                    | FP8            | 20             |    4 |    1 |    1 |       NO          |
| summarization |      4 | 8,000 | 512   |        1         | 1             | 8,384            |          0.75                    | FP8            | 20             |    4 |    1 |    1 |       NO          |
| generation    |      4 | 512   | 8,000 |        1         | 1             | 2,048            |          0.75                    | FP8            | 20             |    4 |    1 |    1 |       NO          |



### Configuration Notes
- **kv_cache_free_gpu_mem_fraction**: fraction of memory allocated to store the kv cache values after loading the model weights.
- **attn_dp_enabled**: This flag in the config.yml dictates whether `Data parallelism` is enabled or disabled for the attention layers.
- You can find more information on the trt-llm build parameters here (https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html)

More details about the inference terms can be found in the [Appendix](../../APPENDIX.md)

### Metric Notes
- **TPS/GPU**: Output Token Throughput per second per GPU
- **TPS/User**: Output Token Throughput per second per user
- **Avg Request Latency**: Average time for a a request to be served
- **TTFT**: Time to First Token
- **TPOT**: Time Per Output Token

# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). You will need this token during the LLMB Installation when preparing your environment.


During the installation process you will be prompted for the token:
```
HuggingFace Token (HF_TOKEN) - Some workloads require this for accessing HuggingFace models and datasets.
You can get your token from: https://huggingface.co/settings/tokens
Note: If you're sure you don't need HF_TOKEN for your selected workloads, this can be left blank.
? Enter your HuggingFace token (HF_TOKEN) or leave blank: <hf-token>
✓ HF_TOKEN configured successfully
```

## Request Access

Access to Llama 4 Maverick must be requested through the [HuggingFace Llama 4](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct). The approval process is not automatic and could take a day or more.


# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/llama4`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).


**Migration Note:**
If you previously used `STAGE_PATH`, replace it with `LLMB_INSTALL` (top-level). All output, logs, and checkpoints will be created under the workload's appropriate `LLMB_WORKLOAD` folder.

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

### Maximum throughput

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run per use case for the maximum throughput scenario (** Recommended **)

# Reasoning
MODE="max_throughput" USE_CASES=reasoning:1000/1000 CONCURRENCY=4096 NUM_REQUESTS=16384 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 8

# Chat
MODE="max_throughput" USE_CASES=chat:128/128 CONCURRENCY=4096 NUM_REQUESTS=16384 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 8

# Summarization
MODE="max_throughput" USE_CASES=summarization:8000/512 KV_CACHE_FRACTION=0.85 CONCURRENCY=1024 NUM_REQUESTS=4096 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 8

# Generation
SBATCH_TIMELIMIT=40:00 MODE="max_throughput" USE_CASES=generation:512/8000 CONCURRENCY=1024 NUM_REQUESTS=2048 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 8
```

### Minimum latency

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run per use case for the min latency scenario (** Recommended **)

# Reasoning
MODE="min_latency" USE_CASES=reasoning:1000/1000 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 4

# Chat
MODE="min_latency" USE_CASES=chat:128/128 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 4

# Summarization
MODE="min_latency" USE_CASES=summarization:8000/512 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 4

# Generation
MODE="min_latency" USE_CASES=generation:512/8000 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 4
```

- Single use cases and their optimized parameters are listed above and work out of the box. Advanced users can add more use_cases in the `setup.sh` 
- Advanced users can learn more about: 
  - [Tuning Max Batch Size and Max Num Tokens](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html#tuning-max-batch-size-and-max-num-tokens) to adjust the inflight batching scheduler 
  -  [Max Tokens in Paged KV Cache and KV Cache Free GPU Memory Fraction](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html#max-tokens-in-paged-kv-cache-and-kv-cache-free-gpu-memory-fraction) to control the maximum number of tokens handled by the KV cache manager


**Multiple USE_CASES:**
- While it is possible to run multiple USE_CASES within a single llmb-run invocation, doing so will increase the total runtime of the SLURM job. To minimize job duration and improve scheduling efficiency, it is recommended to split use cases into separate runs. 
- This approach may be beneficial for advanced LLMB users running multiple short benchmarking experiments, as it allows the model to be loaded only once across multiple experiments.

```bash
# Multi USE_CASE example (** NOT RECOMMENDED **)
 USE_CASES="reasoning:1000/1000 chat:128/128" llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 8
```

**Streaming:**
- You can toggle streaming on and off. When off, users recieve the entire response (all output tokens) back at once instead of receiving output tokens as they are generated
  - **By default, streaming is turned on in this workload**
  -  If turned off -- TTFT (Time to First Token) and TPOT (Time per Output Token) metrics are not applicable, since individual token delivery is bypassed.


```bash
# Example of turning streaming off
STREAMING=false USE_CASES=reasoning:1000/1000 llmb-run single -w inference_llama4 -s 17b --dtype fp8 --scale 8
```

For more details on llmb-run usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

# Direct Method

Alternatively, you can run inference scripts directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the inference commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/llama4/inference`

### Command Template

```shell
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} ./launch.sh 
```

### Results/Log files 
Results for the workload are stored at `$LLMB_INSTALL/workloads/inference_<model>/experiments/<model>_<mode>_TP_EP_PP_CON_USECASE`

You should expect to see result directories for each use case:

- `<model>_<mode>_TP_EP_PP_CON_reasoning`
- `<model>_<mode>_TP_EP_PP_CON_chat`
- `<model>_<mode>_TP_EP_PP_CON_summarization`
- `<model>_<mode>_TP_EP_PP_CON_generation`

Each directory will contain SLURM output and error logs:

Each folder will have output and error logs:
```
Model_TP_EP_PP_CON_<use_case>/
├── Model_TP_EP_PP_CON_<use_case>_streaming-<on/off>_%j.err  # Error logs
├── Model_TP_EP_PP_CON_<use_case>_streaming-<on/off>_%j.out  # Benchmarking output
```

The `PERFORMANCE OVERVIEW` section in the `*.out` file provides key performance metrics:
```
...

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     <value>
Total Output Throughput (tokens/sec):             <value>
Total Token Throughput (tokens/sec):              <value>
Total Latency (ms):                               <value>
Average request latency (ms):                     <value>
Per User Output Throughput [w/ ctx] (tps/user):   <value>
Per GPU Output Throughput (tps/gpu):              <value>
Average time-to-first-token [TTFT] (ms):          <value>
Average time-per-output-token [TPOT] (ms):        <value>
Per User Output Speed (tps/user):                 <value>
...
```

# FAQ & Troubleshooting

Structure of model weights folder `$LLMB_INSTALL/workloads/inference_llama4/Llama-4-Maverick-17B-128E-Instruct-FP8`
The model weights directory will look like below

<details>

<summary>Llama-4-Maverick-17B-128E-Instruct-FP8</summary>

Llama-4-Maverick-17B-128E-Instruct-FP8 weights directory looks like below and should be approximately 376G
Note: `du -sh` includes hidden contents like `.git` and may report a larger total. To verify weights-only, inspect file sizes in this folder with `ls -lh`.
```
total 376G
drwxrwsr-x 3 nlevin infra_rd_gsw 4.0K Jun 30 16:57 .
drwxrwsr-x 6 nlevin infra_rd_gsw 4.0K Jul  1 10:29 ..
drwxrwsr-x 9 nlevin infra_rd_gsw 4.0K Jun 30 16:57 .git
-rw-rw-r-- 1 nlevin infra_rd_gsw 1.6K Jun 30 15:54 .gitattributes
-rw-rw-r-- 1 nlevin infra_rd_gsw  21K Jun 30 15:54 README.md
-rw-rw-r-- 1 nlevin infra_rd_gsw 2.0K Jun 30 15:54 config.json
-rw-rw-r-- 1 nlevin infra_rd_gsw  186 Jun 30 15:54 generation_config.json
-rw-rw-r-- 1 nlevin infra_rd_gsw 1.5K Jun 30 15:54 hf_quant_config.json
-rw-rw-r-- 1 nlevin infra_rd_gsw  11G Jun 30 16:29 model-00001-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:44 model-00002-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:03 model-00003-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:43 model-00004-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:07 model-00005-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:51 model-00006-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:05 model-00007-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:55 model-00008-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:23 model-00009-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:53 model-00010-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:27 model-00011-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:52 model-00012-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 4.7G Jun 30 16:57 model-00013-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:31 model-00014-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:50 model-00015-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:25 model-00016-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:49 model-00017-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:32 model-00018-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:47 model-00019-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:36 model-00020-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:47 model-00021-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:40 model-00022-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:45 model-00023-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:41 model-00024-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:43 model-00025-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:15 model-00026-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:50 model-00027-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:01 model-00028-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:56 model-00029-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:39 model-00030-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:45 model-00031-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 4.7G Jun 30 16:57 model-00032-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:37 model-00033-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:46 model-00034-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:35 model-00035-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:48 model-00036-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:34 model-00037-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:49 model-00038-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:21 model-00039-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:42 model-00040-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:09 model-00041-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:52 model-00042-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:19 model-00043-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:53 model-00044-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:17 model-00045-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:54 model-00046-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:11 model-00047-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:41 model-00048-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw  11G Jun 30 16:13 model-00049-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 5.1G Jun 30 16:46 model-00050-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 4.0G Jun 30 16:56 model-00051-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 2.0G Jun 30 16:55 model-00052-of-00052.safetensors
-rwxrwxr-x 1 nlevin infra_rd_gsw 123K Jun 30 15:54 model.safetensors.index.json
-rwxrwxr-x 1 nlevin infra_rd_gsw  636 Jun 30 15:54 preprocessor_config.json
-rw-rw-r-- 1 nlevin infra_rd_gsw  319 Jun 30 15:54 special_tokens_map.json
-rw-rw-r-- 1 nlevin infra_rd_gsw  27M Jun 30 16:54 tokenizer.json
-rw-rw-r-- 1 nlevin infra_rd_gsw 232K Jun 30 15:54 tokenizer_config.json
```

</details>

