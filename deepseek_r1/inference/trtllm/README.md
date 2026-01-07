# Overview
This recipe provides instructions and scripts for benchmarking the performance of the DeepSeek-R1 model with NVIDIA TRT-LLM (TensorRT-LLM) benchmark suite.

The script uses TRT-LLM release containers to benchmark inference workloads [DeepSeek-R1-FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4) on GB200/ B200 and [DeepSeek-R1-FP8](https://huggingface.co/deepseek-ai/DeepSeek-R1) on H100 platform. 

We consider two benchmarking scenarios
- **Maximum throughput**: the system is configured to generate as many tokens per second as possible. This typically involves large batch sizes, long generation lengths, and aggressive request packing to fully utilize GPU compute. While this approach increases overall efficiency and hardware utilization, it also results in higher latency for individual requests. It's ideal for offline processing, batch jobs, or queued summarization tasks.
- **Minimum latency**: prioritizes fast response times for each individual request. This often involves smaller batch sizes (sometimes just one), shorter generation lengths, and minimal scheduling overhead. While this reduces GPU efficiency and overall throughput, it significantly lowers response time, making it suitable for real-time, interactive applications like chatbots or low-latency APIs.


<div style="background-color: #ffeeba; padding: 10px; border-left: 6px solid #ca9142ff;">
  <strong>⚠️ WARNING:</strong> In this release version we support both Max throughput and Min Latency use cases on GB200 and only support Max throughput use case on H100 and B200.
</div> 


# Performance Measurement and Analysis 
Below, we list the inference configuration and benchmarking performance DeepSeek-R1 model for the two scenarios considered.

## Maximum throughput

### GB200 Inference Use case configs

| Use Case      | GPUs | ISL   | OSL   |   max_batch_size | concurrency | max_num_tokens | kv_cache_free_gpu_mem_fraction | Quantization | num_requests |   TP |   PP |   EP | attn_dp_enabled | chunked_prefill_enabled |
|:--------------|:----:|:-----:|:-----:|:----------------:|:------------|:--------------:|:------------------------------:|:------------:|:------------:|:----:|:----:|:----:|:---------------:|:---:|
| reasoning     |    4 | 1,000 | 1,000 |              768 | 3,072       | 6,000          |         0.85                   | NVFP4        | 30,720       |    4 |    1 |    4 |     Yes         | No |
| chat          |    4 | 128   | 128   |            2,884 | 11,536      | 16,000         |         0.50                   | NVFP4        | 115,360      |    4 |    1 |    4 |     Yes         | No |
| summarization |    4 | 8,000 | 512   |              160 | 960         | 9,000          |         0.69                   | NVFP4        | 9,600        |    4 |    1 |    4 |     Yes         | Yes |
| generation    |    4 | 512   | 8,000 |              224 | 896         | 2,000          |         0.97                   | NVFP4        | 8,960        |    4 |    1 |    4 |     Yes         | Yes |

### B200 Inference Use case configs
| Use Case      | GPUs | ISL   | OSL   |   max_batch_size | concurrency | max_num_tokens | kv_cache_free_gpu_mem_fraction | Quantization | num_requests |   TP |   PP |   EP | attn_dp_enabled |
|:--------------|:----:|:-----:|:-----:|:----------------:|:------------|:--------------:|:------------------------------:|:------------:|:------------:|:----:|:----:|:----:|:---------------:|
| reasoning     |    4 | 1,000 | 1,000 |              786 | 3,144       | 5,000          |         0.85                   | NVFP4        | 31,440       |    4 |    1 |    4 |     Yes         |
| chat          |    4 | 128   | 128   |            3,600 | 4,000      | 12,000         |         0.50                   | NVFP4        | 40,000      |    4 |    1 |    4 |     Yes         |
| summarization |    4 | 8,000 | 512   |              256 | 256        | 4,000          |         0.85                   | NVFP4        | 2,560       |    4 |    1 |    4 |     Yes         |
| generation    |    4 | 512   | 8,000 |              256 |   512       | 2,000          |         0.85                   | NVFP4        | 5,120        |    4 |    1 |    4 |     Yes         |

### H100 Inference Use case configs
| Use Case      | GPUs | ISL   | OSL   |   max_batch_size | concurrency | max_num_tokens | kv_cache_free_gpu_mem_fraction | Quantization | num_requests |   TP |   PP |   EP | attn_dp_enabled |
|:--------------|:----:|:-----:|:-----:|:----------------:|:------------|:--------------:|:------------------------------:|:------------:|:------------:|:----:|:----:|:----:|:---------------:|
| reasoning     |    16 | 1,000 | 1,000 |              384 | 3,072       | 3,000          |         0.65                   | FP8         | 30,720        |    8 |    2 |    8 |     Yes         |
| chat          |    16 | 128   | 128   |              384 | 3,072       | 30,720         |         0.65                   | FP8         | 3,000        |    8 |    2 |    8 |     Yes         |
| summarization |    16 | 8,000 | 512   |              128 | 1024       | 2,000           |         0.65                   | FP8         | 10,240        |    8 |    2 |    8 |     Yes         |
| generation    |    16 | 512   | 8,000 |              450 | 450        | 1,000          |         0.65                   | FP8         | 4,500        |    8 |    2 |    8 |     Yes         |

**Note**
- kv_cache_free_gpu_mem_fraction: fraction of memory allocated to store the kv cache values after loading the model weights.
- attn_dp_enabled: This flag in the config.yml dictates whether `Data parallelism` is enabled or disabled for the attention layers.
- you can find more information on the trt-llm build parameters here (https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html)

More details about the inference terms can be found here [Appendix](../../../APPENDIX.md)


## Minimum latency

### GB200 Inference Use case configs

|      Use Case      | GPUs |   ISL   |   OSL   | max_batch_size  | concurrency | max_num_tokens  | kv_cache_free_gpu_mem_fraction  | Quantization  | num_requests  | TP | PP | EP | attn_dp_enabled  |
|:-------------------|:----:|:-------:|:-------:|:---------------:|:-----------:|:---------------:|:-------------------------------:|:-------------:|:-------------:|:--:|:--:|:--:|:----------------:|
|     reasoning      |  4   | 1,000   | 1,000   |       1         |      1      |      1000       |              0.1                |    NVFP4      |      10       | 4  | 1  | 1  |       No         |
|       chat         |  4   | 128     | 128     |       1         |      1      |       128       |              0.1                |    NVFP4      |      10       | 4  | 1  | 1  |       No         |
|   summarization    |  4   | 8,000   | 512     |       1         |      1      |      8000       |              0.1                |    NVFP4      |      10       | 4  | 1  | 1  |       No         | 
|    generation      |  4   | 512     | 8,000   |       1         |      1      |       512       |              0.1                |    NVFP4      |      10       | 4  | 1  | 1  |       No         |


# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/inference_deepseek-r1`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**
- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

# Running Benchmarks using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL
```

## Maximum throughput scenario

### GB200 and B200
```bash
# Reasoning
MODE="max_throughput" MAX_NUM_TOKENS=6000 MAX_BATCH_SIZE=768 USE_CASE=reasoning:1000/1000 ENABLE_CHUNKED_PREFILL=false llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```
```bash
# Chat
MODE="max_throughput" MAX_NUM_TOKENS=16000 MAX_BATCH_SIZE=2884 KV_CACHE_FRACTION=0.5 ENABLE_CHUNKED_PREFILL=false USE_CASE=chat:128/128 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```
```bash
# Summarization
MODE="max_throughput" MAX_NUM_TOKENS=9000 MAX_BATCH_SIZE=160 KV_CACHE_FRACTION=0.69 USE_CASE=summarization:8000/512 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```
```bash
# Generation
MODE="max_throughput" SBATCH_TIMELIMIT=1:40:00 MAX_BATCH_SIZE=224 KV_CACHE_FRACTION=0.97 USE_CASE=generation:512/8000 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```

### H100
```bash
# Reasoning
MODE="max_throughput" MAX_BATCH_SIZE=384 CONCURRENCY=3072 MAX_NUM_TOKENS=3000 NUM_REQUESTS=30720 USE_CASE=reasoning:1000/1000 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype fp8 --scale 16
```
```bash
# Chat
MODE="max_throughput"  MAX_BATCH_SIZE=384 CONCURRENCY=3072 MAX_NUM_TOKENS=2000 NUM_REQUESTS=3000 USE_CASE=chat:128/128 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype fp8 --scale 16
```
```bash
# Summarization
MODE="max_throughput" MAX_BATCH_SIZE=128 CONCURRENCY=1024 MAX_NUM_TOKENS=2000 NUM_REQUESTS=10000 USE_CASE=summarization:8000/512 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype fp8 --scale 16
```
```bash
# Generation
MODE="max_throughput" SBATCH_TIMELIMIT=1:40:00 MAX_BATCH_SIZE=450 CONCURRENCY=450 MAX_NUM_TOKENS=1000 NUM_REQUESTS=4500 USE_CASE=generation:512/8000 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype fp8 --scale 16
```

## Minimum latency scenario

### GB200
```bash
# Reasoning
MODE="min_latency" MAX_NUM_TOKENS=1000 USE_CASE=reasoning:1000/1000 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```
```bash
# Chat
MODE="min_latency" MAX_NUM_TOKENS=128 USE_CASE=chat:128/128 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```
```bash
# Summarization
MODE="min_latency" MAX_NUM_TOKENS=8000 USE_CASE=summarization:8000/512 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```
```bash
# Generation
MODE="min_latency" MAX_NUM_TOKENS=512 USE_CASE=generation:512/8000 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```

- Single use cases and their optimized parameters are listed above and work out of the box. Advanced users can add more use cases in `setup.sh`.
- Advanced users can learn more about: 
  - [Tuning Max Batch Size and Max Num Tokens](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html#tuning-max-batch-size-and-max-num-tokens) to adjust the inflight batching scheduler 
  -  [Max Tokens in Paged KV Cache and KV Cache Free GPU Memory Fraction](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html#max-tokens-in-paged-kv-cache-and-kv-cache-free-gpu-memory-fraction) to control the maximum number of tokens handled by the KV cache manager
- Use cases such as summarization and generation take significantly more time so the time limits are increased accordingly.

**Streaming:**
- You can toggle streaming on and off. When off, users receive the entire response (all output tokens) back at once instead of receiving output tokens as they are generated.
  - **By default, streaming is turned on in this workload**
  -  If turned off -- TTFT (Time to First Token) and TPOT (Time per Output Token) metrics are not applicable, since individual token delivery is bypassed.

```bash
# Example of turning streaming off on GB200 or B200
STREAMING=false USE_CASE=reasoning:1000/1000 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4

# Example of turning streaming off on H100
STREAMING=false USE_CASE=reasoning:1000/1000 llmb-run submit -w inference_deepseek-r1 -s 671b --dtype fp8 --scale 16
```

For more details on llmb-run usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run inference scripts directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the benchmark commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/deepseek_r1/inference/trtllm/`

### Command Template

```shell
# GB200
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=gb200 ./launch.sh 

# B200
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=b200 ./launch.sh 

# H100
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=h100 ./launch.sh 
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

Structure of model weights folder `$LLMB_INSTALL/workloads/inference_deepseek-r1/DeepSeek-R1-FP4` on GB200 or B200 should look like the section below:

<details>

<summary>DeepSeek-R1-FP4</summary>

Double check that your folder has the same 395GiB size
Note: `du -sh` includes hidden contents like `.git` and may report a larger total. To verify weights-only, inspect file sizes in this folder with `ls -lh`.
```
total 395G
drwxr-xr-x  3 <user> dip  12K Jun 18 19:45 .
drwxr-xr-x 11 <user> dip  12K Jun 26 12:38 ..
-rw-r--r--  1 <user> dip 1.6K Jun 18 18:55 config.json
-rwxr-xr-x  1 <user> dip  13K Jun 18 18:55 configuration_deepseek.py
-rw-r--r--  1 <user> dip  171 Jun 18 18:55 generation_config.json
drwxr-xr-x  9 <user> dip 4.0K Jun 26 13:20 .git
-rw-r--r--  1 <user> dip 1.6K Jun 18 18:55 .gitattributes
-rw-r--r--  1 <user> dip  12K Jun 18 18:55 hf_quant_config.json
-rw-r--r--  1 <user> dip 1.1K Jun 18 18:55 LICENSE
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:00 model-00001-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:02 model-00002-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:36 model-00003-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:36 model-00004-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:26 model-00005-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:06 model-00006-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:26 model-00007-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:39 model-00008-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:27 model-00009-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:25 model-00010-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:10 model-00011-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:02 model-00012-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:40 model-00013-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:22 model-00014-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:20 model-00015-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:07 model-00016-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:29 model-00017-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:36 model-00018-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:20 model-00019-of-00080.safetensors
-rw-r--r--  1 <user> dip 4.8G Jun 18 19:44 model-00020-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:11 model-00021-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.1G Jun 18 19:02 model-00022-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:31 model-00023-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:21 model-00024-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:17 model-00025-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:05 model-00026-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:30 model-00027-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:31 model-00028-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:16 model-00029-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:08 model-00030-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:41 model-00031-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:40 model-00032-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:32 model-00033-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:21 model-00034-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:15 model-00035-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:01 model-00036-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:30 model-00037-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:34 model-00038-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:17 model-00039-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:07 model-00040-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:13 model-00041-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:39 model-00042-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:25 model-00043-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:22 model-00044-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:15 model-00045-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:03 model-00046-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:31 model-00047-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:35 model-00048-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:17 model-00049-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:07 model-00050-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:12 model-00051-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:40 model-00052-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:24 model-00053-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:25 model-00054-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:11 model-00055-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:10 model-00056-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:26 model-00057-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:34 model-00058-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:16 model-00059-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:06 model-00060-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:30 model-00061-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:35 model-00062-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:20 model-00063-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:45 model-00064-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.1G Jun 18 19:01 model-00065-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:12 model-00066-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:35 model-00067-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:18 model-00068-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:06 model-00069-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:44 model-00070-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.1G Jun 18 19:01 model-00071-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:11 model-00072-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:21 model-00073-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:41 model-00074-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:41 model-00075-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:43 model-00076-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:44 model-00077-of-00080.safetensors
-rw-r--r--  1 <user> dip 5.0G Jun 18 19:43 model-00078-of-00080.safetensors
-rw-r--r--  1 <user> dip 3.4G Jun 18 19:42 model-00079-of-00080.safetensors
-rw-r--r--  1 <user> dip 1.8G Jun 18 19:41 model-00080-of-00080.safetensors
-rw-r--r--  1 <user> dip  17M Jun 18 19:41 model.safetensors.index.json
-rw-r--r--  1 <user> dip 4.8K Jun 18 18:55 README.md
-rw-r--r--  1 <user> dip 3.6K Jun 18 18:55 tokenizer_config.json
-rw-r--r--  1 <user> dip 7.5M Jun 18 18:55 tokenizer.json
```

</details>

Structure of model weights folder `$LLMB_INSTALL/workloads/inference_deepseek-r1/DeepSeek-R1-FP8` on H100 should look like the section below:

<details>

<summary>DeepSeek-R1-FP8</summary>

Double check that your folder has the same 642GiB size
Note: `du -sh` includes hidden contents like `.git` and may report a larger total. To verify weights-only, inspect file sizes in this folder with `ls -lh`.
```
total 642G
drwxr-xr-x 3 <user> dip  20K Mar 31 18:25 .
drwxr-xr-x 9 <user> dip 4.0K Apr  3 13:35 ..
-rw-r--r-- 1 <user> dip  17K Mar 31 17:04 checksums.blake3
-rw-r--r-- 1 <user> dip 1.7K Mar 31 17:04 config.json
-rw-r--r-- 1 <user> dip  11K Mar 31 17:04 configuration_deepseek.py
drwxr-xr-x 2 <user> dip 4.0K Mar 31 17:04 figures
-rw-r--r-- 1 <user> dip  171 Mar 31 17:04 generation_config.json
-rw-r--r-- 1 <user> dip 1.1K Mar 31 17:04 LICENSE
-rw-r--r-- 1 <user> dip 4.9G Mar 31 17:14 model-00001-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00002-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00003-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00004-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00005-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00006-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00007-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00008-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00009-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00010-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00011-of-000163.safetensors
-rw-r--r-- 1 <user> dip 1.3G Mar 31 17:07 model-00012-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00013-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00014-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00015-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00016-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00017-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00018-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00019-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:12 model-00020-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:15 model-00021-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00022-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00023-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00024-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00025-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00026-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00027-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00028-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00029-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00030-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00031-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00032-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00033-of-000163.safetensors
-rw-r--r-- 1 <user> dip 1.7G Mar 31 17:16 model-00034-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00035-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00036-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:28 model-00037-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00038-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:21 model-00039-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:23 model-00040-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:24 model-00041-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:24 model-00042-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00043-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00044-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00045-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00046-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00047-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00048-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00049-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00050-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00051-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00052-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00053-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00054-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00055-of-000163.safetensors
-rw-r--r-- 1 <user> dip 1.7G Mar 31 17:24 model-00056-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00057-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:29 model-00058-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:31 model-00059-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:32 model-00060-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:33 model-00061-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:33 model-00062-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:37 model-00063-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00064-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00065-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00066-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00067-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00068-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00069-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00070-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00071-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00072-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00073-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00074-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00075-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00076-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:38 model-00077-of-000163.safetensors
-rw-r--r-- 1 <user> dip 1.7G Mar 31 17:33 model-00078-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:39 model-00079-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:41 model-00080-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:41 model-00081-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:41 model-00082-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:41 model-00083-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:44 model-00084-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:45 model-00085-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:45 model-00086-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:45 model-00087-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:45 model-00088-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:45 model-00089-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:45 model-00090-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:44 model-00091-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:45 model-00092-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 17:45 model-00093-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00094-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:14 model-00095-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:03 model-00096-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00097-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:03 model-00098-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:03 model-00099-of-000163.safetensors
-rw-r--r-- 1 <user> dip 1.7G Mar 31 17:53 model-00100-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:13 model-00101-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00102-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:03 model-00103-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00104-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00105-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:08 model-00106-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:03 model-00107-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00108-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:13 model-00109-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:03 model-00110-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00111-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00112-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:04 model-00113-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:08 model-00114-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:14 model-00115-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:14 model-00116-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:14 model-00117-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:14 model-00118-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:14 model-00119-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:15 model-00120-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:14 model-00121-of-000163.safetensors
-rw-r--r-- 1 <user> dip 1.7G Mar 31 18:08 model-00122-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:15 model-00123-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:15 model-00124-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:15 model-00125-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:15 model-00126-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:15 model-00127-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:15 model-00128-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:15 model-00129-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:18 model-00130-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:19 model-00131-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:19 model-00132-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:23 model-00133-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:23 model-00134-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:24 model-00135-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00136-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00137-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00138-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00139-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00140-of-000163.safetensors
-rw-r--r-- 1 <user> dip 3.0G Mar 31 18:22 model-00141-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:30 model-00142-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00143-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00144-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:29 model-00145-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00146-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00147-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00148-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:25 model-00149-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:29 model-00150-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:30 model-00151-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:30 model-00152-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:31 model-00153-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:32 model-00154-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:32 model-00155-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:32 model-00156-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:32 model-00157-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:32 model-00158-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:32 model-00159-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.9G Mar 31 18:32 model-00160-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:32 model-00161-of-000163.safetensors
-rw-r--r-- 1 <user> dip 4.1G Mar 31 18:32 model-00162-of-000163.safetensors
-rw-r--r-- 1 <user> dip 6.2G Mar 31 18:33 model-00163-of-000163.safetensors
-rw-r--r-- 1 <user> dip  74K Mar 31 18:25 modeling_deepseek.py
-rw-r--r-- 1 <user> dip 8.5M Mar 31 18:25 model.safetensors.index.json
-rw-r--r-- 1 <user> dip  19K Mar 31 17:04 README.md
-rw-r--r-- 1 <user> dip 3.5K Mar 31 18:25 tokenizer_config.json
-rw-r--r-- 1 <user> dip 7.5M Mar 31 18:25 tokenizer.json
```

</details>

If the size of the weights directory is not same as mentioned above or you suspect that weights are corrupted, then you will have to manually delete the DeepSeek wights folder `$LLMB_INSTALL/workloads/inference_deepseek-r1` and restart the setup script. 

## Run time OOM issues
If you encounter an Out of Memory issue during the runs, try to decrease the KV_CACHE_FRACTION to lower value and/or lower the max_batch_size.
Ex: Initial KV_CACHE_FRACTION=0.85 and max_batch_size=420 for 1000/1000 (Reasoning) use case resulted in OOM
Solution is to change KV_CACHE_FRACTION=0.8 in LLMB launch script and try rerunning the recipe.

# Run Nsight Profiling
To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. In addition, you will also need to set the `PROFILE_START` and `PROFILE_END` variables - steps PROFILE_START (inclusive) to PROFILE_END (exclusive) will be profiled.

### Profiling job details:
* **MPI Ranks:** all
* **Job Steps:** PROFILE_START --> PROFILE_END
* **Output Location:** $RESULT_DIR/nsys_profile
* **Filename format:** `profile_${SLURM_JOB_ID}_${SLURM_PROCID}_${SLURM_LOCALID}.nsys-rep`

### Customizing profiling behavior:
* Specify job steps to profile:
	* `PROFILE_START`: start profiling on this job step (Default: 0).
	* `PROFILE_END`: stop profiling on this job step (Default: 1).
* Enable GPU metrics collection:
	* `ENABLE_GPU_METRICS`: Enable GPU metrics collection during Nsight profiling (default: false)
	  - When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
	  - Provides additional GPU utilization, memory usage, and compute efficiency data
	  - May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics enabled:**
```shell
ENABLE_PROFILE=true ENABLE_GPU_METRICS=true PROFILE_START=500 PROFILE_END=505 MODE="max_throughput" MAX_NUM_TOKENS=6000 MAX_BATCH_SIZE=768 USE_CASE=reasoning:1000/1000 ENABLE_CHUNKED_PREFILL=false llmb-run submit -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run. 

When the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each rank. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.
