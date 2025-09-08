# Overview
This recipe provides instructions and scripts for benchmarking the performance of the DeepSeek-R1 model with Nvidia TRT-LLM (TensorRT-LLM) benchmark suite.

The script uses TRT-LLM release containers to benchmark the [DeepSeek-R1-FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4) inference workload. We consider two benchmarking scenarios
- **Maximum throughput**: the system is configured to generate as many tokens per second as possible. This typically involves large batch sizes, long generation lengths, and aggressive request packing to fully utilize GPU compute. While this approach increases overall efficiency and hardware utilization, it also results in higher latency for individual requests. It's ideal for offline processing, batch jobs, or queued summarization tasks.
- **Minimum latency**: prioritizes fast response times for each individual request. This often involves smaller batch sizes (sometimes just one), shorter generation lengths, and minimal scheduling overhead. While this reduces GPU efficiency and overall throughput, it significantly lowers response time, making it suitable for real-time, interactive applications like chatbots or low-latency APIs.

<div style="background-color: #ffeeba; padding: 10px; border-left: 6px solid #f0ad4e;">
  <strong>⚠️ WARNING:</strong> Currently this recipe only supports DeepSeek-R1-FP4 using 4x GB200 GPUs.
</div> 

# Performance Measurement and Analysis 

DeepSeek-R1 model was benchmarked for maximum throughput performance. The goal is to reach the maximum number of output tokens per second per GPU.

## 4x GB200

### Max throughput configurations

| Use Case      | GPUs | ISL   | OSL   |   max_batch_size | concurrency | max_num_tokens | kv_cache_free_gpu_mem_fraction | Quantization | num_requests |   TP |   PP |   EP | attn_dp_enabled |
|:--------------|:----:|:-----:|:-----:|:----------------:|:------------|:--------------:|:------------------------------:|:------------:|:------------:|:----:|:----:|:----:|:---------------:|
| reasoning     |    4 | 1,000 | 1,000 |              816 | 3,264       | 6,000          |         0.85                   | NVFP4        | 32,640       |    4 |    1 |    4 |     Yes         |
| chat          |    4 | 128   | 128   |            3,600 | 14,400      | 16,000         |         0.50                   | NVFP4        | 144,000      |    4 |    1 |    4 |     Yes         |
| summarization |    4 | 8,000 | 512   |              256 | 1,024       | 9,000          |         0.85                   | NVFP4        | 10,240       |    4 |    1 |    4 |     Yes         |
| generation    |    4 | 512   | 8,000 |              256 | 1,024       | 2,000          |         0.85                   | NVFP4        | 10,240       |    4 |    1 |    4 |     Yes         |


### Min latency configurations

|      Use Case      | GPUs |   ISL   |   OSL   | max_batch_size  | concurrency | max_num_tokens  | kv_cache_free_gpu_mem_fraction  | Quantization  | num_requests  | TP | PP | EP | attn_dp_enabled  |
|:-------------------|:----:|:-------:|:-------:|:---------------:|:-----------:|:---------------:|:-------------------------------:|:-------------:|:-------------:|:--:|:--:|:--:|:----------------:|
|     reasoning      |  4   | 1,000   | 1,000   |       1         |      1      |      1000       |              0.1                |    NVFP4      |      10       | 4  | 1  | 1  |       No         |
|       chat         |  4   | 128     | 128     |       1         |      1      |       128       |              0.1                |    NVFP4      |      10       | 4  | 1  | 1  |       No         |
|   summarization    |  4   | 8,000   | 512     |       1         |      1      |      8000       |              0.1                |    NVFP4      |      10       | 4  | 1  | 1  |       No         | 
|    generation      |  4   | 512     | 8,000   |       1         |      1      |       512       |              0.1                |    NVFP4      |      10       | 4  | 1  | 1  |       No         |


### Configuration Notes
- **kv_cache_free_gpu_mem_fraction**: fraction of memory allocated to store the kv cache values after loading the model weights.
- **attn_dp_enabled**: This flag in the config.yml dictates whether `Data parallelism` is enabled or disabled for the attention layers.
- You can find more information on the trt-llm build parameters here (https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html)

More details about the inference terms can be found in the [Appendix](../../APPENDIX.md)

### Metric Notes
- **TPS/GPU**: Output Token Throughput per second per GPU
- **TPS/User**: Output Token Throughput per second per user
- **Average Latency**: Average time for a request to be served
- **TTFT**: Time to First Token
- **TPOT**: Time Per Output Token

# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/deepseek-r1`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).


**Migration Note:**
If you previously used `STAGE_PATH`, replace it with `LLMB_INSTALL` (top-level). All output, logs, and checkpoints will be created under the workload's appropriate `LLMB_WORKLOAD` folder.

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

### Maximum throughput
```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run per use case for the maximum throughput scenario (** Recommended **)

# Reasoning
MODE="max_throughput" MAX_NUM_TOKENS=6000 MAX_BATCH_SIZE=816 USE_CASES=reasoning:1000/1000 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4

# Chat
MODE="max_throughput" MAX_NUM_TOKENS=16000 MAX_BATCH_SIZE=3600 KV_CACHE_FRACTION=0.5 USE_CASES=chat:128/128 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4

# Summarization
MODE="max_throughput" SBATCH_TIMELIMIT=40:00 MAX_NUM_TOKENS=9000 USE_CASES=summarization:8000/512 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4

# Generation
MODE="max_throughput" SBATCH_TIMELIMIT=1:40:00 USE_CASES=generation:512/8000 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```

### Minimum latency
```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run per use case for the minimum latency scenario (** Recommended **)

# Reasoning
MODE="min_latency" MAX_NUM_TOKENS=1000 USE_CASES=reasoning:1000/1000 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4

# Chat
MODE="min_latency" MAX_NUM_TOKENS=128 USE_CASES=chat:128/128 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4

# Summarization
MODE="min_latency" MAX_NUM_TOKENS=8000 USE_CASES=summarization:8000/512 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4

# Generation
MODE="min_latency" MAX_NUM_TOKENS=512 USE_CASES=generation:512/8000 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```

- Single use cases and their optimized parameters are listed above and work out of the box. Advanced users can add more use_cases in the `setup.sh` 
- Advanced users can learn more about: 
  - [Tuning Max Batch Size and Max Num Tokens](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html#tuning-max-batch-size-and-max-num-tokens) to adjust the inflight batching scheduler 
  -  [Max Tokens in Paged KV Cache and KV Cache Free GPU Memory Fraction](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html#max-tokens-in-paged-kv-cache-and-kv-cache-free-gpu-memory-fraction) to control the maximum number of tokens handled by the KV cache manager
- Use cases such as summarization and generation take significantly more time so the time limits are increased accordingly.


**Multiple USE_CASES:**
- While it is possible to run multiple USE_CASES within a single llmb-run invocation, doing so will increase the total runtime of the SLURM job. To minimize job duration and improve scheduling efficiency, it is recommended to split use cases into separate runs. 
- This approach may be beneficial for advanced LLMB users running multiple short benchmarking experiments, as it allows the model to be loaded only once across multiple experiments.

```bash
# Multi USE_CASE example (** NOT RECOMMENDED **)
 USE_CASES="reasoning:1000/1000 chat:128/128" llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```

**Streaming:**
- You can toggle streaming on and off. When off, users recieve the entire response (all output tokens) back at once instead of receiving output tokens as they are generated
  - **By default, streaming is turned on in this workload**
  -  If turned off -- TTFT (Time to First Token) and TPOT (Time per Output Token) metrics are not applicable, since individual token delivery is bypassed.

```bash
# Example of turning streaming off
STREAMING=false USE_CASES=reasoning:1000/1000 llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4
```

For more details on llmb-run usage, see the [llmb-run documentation](../../llmb-run/README.md).

## Direct Method

Alternatively, you can run inference scripts directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/deepseek-r1/inference`

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

Structure of model weights folder `$LLMB_INSTALL/workloads/inference_deepseek-r1/DeepSeek-R1-FP4`
should look like the section below:

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

If the size of the weights directory is not same as mentioned above or you suspect that weights are corrupted, then you will have to manually delete the DeepSeek wights folder `$LLMB_INSTALL/workloads/inference_deepseek-r1/DeepSeek-R1-FP4` and restart the setup script. 

## Run time OOM issues
If you encounter an Out of Memory issue during the runs, try to decrease the KV_CACHE_FRACTION to lower value and/or lower the max_batch_size.
Ex: Initial KV_CACHE_FRACTIONn=0.85 and max_batch_size=420 for 1000/1000 (Reasoning) use case resulted in OOM
Solution is to change KV_CACHE_FRACTION=0.8 in LLMB launch script and try rerunning the recipe.
