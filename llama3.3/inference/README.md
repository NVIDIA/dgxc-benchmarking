# Overview
This recipe provides instructions and scripts for benchmarking the performance of the Llama3.3-70B model with Nvidia TRT-LLM (TensorRT-LLM) benchmark suite.

The script uses TRT-LLM release containers to benchmark the [Llama-3.3-70B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4) inference workload for maximum throughput bencharking.
<div style="background-color: #ffeeba; padding: 10px; border-left: 6px solid #f0ad4e;">
  <strong>⚠️ WARNING:</strong> Currently this recipe only supports `nvidia/Llama-3.3-70B-Instruct-FP4` using a single GB200 GPU.
</div> 

# Benchmarking setup

We consider two benchmarking scenarios:
- **Maximum throughput**: the system is configured to generate as many tokens per second as possible. This typically involves large batch sizes, long generation lengths, and aggressive request packing to fully utilize GPU compute. While this approach increases overall efficiency and hardware utilization, it also results in higher latency for individual requests. It's ideal for offline processing, batch jobs, or queued summarization tasks.
- **Minimum latency**: prioritizes fast response times for each individual request. This often involves smaller batch sizes (sometimes just one), shorter generation lengths, and minimal scheduling overhead. While this reduces GPU efficiency and overall throughput, it significantly lowers response time, making it suitable for real-time, interactive applications like chatbots or low-latency APIs.


# Performance Measurement and Analysis 

Llama3.3-70B model was benchmarked for maximum throughput performance. The goal is to reach the maximum number of output tokens per second per GPU.

## GB200

### Max throughput configurations

| Use Case      |   GPUs | ISL   | OSL   | max_batch_size   | concurrency   | max_num_tokens   | kv_cache_free_gpu_mem_fraction   | Quantization   | num_requests   |   TP |   PP |   EP | attn_dp_enabled   |
|:--------------|:------:|:-----:|:-----:|:----------------:|:-------------:|:----------------:|:--------------------------------:|:--------------:|:--------------:|:----:|:----:|:----:|:-----------------:|
| reasoning     |   1    | 1,000 | 1,000 | 256              | 256           | 16,384           |            0.95                  | NVFP4          | 4,096          |    1 |    1 |    1 |       YES         |
| chat          |   1    | 128   | 128   | 2,048            | 2,048         | 8,192            |            0.95                  | NVFP4          | 4,096          |    1 |    1 |    1 |       YES         |
| summarization |   1    | 8,000 | 512   | 128              | 128           | 8,128            |            0.95                  | NVFP4          | 1,024          |    1 |    1 |    1 |       YES         |
| generation    |   1    | 512   | 8,000 | 128              | 128           | 2,048            |            0.95                  | NVFP4          | 1,000          |    1 |    1 |    1 |       YES         |



### Min latency configurations

| Use Case      |   GPUs | ISL   | OSL   | max_batch_size | concurrency | max_num_tokens  | kv_cache_free_gpu_mem_fraction   | Quantization   | num_requests |   TP |   PP |   EP | attn_dp_enabled  |
|:--------------|:------:|:-----:|:-----:|:--------------:|:-----------:|:---------------:|:--------------------------------:|:--------------:|:------------:|:----:|:----:|:----:|:----------------:|
| reasoning     |   4    | 1,000 | 1,000 | 1              | 1           | 2,048           |            0.75                  | NVFP4          | 20           |    4 |    1 |    1 |       NO         |
| chat          |   4    | 128   | 128   | 1              | 1           | 2,048           |            0.75                  | NVFP4          | 20           |    4 |    1 |    1 |       NO         |
| summarization |   4    | 8,000 | 512   | 1              | 1           | 8,512           |            0.75                  | NVFP4          | 20           |    4 |    1 |    1 |       NO         |
| generation    |   4    | 512   | 8,000 | 1              | 1           | 8,512           |            0.75                  | NVFP4          | 20           |    4 |    1 |    1 |       NO         |



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


During installation process your will be prompted for the token multiple times:
```
HuggingFace Token (HF_TOKEN) - Some workloads require this for accessing HuggingFace models and datasets.
You can get your token from: https://huggingface.co/settings/tokens
Note: If you're sure you don't need HF_TOKEN for your selected workloads, this can be left blank.
? Enter your HuggingFace token (HF_TOKEN) or leave blank: <hf-token>
✓ HF_TOKEN configured successfully
```

When prompted for a username use the string `__token__` and your actual token for the password.

```
Username for 'https://huggingface.co': __token__
Password for 'https://__token__@huggingface.co': <hf_token>
```


## Request Access

Access to Llama3.3 must be requested through the [HuggingFace Llama 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct). The approval process is not automatic and could take a day or more.

# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/llama3.3-70b`.
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

# Run a benchmark with llmb-run per use case (** Recommended **)

# Reasoning
MODE="max_throughput" MAX_NUM_TOKENS=16384 MAX_BATCH_SIZE=256 NUM_REQUESTS=4096 CONCURRENCY=256 USE_CASES=reasoning:1000/1000 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 1

# Chat
MODE="max_throughput" MAX_NUM_TOKENS=8192 MAX_BATCH_SIZE=2048 NUM_REQUESTS=4096 CONCURRENCY=2048 USE_CASES=chat:128/128 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 1

# Summarization
SBATCH_TIMELIMIT=50:00 MODE="max_throughput" MAX_NUM_TOKENS=8128 MAX_BATCH_SIZE=128 NUM_REQUESTS=1024 CONCURRENCY=128 USE_CASES=summarization:8000/512 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 1

# Generation
SBATCH_TIMELIMIT=50:00 MODE="max_throughput" MAX_NUM_TOKENS=2048 MAX_BATCH_SIZE=128 NUM_REQUESTS=1000 CONCURRENCY=128 USE_CASES=generation:512/8000 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 1
```

### Minimum latency

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run per use case (** Recommended **)

# Reasoning
MODE="min_latency" USE_CASES=reasoning:1000/1000 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 4

# Chat
MODE="min_latency" USE_CASES=chat:128/128 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 4

# Summarization
MODE="min_latency" USE_CASES=summarization:8000/512 MAX_NUM_TOKENS=8512 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 4 

# Generation
MODE="min_latency" USE_CASES=generation:512/8000 MAX_NUM_TOKENS=8512 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 4
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
 USE_CASES="reasoning:1000/1000 chat:128/128" llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 1
```

**Streaming:**
- You can toggle streaming on and off. When off, users recieve the entire response (all output tokens) back at once instead of receiving output tokens as they are generated
  - **By default, streaming is turned on in this workload**
  -  If turned off -- TTFT (Time to First Token) and TPOT (Time per Output Token) metrics are not applicable, since individual token delivery is bypassed.

```bash
# Example of turning streaming off
STREAMING=false USE_CASES=reasoning:1000/1000 llmb-run single -w inference_llama3.3 -s 70b --dtype nvfp4 --scale 1
```

For more details on llmb-run usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run inference scripts directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/llama3.3-70b/inference`

### Command Template

```shell
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} ./launch.sh 
```

### Results/Log files 
Results for the workload are stored at `$LLMB_INSTALL/workloads/inference_<model>/experiments/<model>_TP_EP_PP_CON_USECASE`

You should expect to see result directories for each use case:

- `<model>_TP_EP_PP_CON_reasoning`
- `<model>_TP_EP_PP_CON_chat`
- `<model>_TP_EP_PP_CON_summarization`
- `<model>_TP_EP_PP_CON_generation`

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

Structure of model weights folder `$LLMB_INSTALL/workloads/inference_llama3.3-70b/Llama3.3_70B`
should look like the section below:

<details>

<summary>Llama3.3_70B</summary>

Double check that your folder has the same 40GiB
Note: `du -sh` includes hidden contents like `.git` and may report a larger total. To verify weights-only, inspect file sizes in this folder with `ls -lh`.

```
total 40G
drwxrwsr-x 3 user dip 4.0K Jun 29 22:46 .
drwxrwsr-x 6 user dip 4.0K Jun 30 22:39 ..
drwxrwsr-x 3 user dip 4.0K Jun 29 22:45 .cache
-rw-rw-r-- 1 user dip 1.6K Jun 29 22:45 .gitattributes
-rw-rw-r-- 1 user dip 149K Jun 29 22:45 LICENSE.pdf
-rw-rw-r-- 1 user dip 4.8K Jun 29 22:45 README.md
-rw-rw-r-- 1 user dip  940 Jun 29 22:45 config.json
-rw-rw-r-- 1 user dip  184 Jun 29 22:45 generation_config.json
-rw-rw-r-- 1 user dip  269 Jun 29 22:45 hf_quant_config.json
-rw-rw-r-- 1 user dip 4.7G Jun 29 22:46 model-00001-of-00009.safetensors
-rw-rw-r-- 1 user dip 4.6G Jun 29 22:46 model-00002-of-00009.safetensors
-rw-rw-r-- 1 user dip 4.7G Jun 29 22:46 model-00003-of-00009.safetensors
-rw-rw-r-- 1 user dip 4.7G Jun 29 22:46 model-00004-of-00009.safetensors
-rw-rw-r-- 1 user dip 4.7G Jun 29 22:46 model-00005-of-00009.safetensors
-rw-rw-r-- 1 user dip 4.7G Jun 29 22:46 model-00006-of-00009.safetensors
-rw-rw-r-- 1 user dip 4.7G Jun 29 22:46 model-00007-of-00009.safetensors
-rw-rw-r-- 1 user dip 4.7G Jun 29 22:46 model-00008-of-00009.safetensors
-rw-rw-r-- 1 user dip 2.9G Jun 29 22:46 model-00009-of-00009.safetensors
-rw-rw-r-- 1 user dip 216K Jun 29 22:46 model.safetensors.index.json
-rw-rw-r-- 1 user dip  325 Jun 29 22:46 special_tokens_map.json
-rw-rw-r-- 1 user dip  17M Jun 29 22:46 tokenizer.json
-rw-rw-r-- 1 user dip  55K Jun 29 22:46 tokenizer_config.json
```

</details>

If the size of the weights directory is not same as mentioned above or you suspect that weights are corrupted, then you will have to manually delete the Llama3.3-70b weights folder `$LLMB_INSTALL/workloads/inference_llama3.3-70b/Llama3.3_70B` and restart the setup script. 

## Run time OOM issues
If you encounter an Out of Memory issue during the runs, try to decrease the KV_CACHE_FRACTION to lower value and/or lower the max_batch_size.
Ex: Initial KV_CACHE_FRACTIONn=0.85 and max_batch_size=320 for 1000/1000 (Reasoning) use case resulted in OOM
Solution is to change KV_CACHE_FRACTION=0.8 in LLMB launch script and try rerunning the recipe.
