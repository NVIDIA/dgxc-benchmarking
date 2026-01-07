# Overview
This recipe provides instructions and scripts for benchmarking the performance of the GPT-OSS-120B model with NVIDIA dynamo-trtllm + NVIDIA AI Perf benchmark suite.

The script uses dynamo-trtllm release containers to benchmark the [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) inference workload. In this recipe, we benchmark the max-throughput use case only.

- **Maximum throughput**: The system is configured to generate as many tokens per second as possible. This typically involves large batch sizes, long generation lengths, and aggressive request packing to fully utilize GPU compute. While this approach increases overall efficiency and hardware utilization, it also results in higher latency for individual requests. It's ideal for offline processing, batch jobs, or queued summarization tasks.

<div style="background-color: #ffeeba; padding: 10px; border-left: 6px solid #f0ad4e;">
  <strong>⚠️ WARNING:</strong> Currently this recipe only supports GPT-OSS-120B using 4x GB200 GPUs and 4x B200 GPUs. We benchmarked our scripts for ISL/OSL combination of 128/1000 in this recipe.
</div>

# Performance Measurement and Analysis
Below, we list the inference configuration and benchmarking performance of max throughput scenario for GPT-OSS-120B model.

## Maximum throughput

### GB200 Inference Use case configs

| Use Case      | GPUs | ISL   | OSL   |  max_batch_size | concurrency | max_num_tokens | kv_cache_free_gpu_mem_fraction| Quantization |  TP |   PP |   DP | enable-dp-attention |
|:--------------|:-----:|:-----:|:-----:|:----------------:|:-------:|:--------------:|:------------------------------:|:------------:|:----:|:----:|:----:|:---------------:|
| Generation | 4 | 128 | 1000 | 800 | 2048 | 768 | 0.8 | MXFP4 | 4 | 1 | 4 | YES | 



### B200 Inference Use case configs

| Use Case      | GPUs | ISL   | OSL   |   max_batch_size | concurrency | max_num_tokens | kv_cache_free_gpu_mem_fraction | Quantization |  TP |   PP |   DP | enable-dp-attention |
|:--------------|:----:|:-----:|:-----:|:----------------:|:------------:|:--------:|:------------------------------:|:------------:|:----:|:----:|:----:|:---------------:|
| Generation | 4 | 128 | 1000 | 800 | 2048 |  768  | 0.8 | MXFP4 | 4 | 1 | 4 | YES | 


**Note**
- kv_cache_free_gpu_mem_fraction: fraction of memory allocated to store the kv cache values after loading the model weights.
- attn_dp_enabled: This flag in the config.yml dictates whether `Data parallelism` is enabled or disabled for the attention layers.
- You can find more information on the TRT-LLM build parameters (https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html) and NVIDIA Dynamo (https://developer.nvidia.com/dynamo) at the links above.

More details about the inference terms can be found here [Appendix](../../../APPENDIX.md)


# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/inference_gpt-oss-dynamo`.
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
llmb-run submit -w inference_gpt-oss-dynamo -s 120b --dtype mxfp4 --scale 4
```

- Advanced users can learn more about: 
  - [Dynamo benchmarking guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/benchmarks/benchmarking.md) to understand Dynamo benchmarking use cases
  -  [Max Tokens in Paged KV Cache and KV Cache Free GPU Memory Fraction](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html#max-tokens-in-paged-kv-cache-and-kv-cache-free-gpu-memory-fraction) to control the maximum number of tokens handled by the KV cache manager

**Streaming:**
- You can toggle streaming on and off. When off, users receive the entire response (all output tokens) back at once instead of receiving output tokens as they are generated
  - **By default, streaming is turned on in this workload**
  -  If turned off -- TTFT (Time to First Token) and TPOT (Time per Output Token) metrics are not applicable, since individual token delivery is bypassed.

```bash
# Example of turning streaming off on GB200 or B200
STREAMING=false llmb-run submit -w inference_gpt-oss-dynamo -s 120b --dtype mxfp4 --scale 4
```

For more details on llmb-run usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run inference scripts directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**: 
- Ensure your virtual environment is activated before running the inference commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/gpt-oss/inference/slurm/`

### Command Template

```shell
# GB200
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=gb200 ./launch.sh 

# B200
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=b200 ./launch.sh 
```

### Results/Log files 
Results for the workload are stored at `$LLMB_INSTALL/workloads/inference_gpt-oss-dynamo/experiments/<model>_$ISL_$OSL_$MAX_BATCH_SIZE_$CONCURRENCY_$SLURMID`
+Worker and server logs are stored in the sub dir `$LLMB_INSTALL/workloads/inference_gpt-oss-dynamo/experiments/<model>_$ISL_$OSL_$MAX_BATCH_SIZE_$CONCURRENCY_$SLURMID/server_logs` and benchmark logs are located at `$LLMB_INSTALL/workloads/inference_gpt-oss-dynamo/experiments/<model>_$ISL_$OSL_$MAX_BATCH_SIZE_$CONCURRENCY_$SLURMID/benchmark_logs`. We can see the final output log file as a `profile_export_aiperf.csv` in the benchmark_logs folder. 

You should expect to see result directories like this:
```
gpt-oss-dynamo_ISL_OSL_BS_CON_SLURM_JOBID/
|- benchmark_logs
|--- deployment_config.json  
|--- inputs.json  
|--- profile_export_aiperf.csv  
|--- profile_export_aiperf.json  
|--- profile_export.jsonl
├- server_logs
|--- bench.log     
|--- output_server.log
|--- output_workers.log
|--- workers_start.log

```

# FAQ & Troubleshooting

Structure of model weights folder `$LLMB_INSTALL/workloads/inference_gpt-oss-dynamo/gpt-oss-120b` on GB200 or B200 should look like the section below:

<details>

<summary>gpt-oss-120b</summary>

Double check that your folder has the same 61 GiB size
Note: `du -sh` includes hidden contents like `.git` and may report a larger total. To verify weights-only, inspect file sizes in this folder with `ls -lh`.
```
total 61G
drwxrwsr-x 5 <user> dip 4.0K Nov 17 14:47 .
drwxrwsr-x 4 <user> dip 4.0K Nov 20 14:58 ..
drwxrwsr-x 3 <user> dip 4.0K Nov 17 14:37 .cache
-rw-rw-r-- 1 <user> dip  17K Nov 17 14:37 chat_template.jinja
-rw-rw-r-- 1 <user> dip 2.1K Nov 17 14:37 config.json
-rw-rw-r-- 1 <user> dip  177 Nov 17 14:37 generation_config.json
-rw-rw-r-- 1 <user> dip 1.6K Nov 17 14:37 .gitattributes
-rw-rw-r-- 1 <user> dip  12K Nov 17 14:37 LICENSE
drwxrwsr-x 2 <user> dip 4.0K Nov 17 14:47 metal
-rw-rw-r-- 1 <user> dip 4.4G Nov 17 14:40 model-00000-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 3.9G Nov 17 14:41 model-00001-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 4.4G Nov 17 14:37 model-00002-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 3.9G Nov 17 14:41 model-00003-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 4.4G Nov 17 14:40 model-00004-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 3.9G Nov 17 14:38 model-00005-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 4.4G Nov 17 14:38 model-00006-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 3.8G Nov 17 14:38 model-00007-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 4.4G Nov 17 14:40 model-00008-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 3.9G Nov 17 14:41 model-00009-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 4.4G Nov 17 14:39 model-00010-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 3.9G Nov 17 14:40 model-00011-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 3.8G Nov 17 14:41 model-00012-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 4.4G Nov 17 14:43 model-00013-of-00014.safetensors
-rw-rw-r-- 1 <user> dip 3.9G Nov 17 14:44 model-00014-of-00014.safetensors
-rw-rw-r-- 1 <user> dip  54K Nov 17 14:40 model.safetensors.index.json
drwxrwsr-x 2 <user> dip 4.0K Nov 17 14:45 original
-rw-rw-r-- 1 <user> dip 7.0K Nov 17 14:37 README.md
-rw-rw-r-- 1 <user> dip   98 Nov 17 14:43 special_tokens_map.json
-rw-rw-r-- 1 <user> dip 4.2K Nov 17 14:43 tokenizer_config.json
-rw-rw-r-- 1 <user> dip  27M Nov 17 14:43 tokenizer.json
-rw-rw-r-- 1 <user> dip  201 Nov 17 14:37 USAGE_POLICY
```

</details>

