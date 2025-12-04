 Overview
This recipe provides instructions and scripts for benchmarking the performance of the DeepSeek-R1-FP4 model with the sglang benchmark suite.

The script uses sglang release containers to benchmark the [DeepSeek-R1-FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4) inference workload. In this recipe, we are benchmarking Max throughput usecase only.
- **Maximum throughput**: The system is configured to generate as many tokens per second as possible. This typically involves large batch sizes, long generation lengths, and aggressive request packing to fully utilize GPU compute. While this approach increases overall efficiency and hardware utilization, it also results in higher latency for individual requests. It's ideal for offline processing, batch jobs, or queued summarization tasks.

<div style="background-color: #ffeeba; padding: 10px; border-left: 6px solid #f0ad4e;">
  <strong>⚠️ WARNING:</strong> Currently this recipe only supports DeepSeek-R1-FP4 using 4x GB200 GPUs and 8x B200 GPUs.
</div>

# Performance Measurement and Analysis
Below, we list the inference configuration and benchmarking performance of max throughput scenario for DeepSeek-R1-FP4 model.

## Maximum throughput

### GB200 Inference Use case configs

| Use Case      | GPUs | ISL   | OSL   |   max_running_requests | concurrency | chunked prefill tokens | num_prompts | mem-fraction-static | Quantization |  TP |   PP |   DP | enable-dp-attention |
|:--------------|:----:|:-----:|:-----:|:----------------:|:-------:|:-----|:--------------:|:------------------------------:|:------------:|:----:|:----:|:----:|:---------------:|
| reasoning     |    4 | 1,000 | 1,000 |  1,024 | 4,096   | 16384 | 20,000   |       0.8      | NVFP4   |    4 |    1 |    4 |     Yes         |
| chat          |    4 | 128   | 128   |  1,024 | 12,000  | 32768 | 60,000   |       0.8      | NVFP4   |    4 |    1 |    4 |     Yes         |
| summarization |    4 | 8,000 | 512   |  1,024 | 4,096   | 32768 | 20,000   |       0.8      | NVFP4   |    4 |    1 |    4 |     Yes         |
| generation    |    4 | 512   | 8,000 |  1,024 | 1,024   | 32768 | 5,000    |       0.8      | NVFP4   |    4 |    1 |    4 |     Yes         |

### B200 Inference Use case configs

| Use Case      | GPUs | ISL   | OSL   |   max_running_requests | concurrency | chunked prefill tokens | num_prompts | mem-fraction-static | Quantization |  TP |   PP |   DP | enable-dp-attention |
|:--------------|:----:|:-----:|:-----:|:----------------:|:------------:|:------|:--------:|:------------------------------:|:------------:|:----:|:----:|:----:|:---------------:|
| reasoning     |    8 | 1,000 | 1,000 |  4,096 | 4,096   | 32,768  | 20,000   |   0.7   | NVFP4     |    8 |    1 |    8 |     Yes         |
| chat          |    8 | 128   | 128   |  4,096 | 4,096   | 16,384 | 20,000    |   0.7    | NVFP4    |    8 |    1 |    8 |     Yes         |
| summarization |    8 | 8,000 | 512   |  4,096 | 2,048   | 32,768 | 10,000    |   0.8    | NVFP4    |    8 |    1 |    8 |     Yes         |
| generation    |    8 | 512   | 8,000 |  4,096 | 2,048   | 32,768 | 10,000    |   0.8    | NVFP4    |    8 |    1 |    8 |     Yes         |


**Note**
- mem-fraction-static: fraction of memory allocated to store the kv cache values after loading the model weights.
- enable-dp-attention: This flag in the config.yml dictates whether `Data parallelism` is enabled or disabled for the attention layers.
- you can find more information on the sglang parameters here ([sglang arguments](https://docs.sglang.ai/advanced_features/server_arguments.html))

More details about the inference terms can be found here [Appendix](../../../APPENDIX.md)

**Note** Below are the metrics measured for this inference workload
- TPS/GPU: Throughput per second per GPU
- TPS/User: Throughput per second per user
- Average Latency: Average time for a a request to get served
- TTFT: Time for first token
- TPOT: Time between output tokens

# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/inference_deepseek-r1-sglang`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are only guidelines and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

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

# Run a benchmark on GB200 with llmb-run per use case for the maximum throughput scenario (** Recommended **)
# Reasoning
USE_CASES="reasoning:1000/1000" llmb-run single -w inference_deepseek-r1-sglang -s 671b --dtype nvfp4 --scale 4

# Chat
USE_CASES="chat:128/128" llmb-run single -w inference_deepseek-r1-sglang -s 671b --dtype nvfp4 --scale 4

# Summarization
USE_CASES="summarization:8000/512" llmb-run single -w inference_deepseek-r1-sglang -s 671b --dtype nvfp4 --scale 4

# Generation
USE_CASES="generation:512/8000" llmb-run single -w inference_deepseek-r1-sglang -s 671b --dtype nvfp4 --scale 4

# Run a benchmark on B200 with llmb-run per use case for the maximum throughput scenario (** Recommended **)
# Reasoning
USE_CASES="reasoning:1000/1000" llmb-run single -w inference_deepseek-r1-sglang -s 671b --dtype nvfp4 --scale 8

# Chat
USE_CASES="chat:128/128" llmb-run single -w inference_deepseek-r1-sglang -s 671b --dtype nvfp4 --scale 8

# Summarization
USE_CASES="summarization:8000/512" llmb-run single -w inference_deepseek-r1-sglang -s 671b --dtype nvfp4 --scale 8

# Generation
USE_CASES="generation:512/8000" llmb-run single -w inference_deepseek-r1-sglang -s 671b --dtype nvfp4 --scale 8
```

- Single use cases and their optimized parameters are listed above and work out of the box. Advanced users can add more use_cases and configurations in the `launch_server.sh and launch.sh`
- Use cases such as summarization and generation do not work well with large prompts, so the total prompts are decreased for these usescase.


**Multiple USE_CASES:**
- While it is possible to run multiple USE_CASES within a single llmb-run invocation, doing so will increase the total runtime of the SLURM job. To minimize job duration and improve scheduling efficiency, it is recommended to split use cases into separate runs.
- This approach may be beneficial for advanced LLMB users running multiple short benchmarking experiments, as it allows the model to be loaded only once across multiple experiments.

```bash
# Multi USE_CASE example on GB200
# Note that running multiple use cases at once may take longer time and it is advised to adjust the SBATCH time limits accordingly.
 USE_CASES="reasoning:1000/1000 chat:128/128" llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 4

# Multi USE_CASE example on B200
 USE_CASES="reasoning:1000/1000 chat:128/128" llmb-run single -w inference_deepseek-r1 -s 671b --dtype nvfp4 --scale 8
```

For more details on llmb-run usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run inference scripts directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:
- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_REPO/deepseek-r1/inference/sglang`

### Command Template

```shell
#GB200 run command
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=gb200 ./launch.sh

#B200 run command
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} GPU_TYPE=b200 ./launch.sh
```

### Results/Log files
The server logs are located at `$LLMB_INSTALL/workloads/inference_deepseek-r1-sglang/experiments/server_TP_DP_SLURM_JOBID.out` This log shows the model weights loading and details on how tokens are processed.
Benchmark results for the workload are stored at `$LLMB_INSTALL/workloads/inference_deepseek-r1-sglang/experiments/<model>_<mode>_TP_DP_CON_USECASE`

You should expect to see result directories for each use case:

- `<model>_<mode>_TP_DP_CON_reasoning`
- `<model>_<mode>_TP_DP_CON_chat`
- `<model>_<mode>_TP_DP_CON_summarization`
- `<model>_<mode>_TP_DP_CON_generation`

Each directory will contain SLURM output log file per job:

```
Model_TP_DP_CON_<use_case>/
├── Model_TP_DP_CON_<use_case>_SLURM_JOBID.out  # Benchmarking output
```
The `PERFORMANCE OVERVIEW` section in the `*.out` file provides key performance metrics:
```
...

============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    <value>
Max request concurrency:                 <values>
Successful requests:                     <value>
Benchmark duration (s):                  <value>
Total input tokens:                      <value>
Total generated tokens:                  <value>
Total generated tokens (retokenized):    <value>
Request throughput (req/s):              <value>
Input token throughput (tok/s):          <value>
Output token throughput (tok/s):         <value>
Total token throughput (tok/s):          <value>
Concurrency:                             <value>
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   <value>
Median E2E Latency (ms):                 <value>
---------------Time to First Token----------------
Mean TTFT (ms):                          <value>
Median TTFT (ms):                        <value>
P99 TTFT (ms):                           <value>
---------------Inter-Token Latency----------------
Mean ITL (ms):                           <value>
Median ITL (ms):                         <value>
P95 ITL (ms):                            <value>
P99 ITL (ms):                            <value>
Max ITL (ms):                            <value>
==================================================
```

# FAQ & Troubleshooting

Structure of model weights folder `$LLMB_INSTALL/workloads/inference_deepseek-r1-sglang/DeepSeek-R1-FP4` on GB200 or B200 should look like the section below:

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

If the size of the weights directory is not same as mentioned above or you suspect that weights are corrupted, then you will have to manually delete the DeepSeek weights folder `$LLMB_INSTALL/workloads/inference_deepseek-r1-sglang/DeepSeek-R1-FP4` and restart the setup script.