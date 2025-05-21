# Overview 
This recipe provides instructions and scripts for benchmarking the speed performance of NVIDIA Inference Microservices [NIMs](https://build.nvidia.com/models).

The script initializes containerized environments for inference benchmarking. It first deploys a container hosting the NIM service. A second container is launched with the Triton Server SDK and waits for the NIM service to be ready to receive requests. Then the second container executes the GenAI-Perf tool, systematically sending inference requests while capturing real-time performance metrics, including latency and throughput, across multiple concurrency levels.

GenAI-Perf tool is used generate the requests and capture the metrics: [First Token Latency (FTL), Inter-Token Latency (ITL), Request Latency, Request Throughput, and Token Throughput metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html).

For real-time inference its essential to stay within First Token Latency and Inter-Token Latency constraints as they define the user experience. Offline or batch inference generally prioritizes maximum throughput metrics.

# Prerequisites
- `NGC_API_KEY` ([NGC Registry](https://org.ngc.nvidia.com/setup) for access) which provides access  to the following containers:
   -  `nvcr.io/nvidia/tritonserver:25.01-py3-sdk`
   - `nvcr.io/nim/meta/llama3-70b-instruct:1.0.3`
      - The Llama3 70B container 
- Install [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli)
  - `ngc config set` after installation requires `NGC_API_KEY` from first prerequisite. 
- `HF_TOKEN` ([HuggingFace](https://huggingface.co/settings/tokens) for access)
    - Access to Llama 3.x must be requested through [Meta's website](https://llama.meta.com/llama-downloads/) then requested on the [HuggingFace Llama](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B) page. The approval process is not automatic and could take a day or more.

# Prepare Environment
## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**
- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Workload Setup
Follow the steps below on your Slurm login node

1. Set your HF token and NGC key (see [prerequisites](#prerequisites) for access):
      ```shell
      export NGC_API_KEY=<key>
      export HF_TOKEN=<token>
      ```
1. Set your STAGE_PATH, this is the location all artifacts for the benchmarks will be stored.
      ```shell
      export STAGE_PATH=<path> (e.g. /lustre/project/...)
      ```
1. Run the setup script
      - The script downloads the docker images from nvcr.io and saves them as .sqsh files in the $STAGE_PATH:
        - `nvcr.io/nvidia/tritonserver:25.01-py3-sdk` --> `nvidia+tritonserver+25.01.sqsh`
        - `nvcr.io/nim/meta/llama3-70b-instruct:1.0.3` --> `nim+meta+llama3-70b-instruct+1.0.3.sqsh`

      ```shell
      sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
      ```
            
      Check the corresponding `Slurm-<job_id>.out` file for status information on downloading containers.

 2. Download the Model
      -  HuggingFace weights and model engine (this section can take a while)
     
      ```shell
      bash prepare_models.sh
      ```
      Result:
      ```shell
      Getting files to download...
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ • 131.4/131.4 GiB • Remaining: 0:00:00 • 166.8 MB/s • Elapsed: 0:14:23 • Total: 40 - Completed: 40 - Failed: 0

      ------------------------------------------------------------------------------------------------------
      Download status: COMPLETED
      Downloaded local path model: /lustre/../llama3-70b-instruct_vhf
      Total files downloaded: 40
      Total transferred: 131.43 GB
      Started at: 2025-03-05 22:36:53
      Completed at: 2025-03-05 22:51:17
      Duration taken: 14m 23s

      Getting files to download...
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ • 68.4/68.4 GiB • Remaining: 0:00:00 • 141.4 MB/s • Elapsed: 0:07:44 • Total: 10 - Completed: 10 - Failed: 0

      -----------------------------------------------------------------------------------------------------------------------------------------------
      Download status: COMPLETED
      Downloaded local path model: /lustre/.../llama3-70b-instruct_vhf/llama3-70b-instruct_v0.10.0+cbc614f5-h100x4-fp8-throughput
      Total files downloaded: 10
      Total transferred: 68.44 GB
      Started at: 2025-03-05 22:51:23
      Completed at: 2025-03-05 22:59:08
      Duration taken: 7m 44s
      ```

       - If the engines appear to download instantly check that the `Total transferred: <size>GB` is the correct size, older NGC CLI versions will claim it downloaded successfully while only downloading a few KB. If so, install latest NGC CLI.

      In addition you should see in your `$STAGE_PATH` a folder called `llama3-70b-instruct_vhf`, verify that the weights and engine look this below:
      ```shell
      ls $STAGE_PATH/llama3-70b-instruct_vhf/

      config.json						    model-00011-of-00030.safetensors  model-00025-of-00030.safetensors
      generation_config.json					    model-00012-of-00030.safetensors  model-00026-of-00030.safetensors
      LICENSE							    model-00013-of-00030.safetensors  model-00027-of-00030.safetensors
      llama3-70b-instruct_v0.10.0+cbc614f5-h100x4-fp8-throughput  model-00014-of-00030.safetensors  model-00028-of-00030.safetensors
      model-00001-of-00030.safetensors			    model-00015-of-00030.safetensors  model-00029-of-00030.safetensors
      model-00002-of-00030.safetensors			    model-00016-of-00030.safetensors  model-00030-of-00030.safetensors
      model-00003-of-00030.safetensors			    model-00017-of-00030.safetensors  model.safetensors.index.json
      model-00004-of-00030.safetensors			    model-00018-of-00030.safetensors  README.md
      model-00005-of-00030.safetensors			    model-00019-of-00030.safetensors  special_tokens_map.json
      model-00006-of-00030.safetensors			    model-00020-of-00030.safetensors  tokenizer_config.json
      model-00007-of-00030.safetensors			    model-00021-of-00030.safetensors  tokenizer.json
      model-00008-of-00030.safetensors			    model-00022-of-00030.safetensors  trtllm_engine
      model-00009-of-00030.safetensors			    model-00023-of-00030.safetensors  USE_POLICY.md
      model-00010-of-00030.safetensors			    model-00024-of-00030.safetensors
      ```
      ```shell
      ls $STAGE_PATH/llama3-70b-instruct_vhf/trtllm_engine/

      checksums.blake3  LICENSE.txt	metadata.json  rank0.engine  rank2.engine  trt_llm_config.yaml
      config.json  NOTICE.txt     rank1.engine  rank3.engine
      ```

## Run Inference
1. Run launch.sh script 
      
      ```shell
      sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 launch.sh
      ```
      You should then see
      

      `Submitted batch job <job-id>`
      - This will launch the NIM Server and Benchmarking containers. Once the server is ready, the benchmarking container will run a sweep of GenAI-Perf commands
      - Depending on your Slurm system you may require changes in your sbatch parameters
      - You can track the status of your job-id with `squeue --job <job-id>`

      
2. Once your job starts, check progress of your script by viewing the server and benchmarking logs of your job-id
      ```shell
      tail -f $STAGE_PATH/logs/*<job-id>.out
      ```
      You should see the progress of these output logs
      
      ```
      ==> logs/server_<job-id>.out <==
      INFO 02-28 15:38:31.420 server.py:82] Started server process [4179797]
      INFO 02-28 15:38:31.421 on.py:48] Waiting for application startup.
      INFO 02-28 15:38:31.422 on.py:62] Application startup complete.
      INFO 02-28 15:38:31.424 server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

      ==> logs/benchmarking-<job-id>.out <==
                                    NVIDIA GenAI-Perf | LLM Metrics
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
      ┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
      ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
      │          Time to first token (ms) │  16.26 │  12.39 │  17.25 │  17.09 │  16.68 │  16.56 │
      │          Inter token latency (ms) │   1.85 │   1.55 │   2.04 │   2.02 │   1.97 │   1.92 │
      │              Request latency (ms) │ 499.20 │ 451.01 │ 554.61 │ 548.69 │ 526.13 │ 514.19 │
      │            Output sequence length │ 261.90 │ 256.00 │ 298.00 │ 296.60 │ 270.00 │ 265.00 │
      │             Input sequence length │ 550.06 │ 550.00 │ 553.00 │ 551.60 │ 550.00 │ 550.00 │
      │ Output token throughput (per sec) │ 520.87 │    N/A │    N/A │    N/A │    N/A │    N/A │
      │      Request throughput (per sec) │   1.99 │    N/A │    N/A │    N/A │    N/A │    N/A │
      └───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
      ```
      - Logs for benchmarking `benchmarking_<job-id>.out`, and the NIM inference service `server_<job-id>.out` will appear in your `$STAGE_PATH/logs/`
      
      - CSV results per GenAI-Perf command will appear in the `path/to/results`
      
      Example CSV Path:      
      `${STAGE_PATH}/results/llama-3-70b-instruct_{GPUs}_{concurrency}_{use_case}_{ISL}_{OSL}_{job-id}/profile_export_genai_perf.csv`
      
      Example CSV Data:
      ```
      Metric,avg,min,max,p99,p95,p90,p75,p50,p25
      Time To First Token (ms),23.94,22.92,30.75,30.10,27.51,24.26,23.38,23.20,23.02
      Inter Token Latency (ms),11.54,11.45,11.70,11.69,11.64,11.58,11.55,11.53,11.52
      Request Latency (ms),"5,919.16","5,912.73","5,934.45","5,933.31","5,928.74","5,923.03","5,920.17","5,918.56","5,915.29"
      Output Sequence Length,511.90,505.00,516.00,515.91,515.55,515.10,512.75,512.00,511.25
      Input Sequence Length,127.50,126.00,128.00,128.00,128.00,128.00,128.00,128.00,127.00

      Metric,Value
      Output Token Throughput (per sec),86.48
      Request Throughput (per sec),0.17
      ```
