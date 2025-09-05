# Overview 
This recipe provides instructions and scripts for benchmarking the speed performance of the DeepSeek R1 NVIDIA Inference Microservices [NIMs](https://build.nvidia.com/models).

The script initializes containerized environments for inference benchmarking. It first deploys a container hosting the NIM service. A second container is launched with the Triton Server SDK and waits for the NIM service to be ready to receive requests. Then the second container executes the GenAI-Perf tool, systematically sending inference requests while capturing real-time performance metrics, including latency and throughput, across multiple concurrency levels.

GenAI-Perf tool is used generate the requests and capture the metrics: [First Token Latency (FTL), Inter-Token Latency (ITL), Request Latency, Request Throughput, and Token Throughput metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html).

For real-time inference its essential to stay within First Token Latency and Inter-Token Latency constraints as they define the user experience. Offline or batch inference generally prioritizes maximum throughput metrics.

The recipe is supported on NVIDIA H100s. 16 H100s are needed. 

# Prerequisites
- `NGC_API_KEY` ([NGC Registry](https://org.ngc.nvidia.com/setup) for access) which provides access  to the following containers:
   -  `nvcr.io/nvidia/tritonserver:25.01-py3-sdk`
   -  `nvcr.io/nim/deepseek-ai/deepseek-r1:1.7.2`
- Install [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli)
  - `ngc config set` after installation requires `NGC_API_KEY` from first prerequisite. 
- `HF_TOKEN` ([HuggingFace](https://huggingface.co/settings/tokens) for access)


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
2. Set your STAGE_PATH, this is the location all artifacts for the benchmarks will be stored.
      ```shell
      export STAGE_PATH=<path> (e.g. /lustre/project/...)
      ```
3. Run the setup script
      - The script downloads the docker images from nvcr.io and saves them as .sqsh files in the $STAGE_PATH:
        - `nvcr.io/nvidia/tritonserver:25.01-py3-sdk` --> `nvidia+tritonserver+25.01.sqsh`
        - `nvcr.io/nim/deepseek-ai/deepseek-r1:1.7.2` --> `nim+deepseek+r1+1.7.2.sqsh`

      ```shell
      sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
      ```
4. Download the Model
      -  HuggingFace weights and model engine (this section can take a while)
     
      ```shell
      bash prepare_models.sh
      ```
      You should see the model begin to download 641.3GB:
      ```shell
      ⠋ ━━━━━━━━━━━━━━━━━━ • 12.2/641.3 GiB • Remaining: 0:49:49 • 226.0 MB/s • Elapsed: 0:01:01 • Total: 174 - Completed: 7 - Failed: 0

      ```

      When finished you should see in your `$STAGE_PATH` a folder called `deepseek-r1-instruct_vhf-5dde110-nim-fp8`
      - **NOTE: Double check the folder size is 642GB, otherwise you will need to redownload**
                       
      ```shell
      du -sh  $STAGE_PATH/deepseek-r1-instruct_vhf-5dde110-nim-fp8
      ```
      
      ```shell
      642G    <STAGE_PATH>/deepseek-r1-instruct_vhf-5dde110-nim-fp8
      ```

      You should see the following contents in the folder:
      ```shell
      ls -lha $STAGE_PATH/deepseek-r1-instruct_vhf-5dde110-nim-fp8
      ```

      ```shell
      -rw-r--r-- 1 <user> dip  17K Mar 13 09:51 checksums.blake3
      -rw-r--r-- 1 <user> dip 1.7K Mar 13 09:51 config.json
      -rw-r--r-- 1 <user> dip  11K Mar 13 09:51 configuration_deepseek.py
      drwxr-xr-x 2 <user> dip 4.0K Mar 13 09:51 figures
      -rw-r--r-- 1 <user> dip  171 Mar 13 09:51 generation_config.json
      -rw-r--r-- 1 <user> dip 1.1K Mar 13 09:51 LICENSE
      -rw-r--r-- 1 <user> dip 4.9G Mar 13 10:02 model-00001-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00002-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00003-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00004-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00005-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00006-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00007-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00008-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00009-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00010-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00011-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 1.3G Mar 13 09:54 model-00012-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00013-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00014-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00015-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00016-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00017-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00018-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00019-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:01 model-00020-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:03 model-00021-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00022-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00023-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00024-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00025-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00026-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00027-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00028-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00029-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00030-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00031-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00032-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00033-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 1.7G Mar 13 10:04 model-00034-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00035-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00036-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00037-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:08 model-00038-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:09 model-00039-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:10 model-00040-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:11 model-00041-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:11 model-00042-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00043-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00044-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00045-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00046-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00047-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00048-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00049-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00050-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00051-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00052-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00053-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00054-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00055-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 1.7G Mar 13 10:12 model-00056-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00057-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00058-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:17 model-00059-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:18 model-00060-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:19 model-00061-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:20 model-00062-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:20 model-00063-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00064-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00065-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00066-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00067-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00068-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00069-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00070-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00071-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00072-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00073-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00074-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00075-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00076-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00077-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 1.7G Mar 13 10:20 model-00078-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:24 model-00079-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:25 model-00080-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:25 model-00081-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:26 model-00082-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:26 model-00083-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:26 model-00084-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00085-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00086-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00087-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00088-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00089-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00090-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00091-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00092-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:28 model-00093-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00094-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00095-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00096-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00097-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00098-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00099-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 1.7G Mar 13 10:31 model-00100-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00101-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00102-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00103-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00104-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00105-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00106-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00107-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00108-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00109-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00110-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00111-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00112-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:35 model-00113-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:38 model-00114-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00115-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00116-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00117-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00118-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00119-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00120-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00121-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 1.7G Mar 13 10:38 model-00122-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00123-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00124-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00125-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00126-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00127-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00128-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00129-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00130-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00131-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00132-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:42 model-00133-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:45 model-00134-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:46 model-00135-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00136-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00137-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00138-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00139-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00140-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 3.0G Mar 13 10:48 model-00141-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00142-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00143-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00144-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00145-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00146-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00147-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00148-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00149-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00150-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00151-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00152-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:50 model-00153-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:52 model-00154-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:52 model-00155-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:53 model-00156-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:53 model-00157-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:53 model-00158-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:53 model-00159-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.9G Mar 13 10:54 model-00160-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:53 model-00161-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 4.1G Mar 13 10:53 model-00162-of-000163.safetensors
      -rw-r--r-- 1 <user> dip 6.2G Mar 13 10:54 model-00163-of-000163.safetensors
      -rw-r--r-- 1 <user> dip  74K Mar 13 10:50 modeling_deepseek.py
      -rw-r--r-- 1 <user> dip 8.5M Mar 13 10:50 model.safetensors.index.json
      -rw-r--r-- 1 <user> dip  19K Mar 13 09:51 README.md
      -rw-r--r-- 1 <user> dip 3.5K Mar 13 10:50 tokenizer_config.json
      -rw-r--r-- 1 <user> dip 7.5M Mar 13 10:50 tokenizer.json

      ```

# Run Inference
1. Run launch.sh script       
      ```shell
      sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 2 launch.sh
      ```
      You should then see
      

      `Submitted batch job <job-id>`
      - This will launch the NIM Server and Benchmarking containers. Once the server is ready, the benchmarking container will run a sweep of GenAI-Perf commands
      - Depending on your Slurm system you may require changes in your sbatch parameters
      - You can track the status of your job-id with `squeue --job <job-id>`

      
2. Once your job starts, check progress of your script by viewing the server and benchmarking logs of your job
      - Logs for benchmarking `benchmarking_<job-id>.out`, and the NIM inference service `server_<job-id>.out` will appear in your `$STAGE_PATH/logs/`

      The server should show that its loading tensors:

      ```shell
      tail $STAGE_PATH/logs/server_<job-id>.out
      ```

      ```shell
      Loading safetensors checkpoint shards:   0% Completed | 0/163 [00:00<?, ?it/s]
      Loading safetensors checkpoint shards:   1% Completed | 1/163 [00:10<27:09, 10.06s/it]
      Loading safetensors checkpoint shards:   1% Completed | 2/163 [00:18<24:37,  9.17s/it]
      Loading safetensors checkpoint shards:   2% Completed | 3/163 [00:26<23:20,  8.75s/it]      
      ```
      - Note: This loading tensors step shown above can take up to 40 minutes, there is a sleep timer in run_benchmark.sh which will wait until the loading completes. 
      
      When the server is ready for benchmarking you will see these lines at the end of the server logs:
      ```
      INFO 02-28 15:38:31.420 server.py:82] Started server process [4179797]
      INFO 02-28 15:38:31.421 on.py:48] Waiting for application startup.
      INFO 02-28 15:38:31.422 on.py:62] Application startup complete.
      INFO 02-28 15:38:31.424 server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
      ```

      Once model weights have been loaded, the GenAI-Perf commands will start appearing in the benchmarking logs

      ```shell
      tail $STAGE_PATH/logs/benchmarking_<job-id>.out
      ```

      ```bash
      Concurrency: 1
      Use Case: chat
      ISL: 128
      OSL: 128
      ----------------
      2025-03-29 21:06 [INFO] genai_perf.parser:1078 - Detected passthrough args: ['-v', '--max-threads=1', '--request-count', '20']
      2025-03-29 21:06 [INFO] genai_perf.parser:115 - Profiling these models: deepseek-ai/deepseek-r1
      2025-03-29 21:06 [INFO] genai_perf.subcommand.common:208 - Running Perf Analyzer : 'perf_analyzer -m deepseek-ai/deepseek-r1 --async --input-data /lustre/project/results/deepseek-r1_16_1_chat_128_128_1743305976/inputs.json -i http --concurrency-range 1 --endpoint v1/chat/completions --service-kind openai -u 10.52.48.35:8000 --request-count 0 --warmup-request-count 0 --profile-export-file /lustre/project/results/deepseek-r1_16_1_chat_128_128_1743305976/profile_export.json --measurement-interval 10000 --stability-percentage 999 -v --max-threads=1 --request-count 20'
      Concurrency: 1
      ```
      A table will be printed for each GenAI-Perf command:
      ```bash
      # example output format
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
      

3. When the job finishes check your results
      - You should see 4 folders for each of the concurrency levels we test by default
      ```bash
      ls $STAGE_PATH/results
      ...
      # <model>_<num-GPUs>_<concurrency>_<use-case>_<ISL>_<OSL>_<job-id>_<time-stamp>
      deepseek-r1_16_1_chat_128_128_<job-id>_<time-stamp>
      deepseek-r1_16_5_chat_128_128_<job-id>_<time-stamp>
      deepseek-r1_16_10_chat_128_128_<job-id>_<time-stamp>
      deepseek-r1_16_25_chat_128_128_<job-id>_<time-stamp>
      ```
      See results from each test in the `profile_export_genai_perf.csv` or `profile_export_genai_perf.json` 
      ```bash
      cat deepseek-r1_16_1_chat_128_128_<job-id>_<time-stamp>/profile_export_genai_perf.csv
      ```
      Example CSV format:
      ```bash
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

# FAQ & Troubleshooting
      
## Benchmark job hangs

#### Problem
Model weights loading doesn't complete and job stops abruptly before benchmark can run. No results are generated.

#### Symptoms
No error messages are generated, only partial progress reported in the log files. 

#### Resolution
In run-benchmark.sh we start the Gen-AI perf run and wait for the weights to load. The tensor load times might vary from cluster to cluster. On some clusters we observed per tensor load time as high as 12-13 seconds. Therefore to load 163 tensors we need 2100 seconds. Hence, the solution would be to increase sleep time to 2400 seconds as below: 

E.g., to apply higher wait time use `SERVER_SLEEP_TIME` variable when launching benchmark:

`sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 2 --export=SERVER_SLEEP_TIME=2400 launch.sh`


## Server fails to load

#### Problem
Server loading failed due to errors

#### Symptoms
`server_0_<jobid>.out` file contains error message as below

```
RuntimeError: CUDA error: invalid device ordinal
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

#### Resolution

These usually mean that one of the GPUs is hanging. Possible resolutions: 
  * re-running the job on a different set of nodes
  * rebooting affected nodes.

## Download model weights fails

#### Problem
```shell
ngc registry model download-version nim/deepseek-ai/deepseek-r1-instruct:hf-5dde110-nim-fp8 --dest $STAGE_PATH

...


./ngc registry model download-version nim/deepseek-ai/deepseek-r1-instruct:hf-5dde110-nim-fp8 --dest /mnt/localdisk/mlperf/ds/ngc-cli
⠸ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ • 0/0 bytes • Remaining: -:--:-- • ? • Elapsed: 0:00:00 •
Client Error: 404 Response: 404 NOT_FOUND, ProblemDetail[type='urn:kaizen:problem-details:artifact-not-found', title='Not Found', status=404, detail='Model not found', instance='null', properties='null'] - Request Id: 1bae37d7-7818-41a8-9689-21103820908b Url: https://api.ngc.nvidia.com/v2/models/nim/deepseek-ai/deepseek-r1-instruct?resolve-labels=false
```

#### Symptoms
```shell
ngc registry model list *deepseek-r1-instruct*

# Empty Table --> NOT WORKING

+-----------------+-----------------+----------------+-------------+-------------+-----------+---------------+------------+-------------+------------+------------+
| Name            | Repository      | Latest Version | Application | Framework  | Precision | Last Modified | Permission | Access Type | Associated | Has Signed |
|                 |                 |                |             |            |           |               |            |             | Products   | Version    |
+-----------------+-----------------+----------------+-------------+-------------+-----------+---------------+------------+-------------+------------+------------+
|                 |                 |                |             |            |           |               |            |             |            |            |
+-----------------+-----------------+----------------+-------------+-------------+-----------+---------------+------------+-------------+------------+------------+

```

#### Resolution
Configure NGC with a valid API Key

```shell
ngc config set
```
Follow the CLI prompts:
```shell
Enter API key [****]. Choices: [<VALID_APIKEY>, 'no-apikey']: 
```
You should now be able to see values in the table:
```shell
ngc registry model list *deepseek-r1-instruct*

+-------------+--------------+--------------+-------------+-----------+-----------+--------------+------------+-------------+------------+------------+
| Name        | Repository   | Latest       | Application | Framework | Precision | Last         | Permission | Access Type | Associated | Has Signed |
|             |              | Version      |             |           |           | Modified     |            |             | Products   | Version    |
+-------------+--------------+--------------+-------------+-----------+-----------+--------------+------------+-------------+------------+------------+
| DeepSeek-R1 | nvstaging/ni | nvfp4_allmoe | Other       | Other     | N/A       | Mar 13, 2025 | unlocked   |             |            | False      |
|             | m/deepseek-  |              |             |           |           |              |            |             |            |            |
|             | r1-instruct  |              |             |           |           |              |            |             |            |            |
+-------------+--------------+--------------+-------------+-----------+-----------+--------------+------------+-------------+------------+------------+
```
