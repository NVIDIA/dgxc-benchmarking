# Overview

This recipe contains information and scripts to produce performance results for the Llama 3.1 training workload. The scripts help perform environment setup, dataset setup, and launch benchmark jobs.
This variant of the workload is best-suited for GPU clusters with

* At least 576 GPUs with at least 80 GB memory each. Training of this 405-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
  * 32 GPUs with at least 80GB memory is the minimum when running proxy configs: <576 GPUs.
* H100 GPUs. This workload runs with BF16 or FP8, which are both supported by H100 GPUs.

# Expected Performance

Performance for Llama 3.1 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/405b/$JOB_TOTAL_GPUS` folder.

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep train_step_timing results/*.out
Epoch 0: : 100%|██████████| 50/50 [16:26<00:00, v_num=gjbq, reduced_train_loss=11.70, global_step=49.00, consumed_samples=12600.0, train_step_timing in s=12.80]
```

To obtain throughput as a tokens per second measurement, follow this formula:
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 252 / 12.80 = 161280

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days)
```
E.g. 1e12 / 161280 / 86400 = 71.63 days


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is 1979 TFLOPS and for H100 BF16 is 989 TFLOPS.

The model flops for Llama 3.1 405b for GBS=1 is 2.17E+16. Calculation shown [here](#notes).

E.g. Llama 3.1 405b FP8 on 576x H100 GPUs (GBS=252)
```shell
peak FP8 FLOPS for H100 = 1979 TFLOPS
training step time = 11.24
model flops = 2.17E+16
MFU = 252 * 2.17E+16 / 11.24 / 576 / 1979E+12 = 42.71%
```

| Llama 3.1 405b BF16 (TP=8, PP=9, CP=2, VP=7, MBS=1, GA=63) | Throughput on 32x H100 GPUs  | Throughput on 96x H100 GPUs  | Throughput on 192x H100 GPUs | Throughput on 576x H100 GPUs | Throughput on 1152x H100 GPUs | Throughput on 2304x H100 GPUs |
|---|---|---|---|---|---|---|
| Layers                                 | 7      | 21     | 42     | 126    | 126    | 126    | 
| GBS                                    | 126    | 126    | 252    | 252    | 504    | 1008   | 
| PP                                     | 1      | 3      | 3      | 9      | 9      | 9      | 
| VP                                     | n/a    | 7      | 7      | 7      | 7      | 7      | 
| Training step time (seconds per step)  | 9.00   | 8.87   | 17.30  | 17.50  | 17.60  | 17.70  | 
| Throughput in tokens per second        | 114637 | 116342 | 119328 | 117964 | 234589 | 466527 | 
| Model flops utilization                | 57.65% | 55.41% | 56.03% | 54.85% | 54.54% | 54.23% | 
| Time to train 1T tokens in days        | n/a    | n/a    | n/a    | 98.11  | 49.34  | 24.81  | 

| Llama 3.1 405b FP8 (TP=8, PP=9, CP=2, VP=7, MBS=1, GA=63) | Throughput on 32x H100 GPUs  | Throughput on 96x H100 GPUs  | Throughput on 192x H100 GPUs | Throughput on 576x H100 GPUs | Throughput on 1152x H100 GPUs | Throughput on 2304x H100 GPUs |
|---|---|---|---|---|---|---|
| Layers                                 | 7      | 21     | 42     | 126    | 126    | 126    | 
| GBS                                    | 126    | 126    | 252    | 252    | 504    | 1008   | 
| PP                                     | 1      | 3      | 3      | 9      | 9      | 9      | 
| VP                                     | n/a    | 7      | 7      | 7      | 7      | 7      | 
| Training step time (seconds per step)  | 5.75   | 5.63   | 10.8   | 11.00  | 11.10  | 11.20  | 
| Throughput in tokens per second        | 179511 | 183337 | 191146 | 187671 | 371961 | 737280 | 
| Model flops utilization                | 45.12% | 43.65% | 44.87% | 43.61% | 43.22% | 42.83% | 
| Time to train 1T tokens in days        | n/a    | n/a    | n/a    | 61.67  | 31.12  | 15.70  | 

For proxy configs (<576 GPUs scales) we don't provide time to train estimates to avoid misleading conclusions. Proxy configs are not realistic and were created to allow fit of Llama model to smaller number of GPUs than intended.

# Prerequisites

This recipe requires access to Llama 3.1. Instructions are below if needed.

# Request Access
A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens) in order to run the training script. Add the generated token to your environment via `export HF_TOKEN=<your token>`.

Access to Llama 3.1 must be requested through [Meta's website](https://llama.meta.com/llama-downloads/) then requested on the [HuggingFace Llama](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B) page. The approval process is not automatic and could take a day or more.

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
Create a staging area by running the attached setup.sh. The script converts the docker image from `nvcr.io/nvidia/nemo:24.12` to the `nvidia+nemo+24.12.sqsh` file under the `$STAGE_PATH` folder and copies NeMo Launcher code from the container. 

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)
# Set HuggingFace token
export HF_TOKEN=<your token>

# Run the setup
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
```
Check the corresponding `slurm-<job_id>.out` file for status information.

**Notes:**
- Slurm parameters might not be applicable to all environments. Please consult with your system administrator and update or remove parameters as needed.
- Setup script expects your `$PYTHONUSERBASE/bin` folder (typically ~/.local/bin) to be in your PATH. You can check this with `python3 -m site --user-base`. If you encounter error messages such as:
    ```shell
    WARNING: The script huggingface-cli is installed in '$PYTHONUSERBASE/bin' which is not on PATH.
    Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
    ..snip..
    /cm/local/apps/slurm/var/spool/job1490931/slurm_script: line xx: huggingface-cli: command not found
    ```
    Please add `$PYTHONUSERBASE/bin` to your PATH, and retry.

**Important:** `STAGE_PATH` used in this step must be used when running the workload.

# Prepare Dataset
Llama 3.1 405B uses synthetic data for training. A dataset does not need to be prepared. Note that Llama3.1 405B uses the GPT2BPETokenizer as a proxy.

# Run Training

NeMo Launcher is using the Hydra framework to process command line arguments and pass them down as hyperparameters to a multi-node job performing the training.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/405b/$JOB_TOTAL_GPUS` folder.

Below is a command template for launching Llama 3.1 405b model training.
```shell
DTYPE=<fp8/bf16> MODEL_SIZE=405b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```
Where:
- `DTYPE` and `MODEL_SIZE` are **required** environment variables.
  - `DTYPE` can be either `fp8` or `bf16`.
  - `MODEL_SIZE` should be `405b` in this case.
- `NUM_NODES` can be calculate by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 576 GPUs scale, `NUM_NODES` should be `576 / 8 = 72`.
- [Slurm Settings](#slurm) for more information on Slurm parameters.

The following applies only to the full model scales: 576, 1152, 2304 GPUs. Configurations and Global batch size changes for proxy configs <576 GPUs.
>>>
It is important to maintain these values for model parallelism settings in order to accurately assess performance results for completed jobs against expected baseline for the non-proxy 405b configurations:
* `training.model.tensor_model_parallel_size=8`
* `training.model.pipeline_model_parallel_size=9`
* `training.model.virtual_pipeline_model_parallel_size=7`
* `training.model.context_parallel_size=2`

Global batch size (`training.model.global_batch_size`) value should scale with total number GPUs. The starting global batch size for 576 GPUs is 252, therefore it should set to `<number of total gpus> * 252 / 576`.
>>>

# Profiling
We have two profiling methods supported: Nsight, and NCCL Trace.

Due to overhead while profiling: the results generated with these settings is not valid for comparison. 'Performance' and 'Profiling' runs should be done separately.

**Note:** Profiling and NCCL Trace are currently mutually exclusive.

## Run Nsight Profiling

Nsight Systems is included in our containers. To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Default Profiling Settings:
* **MPI Ranks:** 0-15
* **Job Steps:** 20-30
* **Output Location:** .nsys-rep files are saved in the nsys folder within the existing results directory.
* **Filename format:** `${MODEL}-${MODEL_SIZE}-${DTYPE}_${NUM_GPUS}g_${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURM_LOCALID}.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true DTYPE=<fp8/bf16> MODEL_SIZE=405b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```
### Customizing profiling behavior:
* Specify job steps to profile:
  * `RUN_CONF_PROFILE_START_STEP`: start profiling on this job step.
  Default: 20
  * `RUN_CONF_PROFILE_STOP_STEP`: stop profiling on this job step.
  Default: 30
* Select MPI ranks to profile:
  * `RUN_CONF_PROFILE_RANKS`: Comma-separated list of MPI ranks to profile.
    Example: "0,1,2,3"
    Default: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
* Enable GPU device metrics capture:
  * `RUN_CONF_PROFILE_GPU_METRICS`: boolean, set to 'true' to capture device metrics.
  - Default: false
  - **Note:** Additional system configuration is required for GPU device metrics to work.
* Enable CPU metrics capture:
  * `RUN_CONF_PROFILE_CPU`: boolean, set to 'true' to capture CPU metrics.
  - Default: false

**Example customized profiling command:**
```shell
ENABLE_PROFILE=true RUN_CONF_PROFILE_GPU_METRICS=true RUN_CONF_PROFILE_RANKS="0" DTYPE=<fp8/bf16> MODEL_SIZE=405b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```

### Troubleshooting:

If you encounter issues, try the defaults `ENABLE_PROFILE=true` first as these should be broadly applicable to most systems.

### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

## Run NCCL Trace

NCCL traces provide a breakdown of the communication pattern outlining both the type of NCCL calls being made and their message sizes.

To collect NCCL Trace information, set variable `ENABLE_NCCL_TRACE=true` when submitting your job.

**Defaults:**
* File Size: NCCL Trace generates large files, therefore profiling is limited to the first 10 steps.
* Output Location: Trace files are saved to a separate directory with nccl-trace appended to the version string.
* Output Directory: `$STAGE_PATH/results/$GSW_VERSION-nccl-trace/$DTYPE/${MODEL_SIZE}/$JOB_TOTAL_GPUS`

**Example command:**
```shell
ENABLE_NCCL_TRACE=true DTYPE=<fp8/bf16> MODEL_SIZE=405b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```

# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = 12 * (number of layers) * (hidden size)^2 * (1 + (number of query groups)/(number of attention heads) + (sequence length)/(hidden size))
    mlp flops = 18 * (number of layers) * (FFN size) * (hidden size)
    embedding flops = 6 * (vocab size) * (hidden size)

Llama 3.1 405b calculation:
    sequence length = 8192
    attention flops = 12 * 126 * 16384^2 * (1 + 16/128 + 8192/16384) = 659,545,915,392
    mlp flops = 18 * 126 * 53248 * 16384 = 1,978,637,746,176
    embedding flops = 6 * 128256 * 16384 = 12,608,077,824

    model flops = 8192 * (659,545,915,392 + 1,978,637,746,176 + 12,608,077,824) = 2.17E16
```
