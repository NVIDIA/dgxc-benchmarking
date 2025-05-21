# Overview

This recipe contains information and scripts to produce performance results for the Llama 3.1 training workload. The scripts help perform environment setup, dataset setup, and launch benchmark jobs.
This variant of the workload is best-suited for GPU clusters with

* At least 64 GPUs with at least 80 GB memory each. Training of this 70-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* H100 GPUs. This workload runs with BF16 or FP8, which are both supported by H100 GPUs.

# Expected Performance

Performance for Llama 3.1 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/results/$GSW_VERSION/${DTYPE}/70b/$JOB_TOTAL_GPUS` folder. 

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep train_step_timing results/*.out
Epoch 0: : 100%|██████████| 100/100 [20:15<00:00, reduced_train_loss=6.370, global_step=99.00, consumed_samples=12800.0, train_step_timing in s=11.20]
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 128 / 11.20 = 93623

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 93623 / 86400 = 123.62 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is 1979 TFLOPS and for H100 BF16 is 989 TFLOPS.

The model flops for Llama 3.1 70b for GBS=1 is 3.94E+15. Calculation shown [here](#notes).

E.g. Llama 3.1 70b FP8 on 64x H100 GPUs (GBS=128)
```shell
peak FLOPS for H100 = 1979 TFLOPS 
training step time = 11.20
model flops = 3.94E+15
MFU = 128 * 3.94E+15 / 11.20 / 64 / 1979E+12 = 35.55% 
```

| Llama 3.1 70b BF16 (TP=4, PP=4, CP=2, VP=5, MBS=1, GA=64) | Throughput on 64x H100 GPUs (GBS=128) | Throughput on 128x H100 GPUs (GBS=256) | Throughput on 256x H100 GPUs (GBS=512) | Throughput on 512x H100 GPUs (GBS=1024) | Throughput on 1024x H100 GPUs (GBS=2048) | Throughput on 2048x H100 GPUs (GBS=4096)
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 14.61  | 14.70  | 14.72   | 14.80  | 14.86   | 14.84
| Throughput in tokens per second       | 71756  | 142663 | 284881  | 566797 | 1129246 | 2260623
| Model flops utilization               | 54.52% | 54.20% | 54.12%  | 53.84% | 53.63%  | 53.68%
| Time to train 1T tokens in days       | 161.26 | 81.13  | 40.62   | 20.42  | 10.25   | 5.12

| Llama 3.1 70b FP8 (TP=4, PP=4, CP=2, VP=5, MBS=1, GA=64) | Throughput on 64x H100 GPUs (GBS=128) | Throughput on 128x H100 GPUs (GBS=256) | Throughput on 256x H100 GPUs (GBS=512) | Throughput on 512x H100 GPUs (GBS=1024) | Throughput on 1024x H100 GPUs (GBS=2048) | Throughput on 2048x H100 GPUs (GBS=4096)
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step)   | 10.46  | 10.52  | 10.75  | 10.76  | 10.83   | 10.99
| Throughput in tokens per second         | 100275 | 199292 | 390167 | 779393 | 1549142 | 3053178
| Model flops utilization                 | 38.08% | 37.84% | 37.04% | 37.00% | 36.77%  | 36.23%
| Time to train 1T tokens in days         | 115.46 | 58.06  | 29.66  | 14.85  | 7.47    | 3.79

# Prerequisites

This recipe requires access to Llama 3.1. Instructions are below if needed.

# Request Access
A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens) in order to run the training script. Add the generated token to your environment via ```export HF_TOKEN=<your token>```.

Access to Llama 3.1 must be requested through [Meta's website](https://llama.meta.com/llama-downloads/) then requested on the [HuggingFace Llama](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B) page. The approval process is not automatic and could take a day or more.

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
Create a staging area by running the attached setup.sh. The script converts the docker image from `nvcr.io/nvidia/nemo:24.12` to the `nvidia+nemo+24.12.sqsh` file under the $STAGE_PATH folder and copies NeMo Launcher code from the container. The setup script also downloads Llama3 tokenizer related files from HuggingFace [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) repo using `HF_TOKEN` obtained in the previous step.

**Note:** Llama3.1 8B and 70B use the same Llama3 tokenizer.

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
Pre-training Llama3.1 requires a text-based dataset to be downloaded and pre-processed for the NeMo Framework to ingest the data optimally. [The Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted) is often used as the dataset for pre-training models. The NeMo Framework contains helper scripts to download and pre-process the dataset. The following steps outline how to download and pre-process the dataset on DGX Cloud with an explanation of key points after.

Make sure `$STAGE_PATH/llama3.1-dataset/llama` contains tokenizer files downloaded from previous step.

Submit the `generate_dataset.sh` script. The script launches several Slurm jobs that will download the dataset from The Pile, pre-process it and save it in a form suitable for subsequent training. The resulting dataset files will be saved under the `$STAGE_PATH/llama3.1-dataset` folder. The dataset creation may use up to 100GB. Make sure you have sufficient disk space available.

**Important:** You only need to run this step once. The same dataset can be used for Llama3.1 8b and 70b.

```shell
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./generate_dataset.sh
```

If the dataset generation step was successful there should be 2 idx and 2 bin files in the $STAGE_PATH/llama3.1-dataset folder.

```shell
my-llama_00_text_document.bin
my-llama_00_text_document.idx
my-llama_01_text_document.bin
my-llama_01_text_document.idx
```

If that is not the case, check the log files in: `$STAGE_PATH/results.data_preparation`


# Run Training

NeMo Launcher is using the Hydra framework to process command line arguments and pass them down as hyperparameters to a multi-node job performing the training.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `$STAGE_PATH/results/$GSW_VERSION/${DTYPE}/70b/$JOB_TOTAL_GPUS` folder.

Below is a command template for launching Llama 3.1 70b model training.
```shell
DTYPE=<fp8/bf16> MODEL_SIZE=70b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```

Where:
- `DTYPE` and `MODEL_SIZE` are **required** environment variables.
  - `DTYPE` can be either `fp8` or `bf16`.
  - `MODEL_SIZE` should be `70b` in this case.
- `NUM_NODES` can be calculate by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 128 GPUs scale, `NUM_NODES` should be `128 / 8 = 16`.
- [Slurm Settings](#slurm) for more information on Slurm parameters.

It is important to maintain these values for model parallelism settings in order to accurately assess performance results for completed jobs against expected baseline:
* `training.model.tensor_model_parallel_size=4`
* `training.model.pipeline_model_parallel_size=4`
* `training.model.virtual_pipeline_model_parallel_size=5`
* `training.model.context_parallel_size=2`

Global batch size ( training.model.global_batch_size) value should be set to ```<number of nodes> * 16. E.g., 16 * 16 = 256 (in the example above)```.

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
ENABLE_PROFILE=true DTYPE=<fp8/bf16> MODEL_SIZE=70b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
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
ENABLE_PROFILE=true RUN_CONF_PROFILE_GPU_METRICS=true RUN_CONF_PROFILE_RANKS="0" DTYPE=<fp8/bf16> MODEL_SIZE=70b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
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
ENABLE_NCCL_TRACE=true DTYPE=<fp8/bf16> MODEL_SIZE=70b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```

# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = 12 * (number of layers) * (hidden size)^2 * (1 + (number of query groups)/(number of attention heads) + (sequence length)/(hidden size))
    mlp flops = 18 * (number of layers) * (FFN size) * (hidden size)
    embedding flops = 6 * (vocab size) * (hidden size)

Llama 3.1 70b calculation: 
    sequence length = 8192
    attention flops = 12 * 80 * 8192^2 * (1 + 8/64 + 8192/8192) = 136,902,082,560
    mlp flops = 18 * 80 * 28672 * 8192 = 338,228,674,560
    embedding flops = 6 * 128256 * 8192 = 6,304,038,912

    model flops = 8192 * (136,902,082,560 + 338,228,674,560 + 6,304,038,912) = 3.94E+15
```

