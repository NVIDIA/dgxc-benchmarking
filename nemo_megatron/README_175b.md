# Overview

This recipe contains information and scripts to produce performance results for the Nemo Megatron GPT3 175B training workload. The scripts help perform environment setup and launch benchmark jobs.

This variant of the workload is best-suited for GPU clusters with:
* At least 128 GPUs with at least 80 GB memory each. Training of this 175-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* This workload runs with BF16 or FP8 precision. 


# Expected Performance

Once the training job has finished successfully it's performance measurement metric is it's training throughput which is based on time it took to complete each training step.

The example below is taken from the end of the output log file - see `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/175b/$JOB_TOTAL_GPUS/*.out`, where JOB_TOTAL_GPUS=128, the training step time was measured as **5.9** seconds during step number 50.
```shell
grep train_step_timing results/$GSW_VERSION/fp8/175b/128/*.out
Epoch 0: : 100%|██████████| 50/50 [31:51<00:00, reduced_train_loss=6.130, global_step=50.0, consumed_samples=76800.0, train_step_timing in s=5.900, val_loss=6.250]
```

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

 To obtain throughput as a tokens per second measurement, follow this formula: 
 ```shell
 (sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
 ```

 E.g. 2048 * 256 / 5.90 = 88862.37

To calculate time to train estimate:
```shell
 (total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
 E.g. 1e12 / 88862.37 / 86400 = 130.25 days

 
To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is 1979 TFLOPS and for H100 BF16 is 989 TFLOPS.

The model flops for GPT 3 175b for GBS=1 is 2.20E+15. Calculation shown [here](#notes).

E.g. GPT 3 175b FP8 on 128x H100 GPUs (GBS=256)
```shell
peak FLOPS for H100 = 1979 TFLOPS
training step time = 5.90 s
model flops = 2.20E+15

MFU = 256 * 2.20E+15 / 5.90 / 128 / 1979E+12 = 37.68%
```

| NeMo Megatron BF16 (TP=4,PP=8, MBS=1, VP=12, SEQ=2048) | 128x H100 GPUs (GBS=256) | 256x H100 GPUs (GBS=512) | 512x H100 GPUs (GBS=1024) | 1024x H100 GPUs (GBS=2048) | 2048x H100 GPUs (GBS=4096)
|---|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 8.309 | 8.408 | 8.42 | 8.453 | 8.49
| Throughput in tokens per second | 63099 | 124712 | 249068 | 496191 | 988057 
| Model flops utilization | 53.54% | 52.91% | 52.84% | 52.63% | 52.40% 
| Time to train 1T tokens in days | 183 | 93 | 46 | 23 | 12

| NeMo Megatron FP8 (TP=4,PP=8, MBS=1, VP=12, SEQ=2048) | 128x H100 GPUs (GBS=256) | 256x H100 GPUs (GBS=512) | 512x H100 GPUs (GBS=1024) | 1024x H100 GPUs (GBS=2048) | 2048x H100 GPUs (GBS=4096)
|---|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 5.626 | 5.725 | 5.774 | 5.82 | 5.96
| Throughput in tokens per second | 93190 | 183157 | 363206 | 720671 | 1407485
| Model flops utilization | 39.52% | 38.84% | 38.51% | 38.20% | 37.30%
| Time to train 1T tokens in days | 124 | 63 | 32 | 16 | 8


# Request Access

No special pre-requisites or access required for this recipe.


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
Create a staging area by running the setup.sh script. The script saves the container image from the registry in the $STAGE_PATH folder and copies the NeMo Launcher code from the container to the staging directory. 

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)

# Run the setup
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
```
Check the corresponding `slurm-<job_id>.out` file for status information.

**Important:** `STAGE_PATH` used in this step must be used when running the workload.

# Prepare Dataset
Pre-training a GPT-3 model requires a text-based dataset to be downloaded and pre-processed for the NeMo Framework to ingest the data optimally. [The Pile](https://pile.eleuther.ai/) is often used as the dataset for pre-training models. The NeMo Framework contains helper scripts to download and pre-process the dataset. The following steps outline how to download and pre-process the dataset on DGX Cloud with an explanation of key points after.

Submit the generate_dataset.sh script. The script launches several Slurm jobs that will download the dataset from The Pile, pre-process it and save it in a form suitable for subsequent training. The resulting dataset files will be saved under the $STAGE_PATH/gpt3-dataset folder. The dataset creation may use up to 250GB. Make sure you have sufficient disk space available.


```shell
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./generate_dataset.sh
```

If the dataset generation step was successful there should be 4 idx and 4 bin files in the $STAGE_PATH/gpt3-dataset folder.

```shell
my-gpt3_00_text_document.bin
my-gpt3_00_text_document.idx
my-gpt3_01_text_document.bin
my-gpt3_01_text_document.idx
my-gpt3_02_text_document.bin
my-gpt3_02_text_document.idx
my-gpt3_03_text_document.bin
my-gpt3_03_text_document.idx
```

You can run check_dataset.sh script to perform initial checks of the generated dataset state:
```shell
bash ./check_dataset.sh
Dataset generation completed successfully at: <$STAGE_PATH/gpt3-dataset>
```

If that is not the case, check the log files under: `$STAGE_PATH/results.data_preparation`


# Run Training
Once the environment has been prepared, it is time to train a model. The NeMo Framework contains many predefined configuration files for various models including the 175 billion parameter GPT-3 model. This section will demonstrate how to initiate training on the model.

NeMo uses the Hydra framework to process command line arguments and the base config in the gpt3_175b_hydra.yaml file and passes them down as hyper parameters to a multi-node job performing the training.

Run the launch.sh script to start NeMo Megatron 175b model training. Minimum required number of nodes is 16 (or 128 GPUs). The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/175b/$JOB_TOTAL_GPUS` folder.

Below is a command template for launching NeMo Megatron model training.
```shell
DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```
Where:
- `DTYPE` is a **required** environment variable.
    - `DTYPE` can be either `fp8` or `bf16`.
- `NUM_NODES` can be calculate by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 256 GPUs scale, `NUM_NODES` should be `256 / 8 = 32`.
- [Slurm Settings](#slurm) for more information on Slurm parameters.

It is important to maintain these values for model parallelism settings in order to accurately assess performance results for completed jobs against expected baseline, which can be seen in the gpt3_175b_hydra.yaml:
* training.model.tensor_model_parallel_size=4
* training.model.pipeline_model_parallel_size=8
* training.model.micro_batch_size=1 
* training.model.virtual_pipeline_model_parallel_size=12

Global batch size ( training.model.global_batch_size) value should be set to ```<number of nodes> * 16. E.g., 16 * 16 = 256 (in the example above).```

# Profiling
We have two profiling methods supported: Nsight, and NCCL Trace.

Due to overhead while profiling: the results generated with these settings is not valid for comparison. 'Performance' and 'Profiling' runs should be done separately.

**Note:** Profiling and NCCL Trace are currently mutually exclusive.

## Run Nsight Profiling

Nsight Systems is included in our containers. To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Default Profiling Settings:
* **MPI Ranks:** 0-8
* **Job Steps:** 20-30
* **Output Location:** .nsys-rep files are saved in the nsys folder within the existing results directory.
* **Filename format:** `${MODEL}-${MODEL_SIZE}-${DTYPE}_${NUM_GPUS}g_${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURM_LOCALID}.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
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
	  Default: "0,1,2,3,4,5,6,7"
* Enable GPU device metrics capture:
	* `RUN_CONF_PROFILE_GPU_METRICS`: boolean, set to 'true' to capture device metrics.
	- Default: false
	- **Note:** Additional system configuration is required for GPU device metrics to work.
* Enable CPU metrics capture:
	* `RUN_CONF_PROFILE_CPU`: boolean, set to 'true' to capture CPU metrics.
	- Default: false

**Example customized profiling command:**
```shell
ENABLE_PROFILE=true RUN_CONF_PROFILE_GPU_METRICS=true RUN_CONF_PROFILE_RANKS="0" DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
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
ENABLE_NCCL_TRACE=true DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```

# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = (24 * (number of layers) * (hidden size)^2) + (12 * (number of layers) * (hidden size) * (sequence length))
    mlp flops = 48 * (number of layers) * (hidden size)^2
    embedding flops = 6 * (vocab size) * (hidden size)

GPT 3 175b calculation:
    sequence length = 2048
    attention flops = 24 * 96 * 12288^2 + 12 * 96 * 12288 * 2048 = 376,883,380,224
    mlp flops = 48 * 96 * 12288^2 = 695,784,701,952
    embedding flops = 6 * 51200 * 12288 = 3,774,873,600

    model flops = 2048 * (376,883,380,224 + 695,784,701,952 + 3,774,873,600) = 2.20E+15
```
