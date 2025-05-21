# Overview

This recipe contains information and scripts to produce performance results for the Maxtext Llama3 70B training workload. The scripts help perform environment setup and launch benchmark jobs.

This variant of the workload is best-suited for GPU clusters with:
* At least 128 GPUs with at least 80 GB memory each. Training of this 70-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* This workload runs with BF16 or FP8 precision. FP8 is only supported by H100 GPUs. BF16 recipes are suitable for both A100 and H100 GPUs.

| Size | Precision | GPUs | SeqLen | Layers | dcn_TP | dcn_FSDP | dcn_DP | ici_TP | ici_FSDP | ici_DP | per_device_batch_size | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:------:|:--------:|:------:|:------:|:--------:|:------:|:---------------------:|:----:|:---:|
| 70b  | BF16      | 128  | 8192   | 80     | 1   | 16  | 1   | 1   | 8   | 1   | 2   | 256  | 1 |
| 70b  | BF16      | 256  | 8192   | 80     | 1   | 32  | 1   | 1   | 8   | 1   | 2   | 512  | 1 |
| 70b  | BF16      | 512  | 8192   | 80     | 1   | 16  | 4   | 1   | 8   | 1   | 2   | 1024 | 1 |
| 70b  | BF16      | 1024 | 8192   | 80     | 1   | 16  | 8   | 1   | 8   | 1   | 2   | 2048 | 1 |
| 70b  | BF16      | 2048 | 8192   | 80     | 1   | 16  | 16  | 1   | 8   | 1   | 2   | 4096 | 1 |
| 70b  | FP8       | 128  | 8192   | 80     | 1   | 16  | 1   | 1   | 8   | 1   | 2   | 256  | 1 |
| 70b  | FP8       | 256  | 8192   | 80     | 1   | 32  | 1   | 1   | 8   | 1   | 2   | 512  | 1 |
| 70b  | FP8       | 512  | 8192   | 80     | 1   | 16  | 4   | 1   | 8   | 1   | 2   | 1024 | 1 |
| 70b  | FP8       | 1024 | 8192   | 80     | 1   | 16  | 8   | 1   | 8   | 1   | 2   | 2048 | 1 |
| 70b  | FP8       | 2048 | 8192   | 80     | 1   | 16  | 16  | 1   | 8   | 1   | 2   | 4096 | 1 |

# Expected Performance

Performance for Maxtext Llama3 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/70b/$JOB_TOTAL_GPUS` folder. 

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep 'completed step:' results/$GSW_VERSION/bf16/70b/128/log-maxtext_llama3_70b_128_2102379.out | tail -1
completed step: 49, seconds: 15.570, TFLOP/s/device: 506.597, Tokens/s/device: 1052.266, total_weights: 2097152, loss: 0.000
```
To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 256 / 15.570 = 134691.84

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 134691.84 / 86400 = 85.93 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is 1979 TFLOPS and for H100 BF16 is 989 TFLOPS.

The model flops for Llama3 70b for GBS=1 is 3.94E+15. Calculation shown [here](#notes).

E.g. Llama3 70b BF16 on 128x H100 GPUs (GBS=256)
```shell
peak FLOPS for H100 = 989 TFLOPS 
training step time = 15.570 s 
model flops = 3.94E+15
MFU = 256 * 3.94E+15 / 15.570 / 128 / 989E+12 = 51.17% 
```


| Maxtext Llama3 70b BF16 | Throughput on 128x H100 GPUs | Throughput on 256x H100 GPUs | Throughput on 512x H100 GPUs | Throughput on 1024x H100 GPUs | Throughput on 2048x H100 GPUs | 
|---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 15.563 | 15.444 | 15.519 | 15.603 | 15.898 |
| Throughput in tokens per second | 134752.43 | 271581.46 | 540537.92 | 1075255.78 | 2110607.12 |
| Model flops utilization | 51.20% | 51.59% | 51.34% | 51.06% | 50.12% |
| Time to train 1T tokens in days | 85.89 | 42.62 | 21.41 | 10.76 | 5.48 |

| Maxtext Llama3 70b FP8 | Throughput on 128x H100 GPUs | Throughput on 256x H100 GPUs | Throughput on 512x H100 GPUs | Throughput on 1024x H100 GPUs | Throughput on 2048x H100 GPUs | 
|---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 9.074 | 9.488 | 9.244 | 9.483 | 9.339 |
| Throughput in tokens per second  | 231116.60 | 442064.08 | 907465.17 | 1769188.65 | 3592936.29 |
| Model flops utilization  | 43.88% | 41.97% | 43.07% | 41.99% | 42.64% |
| Time to train 1T tokens in days | 50.08 | 26.18 | 12.75 | 6.54 | 3.22 |

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
Create a staging area by running the attached setup.sh. The script converts the docker image from the registry to a sqsh file under the $STAGE_PATH folder.

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/maxtext)

# Run the setup
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
```
Check the corresponding `slurm-<job_id>.out` file for status information.

**Important:** `STAGE_PATH` used in this step must be used when running the workload.

# Prepare Dataset
Since Maxtext Llama training only uses synthetic datasets, this step is omitted.

# Run Slurm Training
Once the environment has been prepared, it is time to train a model. 

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/70b/$JOB_TOTAL_GPUS` folder.

Below is a slurm command template for launching Maxtext Llama3 70b model training.
```shell
DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```
Where:
- `DTYPE` is a **required** environment variable.
	- `DTYPE` can be either `fp8` or `bf16`.
- `NUM_NODES` can be calculated by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 256 GPUs scale, `NUM_NODES` should be `256 / 8 = 32`.
- [Slurm Settings](#slurm) for more information on Slurm parameters.

Global batch size ( training.model.global_batch_size) value should be set to `<number of nodes> * <gpus per node> * <mbs> . E.g., 16 * 8 * 2 = 256 (in the example above)` since we are using FSDP+DP.

# Profiling

Nsight profiling is used as part of this benchmark. See [Run Nsight Profiling](#run-nsight-profiling) for more info.

## Run Nsight Profiling

The workload uses [PGLE](See https://github.com/google/paxml?tab=readme-ov-file#run-pgle-workflow-on-gpu) and turns on Nsight profiling by default. The profiles are located at `$STAGE_PATH/results/$GSW_VERSION/${DTYPE}/70b/$JOB_TOTAL_GPUS/nsys`.

Each job generates two nsys-rep files:
-  PGLE profiling run at the start of the benchmark. 
-  Profile taken during the performace run. Use **this** profile when debugging performance issues.

Profiling for performance runs in MaxText is performed **only on Global Rank 0**. This is intentional, as MaxText enables profiling exclusively for Global Rank 0, as detailed in the [profiler source code](https://github.com/AI-Hypercomputer/maxtext/blob/63c0c278f882469feebd3926fca82a8a6ff853d2/MaxText/profiler.py#L48).

### Profiling Details
- **Ranks Profiled:** 0 (Global Rank 0 only)
- **Job Steps Profiled:** 10–15
- **Output Location:**
  PGLE profiling run files are saved in the `pgle` folder within the existing results directory.
  Performance run files are saved in the `nsys` folder within the existing results directory.
* **Filename format:** `${MODEL}-${MODEL_SIZE}-${DTYPE}_${JOB_TOTAL_GPUS}g_${SLURM_JOB_ID}_%q{SLURM_NODEID}_%q{SLURM_PROCID}.nsys-rep`

### Customizing profiling behavior:
* Specify job steps to profile:
	* `RUN_CONF_PROFILE_START_STEP`: start profiling on this job step. Default: 10
	* `RUN_CONF_PROFILE_STOP_STEP`: stop profiling on this job step. Default: 15
* Enable GPU device metrics capture:
	* `RUN_CONF_PROFILE_GPU_METRICS`: boolean, set to 'false' to capture device metrics.
	- Default: false
	- **Note:** Additional system configuration is required for GPU device metrics to work.

**Example customized profiling command:**
```shell
ENABLE_PROFILE=true RUN_CONF_PROFILE_GPU_METRICS=true DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```


### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (normalization flops) + (embedding flops))

model flops breakdown:
    attention flops = 12 * (number of layers) * (hidden size)^2 * (1 + (number of query groups)/(number of attention heads) + (sequence length)/(hidden size))
    mlp flops = 18 * (number of layers) * (FFN size) * (hidden size)
    embedding flops = 6 * (vocab size) * (hidden size)

Llama3 70b calculation: 
    sequence length = 8192
    attention flops = 12 * 80 * 8192^2 * (1 + 8/64 + 8192/8192) = 136,902,082,560
    mlp flops = 18 * 80 * 28672 * 8192 = 338,228,674,560
    embedding flops = 6 * 128256 * 8192 = 6,304,038,912

    model flops = 8192 * (136,902,082,560 + 338,228,674,560 + 6,304,038,912) = 3.94E+15
```
