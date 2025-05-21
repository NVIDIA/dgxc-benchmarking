# Overview

This recipe contains information and scripts to produce performance results for the Maxtext Llama2 70B training workload. The scripts help perform environment setup and launch benchmark jobs.

This variant of the workload is best-suited for GPU clusters with:
* At least 64 GPUs with at least 80 GB memory each. Training of this 70-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* This workload runs with BF16 or FP8 precision. FP8 is only supported by H100 GPUs. BF16 recipes are suitable for both A100 and H100 GPUs.

# Expected Slurm Performance

Performance for Maxtext Llama2 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/70b/$JOB_TOTAL_GPUS` folder. 

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep 'completed step:' results/$GSW_VERSION/bf16/70b/64/log-maxtext-64_70b_1407993.out | tail -1
completed step: 49, seconds: 7.908, TFLOP/s/device: 460.480, Tokens/s/device: 1035.970, total_weights: 524288, loss: 0.000
```
To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 4096 * 128 / 7.908 = 66298.43

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 66298.43 / 86400 = 174.57 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is 1979 TFLOPS and for H100 BF16 is 989 TFLOPS.

The model flops for Llama 2 70b for GBS=1 is 1.82E+15. Calculation shown [here](#notes).

E.g. Llama 2 70b BF16 on 64x H100 GPUs (GBS=128)
```shell
peak FLOPS for H100 = 989 TFLOPS 
training step time = 7.908 s 
model flops = 1.82E+15
MFU = 128 * 1.82E+15 / 7.908 / 64 / 989E+12 = 46.54% 
```


| Maxtext LLama2 70b BF16 (FSDP+DP MBS=2) | Throughput on 64x H100 GPUs (FSDP=64 GBS=128) | Throughput on 128x H100 GPUs (FSDP=128 GBS=256) | Throughput on 256x H100 GPUs (FSDP=256 GBS=512) | Throughput on 512x H100 GPUs (FSDP=128 GBS=1024) | Throughput on 1024x H100 GPUs (FSDP=128 GBS=2048) | Throughput on 2048x H100 GPUs (FSDP=128 GBS=4096) | 
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 6.999 | 7.072 | 7.153 | 7.091 | 7.084 | 7.066 |
| Throughput in tokens per second | 74908.99 | 148271.49 | 293184.96 | 591496.83 | 1184162.62 | 2374358.34 |
| Model flops utilization | 52.59% | 52.04% | 51.45% | 51.90% | 51.95% | 52.09% |
| Time to train 1T tokens in days | 154.51 | 78.06 | 39.48 | 19.57 | 9.77 | 4.87 |

| Maxtext LLama2 70b FP8 (FSDP+DP MBS=2) | Throughput on 64x H100 GPUs (FSDP=64 GBS=128) | Throughput on 128x H100 GPUs (FSDP=128 GBS=256) | Throughput on 256x H100 GPUs (FSDP=256 GBS=512) | Throughput on 512x H100 GPUs (FSDP=128 GBS=1024) | Throughput on 1024x H100 GPUs (FSDP=128 GBS=2048) | Throughput on 2048x H100 GPUs (FSDP=128 GBS=4096) | 
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 4.324 | 4.556 | 4.959 | 5.043 | 5.062 | 4.93 |
| Throughput in tokens per second  | 121250.69 | 230152.77 | 422898.16 | 831708.11 | 1657172.66 | 3403086.41 |
| Model flops utilization  | 42.54% | 40.37% | 37.09% | 36.47% | 36.34% | 37.31% |
| Time to train 1T tokens in days | 95.46 | 50.29 | 27.37 | 13.92 | 6.98 | 3.40 |


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
Create a staging area by running the attached setup.sh. The script converts the docker image from `ghcr.io/nvidia/jax:maxtext-2024-12-09` to the `nvidia+jax+maxtext-2024.12.09.sqsh` file under the $STAGE_PATH folder. 

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)

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

Below is a slurm command template for launching Maxtext Llama2 70b model training.
```shell
DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} $STAGE_PATH/launch.sh
```
Where:
- `DTYPE` is a **required** environment variable.
	- `DTYPE` can be either `fp8` or `bf16`.
- `NUM_NODES` can be calculated by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 256 GPUs scale, `NUM_NODES` should be `256 / 8 = 32`.
- [Slurm Settings](#slurm) for more information on Slurm parameters.

Global batch size ( training.model.global_batch_size) value should be set to `<number of nodes> * <gpus per node> * <mbs> . E.g., 8 * 8 * 2 = 128 (in the example above)` since we are using FSDP+DP.

# Profiling

Nsight profiling is used as part of this benchmark. See [Run Nsight Profiling](#run-nsight-profiling) for more info.

## Run Nsight Profiling

The workload uses [PGLE](See https://github.com/google/paxml?tab=readme-ov-file#run-pgle-workflow-on-gpu) and turns on Nsight profiling by default. The profiles are located at `$STAGE_PATH/results/$GSW_VERSION/${DTYPE}/70b/$JOB_TOTAL_GPUS/nsys`.

Each job generates two nsys-rep files:
- `*-prof.nsys-rep` is the PGLE profiling run at the start of the benchmark.
- `*-perf.nsys-rep` is the profile taken during the performace run. Use **this** profile when debugging performance issues.

Options to customize profiling behaviour are not currently available for this workload. GPU device metrics capture is not currently supported for this workload.

### Viewing results

[How to consume Nsight profiling results](../common/nsys-profile.md)

# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (normalization flops) + (embedding flops))

model flops breakdown:
    attention flops = 12 * (number of layers) * (hidden size)^2 * (1 + (number of query groups)/(number of attention heads) + (sequence length)/(hidden size))
    mlp flops = 18 * (number of layers) * (FFN size) * (hidden size)
    embedding flops = 6 * (vocab size) * (hidden size)

Llama 2 70b calculation: 
    sequence length = 4096
    attention flops = 12 * 80 * 8192^2 * (1 + 8/64 + 4096/8192) = 104,689,827,840
    mlp flops = 18 * 80 * 28672 * 8192 = 338,228,674,560
    embedding flops = 6 * 32000 * 8192 = 1,572,864,000

    model flops = 4096 * (104,689,827,840 + 338,228,674,560 + 1,572,864,000) = 1.82E+15
```
