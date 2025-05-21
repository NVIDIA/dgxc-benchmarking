# Expected Performance

The performance of paxml GPT training is measured in training_step_time(seconds), which is the time taken for a training step to complete.

This metric is logged for every training step in the output .out file which is located in the logs folder at `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/175b/$JOB_TOTAL_GPUS`.

In the example below taken from the slurm output log file, the training step time was measured as 7.47 seconds during one of the later steps.
```shell
grep 'train_step() took' *.out | tail -1
I0411 15:03:26.866614 140737350317184 programs.py:366] [PAX STATUS]: train_step() took 7.47 seconds.
```

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value. 

To obtain throughput as a tokens per second measurement, follow this formula: 
 ```shell
 (sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
 ```

 E.g. 2048 * 256 / 7.47 = 70185.80

To calculate time to train estimate for 1T tokens:
```shell
 (total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
 E.g. 1e12 / 70185.80 / 86400 = 164.9 days

 
 To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is 1979 TFLOPS.

The model flops for GPT 3 175B for GBS=1 is 2.20E+15. Example calculation shown [here](#notes).

E.g. GPT 3 175B FP8 on 128x H100 GPUs (GBS=256)
```shell
peak FLOPS for H100 = 1979 TFLOPS
training step time = 7.47 s
model flops = 2.20E+15

MFU = 256 * 2.20E+15 / 7.47 / 128 / 1979E+12 = 29.76% 
```

| Paxml GPT175B 24.11 FP8 (FSDP+DP, MBS=2) | 128x H100 GPUs (GBS=256) | 256x H100 GPUs (GBS=512) | 512x H100 GPUs (GBS=1024) | 1024x H100 GPUs (GBS=2048) | 2048x H100 GPUs (GBS=4096)
|---|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 7.47  | 7.58  | 7.60  | 7.89  | 8.02
| Throughput in tokens per second | 70185.8 | 138334.56  | 275941.05 | 531597.46 | 1045961.09
| Model flops utilization | 29.76% | 29.33% | 29.25% | 28.17% | 27.72%
| Time to train 1T tokens(days) | 164.9 | 83.67 |  41.94  |  21.77 | 11.07


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
Create a staging area by running the setup.sh script. The script saves the container image from the registry in the $STAGE_PATH folder and copies copies config file paxml_mod_configs.py to the $STAGE_PATH directory. 

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)

# Run the setup
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
```
Check the corresponding `slurm-<job_id>.out` file for status information.

**Important:** `STAGE_PATH` used in this step must be used when running the workload.

# Prepare Dataset
Since Paxml GPT training can be run using synthetic datasets, this step is omitted.

# Run Training
Once the environment has been prepared, it is time to train a model.

Training is run for 100 steps and will stop afterwards. Log files and results will be located under `$STAGE_PATH/results/$GSW_VERSION/${DTYPE}/175b/$JOB_TOTAL_GPUS` folder.

Below is a command template for launching Paxml model training. **BF16 is not currently supported** for 175b.
```shell
DTYPE=fp8 MODEL_SIZE=175b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```
Where:
- `DTYPE` and `MODEL_SIZE` are **required** environment variables.
	- `DTYPE` must be `fp8`.
	- `MODEL_SIZE` should be `175b` in this case.
- `NUM_NODES` can be calculate by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 128 GPUs scale, `NUM_NODES` should be `128 / 8 = 16`.
- [Slurm Settings](#slurm) for more information on Slurm parameters.

# Profiling
We have one profiling method supported: Nsight.

Due to overhead while profiling: the results generated with these settings is not valid for comparison. 'Performance' and 'Profiling' runs should be done separately.

## Run Nsight Profiling

Nsight Systems is included in our containers. To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Default Profiling Settings:
* **MPI Ranks:** 0-8
* **Duration:** 60 seconds with a 300 second delay at job start.
* **Output Location:** .nsys-rep files are saved in the nsys folder within the existing results directory.
* **Filename format:** `${MODEL}-${MODEL_SIZE}-${DTYPE}_${NUM_GPUS}g_${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURM_LOCALID}.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true DTYPE=fp8 MODEL_SIZE=175b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```
### Customizing profiling behavior:
* Specify job steps to profile:
    * `RUN_CONF_PROFILE_DELAY`: start profiling after this many seconds after job start.
      Default: 300
    * `RUN_CONF_PROFILE_DURATION`: run profiling for this duration in seconds.
      Default: 60

Additional settings for profile ranks, and cpu/gpu metrics are not currenty supported. 

**Example customized profiling command:**
```shell
ENABLE_PROFILE=true RUN_CONF_PROFILE_GPU_METRICS=true RUN_CONF_PROFILE_RANKS="0" DTYPE=fp8 MODEL_SIZE=175b sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```

### Troubleshooting:

If you encounter issues, try the defaults `ENABLE_PROFILE=true` first as these should be broadly applicable to most systems.

### Viewing results

[How to consume Nsight profiling results](../common/nsys-profile.md)


# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = (24 * (number of layers) * (hidden size)^2) + (12 * (number of layers) * (hidden size) * (sequence length))
    mlp flops = 48 * (number of layers) * (hidden size)^2
    embedding flops = 6 * (vocab size) * (hidden size)

GPT 3 175b calculation:
    sequence length = 2048
    attention flops = 24 * 96 * 12288^2 + 12 * 96 * 12288 * 2048 = 376883380224
    mlp flops = 48 * 96 * 12288^2 = 695784701952
    embedding flops = 6 * 51200 * 12288 = 3774873600

    model flops = 2048 * (376883380224 + 695784701952 + 3774873600) = 2.20E+15
```
