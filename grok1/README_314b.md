# Overview

This recipe contains information and scripts to produce performance results for the Grok 1 training workload. The scripts help perform environment setup and launch benchmark jobs.
This variant of the workload is best-suited for GPU clusters with

* At least 8 GPUs with at least 80 GB memory each. Training of this 314-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* H100 or GH200 GPUs. This workload runs with FP8 and BF16 precision.

| GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 8    | 4096   | 2      | 4   | 1   | 1   | 2   | 1   | NA  | 1   | 1024 | 128 |
| 16   | 4096   | 4      | 4   | 1   | 1   | 4   | 1   | NA  | 1   | 1024 | 128 |
| 32   | 4096   | 4      | 4   | 1   | 1   | 8   | 1   | NA  | 1   | 1024 | 128 |
| 64   | 8192   | 8      | 4   | 1   | 2   | 8   | 1   | NA  | 1   | 1024 | 128 |
| 128  | 8192   | 16     | 4   | 2   | 2   | 8   | 1   | 8   | 1   | 1024 | 128 |
| 256  | 8192   | 32     | 4   | 4   | 2   | 8   | 1   | 8   | 1   | 1024 | 128 |
| 512  | 8192   | 64     | 4   | 8   | 2   | 8   | 1   | 8   | 1   | 1024 | 128 |
| 1024 | 8192   | 64     | 4   | 8   | 2   | 8   | 2   | 8   | 1   | 2048 | 128 |
| 2048 | 8192   | 64     | 4   | 8   | 2   | 8   | 4   | 8   | 1   | 4096 | 128 |

# Expected Performance

Performance for Grok 1 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/314b/$JOB_TOTAL_GPUS` folder. 

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep train_step_timing *.out
Epoch 0: : 100%|██████████| 50/50 [23:22<00:00, reduced_train_loss=0.186, global_step=49.00, consumed_samples=51200.0, train_step_timing in s=24.00]
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 4096 / 22 = 1525201

To calculate time to train with 1T tokens estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 1525201 / 86400 = 7.59 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) /peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is 1979 TFLOPS and for H100 BF16 is 989 TFLOPS.

The model flops for Grok 1 for GBS=1 per GPU for 2048 GPUs is 4.27E+15.

E.g. Grok 1 BF16 on 2048x H100 GPUs (GBS=4096)
```shell
peak FLOPS for H100 = 989 TFLOPS
training step time = 22
model flops = 4.27E+15

MFU = 4096 * 4.27E+15 / 22 / 2048 / 989E+12 = 39.3%
```

| Grok 1 314b Dev BF16 | Throughput on 8x H100 GPUs | Throughput on 16x H100 GPUs | Throughput on 32x H100 GPUs | Throughput on 64x H100 GPUs | Throughput on 128x H100 GPUs | Throughput on 256 H100 GPUs | Throughput on 512 H100 GPUs | Throughput on 1024 H100 GPUs | Throughput on 2048 H100 GPUs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 24.03 | 21.52 | 10.64 | 20.64 | 20.64 | 20.7 | 20.83 | 21 | 21.09
| Throughput in tokens per second       | 174544 | 194903 | 394202 | 406425 | 406425 | 405247 | 402718 | 798915 | 1591011
| Model flops utilization               | 45.4% | 44.9% | 45.4% | 44.5% | 43.0% | 42.1% | 41.5% | 41.1% | 41.0%
| Time to train 1T tokens in days       | NA | NA | NA | NA | NA | NA | 28.74 | 14.49 | 7.27

| Grok 1 314b Dev FP8 | Throughput on 8x H100 GPUs | Throughput on 16x H100 GPUs | Throughput on 32x H100 GPUs | Throughput on 64x H100 GPUs | Throughput on 128x H100 GPUs | Throughput on 256 H100 GPUs | Throughput on 512 H100 GPUs | Throughput on 1024 H100 GPUs | Throughput on 2048 H100 GPUs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 19.89 | 17.08 | 8.3 | 16.21 | 16.48 | 16.55 | 16.69 | 16.89 | 17.04
| Throughput in tokens per second       | 210875 | 245568 | 505338 | 517496 | 509017 | 506865 | 502613 | 993322 | 1969157
| Model flops utilization               | 27.4% | 28.3% | 29.1% | 28.3% | 26.9% | 26.3% | 25.9% | 25.6% | 25.3%
| Time to train 1T tokens in days       | NA | NA | NA | NA | NA | NA | 23.03 | 11.65 | 5.88

For proxy configs (<512 GPUs scales) we don't provide time to train estimates to avoid misleading conclusions. Proxy configs are not realistic and were created to allow fit of Grok model to smaller number of GPUs than intended.

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
Create a staging area by running the attached setup.sh. The script converts the docker image from nvcr.io/nvidia/nemo:dev (sha256:9c876dce0621f954d1733063e2af642682b188d757fb280cc55755fb8e194400) to the nvidia+nemo+dev.sqsh file under the $STAGE_PATH folder and copies NeMo Launcher code from the container.

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)

# Run the setup
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
```
Check the corresponding `slurm-<job_id>.out` file for status information.

**Important:** `STAGE_PATH` used in this step must be used when running the workload.

# Dataset
Grok 1 uses synthetic data. A dataset does not need to be downloaded.

# Run Training
Once the environment has been prepared, it is time to train a model. This section will demonstrate how to initiate training the model.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/314b/$JOB_TOTAL_GPUS` folder.

Below is a command template for launching Grok 1 model training.
```shell
DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```
Where:
- `DTYPE` is a **required** environment variable.
	- `DTYPE` can be either `fp8` or `bf16`.
- `NUM_NODES` can be calculated by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 256 GPUs scale, `NUM_NODES` should be `256 / 8 = 32`.
- [Slurm Settings](#slurm) for more information on Slurm parameters.

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
	  Default: "0,1,2,3,4,5,7"
* Enable GPU device metrics capture:
	* `RUN_CONF_PROFILE_GPU_METRICS`: boolean, set to 'true' to capture device metrics.
	- Default: false
	- **Note:** Additional system configuration is required for GPU device metrics to work.
* Enable CPU metrics capture:
	* `RUN_CONF_PROFILE_CPU`: boolean, set to 'true' to capture CPU metrics.
	- Default: false

**Example customized profiling command:**
```
ENABLE_PROFILE=true RUN_CONF_PROFILE_GPU_METRICS=true RUN_CONF_PROFILE_RANKS="0" DTYPE=<fp8/bf16> sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N ${NUM_NODES} ./launch.sh
```

### Troubleshooting:

If you encounter issues, try the defaults `ENABLE_PROFILE=true` first as these should be broadly applicable to most systems.

### Viewing results

[How to consume Nsight profiling results](../common/nsys-profile.md)


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
