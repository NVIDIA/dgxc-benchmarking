# Overview

This recipe contains information and scripts to produce performance results for the Nemotron 4 15B training workload. The scripts help perform environment setup and launch benchmark jobs.

This variant of the workload is best-suited for GPU clusters with:
* At least 8 GPUs with at least 80 GB memory each. Training of this 15-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* This workload runs with BF16 or FP8 precision. FP8 is only supported by H100 GPUs. BF16 recipes are suitable for both A100 and H100 GPUs.
* The minimum number of required H100 GPUs is 8 (fp8 precision) and 16 (bf16 precision). Out Of Memory errors could be encountered if using fewer number of GPUs.


# Expected Slurm Performance

Performance for Nemotron 4 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/15b/$JOB_TOTAL_GPUS` folder. 

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep train_step_timing results/*.out
Epoch 0: : 100%|██████████| 100/100 [10:48<00:00, reduced_train_loss=0.0172, global_step=99.00, consumed_samples=25600.0, train_step_timing in s=3.130]
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 4096 * 256 / 3.13 = 335008

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 335008 / 86400 = 34.55 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is **1979** TFLOPS and for H100 BF16 is **989** TFLOPS.

The model flops for NeMoTron4 15b for GBS=1 is 434.5e12. Calculation shown [here](#notes).

E.g. NeMotron4 15b BF16 on 64x H100 GPUs (GBS=256)
```shell
peak FLOPS for H100 BF16 = 989 TFLOPS
training step time = 2.86 s
model flops = 434.5e12

MFU = 256 * 434.5e12 / 2.86 / 64 / 989e+12 = 61%
```

| Nemotron4 15b 24.11 BF16 (TP=4, PP=1, MBS=4, GA=4) | Throughput on 16x H100 GPUs (GBS=64) | Throughput on 32x H100 GPUs (GBS=128) | Throughput on 64x H100 GPUs (GBS=256) | Throughput on 128x H100 GPUs (GBS=512) | Throughput on 256x H100 GPUs (GBS=1024) | Throughput on 512x H100 GPUs (GBS=2048) | Throughput on 1024x H100 GPUs (GBS=4096) | Throughput on 2048x H100 GPUs (GBS=8192) |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 2.83 | 2.86 | 2.86 | 2.89 | 2.88 | 2.92 | 2.92 | 2.96 |
| Throughput in tokens per second | 92630 | 183317 | 366635 | 725658 | 1456356 | 2872811 | 5745622 | 11335957 |
| Model flops utilization | 62.10% | 61.45% | 61.45% | 60.81% | 61.02% | 60.18% | 60.18% | 59.37% |
| Time to train 1T tokens in days | 124.95 | 63.14 | 31.57 | 15.95 | 7.95 | 4.03 | 2.01 | 1.02 |

| Nemotron4 15b 24.11 FP8 (TP=4, PP=1, MBS=4, GA=4) | Throughput on 16x H100 GPUs (GBS=64) | Throughput on 32x H100 GPUs (GBS=128) | Throughput on 64x H100 GPUs (GBS=256) | Throughput on 128x H100 GPUs (GBS=512) | Throughput on 256x H100 GPUs (GBS=1024) | Throughput on 512x H100 GPUs (GBS=2048) | Throughput on 1024x H100 GPUs (GBS=4096) | Throughput on 2048x H100 GPUs (GBS=8192) |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 2.08 |  2.07 | 2.09 | 2.08  | 2.1 | 2.12 | 2.12 | 2.18 |
| Throughput in tokens per second |126031 | 253279 | 501711 | 1008246 | 1997288 | 3956891 | 7913781|15391941|
| Model flops utilization | 42.22% | 42.43% | 42.02% | 42.22% | 41.82% | 41.43% | 41.43% | 40.29% |
| Time to train 1T tokens in days | 91.84 | 45.70 | 23.07 | 11.48 | 5.79 | 2.93 | 1.46 | 0.75 |


# Prepare Slurm Environment

Create a staging area by running the attached setup.sh. The script converts the docker image from nvcr.io/nvidia/nemo:24.09 to the nvidia+nemo+24.09.sqsh file under the $STAGE_PATH folder and copies NeMo Launcher code from the container. 

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)
# Set the Slurm partition to launch against
export SLURM_PARTITION="batch"
# Set the Slurm account to launch against
export SLURM_ACCOUNT="account_name"
# number of GPUs per node. For H100: 8, For GH200: 2
export SLURM_GPUS_PER_NODE=8 
# Set the number of GPUs per node according to Slurm's gres, this is usually 8 or null - https://slurm.schedmd.com/gres.html
export SLURM_GPUS_PER_NODE=null

# Run the setup
bash ./setup.sh
```

# Prepare Dataset
Since Nemotron training only uses synthetic datasets, this step is omitted.

# Run Slurm Training

Once the environment has been prepared, it is time to train a model. NeMo Framework contains many predefined configuration files for various models including the 15 billion parameter Nemotron 4 model. This section will demonstrate how to initiate training the model. You can see the [default configuration for Nemotron 15b](https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/24.05/launcher_scripts/conf/training/nemotron/nemotron_15b.yaml) in the NeMo-Framework-Launcher Github repository. We will modify some of these parameters with our launch command.

NeMo Launcher is using the Hydra framework to process command line arguments and pass them down as hyper parameters to a multi-node job performing the training.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/15b/$JOB_TOTAL_GPUS` folder.

Below is a slurm command template for launching Nemotron 4 15b model training.
```shell
DTYPE=<fp8/bf16> MODEL_SIZE=15b sbatch -A ${SLURM_ACCOUNT} -p ${SLURM_PARTITION} -N ${NUM_NODES} $STAGE_PATH/launch.sh
```
Where:
- `DTYPE` and `MODEL_SIZE` are **required** environment variables.
	- `DTYPE` can be either `fp8` or `bf16`.
	- `MODEL_SIZE` should be `15b` in this case.
- `NUM_NODES` can be calculate by `N_GPUS / SLURM_GPUS_PER_NODE`, `SLURM_GPUS_PER_NODE` is 8 for DGX H100, therefore for 128 GPUs scale, `NUM_NODES` should be `128 / 8 = 16`.

**Note:** that it might be necessary to pass `--gres=gpu:8` to sbatch for certain clusters on encountering errors like GPU not found. See https://slurm.schedmd.com/gres.html

It is important to maintain these values for model parallelism settings in order to accurately assess performance results for completed jobs against expected baseline:
* training.model.tensor_model_parallel_size=4
* training.model.pipeline_model_parallel_size=1
* training.model.micro_batch_size=4

Global batch size ( training.model.global_batch_size) value should be set to ```<number of nodes> * 32. E.g., 16 * 32 = 512 (in the example above)```.


# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = (24 * (number of layers) * (hidden size)^2) + (12 * (number of layers) * (hidden size) * (sequence length))
    mlp flops = 48 * (number of layers) * (hidden size)^2
    embedding flops = 6 * (vocab size) * (hidden size)

Nemotron4 15b calculation:
    sequence length = 4096
    number of layers = 32
    hidden size = 6144
    vocab size = 256000 
    attention flops = 24 * 32 * 6144^2 + 12 * 32 * 6144 * 4096 = 38666279738
    mlp flops = 48 * 32 * 6144^2 = 57982058496
    embedding flops = 6 * 256000 * 6144 = 9437184000

    model flops = 4096 * (38666279738 + 57982058496 + 9437184000) = 434,526,299,070,464 = 434.5e12
```
