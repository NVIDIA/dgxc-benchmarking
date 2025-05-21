# Overview

This recipe contains information and scripts to produce performance results for the Nemotron 4 340B training workload. The scripts help perform environment setup and launch benchmark jobs.

This variant of the workload is best-suited for GPU clusters with:
* At least 128 GPUs with at least 80 GB memory each. Training of this 340-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* This workload supports BF16 or FP8 precision. FP8 is only supported by H100 GPUs. BF16 recipes are suitable for both A100 and H100 GPUs.

# Expected Performance

Performance for Nemotron 4 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/340b/$JOB_TOTAL_GPUS` folder. 

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep train_step_timing results/*.out
Epoch 0: : 100%|██████████| 100/100 [07:57<00:00, reduced_train_loss=7.310, global_step=99.00, consumed_samples=25600.0, train_step_timing in s=3.590]
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 4096 * 256 / 3.59 = 292082

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 292082 / 86400 = 39.6 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is **1979** TFLOPS and for H100 BF16 is **989** TFLOPS.

The model flops for NeMoTron4 340b for GBS=1 is 1.01e16. Calculation shown [here](#notes).

E.g. NeMotron4 340b BF16 on 128x H100 GPUs (GBS=32)
```shell
peak FLOPS for H100 BF16 = 989 TFLOPS
training step time = 4.77 s
model flops = 1.01e16

MFU = 32 * 1.01e16 / 4.77 / 128 / 989e+12 = 53.52%
```

| Nemotron4 340b 24.11 BF16 (TP=8, PP=8, MBS=1, GA=16, VP=12) | Throughput on 128x H100 GPUs (GBS=32) | Throughput on 256x H100 GPUs (GBS=64) | Throughput on 512x H100 GPUs (GBS=128) | Throughput on 1024x H100 GPUs (GBS=256) | Throughput on 2048x H100 GPUs (GBS=512) | 
|---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) |  4.61 | 4.67 | 4.7 | 4.73 |  4.84 |
| Throughput in tokens per second | 28432 | 56134 | 111551 | 221686 | 433296 |
| Model flops utilization | 55.38% | 54.67% | 54.32% | 53.98% | 52.75% |
| Time to train 1T tokens in days | 407 | 206 | 104 | 52 | 27 |

| Nemotron4 340b 24.11 FP8 (TP=8, PP=8, MBS=1, GA=16, VP=12) | Throughput on 128x H100 GPUs (GBS=32) | Throughput on 256x H100 GPUs (GBS=64) | Throughput on 512x H100 GPUs (GBS=128) | Throughput on 1024x H100 GPUs (GBS=256) | Throughput on 2048x H100 GPUs (GBS=512) |
|---|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 3.16 | 3.21 | 3.26 | 3.34 | 3.49 |
| Throughput in tokens per second | 41478 | 81665 | 160825 | 313945 | 600903 |
| Model flops utilization | 40.38% | 39.75% | 39.14% | 38.20% | 36.56% |
| Time to train 1T tokens in days | 279 | 142 | 72 | 37 | 19 |


# Prepare Environment

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

# Run Training

Once the environment has been prepared, it is time to train a model. NeMo Framework contains many predefined configuration files for various models including the 340 billion parameter Nemotron 4 model. This section will demonstrate how to initiate training the model. You can see the [default configuration for Nemotron 340b](https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/main/launcher_scripts/conf/training/nemotron/nemotron_340b.yaml) in the NeMo-Framework-Launcher Github repository. We will modify some of these parameters with our launch command.

NeMo Launcher is using the Hydra framework to process command line arguments and pass them down as hyper parameters to a multi-node job performing the training.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/340b/$JOB_TOTAL_GPUS` folder.

Below is a slurm command template for launching Nemotron 4 340b model training.
```shell
DTYPE=<fp8/bf16> MODEL_SIZE=340b sbatch -A ${SLURM_ACCOUNT} -p ${SLURM_PARTITION} -N ${NUM_NODES} $STAGE_PATH/launch.sh
```
Where:
- `DTYPE` and `MODEL_SIZE` are **required** environment variables.
	- `DTYPE` can be either `fp8` or `bf16`.
	- `MODEL_SIZE` should be `340b` in this case.
- `NUM_NODES` can be calculate by `N_GPUS / SLURM_GPUS_PER_NODE`, `SLURM_GPUS_PER_NODE` is 8 for DGX H100, therefore for 128 GPUs scale, `NUM_NODES` should be `128 / 8 = 16`.

**Note:** that it might be necessary to pass `--gres=gpu:8` to sbatch for certain clusters on encountering errors like GPU not found. See https://slurm.schedmd.com/gres.html

It is important to maintain these values for model parallelism settings in order to accurately assess performance results for completed jobs against expected baseline:
* training.model.tensor_model_parallel_size=8
* training.model.pipeline_model_parallel_size=8
* training.model.micro_batch_size=1
* training.model.virtual_pipeline_model_parallel_size=12

Global batch size ( training.model.global_batch_size) value should be set to ```<number of nodes> * 2. E.g., 128 * 2 = 256 (in the example above)```.




# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = (24 * (number of layers) * (hidden size)^2) + (12 * (number of layers) * (hidden size) * (sequence length))
    mlp flops = 48 * (number of layers) * (hidden size)^2
    embedding flops = 6 * (vocab size) * (hidden size)

Nemotron4 340b calculation:
    sequence length = 4096
    number of layers = 96
    hidden size = 18432
    vocab size = 256000 
    attention flops = 24 * 96 * 18432^2 + 12 * 96 * 18432 * 4096 = 869730877440
    mlp flops = 48 * 96 * 18432^2 = 1565515579392
    embedding flops = 6 * 256000 * 18432 = 28311552000

    model flops = 4096 * (869730877440 + 1565515579392 + 28311552000) = 1.01e16
```
