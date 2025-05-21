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

Create a staging area by running the setup.sh script. The script saves the container image from the registry in the $STAGE_PATH folder and copies copies config file paxml_mod_configs.py to the $STAGE_PATH directory. 


```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)
# Set the Slurm partition to launch against
export SLURM_PARTITION="batch"
# Set the Slurm account to launch against
export SLURM_ACCOUNT="account_name"
# Set the number of GPUs per node according to Slurm's gres, this is usually 8 or null - https://slurm.schedmd.com/gres.html
export SLURM_GPUS_PER_NODE=null

# Run the setup
bash ./setup.sh
```

# Prepare Dataset
Since Paxml GPT training can be run using synthetic datasets, this step is omitted.

# Run Training
Once the environment has been prepared, it is time to train a model.

Training is run for 100 steps and will stop afterwards. Log files and results will be located under `$STAGE_PATH/results/$GSW_VERSION/${DTYPE}/175b/$JOB_TOTAL_GPUS` folder.

Below is a command template for launching Paxml model training.
```shell
DTYPE=fp8 MODEL_SIZE=175b sbatch -A ${SLURM_ACCOUNT} -p ${SLURM_PARTITION} -N ${NUM_NODES} ./launch.sh
```
Where:
- `DTYPE` and `MODEL_SIZE` are **required** environment variables.
	- `DTYPE` must be `fp8`.
	- `MODEL_SIZE` should be `175b` in this case.
- `NUM_NODES` can be calculate by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 128 GPUs scale, `NUM_NODES` should be `128 / 8 = 16`.

**Note:** that it might be necessary to pass `--gres=gpu:8` to sbatch for certain clusters on encountering errors like GPU not found. See https://slurm.schedmd.com/gres.html

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
