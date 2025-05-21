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

E.g. 8192 * 252 / 12.84 = 160778

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days)
```
E.g. 1e12 / 160778 / 86400 = 71.99 days


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

| Llama 3.1 405b 24.09 BF16 (TP=8, PP=9, CP=2, VP=7, MBS=1, GA=63) | Throughput on 32x H100 GPUs  | Throughput on 96x H100 GPUs  | Throughput on 192x H100 GPUs | Throughput on 576x H100 GPUs | Throughput on 1152x H100 GPUs | Throughput on 2304x H100 GPUs |
|---|---|---|---|---|---|---|
| Layers                                 | 7      | 21     | 42     | 126    | 126    | 126    | 
| GBS                                    | 126    | 126    | 252    | 252    | 504    | 1008   | 
| PP                                     | 1      | 3      | 3      | 9      | 9      | 9      | 
| VP                                     | n/a    | 7      | 7      | 7      | 7      | 7      | 
| Training step time (seconds per step)  | 8.85   | 8.7  5 | 16.93  | 17.20  | 17.52  | 17.62  | 
| Throughput in tokens per second        | 116632 | 117965 | 121936 | 120022 | 235660 | 468646 | 
| Model flops utilization                | 58.63% | 56.17% | 57.25% | 55.78% | 54.76% | 54.45% | 
| Time to train 1T tokens in days        | n/a    | n/a    | n/a    | 96.43  | 49.11  | 24.7   | 

| Llama 3.1 405b 24.09 FP8 (TP=8, PP=9, CP=2, VP=7, MBS=1, GA=63) | Throughput on 32x H100 GPUs  | Throughput on 96x H100 GPUs  | Throughput on 192x H100 GPUs | Throughput on 576x H100 GPUs | Throughput on 1152x H100 GPUs | Throughput on 2304x H100 GPUs |
|---|---|---|---|---|---|---|
| Layers                                 | 7      | 21     | 42     | 126    | 126    | 126    | 
| GBS                                    | 126    | 126    | 252    | 252    | 504    | 1008   | 
| PP                                     | 1      | 3      | 3      | 9      | 9      | 9      | 
| VP                                     | n/a    | 7      | 7      | 7      | 7      | 7      | 
| Training step time (seconds per step)  | 5.80   | 5.71   | 11.00  | 11.24  | 11.31  | 12.35  | 
| Throughput in tokens per second        | 178118 | 180674 | 187740 | 183664 | 365055 | 668626 | 
| Model flops utilization                | 44.77% | 43.01% | 44.07% | 42.71% | 42.45% | 38.87% | 
| Time to train 1T tokens in days        | n/a    | n/a    | n/a    | 63.02  | 31.71  | 17.31  | 

For proxy configs (<576 GPUs scales) we don't provide time to train estimates to avoid misleading conclusions. Proxy configs are not realistic and were created to allow fit of Llama model to smaller number of GPUs than intended.

# Prerequisites

This recipe requires access to Llama 3.1. Instructions are below if needed.

# Request Access
A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens) in order to run the training script. Add the generated token to your environment via `export HF_TOKEN=<your token>`.

Access to Llama 3.1 must be requested through [Meta's website](https://llama.meta.com/llama-downloads/) then requested on the [HuggingFace Llama](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B) page. The approval process is not automatic and could take a day or more.

# Prepare Environment

Create a staging area by running the attached setup.sh. The script converts the docker image from `nvcr.io/nvidia/nemo:24.09` to the `nvidia+nemo+24.09.sqsh` file under the `$STAGE_PATH` folder and copies NeMo Launcher code from the container. 

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)
# Set the Slurm partition to launch against
export SLURM_PARTITION="batch"
# Set the Slurm account to launch against
export SLURM_ACCOUNT="account_name"
# Set the number of GPUs per node according to Slurm's gres, this is usually 8 or null - https://slurm.schedmd.com/gres.html
export SLURM_GPUS_PER_NODE=null
# Set HuggingFace token
export HF_TOKEN=<your token>

# Run the setup
bash ./setup.sh
```

# Prepare Dataset
Llama 3.1 405B uses synthetic data for training. A dataset does not need to be prepared. Note that Llama3.1 405B uses the GPT2BPETokenizer as a proxy.

# Run Training

NeMo Launcher is using the Hydra framework to process command line arguments and pass them down as hyperparameters to a multi-node job performing the training.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/405b/$JOB_TOTAL_GPUS` folder.

Below is a command template for launching Llama 3.1 405b model training.
```shell
DTYPE=<fp8/bf16> MODEL_SIZE=405b sbatch -A ${SLURM_ACCOUNT} -p ${SLURM_PARTITION} -N ${NUM_NODES} ./launch.sh
```
Where:
- `DTYPE` and `MODEL_SIZE` are **required** environment variables.
	- `DTYPE` can be either `fp8` or `bf16`.
	- `MODEL_SIZE` should be `405b` in this case.
- `NUM_NODES` can be calculate by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 576 GPUs scale, `NUM_NODES` should be `576 / 8 = 72`.

**Note:** it might be necessary to pass `--gres=gpu:8` to sbatch for certain clusters on encountering errors like GPU not found. See https://slurm.schedmd.com/gres.html

The following applies only to the full model scales: 576, 1152, 2304 GPUs. Configurations and Global batch size changes for proxy configs <576 GPUs.
>>>
It is important to maintain these values for model parallelism settings in order to accurately assess performance results for completed jobs against expected baseline for the non-proxy 405b configurations:
* `training.model.tensor_model_parallel_size=8`
* `training.model.pipeline_model_parallel_size=9`
* `training.model.virtual_pipeline_model_parallel_size=7`
* `training.model.context_parallel_size=2`

Global batch size (`training.model.global_batch_size`) value should scale with total number GPUs. The starting global batch size for 576 GPUs is 252, therefore it should set to `<number of total gpus> * 252 / 576`.
>>>

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

    model flops = 8129 * (659,545,915,392 + 1,978,637,746,176 + 12,608,077,824) = 2.17E16
```
