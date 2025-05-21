# Expected Performance

Once the training job has finished successfully it's performance measurement metric is it's training throughput which is based on time it took to complete each training step.

The example below is taken from the end of the output log file - see `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/175b/$JOB_TOTAL_GPUS/*.out`, where NUM_NODES=16, the training step time was measured as **5.9** seconds during step number 50.
```shell
grep train_step_timing results/fp8/175b/16/*.out
Epoch 0: : 100%|██████████| 50/50 [31:51<00:00, reduced_train_loss=6.130, global_step=50.0, consumed_samples=76800.0, train_step_timing in s=5.900, val_loss=6.250]
```

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

 To obtain throughput as a tokens per second measurement, follow this formula: 
 ```shell
 (sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
 ```

 E.g. 2048 * 256 / 5.80 = 90394.48

To calculate time to train estimate:
```shell
 (total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
 E.g. 1e12 / 90394.48 / 86400 = 128.04 days

 
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

| NeMo Megatron 24.11 BF16 (TP=4,PP=8, MBS=1, VP=12, SEQ=2048) | 128x H100 GPUs (GBS=256) | 256x H100 GPUs (GBS=512) | 512x H100 GPUs (GBS=1024) | 1024x H100 GPUs (GBS=2048) | 2048x H100 GPUs (GBS=4096)
|---|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 8.91   | 8.95   | 9.09  | 9.01  | 9.02 
| Throughput in tokens per second | 58843 | 117159 | 230710 | 465517 | 930001 
| Model flops utilization | 49.93% | 49.71% | 48.94% | 49.38% | 49.32% | 
| Time to train 1T tokens in days | 197 | 99 | 50 | 25 | 12

| NeMo Megatron 24.11 FP8 (TP=4,PP=8, MBS=1, VP=12, SEQ=2048) | 128x H100 GPUs (GBS=256) | 256x H100 GPUs (GBS=512) | 512x H100 GPUs (GBS=1024) | 1024x H100 GPUs (GBS=2048) | 2048x H100 GPUs (GBS=4096)
|---|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 5.92   | 5.96   | 6.44  | 6.72  | 6.24
| Throughput in tokens per second | 88562 | 175936 | 325645 | 624152 | 1344328
| Model flops utilization | 37.56% | 37.30% | 34.52% | 33.09% | 35.63% 
| Time to train 1T tokens in days | 131 | 66 | 36 | 19 | 9


# Prepare Environment

Create a staging area by running the setup.sh script. The script saves the container image from the registry in the $STAGE_PATH folder and copies the NeMo Launcher code from the container to the staging directory. 

``` shell
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
Pre-training a GPT-3 model requires a text-based dataset to be downloaded and pre-processed for the NeMo Framework to ingest the data optimally. [The Pile](https://pile.eleuther.ai/) is often used as the dataset for pre-training models. The NeMo Framework contains helper scripts to download and pre-process the dataset. The following steps outline how to download and pre-process the dataset on DGX Cloud with an explanation of key points after.

Run the generate_dataset.sh script. The script launches several Slurm jobs that will download the dataset from The Pile, pre-process it and save it in a form suitable for subsequent training. The resulting dataset files will be saved under the $STAGE_PATH/gpt3-dataset folder. The dataset creation may use up to 250GB. Make sure you have sufficient disk space available.


``` shell
bash ./generate_dataset.sh
```

If the dataset generation step was successful there should be 4 idx and 4 bin files in the $STAGE_PATH/gpt3-dataset folder.

``` shell
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
``` shell
bash ./check_dataset.sh
Dataset generation completed successfully at: <$STAGE_PATH/gpt3-dataset>
```

If that is not the case, check the log files under: $STAGE_PATH/results.data_preparation 



# Run Training
Once the environment has been prepared, it is time to train a model. The NeMo Framework contains many predefined configuration files for various models including the 175 billion parameter GPT-3 model. This section will demonstrate how to initiate training on the model.

NeMo uses the Hydra framework to process command line arguments and the base config in the gpt3_175b_hydra.yaml file and passes them down as hyper parameters to a multi-node job performing the training.

Run the launch.sh script to start NeMo Megatron 175b model training. Minimum required number of nodes is 16 (or 128 GPUs). The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under `$STAGE_PATH/results/$GSW_VERSION/$DTYPE/175b/$JOB_TOTAL_GPUS` folder.

Below is a command template for launching NeMo Megatron model training.
```shell
DTYPE=<fp8/bf16> sbatch -A ${SLURM_ACCOUNT} -p ${SLURM_PARTITION} -N ${NUM_NODES} ./launch.sh
```
Where:
- `DTYPE` is a **required** environment variable.
    - `DTYPE` can be either `fp8` or `bf16`.
- `NUM_NODES` can be calculate by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 256 GPUs scale, `NUM_NODES` should be `256 / 8 = 32`.

**Note:** it might be necessary to pass ` --gres=gpu:8 ` to sbatch for certain clusters on encountering errors like GPU not found. See https://slurm.schedmd.com/gres.html

It is important to maintain these values for model parallelism settings in order to accurately assess performance results for completed jobs against expected baseline, which can be seen in the gpt3_175b_hydra.yaml:
* training.model.tensor_model_parallel_size=4
* training.model.pipeline_model_parallel_size=8
* training.model.micro_batch_size=1 
* training.model.virtual_pipeline_model_parallel_size=12

Global batch size ( training.model.global_batch_size) value should be set to ```<number of nodes> * 16. E.g., 16 * 16 = 256 (in the example above).```

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
