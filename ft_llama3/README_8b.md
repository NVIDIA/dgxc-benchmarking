# Overview

This recipe contains information and scripts to produce performance results for the Llama3 8B Supervised Fine-Tuning(SFT) and Low Rank Adaptation(LoRA) schemes of finetuning workloads. This workload supports BF16 and FP8 precision. It is best-suited for clusters with at least 8 H100 GPUs with at least 80 GB memory each. Fine-tuning of this 8-billion parameter variant of the workload will not fit on fewer GPUs with less memory.

| GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS |GBS | 
|------|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8    | 4096   | 1   | 1   | 1   | NA  | 8  | NA   | 1 | 32 |
| 16   | 4096   | 1   | 1   | 1   | NA  | 8  | NA   | 1 | 64 |
| 32   | 4096   | 1   | 1   | 1   | NA  | 8  | NA   | 1 | 128 |


# Expected Performance

Performance for Llama 3 fine-tuning is measured by seconds per iteration, or in other words seconds per training step. This metric is logged in a `log-*.out` which is generated in `$STAGE_PATH/logs/experiments/<experiment_name>/<experiment_name>_<timestamp>/<experiment_name>/` folder. $STAGE_PATH will be defined in the later stages of this document.

Example of an output file for LoRA finetuning run with bf16 precison of Llama3 8B model on 16 GPUs, where experiment_name = lora_nemo_llama3_8b_bf16_16:
```shell
$STAGE_PATH/logs/experiments/lora_nemo_llama3_8b_bf16_16/lora_nemo_llama3_8b_bf16_16_1771198/lora_nemo_llama3_8b_bf16_16/log-<slurm-account>.lora_nemo_llama3_8b_bf16_16_1771198_0.out
```

Since the performance fluctuates significantly at the beginning due to high warmup step times, we are using average time from the last 20 training steps to obtain throughput value.

Example output line
```shell
peak_memory_usage: 35475423232 | memory_allocated: 16209118208 | reduced_train_loss: 0.208 | train_step_timing in s: 0.9181 | consumed_samples: 3168
```
To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 128 / 0.9181 = 1142115

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 1142115 / 86400 = 10.13 days 

To calculate the model flops utilization (MFU) for SFT finetuning:
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

To calculate the model flops utilization (MFU) for LoRA finetuning use the same formula as SFT above but multiplied by 2/3. 

E.g. For Llama 8B BF16 SFT finetuning run on 8 gpus with batch size 32

32 * 1.84E+14 / 1.32 / 8 / 9.89E+14 = 56.38%

**Note:** Model flops for llama 8B finetuning 1.84E+14. The peak theoretical throughput for H100 FP8 is 1979 TFLOPS and for H100 BF16 is 989 TFLOPS.

| Llama3 8b Finetuning SFT BF16 | Throughput on 8x H100 GPUs | Throughput on 16x H100 GPUs | Throughput on 32x H100 GPUs | 
|---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 1.32 | 1.32 | 1.33 |  | |  |   
| Throughput in tokens per second | 99296  | 198593 | 397184 |  |  | |  |   
| Model flops utilization | 56.38% | 56.38% | 55.95% |  | |  |  
| Time to train 1T tokens in days | 116 | 58 | 29 |  | |  | 

| Llama3 8b Finetuning SFT FP8 | Throughput on 8x H100 GPUs | Throughput on 16x H100 GPUs | Throughput on 32x H100 GPUs |
|---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 0.95 | 0.96 | 0.96 |  | |  |   
| Throughput in tokens per second | 137970 | 273066 | 546133 |  |  | |  |   
| Model flops utilization | 39.16% | 38.7% | 38.7% |  | |  |  
| Time to train 1T tokens in days | 84 | 42 | 21 |  | |  |  

| Llama3 8b Finetuning LORA BF16 | Throughput on 8x H100 GPUs | Throughput on 16x H100 GPUs | Throughput on 32x H100 GPUs |
|---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 0.92 | 0.924 | 0.92 | | |  |  
| Throughput in tokens per second | 142469 | 284939 | 569878 |  |  | |  |   
| Model flops utilization | 53.9% | 53.7% | 53.9% |  | |  |  
| Time to train 1T tokens in days | 81 | 40.5 | 20 |  | |  | 

| Llama3 8b Finetuning LORA FP8 | Throughput on 8x H100 GPUs | Throughput on 16x H100 GPUs | Throughput on 32x H100 GPUs |
|---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 0.68 | 0.68 | 0.69 |   
| Throughput in tokens per second | 192752 | 385505 | 771011 | 
| Model flops utilization | 36.4% | 36.4% | 35.9% |   
| Time to train 1T tokens in days | 60 | 30 | 15 | 

# Prerequisites

This recipe requires access to Llama3 and its model weights from Huggingface. Instructions are below if needed.

Python virtual environment must be created using Python v.3.10.12 or newer before running the workload.

# Request Access
A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens) the huggingface token is used to run the fine-tuning scripts. Add the generated token to your environment via ```export HF_TOKEN=<your token>```.

Access to Llama3 must be requested through [Meta's website](https://llama.meta.com/llama-downloads/) then requested on the [HuggingFace Llama](https://huggingface.co/meta-llama/Meta-Llama-3-8B) page. The approval process is not automatic and could take a day or more.

# Prepare Environment

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**
- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
	- Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Workload Setup
Create a staging area by running the attached setup.sh. The script clones two repos "Megatron-LM" and "NeMo" and builds Megatron-Lm, which is needed to run our benchmark scripts. 


### Set the environment variables
```
export STAGE_PATH=<path to your shared file system folder>
export HF_TOKEN=<your_HF_token_value>
```

**Important:** `STAGE_PATH` used in this step must be used when running the workload.

### Prepare python virtual environment

The workload relies on python virtual environment in order ensure there are no conflicts between required dependencies and user's packages. We require Python 3.10.12 or newer for the workload to work.

There are multiple choices available to set up virtual environment: 
* conda
* python venv


#### Conda 

To install and activate conda virtual environment
```shell
# pick INSTALL_PATH with sufficient disk space
INSTALL_PATH=~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $INSTALL_PATH/miniconda.sh
bash $INSTALL_PATH/miniconda.sh -b -p $INSTALL_PATH/miniconda3
$INSTALL_PATH/miniconda3/bin/conda init
source ~/.bashrc
```

When you are finished running this benchmark you can deactivate the environment, run this command
```shell
conda deactivate
```

#### Python venv

To install and activate python venv 
```shell
python3 -m venv $STAGE_PATH/test_sft
source $STAGE_PATH/test_sft/bin/activate
```
When you are finished running this benchmark you can deactivate the environment, run this command
```shell
deactivate
```

### Setup script 
The setup.sh script will convert the docker file into .sqsh file and will store into the `STAGE_PATH`. Next step is to clone Megatron-LM and NeMo repositories and install all the necessary packages. 

The following command will get the environment set up for you. 

### Run setup.sh
Ensure that your python environment has been created and activated before running the setup.sh script.
```shell
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
```
Above command will launch a slurm job and must finish succesfully before you can proceed to the next step.

After running the setup script, directory structure of `STAGE_PATH` should look like below
```
|--|STAGE_PATH
|--|--|HF_ckpt
|--|--|logs
|--|--|Megatron-LM
|--|--|NeMo
```

# Run fine-tuning

Ensure that your python environment has been created and activated before running the launch.sh script.

Environment variable `FT_TYPE` allows user to choose between the two types of fine-tuning schemes (`lora` or `sft`). The fine-tuning will run for the first 100 steps and will stop afterwards. Log files and results will be located under the `$STAGE_PATH/logs/...` folder. 

Below is a an example on how to launch the workload:

```shell
DTYPE=fp8 MODEL_SIZE=8b FT_TYPE=lora JOB_TOTAL_GPUS=16 RUN_CONF_GPUS_PER_NODE=8 ./launch.sh
```
Where:
- `DTYPE`, `MODEL_SIZE`, `JOB_TOTAL_GPUS` and `FT_TYPE` can be changed by the user.
  - `DTYPE` can be either `fp8` or `bf16`.
  - `MODEL_SIZE` should be `8b` in this case.
  - `FT_TYPE` can be either `sft` or `lora`
  - `JOB_TOTAL_GPUS` can start from `8` to `32`
- [Slurm Settings](#slurm) for more information on Slurm parameters.

Below is the summary of configurations we support 
| Llama Model Type | Fine-tuning scheme | Precision | Scale(#gpus) | Run Command with scale options |
|:---:|:---:|:---:|:---:|:--|
| 8b | SFT | FP8 | 8, 16, 32 |`MODEL_SIZE=8b FT_TYPE=sft DTYPE=fp8 JOB_TOTAL_GPUS=<8/16/32> ./launch.sh` |
| 8b | SFT | BF16 | 8, 16, 32 |`MODEL_SIZE=8b FT_TYPE=sft DTYPE=bf16 JOB_TOTAL_GPUS=<8/16/32> ./launch.sh` |
| 8b | LoRA | FP8 | 8, 16, 32 |`MODEL_SIZE=8b FT_TYPE=lora DTYPE=fp8 JOB_TOTAL_GPUS=<8/16/32> ./launch.sh` |
| 8b | LoRA | BF16 | 8, 16, 32 |`MODEL_SIZE=8b FT_TYPE=lora DTYPE=bf16 JOB_TOTAL_GPUS=<8/16/32> ./launch.sh` |

**Note:** Running multiple/concurrent jobs using the same model can cause inconsistent or unintended results. 

# Advanced information for the user
Chapters below are for informational purpose only.
## Model Weights 

Model weights will be downloaded from Huggingface during the first finetuning run. We have a block of script which takes care of downloading the model weights and converts them to NeMo desirable format. A Slurm executor block with fixed duration is set for the download of weights. 

**Note:** Downloading and checkpoint converting steps are to be done only once per model. Subsequent runs use the cached weights.

**Note:** The model weights require approximately 20GB of space, so make sure you have enough space to download the weights and run the scripts.

The model weights and context are stored in the `HF_HOME` location. 
The directory structure looks like below:

```
-HF_ckpt
|--|models
|--|--|meta-llama
|--|--|--|Meta-Llama-3-8B
|--|--|--|--|context
|--|--|--|--|weights
```
The directory contains tokenizer files downloaded from previous steps. The weights are generic for each model and will be used by all the model precisions and fine-tuning schemes.

## Common Environment Issues
- Slurm parameters might not be applicable to all environments. Please consult with your system administrator and update or remove parameters as needed.

