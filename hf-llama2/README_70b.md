# Overview

This recipe contains information and scripts to produce performance results for the Llama 2 Hugging Face fine-tuning training workload using PEFT and FSDP. The scripts help perform environment setup, dataset setup, and launch benchmark jobs.
This variant of the workload is best-suited for GPU clusters with

* At least 8 GPUs with at least 80 GB memory each. Fine tuning of this 70-billion parameter variant of the workload will not fit on fewer GPUs with less memory.
* H100 GPUs. This workload runs with BF16, which is supported by H100 GPUs.

# Expected Performance

Performance for HF Llama 2 fine tuning is measured by train samples per second, which is logged in the .out file associated with the job.

```shell
grep train_samples_per_second log-hf_llama2_70b_32_652934.out
{'train_runtime': 2577.7505, 'train_samples_per_second': 95.339, 'train_steps_per_second': 0.012, 'train_loss': 1.0156359354654947, 'epoch': 0.9}
wandb: train_samples_per_second 95.339
```
| LLAMA2 70b 24.02 BF16 | Train samples per second on 8x H100 GPUs | Train samples per second on 16x H100 GPUs  | Train samples per second on 32x H100 GPUs  | Train samples per second on 64x H100 GPUs  | Train samples per second on 128x H100 GPUs  | Train samples per second on 256x H100 GPUs  | Train samples per second on 512x H100 GPUs
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Training samples per second | 1.247 | 3.664 | 9.432 | 21.905 | 47.342 | 95.339 |154.931

# Prerequisites

This recipe requires access to Hugging Face Llama 2. Instructions are below if needed.

# Prepare Environment

Create a staging area by running the setup.sh script. The script converts the docker image from nvcr.io/nvidia/pytorch:24.02-py3 to the nvidia+pytorch+24.02.sqsh file under the $STAGE_PATH folder and downloads DHS-LLM workshop source code.

```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/<userid>)
# Set the Slurm partition to use
export SLURM_PARTITION="batch"
# Set the Slurm account to use
export SLURM_ACCOUNT="account_name"

# Run the setup
bash ./setup.sh
```

# Request Access
Access to Llama 2 must be requested through [Meta's website](https://llama.meta.com/llama-downloads/) then requested on the [Hugging Face Llama](https://huggingface.co/meta-llama/Llama-2-70b-hf) page. The approval process is not automatic and could take a day or more.

# Prepare Dataset
To download the model and dataset you will need to create a Hugging Face access token with READ privileges. You will use your HF user name and access token as the user/password for the git clones. For more information see: https://huggingface.co/docs/hub/en/security-tokens

### Download the model:
*Note:* Cloning the model can take well over an hour, and you will be prompted _twice_ for user/password. After the second prompt it'll appear as if it's hung.
```shell
cd $STAGE_PATH

# Only needs to be peformed once
git lfs install

git clone https://huggingface.co/meta-llama/Llama-2-70b-hf
```

If the model download step was successful there should these files in the $STAGE_PATH/Llama-2-70b-hf folder.

```shell
LICENSE.txt                config.json                       model-00003-of-00015.safetensors  model-00007-of-00015.safetensors  model-00011-of-00015.safetensors  model-00015-of-00015.safetensors  pytorch_model-00003-of-00015.bin  pytorch_model-00007-of-00015.bin  pytorch_model-00011-of-00015.bin  pytorch_model-00015-of-00015.bin  tokenizer.model
README.md                  generation_config.json            model-00004-of-00015.safetensors  model-00008-of-00015.safetensors  model-00012-of-00015.safetensors  model.safetensors.index.json      pytorch_model-00004-of-00015.bin  pytorch_model-00008-of-00015.bin  pytorch_model-00012-of-00015.bin  pytorch_model.bin.index.json      tokenizer_config.json
Responsible-Use-Guide.pdf  model-00001-of-00015.safetensors  model-00005-of-00015.safetensors  model-00009-of-00015.safetensors  model-00013-of-00015.safetensors  pytorch_model-00001-of-00015.bin  pytorch_model-00005-of-00015.bin  pytorch_model-00009-of-00015.bin  pytorch_model-00013-of-00015.bin  special_tokens_map.json
USE_POLICY.md              model-00002-of-00015.safetensors  model-00006-of-00015.safetensors  model-00010-of-00015.safetensors  model-00014-of-00015.safetensors  pytorch_model-00002-of-00015.bin  pytorch_model-00006-of-00015.bin  pytorch_model-00010-of-00015.bin  pytorch_model-00014-of-00015.bin  tokenizer.json
```

### Download the dataset:
```shell
git clone https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

```

If the dataset clone step was successful there should be these files in the $STAGE_PATH/ultrachat_200k/data folder
```shell
test_gen-00000-of-00001-3d4cd8309148a71f.parquet  test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet  train_gen-00000-of-00003-a6c9fb894be3e50b.parquet  train_gen-00001-of-00003-d6a0402e417f35ca.parquet  train_gen-00002-of-00003-c0db75b92a2f48fd.parquet  train_sft-00000-of-00003-a3ecf92756993583.parquet  train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet  train_sft-00002-of-00003-ee46ed25cfae92c6.parquet
```

More information on the model and dataset can be found https://huggingface.co/meta-llama/Llama-2-70b-hf and https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k respectively.

# Run Training

Once the environment has been prepared, it is time to train a model. Run the launch_70b.sh script with sbatch for launching Hugging Face LLAMA2 70b model training on 1 to 64 nodes with BF16 precision.
Log files will be located under `${STAGE_PATH}/results/$GSW_VERSION/bf16/70b/$JOB_TOTAL_GPUS`.

```shell
sbatch -A ${SLURM_ACCOUNT} -p ${SLURM_PARTITION} -N ${NUM_NODES} ./launch_70b.sh
```
Where:
- `NUM_NODES` can be calculate by `N_GPUS / N_GPUS_PER_NODE`, `N_GPUS_PER_NODE` is 8 for DGX H100, therefore for 256 GPUs scale, `NUM_NODES` should be `256 / 8 = 32`

**Note:** that it might be necessary to pass `--gres=gpu:8` to sbatch for certain clusters on encountering errors like GPU not found. See https://slurm.schedmd.com/gres.html

# Notes

accelerate launches on every node and pip install requirements.txt is run as part of srun command to ensure compute nodes have same environment. PYTHONPATH is set for this.
