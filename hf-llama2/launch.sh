#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For each dataset a user elects to use, the user is responsible for
# checking if the dataset license is fit for the intended purpose.

#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

# bash strict mode: http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -eu -o pipefail

export FRAMEWORK=hf
export MODEL=llama2
export MODEL_SIZE=70b
export GSW_VERSION=25.01
export FW_VERSION=24.02

export IMAGE=$STAGE_PATH/nvidia+pytorch+${FW_VERSION}.sqsh

# Workload requires 1 task per node during launch.
export NTASKS_PER_NODE=${RUN_CONF_GPU_PER_NODE:-8}
# Only BF16 supported.
export DTYPE=bf16

export JOB_TOTAL_GPUS=${SBATCH_GPUS:-$(( ${SLURM_JOB_NUM_NODES} * ${NTASKS_PER_NODE} ))}

export RESULT_DIR=$STAGE_PATH/results/$GSW_VERSION/$DTYPE/$MODEL_SIZE/$JOB_TOTAL_GPUS
export RESULT_FILES_NAME=log-${FRAMEWORK}_${MODEL}_${MODEL_SIZE}_${JOB_TOTAL_GPUS}
# ! breaking convention ! - no gsw specific container.

export MODEL_PATH=$STAGE_PATH/Llama-2-70b-hf
export DATASET_PATH=$STAGE_PATH/ultrachat_200k/data

mkdir -p $RESULT_DIR

# Workload configuration environment variables
MASTER_PORT=29500
export WANDB_MODE=offline
export HF_HOME=$STAGE_PATH/hf_home
mkdir -p $HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME
export WANDB_CACHE_DIR=$HF_HOME
export PYTHONPATH=$STAGE_PATH/DHS-LLM-Workshop/chat_assistant/sft/training

INSTALLER="cd $STAGE_PATH/DHS-LLM-Workshop/chat_assistant/sft/training; pip install -r $STAGE_PATH/requirements.txt;"

LAUNCHER="accelerate launch \
    --config_file $STAGE_PATH/DHS-LLM-Workshop/chat_assistant/sft/training/configs/fsdp_config.yaml \
    --main_process_ip \${SLURM_LAUNCH_NODE_IPADDR} \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $JOB_TOTAL_GPUS \
    --num_machines $SLURM_JOB_NUM_NODES \
    "

PROGRAM="\
$STAGE_PATH/DHS-LLM-Workshop/chat_assistant/sft/training/train.py \
--seed 100 \
--model_name_or_path "$MODEL_PATH" \
--dataset_name "$DATASET_PATH" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--max_steps 30 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "Llama-2-70b-sft-fsdp" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj" \
--use_4bit_quantization False
"

# SRUN_OUTPUT and SRUN_ERROR are Slurm environment variables to control output/error file locations.
export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-"pmix"}
export SRUN_OUTPUT=${SRUN_OUTPUT-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.out}
export SRUN_ERROR=${SRUN_ERROR-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.err}

srun \
  --container-image="$IMAGE" \
  --container-mounts="$RESULT_DIR,$STAGE_PATH" \
  --container-writable \
  --container-env=PYTHONPATH,WANDB_CACHE_DIR,HF_HOME,HF_DATASETS_CACHE,TORCH_HOME,WANDB_MODE \
  --no-container-mount-home bash -c "$INSTALLER $LAUNCHER $PROGRAM"
