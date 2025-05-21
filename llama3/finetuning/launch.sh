#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

#export finetuning required variables
export STAGE_PATH=${STAGE_PATH?"Required variable STAGE_PATH"}
export HF_TOKEN=${HF_TOKEN?"Required variable HF_TOKEN"}
export NEMO_HOME=${NEMO_HOME:-$STAGE_PATH/HF_ckpt}
export HF_HOME=${HF_HOME:-$NEMO_HOME}
export NEMORUN_HOME=${NEMORUN_HOME:-$STAGE_PATH/logs}

export IMAGE=${RUN_CONF_IMAGE:-$STAGE_PATH/nvidia+nemo+24.12.sqsh}
export DTYPE=${DTYPE:-bf16}
export MODEL_SIZE=${MODEL_SIZE:-8b}
export SFT_SCHEME=${SFT_SCHEME:-'lora'}
export CONTAINER_MOUNTS="$STAGE_PATH/Megatron-LM:/opt/megatron-lm"
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-8}
export RUN_CONF_GPUS_PER_NODE=${RUN_CONF_GPUS_PER_NODE:-8}
NUM_NODES=$(( ${JOB_TOTAL_GPUS} / ${RUN_CONF_GPUS_PER_NODE} ))

#run command
python3 $STAGE_PATH/NeMo/scripts/llm/performance/llama3_finetune.py --account $SBATCH_ACCOUNT --partition ${SBATCH_PARTITION} -i $IMAGE -cm $CONTAINER_MOUNTS -m $MODEL_SIZE -c $DTYPE -hf $HF_TOKEN -f $SFT_SCHEME -n $NUM_NODES -ng $RUN_CONF_GPUS_PER_NODE

