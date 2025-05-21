#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
export FT_TYPE=${FT_TYPE:-'lora'}
export CONTAINER_MOUNTS="$STAGE_PATH/Megatron-LM:/opt/megatron-lm"
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-8}
export RUN_CONF_GPUS_PER_NODE=${RUN_CONF_GPUS_PER_NODE:-8}
NUM_NODES=$(( ${JOB_TOTAL_GPUS} / ${RUN_CONF_GPUS_PER_NODE} ))

#run command
python3 $STAGE_PATH/NeMo/scripts/llm/performance/llama3_finetune.py --account $SBATCH_ACCOUNT --partition ${SBATCH_PARTITION} -i $IMAGE -cm $CONTAINER_MOUNTS -m $MODEL_SIZE -c $DTYPE -hf $HF_TOKEN -f $FT_TYPE -n $NUM_NODES -ng $RUN_CONF_GPUS_PER_NODE

