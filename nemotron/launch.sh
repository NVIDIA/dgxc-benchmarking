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

export FW_VERSION=25.02.01
export GSW_VERSION=25.04.01

export OPENBLAS_NUM_THREADS=1 # optional, to avoid resource contention at the frontend node.
export NEMORUN_HOME=$STAGE_PATH
export LLMB_PATH=$PWD

DTYPE=${DTYPE:-fp8}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-16}
MODEL_SIZE=${MODEL_SIZE:-15b}
MODEL_SIZE=${MODEL_SIZE,,}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}

MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-false}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT,,}

export PROFILE_START_STEP=${PROFILE_START_STEP:-20}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-25}

if [ $MODEL_SIZE = 15b ]; then
  TP=2
  PP=1
  CP=1
  GBS=$(( $JOB_TOTAL_GPUS * 4 ))
  MBS=2
  VP=0
  NUM_LAYERS=32
  HIDDEN_SIZE=6144
elif [ $MODEL_SIZE = 340b ]; then
  TP=8
  PP=8
  CP=1
  GBS=$(( $JOB_TOTAL_GPUS / 4 ))
  MBS=1
  VP=12
  NUM_LAYERS=96
  HIDDEN_SIZE=18432
fi

CONFIG_OVERRIDES=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -gb $GBS \
  -mb $MBS \
  --num_layers $NUM_LAYERS \
  --hidden_size $HIDDEN_SIZE \
"

if [ $VP != 0 ]; then
  CONFIG_OVERRIDES+=" -vp $VP " 
fi

if [ $PROFILE_ENABLED = true ]; then
  CONFIG_OVERRIDES+=" -en " 
  CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
  CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
  MAX_STEPS=$PROFILE_STOP_STEP
fi

if [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
  MAX_STEPS=1
fi

if [ $CPU_PER_TASK_PINNING -gt 0 ]; then
  CONFIG_OVERRIDES+=" --cpu_pinning $CPU_PER_TASK_PINNING "
fi

pushd $LLMB_PATH/NeMo

python -m scripts.performance.llm.pretrain_nemotron4_${MODEL_SIZE} \
  --account $SBATCH_ACCOUNT \
  --partition $SBATCH_PARTITION \
  --log_dir ${NEMORUN_HOME} \
  --gpu h100 \
  --container_image $STAGE_PATH/nvidia+nemo+$FW_VERSION.sqsh \
  --compute_dtype $DTYPE \
  --num_gpus $JOB_TOTAL_GPUS \
  --gpus_per_node 8 \
  --max_steps $MAX_STEPS \
  $CONFIG_OVERRIDES

popd
