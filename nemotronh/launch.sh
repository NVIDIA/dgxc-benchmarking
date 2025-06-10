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

export WORKLOAD_TYPE=pretraining
export MODEL_NAME=nemotronh
export FW_VERSION=25.04.01
export GSW_VERSION=25.05

export OPENBLAS_NUM_THREADS=1 # optional, to avoid resource contention at the frontend node.

# Ensure STAGE_PATH is not set as it's been replaced by LLMB_INSTALL
if [ -n "${STAGE_PATH+x}" ]; then
  echo "Error: STAGE_PATH is deprecated and should not be set. Please use LLMB_INSTALL instead."
  exit 1
fi

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

DTYPE=${DTYPE:-fp8}
DTYPE=${DTYPE,,}
GPU_TYPE=${GPU_TYPE:-h100}
GPU_TYPE=${GPU_TYPE,,}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-256}
MODEL_SIZE=${MODEL_SIZE:-56b}
MODEL_SIZE=${MODEL_SIZE,,}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
NCCLTRACE_ENABLED=${ENABLE_NCCLTRACE:-false}
NCCLTRACE_ENABLED=${NCCLTRACE_ENABLED,,}

MAX_STEPS=${MAX_STEPS:-50}
TIME_LIMIT=${TIME_LIMIT:-"00:25:00"}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}

export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

if [ $MODEL_SIZE = 56b ]; then
  TP=8
  PP=1
  CP=1
  GBS=$(( $JOB_TOTAL_GPUS * 3 ))
  MBS=1
  VP=0
  NUM_LAYERS=118
  HIDDEN_SIZE=8192
fi

CONFIG_OVERRIDES=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -gb $GBS \
  -mb $MBS \
  --num_layers $NUM_LAYERS \
  --hidden_size $HIDDEN_SIZE \
  -ep 1 \
"

if [ $PROFILE_ENABLED = true ] && [ $NCCLTRACE_ENABLED = true ]; then
  echo "Cannot both profile and get NCCL traces"
  exit 1
fi

if [[ "$PROFILE_ENABLED" = "true" ]]; then
  CONFIG_OVERRIDES+=" --enable_nsys"
  CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
  CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
  MAX_STEPS=$PROFILE_STOP_STEP
elif [[ "$NCCLTRACE_ENABLED" = "true" ]]; then
  CONFIG_OVERRIDES+=" -nt "
  MAX_STEPS=5
  TIME_LIMIT="00:15:00"
fi

pushd $LLMB_WORKLOAD/NeMo

python -m scripts.performance.llm.pretrain_nemotronh_${MODEL_SIZE} \
  --gpu h100 \
  --container_image $IMAGE \
  --compute_dtype $DTYPE \
  --num_gpus $JOB_TOTAL_GPUS \
  --gpus_per_node $GPUS_PER_NODE \
  --max_steps $MAX_STEPS \
  $CONFIG_OVERRIDES \
  slurm \
  --account $SBATCH_ACCOUNT \
  --partition $SBATCH_PARTITION \
  --log_dir ${NEMORUN_HOME} \
  --time_limit $TIME_LIMIT

popd
