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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=nemotron-h
export FW_VERSION=25.09.00

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

DTYPE=${DTYPE:-fp8}
DTYPE=${DTYPE,,}
FP8_RECIPE=${FP8_RECIPE:-cs}
FP8_RECIPE=${FP8_RECIPE,,}
GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}
MODEL_SIZE=${MODEL_SIZE:-56b}
MODEL_SIZE=${MODEL_SIZE,,}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

MAX_STEPS=${MAX_STEPS:-50}
TIME_LIMIT=${TIME_LIMIT:-"00:30:00"}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}

export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

CONTAINER_MOUNTS=""
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

if [ $MODEL_SIZE = 56b ]; then
    if [ $GPU_TYPE = gb300 ]; then
        DEF_GPUS_PER_NODE=4
        TP=2
        MBS=1
    elif [ $GPU_TYPE = gb200 ]; then
        DEF_GPUS_PER_NODE=4
        TP=4
        MBS=2
    elif [ $GPU_TYPE = b200 ]; then
        DEF_GPUS_PER_NODE=8
        TP=4
        MBS=2
    elif [ $GPU_TYPE = h100 ]; then
        DEF_GPUS_PER_NODE=8
        TP=8
        MBS=1
    fi

    GPUS_PER_NODE=${GPUS_PER_NODE:-$DEF_GPUS_PER_NODE}

    PP=1
    CP=1
    GBS=${GBS:-$((JOB_TOTAL_GPUS * 3))}
    # VP=0 # For reference
    CUDA_GRAPH=true
    NUM_LAYERS=118
    HIDDEN_SIZE=8192
else
    echo "MODEL_SIZE: ${MODEL_SIZE} is unsupported."
fi

CONFIG_OVERRIDES=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -gb $GBS \
  -mb $MBS \
  --cuda_graphs $CUDA_GRAPH \
  --num_layers $NUM_LAYERS \
  --hidden_size $HIDDEN_SIZE \
  -ep 1 \
  -fr $FP8_RECIPE \
"

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --enable_nsys"
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    MAX_STEPS=$PROFILE_STOP_STEP
    if [[ $GPU_METRICS_ENABLED == true ]]; then
        CONFIG_OVERRIDES+=" -pgm "
    fi
fi

if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

# Add additional SLURM parameters if provided
SLURM_ARGS=""
if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
    SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
fi

pushd $LLMB_WORKLOAD/NeMo

python3 -m scripts.performance.llm.pretrain_nemotronh_${MODEL_SIZE} \
    --gpu $GPU_TYPE \
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
    --time_limit $TIME_LIMIT \
    $SLURM_ARGS

popd
