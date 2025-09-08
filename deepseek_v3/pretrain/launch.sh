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
export MODEL_NAME=deepseek-v3
export FW_VERSION=25.07.01
export GSW_VERSION=25.08

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

# Ensure STAGE_PATH is not set as it's been replaced by LLMB_INSTALL
if [ -n "${STAGE_PATH+x}" ]; then
    echo "Error: STAGE_PATH is deprecated and should not be set. Please use LLMB_INSTALL instead."
    exit 1
fi

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

export NEMORUN_HOME=$LLMB_WORKLOAD
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD}
export HF_HOME=${HF_HOME:-$LLMB_WORKLOAD}
export HF_TOKEN=${HF_TOKEN?"Required variable HF_TOKEN"}

export MAX_STEPS=${MAX_STEPS:-50}
export DTYPE=${DTYPE:-bf16}
export DTYPE=${DTYPE,,}
export GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
export GPU_TYPE=${GPU_TYPE,,}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}
export TIME_LIMIT=${TIME_LIMIT:-"00:55:00"}

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}

export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

if [[ $DTYPE == "fp8" ]]; then
    export FP8_RECIPE=${FP8_RECIPE:-ds}
    export FP8_RECIPE=${FP8_RECIPE,,}

    if [[ $FP8_RECIPE == "mxfp8" && ($GPU_TYPE == "gb200" || $GPU_TYPE == "b200") && $JOB_TOTAL_GPUS -eq 128 ]]; then
        echo "Error: $FP8_RECIPE is not supported for $GPU_TYPE with $JOB_TOTAL_GPUS GPUs."
        exit 1
    fi
fi

CONTAINER_MOUNTS=""
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

CONFIG_OVERRIDES=""

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" -en "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    MAX_STEPS=$PROFILE_STOP_STEP
    if [[ $GPU_METRICS_ENABLED == true ]]; then
        CONFIG_OVERRIDES+=" -pgm "
    fi
fi

if [[ $GPU_TYPE == "h100" ]]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    TP=1
    EP=64
    PP=8
    VP=2
    RM="mla_up_proj mlp moe"
    if [[ $DTYPE == "fp8" && ($FP8_RECIPE == "ss" || $JOB_TOTAL_GPUS -eq 512) ]]; then
        TP=2
    fi
elif [[ $GPU_TYPE == "gb200" ]]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    TP=1
    EP=32
    if [[ $JOB_TOTAL_GPUS -eq 128 ]]; then
        PP=4
        VP=1
    else
        PP=8
        VP=2
    fi
    RM="mla_up_proj"
elif [[ $GPU_TYPE == "b200" ]]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    TP=1
    EP=8
    PP=16
    VP=1
    RM="mla_up_proj"
fi

CP=1
ETP=1
AOL=0
MBS=1
GBS=$((JOB_TOTAL_GPUS * 8))
CG=false
NUM_LAYERS=61
HIDDEN_SIZE=7168

CONFIG_OVERRIDES+=" -tp $TP \
  -ep $EP \
  -pp $PP \
  -vp $VP \
  -cp $CP \
  -et $ETP \
  -ol $AOL \
  -mb $MBS \
  -gb $GBS \
  -rm $RM \
  --cuda_graphs $CG \
  --num_layers $NUM_LAYERS \
  --hidden_size $HIDDEN_SIZE \
"

if [[ $DTYPE == "fp8" ]]; then
    CONFIG_OVERRIDES+=" -fr $FP8_RECIPE "
fi

if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

#run command
pushd $LLMB_WORKLOAD/NeMo

python3 -m scripts.performance.llm.pretrain_deepseek_v3 \
    --container_image $IMAGE \
    --compute_dtype $DTYPE \
    --hf_token $HF_TOKEN \
    --gpu $GPU_TYPE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --max_steps $MAX_STEPS \
    ${CONFIG_OVERRIDES} \
    slurm \
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT

popd
