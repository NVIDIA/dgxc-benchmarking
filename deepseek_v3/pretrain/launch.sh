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
export FW_VERSION=25.09.00
export GSW_VERSION=25.10

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

export NEMORUN_HOME=$LLMB_WORKLOAD
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD}

export DTYPE=${DTYPE:-bf16}
export DTYPE=${DTYPE,,}
export GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
export GPU_TYPE=${GPU_TYPE,,}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}
export TIME_LIMIT=${TIME_LIMIT:-"00:40:00"}

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

# Add additional SLURM parameters if provided
SLURM_ARGS=""
if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
    SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
fi

CONTAINER_MOUNTS=""
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

CONFIG_OVERRIDES=""
if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --enable_nsys "
    if [[ $GPU_METRICS_ENABLED == true ]]; then
        CONFIG_OVERRIDES+=" -pgm "
    fi
fi

if [[ $DTYPE == "fp8" ]]; then
    export FP8_RECIPE=${FP8_RECIPE:-cs}
    export FP8_RECIPE=${FP8_RECIPE,,}
    CONFIG_OVERRIDES+=" --fp8_recipe $FP8_RECIPE "
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

if [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "gb300" ]]; then
    GPUS_PER_NODE=4
else
    GPUS_PER_NODE=8
fi

if [[ $GPU_TYPE == "h100" ]] && [[ $JOB_TOTAL_GPUS -eq 512 ]] && [[ $DTYPE == "fp8" ]] && [[ $FP8_RECIPE == "ss" ]]; then
    echo "FP8-SS requires a minimum of 1024 GPUs on H100."
    exit 1
fi

if [[ $DTYPE == "fp8" ]]; then
    if [[ $FP8_RECIPE == "ss" && $GPU_TYPE != "h100" ]]; then
        echo "FP8-SS is only supported on H100."
        exit 1
    elif [[ $FP8_RECIPE == "mx" && $GPU_TYPE == "h100" ]]; then
        echo "FP8-MX is only supported for B200, GB200 and GB300 and not supported on H100."
        exit 1
    fi
fi

#run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python3 -m scripts.performance.setup_experiment \
    --container_image $IMAGE \
    --compute_dtype $DTYPE \
    --gpu $GPU_TYPE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --model_name deepseek \
    --model_size v3 \
    ${CONFIG_OVERRIDES} \
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT \
    $SLURM_ARGS

popd
