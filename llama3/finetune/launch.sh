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

# Check Bash version
if [ "${BASH_VERSION:0:1}" -lt 4 ] || [ "${BASH_VERSION:0:1}" -eq 4 ] && [ "${BASH_VERSION:2:1}" -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

# Required environment variables
: "${SBATCH_ACCOUNT:?Required variable SBATCH_ACCOUNT}"
: "${SBATCH_PARTITION:?Required variable SBATCH_PARTITION}"
: "${HF_TOKEN:?Required variable Hugging Face token}"
: "${GPU_TYPE:?Required variable GPU_TYPE (Eg: gb200, b200, h100)}"
: "${LLMB_INSTALL:?Required variable LLMB_INSTALL}"

export WORKLOAD_TYPE=finetune
export MODEL_NAME=llama3
export FW_VERSION=25.09.00
export GSW_VERSION=25.10

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD

export IMAGE=${IMAGE:-${LLMB_INSTALL}/images/nvidia+nemo+${FW_VERSION}.sqsh}

export NEMORUN_HOME=$LLMB_WORKLOAD
if [ ! -d "$LLMB_WORKLOAD/checkpoint_and_dataset" ]; then
    echo "Error: checkpoint_and_dataset folder not found in $LLMB_WORKLOAD"
    echo "Please ensure you have downloaded both the dataset and checkpoint data. See the README for more details."
    exit 1
fi
export HF_HOME=${HF_HOME:-$LLMB_WORKLOAD/checkpoint_and_dataset}
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD/checkpoint_and_dataset}

#Default values
GPU_TYPE=${GPU_TYPE,,}
DTYPE=${DTYPE:-bf16}
DTYPE=${DTYPE,,}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}
MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

if [[ $DTYPE == "fp8" ]]; then
    export FP8_RECIPE=${FP8_RECIPE:-cs}
    export FP8_RECIPE=${FP8_RECIPE,,}
fi

CONTAINER_MOUNTS=""
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

NUM_LAYERS=${NUM_LAYERS:-80}
HIDDEN_SIZE=${HIDDEN_SIZE:-8192}

if [ $GPU_TYPE = "gb200" ]; then
    #Default launch configs for GB200
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    #Parallelism settings for GB200
    TP=${TP:-1}
    PP=${PP:-4}
    CP=${CP:-1}
    VP=${VP:-20}
    MBS=${MBS:-1}
    GBS=${GBS:-64}
    CUDA_GRAPH=${CUDA_GRAPH:-true}
    TIME_LIMIT=${TIME_LIMIT:-"00:15:00"}
elif [ $GPU_TYPE = "b200" ]; then
    #Default launch config for B200
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    #Parallelism settings for B200
    TP=${TP:-1}
    PP=${PP:-4}
    CP=${CP:-1}
    VP=${VP:-20}
    MBS=${MBS:-1}
    GBS=${GBS:-32}
    CUDA_GRAPH=${CUDA_GRAPH:-false}
    TIME_LIMIT=${TIME_LIMIT:-"00:15:00"}
elif [ $GPU_TYPE = "h100" ]; then
    #Default launch config for H100
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    #Parallelism settings for H100
    TP=${TP:-2}
    PP=${PP:-4}
    CP=${CP:-1}
    VP=${VP:-20}
    MBS=${MBS:-1}
    GBS=${GBS:-32}
    CUDA_GRAPH=${CUDA_GRAPH:-false}
    TIME_LIMIT=${TIME_LIMIT:-"00:15:00"}
else
    echo "$GPU_TYPE not supported"
    exit 1
fi

CONFIG_OVERRIDES=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -vp $VP \
  -ep 1 \
  -gb $GBS \
  -mb $MBS \
  --cuda_graphs $CUDA_GRAPH \
  --num_layers $NUM_LAYERS \
  --hidden_size $HIDDEN_SIZE \
"

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" -en "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    if [[ $GPU_METRICS_ENABLED == true ]]; then
        CONFIG_OVERRIDES+=" -pgm "
    fi
    MAX_STEPS=$PROFILE_STOP_STEP
fi

if [[ $DTYPE == "fp8" ]]; then
    CONFIG_OVERRIDES+=" -fr $FP8_RECIPE "
fi

if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

SCRIPT_NAME="scripts.performance.llm.finetune_llama3_70b"

# Add additional SLURM parameters if provided
SLURM_ARGS=""
if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
    SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
fi

pushd "${LLMB_WORKLOAD}/NeMo"

python3 -m ${SCRIPT_NAME} \
    --hf_token "${HF_TOKEN}" \
    --gpu "${GPU_TYPE}" \
    --num_gpus "${JOB_TOTAL_GPUS}" \
    --container_image "${IMAGE}" \
    --compute_dtype "${DTYPE}" \
    --gpus_per_node "${GPUS_PER_NODE}" \
    --max_steps "${MAX_STEPS}" \
    --finetuning "lora" \
    --skip_dataset_download \
    --skip_import_checkpoint \
    $CONFIG_OVERRIDES \
    slurm \
    --account "${SBATCH_ACCOUNT}" \
    --partition "${SBATCH_PARTITION}" \
    --log_dir "${NEMORUN_HOME}" \
    --time_limit $TIME_LIMIT \
    $SLURM_ARGS

popd
