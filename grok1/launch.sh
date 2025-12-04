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
export MODEL_NAME=grok1
export MODEL_SIZE=314b
export GSW_VERSION=25.10
export FW_VERSION=25.09.00

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.
export HF_TOKEN=${HF_TOKEN?"Required variable HF_TOKEN"}

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD
GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}
DTYPE=${DTYPE:-fp8}
DTYPE=${DTYPE,,}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
ENABLE_GPU_METRICS=${ENABLE_GPU_METRICS:-false}
ENABLE_GPU_METRICS=${ENABLE_GPU_METRICS,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

TIME_LIMIT=${TIME_LIMIT:-"00:35:00"}
MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-false}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT,,}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

CONTAINER_MOUNTS=""
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

if [[ $CLUSTER_TYPE != "slurm" ]]; then
    echo "Only SLURM is supported for this workload"
    exit 1
fi

if [ $GPU_TYPE = "gb200" ] || [ $GPU_TYPE = "gb300" ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    TP=${TP:-4}
    PP=${PP:-1}
    CP=${CP:-1}
    EP=${EP:-8}
    VP=${VP:-1}
    ET=${ET:-4}
    GBS=${GBS:-}
    if [ -z "$GBS" ]; then
        GBS=$((JOB_TOTAL_GPUS * 2))
    fi
    MBS=${MBS:-1}
    CUDA_GRAPH=${CUDA_GRAPH:-true}
elif [ $GPU_TYPE = "b200" ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    TP=${TP:-4}
    PP=${PP:-4}
    CP=${CP:-1}
    EP=${EP:-8}
    VP=${VP:-8}
    ET=${ET:-1}
    GBS=${GBS:-}
    if [ -z "$GBS" ]; then
        GBS=$((JOB_TOTAL_GPUS * 2))
    fi
    MBS=${MBS:-1}
    CUDA_GRAPH=${CUDA_GRAPH:-false}
elif [ $GPU_TYPE = "h100" ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    TP=${TP:-4}
    PP=${PP:-8}
    CP=${CP:-2}
    EP=${EP:-8}
    VP=${VP:-8}
    ET=${ET:-1}
    GBS=${GBS:-}
    if [ -z "$GBS" ]; then
        GBS=$((JOB_TOTAL_GPUS * 2))
    fi
    MBS=${MBS:-1}
    CUDA_GRAPH=${CUDA_GRAPH:-false}
else
    echo "$GPU_TYPE not supported"
    exit 1
fi

NUM_LAYERS=${NUM_LAYERS:-64}
HIDDEN_SIZE=${HIDDEN_SIZE:-6144}

CONFIG_OVERRIDES=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -ep $EP \
  -vp $VP \
  -et $ET \
  -gb $GBS \
  -mb $MBS \
  --cuda_graphs $CUDA_GRAPH \
  --num_layers $NUM_LAYERS \
  --hidden_size $HIDDEN_SIZE \
"

if [ $PROFILE_ENABLED = true ]; then
    CONFIG_OVERRIDES+=" -en "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    if [[ $ENABLE_GPU_METRICS == true ]]; then
        CONFIG_OVERRIDES+=" -pgm "
    fi
    MAX_STEPS=$PROFILE_STOP_STEP
fi

if [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
    MAX_STEPS=1
fi

if [ $CPU_PER_TASK_PINNING -gt 0 ]; then
    CONFIG_OVERRIDES+=" --cpu_pinning $CPU_PER_TASK_PINNING "
fi

if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

if [[ $DTYPE == fp8 ]]; then
    CONFIG_OVERRIDES+=" -fr cs "
fi

pushd $LLMB_WORKLOAD/NeMo

if [ $CLUSTER_TYPE = slurm ]; then
    # Add additional SLURM parameters if provided
    SLURM_ARGS=""
    if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
        SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
    fi

    python -m scripts.performance.llm.pretrain_grok1_314b \
        --gpu $GPU_TYPE \
        --container_image $IMAGE \
        --compute_dtype $DTYPE \
        --num_gpus $JOB_TOTAL_GPUS \
        --gpus_per_node $GPUS_PER_NODE \
        --max_steps $MAX_STEPS \
        --hf_token $HF_TOKEN \
        $CONFIG_OVERRIDES \
        slurm \
        --account $SBATCH_ACCOUNT \
        --partition $SBATCH_PARTITION \
        --time_limit $TIME_LIMIT \
        --log_dir ${NEMORUN_HOME} \
        $SLURM_ARGS
else
    python -m scripts.performance.llm.pretrain_grok1_314b \
        --gpu $GPU_TYPE \
        --container_image nvcr.io/nvidia/nemo:$FW_VERSION \
        --compute_dtype $DTYPE \
        --num_gpus $JOB_TOTAL_GPUS \
        --gpus_per_node $GPUS_PER_NODE \
        --max_steps $MAX_STEPS \
        --hf_token $HF_TOKEN \
        $CONFIG_OVERRIDES \
        runai \
        --base_url $BASE_URL \
        --app_id $APP_ID \
        --app_secret $APP_SECRET \
        --project_name $PROJECT_NAME \
        --pvc_nemo_run_dir $PVC_DIR

fi

popd
