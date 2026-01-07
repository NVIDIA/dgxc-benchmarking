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
export MODEL_NAME=nemotron4-15b
export FW_VERSION=25.09.00 # Nemo Version

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}
CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}
CLUSTER_TYPE=${CLUSTER_TYPE,,}
DTYPE=${DTYPE:-fp8}
DTYPE=${DTYPE,,}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
ENABLE_GPU_METRICS=${ENABLE_GPU_METRICS:-false}
ENABLE_GPU_METRICS=${ENABLE_GPU_METRICS,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}
STRONG_SCALING=${STRONG_SCALING:-false}
STRONG_SCALING=${STRONG_SCALING,,}
MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-false}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT,,}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

# Mount Hugging Face cache for tokenizers
export HF_HOME="$LLMB_INSTALL/.cache/huggingface"

# For Slurm, mount HF_HOME as a path; for Run:AI, use HF_TOKEN instead
if [[ $CLUSTER_TYPE == "slurm" ]]; then
    CONTAINER_MOUNTS="$HF_HOME"
else
    CONTAINER_MOUNTS=""
fi

if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

if [[ $STRONG_SCALING == true ]]; then
    echo "Strong scaling is only supported with model size of 340b. Current model size is 15b"
    exit 1
fi

export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

if [ $GPU_TYPE = gb200 ] || [ $GPU_TYPE = b200 ] || [ $GPU_TYPE = gb300 ]; then
    if [ $GPU_TYPE = b200 ]; then
        GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    else
        GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    fi

    TP=1
    PP=1
    CP=1
    GBS=${GBS:-$((JOB_TOTAL_GPUS * 4))}
    MBS=2
    VP=1
    NUM_LAYERS=32
    HIDDEN_SIZE=6144
    CUDA_GRAPH=${CUDA_GRAPH:-true}
elif [ $GPU_TYPE = h100 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    TP=2
    PP=1
    CP=1
    GBS=${GBS:-$((JOB_TOTAL_GPUS * 4))}
    MBS=2
    VP=0
    NUM_LAYERS=32
    HIDDEN_SIZE=6144
    CUDA_GRAPH=${CUDA_GRAPH:-false}
else
    echo "${GPU_TYPE} is unsupported for this workload."
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

CONFIG_OVERRIDES+=" -fsdp 0 --cuda_graphs $CUDA_GRAPH "

if [ $VP != 0 ]; then
    CONFIG_OVERRIDES+=" -vp $VP "
fi

if [ $PROFILE_ENABLED = true ]; then
    CONFIG_OVERRIDES+=" -en "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    if [[ $ENABLE_GPU_METRICS == true ]]; then
        CONFIG_OVERRIDES+=" -pgm "
    fi
    MAX_STEPS=$PROFILE_STOP_STEP
fi

if [[ $ENABLE_CHECKPOINT == true ]]; then
    CONFIG_OVERRIDES+=" --checkpoint_save=True "
else
    CONFIG_OVERRIDES+=" --checkpoint_save=False "
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

if [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
    MAX_STEPS=1
    CONFIG_OVERRIDES+=" --checkpoint_load_path=$LOAD_CHECKPOINT_PATH "
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${LOAD_CHECKPOINT_PATH}"
fi

# Add custom mounts for both Slurm and Run:AI
if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

if [[ $DTYPE == fp8 ]]; then
    CONFIG_OVERRIDES+=" -fr cs "
fi

pushd $LLMB_WORKLOAD/NeMo

if [ $CLUSTER_TYPE = slurm ]; then
    # After all overrides - STRONG_SCALING is significantly slower at 128 than weak scaling.
    TIME_LIMIT=${TIME_LIMIT:-"00:25:00"}

    # Add additional SLURM parameters if provided
    SLURM_ARGS=""
    if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
        SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
    fi

    python3 -m scripts.performance.llm.pretrain_nemotron4_15b \
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
else
    python3 -m scripts.performance.llm.pretrain_nemotron4_15b \
        --gpu $GPU_TYPE \
        --container_image nvcr.io/nvidia/nemo:$FW_VERSION \
        --compute_dtype $DTYPE \
        --num_gpus $JOB_TOTAL_GPUS \
        --gpus_per_node $GPUS_PER_NODE \
        --max_steps $MAX_STEPS \
        --hf_token ${HF_TOKEN:?HF_TOKEN is required} \
        $CONFIG_OVERRIDES \
        runai \
        --base_url $BASE_URL \
        --app_id $APP_ID \
        --app_secret $APP_SECRET \
        --project_name $PROJECT_NAME \
        --pvc_nemo_run_dir $PVC_DIR
fi

popd
