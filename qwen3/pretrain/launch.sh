#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#Required environment variables
: "${LLMB_INSTALL:?Required variable LLMB_INSTALL}"

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=qwen3
export MODEL_SIZE=${MODEL_SIZE:-235b}
export MODEL_SIZE=${MODEL_SIZE,,}

# Map short model size to full model size for --model_size argument
if [ $MODEL_SIZE = "235b" ]; then
    MODEL_SIZE_FULL="235b_a22b"
elif [ $MODEL_SIZE = "30b" ]; then
    MODEL_SIZE_FULL="30b_a3b"
else
    echo "Unsupported MODEL_SIZE: $MODEL_SIZE"
    exit 1
fi

export FW_VERSION=25.11.01

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
export MAX_STEPS=${MAX_STEPS:-50}
export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

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

# Mount Hugging Face cache for tokenizers
export HF_HOME="$LLMB_INSTALL/.cache/huggingface"
CONTAINER_MOUNTS="$HF_HOME"
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

CUDA_GRAPH=${CUDA_GRAPH:-""}

if [ $GPU_TYPE = "gb300" ]; then
    if [ $MODEL_SIZE = "30b" ]; then
        #Parallelism settings for GB300
        TP=${TP:-1}
        PP=${PP:-1}
        CP=${CP:-1}
        VP=${VP:-1}
        EP=${EP:-8}
        ETP=${ETP:-1}
        MBS=${MBS:-8}
        GBS=${GBS:-$((JOB_TOTAL_GPUS * 64))}
        CUDA_GRAPH=${CUDA_GRAPH:-'--cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess'}
    elif [ $MODEL_SIZE = "235b" ]; then
        TP=${TP:-1}
        PP=${PP:-1}
        CP=${CP:-1}
        VP=${VP:-1}
        EP=${EP:-64}
        ETP=${ETP:-1}
        MBS=${MBS:-2}
        GBS=${GBS:-$((JOB_TOTAL_GPUS * 16))}
        CUDA_GRAPH=${CUDA_GRAPH:-'--cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess'}
    fi
elif [ $GPU_TYPE = "gb200" ]; then
    if [ $MODEL_SIZE = "30b" ]; then
        #Parallelism settings for GB200
        TP=${TP:-1}
        PP=${PP:-1}
        CP=${CP:-1}
        VP=${VP:-1}
        EP=${EP:-8}
        ETP=${ETP:-1}
        MBS=${MBS:-4}
        GBS=${GBS:-$((JOB_TOTAL_GPUS * 64))}
        if [ $DTYPE = "bf16" ]; then
            CUDA_GRAPH=${CUDA_GRAPH:-'--cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess,attn'}
        else
            CUDA_GRAPH=${CUDA_GRAPH:-'--cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess'}
        fi
    elif [ $MODEL_SIZE = "235b" ]; then
        TP=${TP:-1}
        PP=${PP:-8}
        CP=${CP:-1}
        VP=${VP:-1}
        EP=${EP:-8}
        ETP=${ETP:-1}
        MBS=${MBS:-1}
        GBS=${GBS:-$((JOB_TOTAL_GPUS * 16))}
        CUDA_GRAPH=${CUDA_GRAPH:-'--cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess,attn'}
    fi
elif [ $GPU_TYPE = "b200" ]; then
    if [ $MODEL_SIZE = "30b" ]; then
        #Parallelism settings for B200
        TP=${TP:-1}
        PP=${PP:-1}
        CP=${CP:-1}
        VP=${VP:-1}
        EP=${EP:-8}
        ETP=${ETP:-1}
        MBS=${MBS:-1}
        GBS=${GBS:-$((JOB_TOTAL_GPUS * 64))}
        CUDA_GRAPH=${CUDA_GRAPH:-'--cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess'}
    elif [ $MODEL_SIZE = "235b" ]; then
        TP=${TP:-1}
        PP=${PP:-8}
        CP=${CP:-1}
        VP=${VP:-4}
        EP=${EP:-8}
        ETP=${ETP:-1}
        MBS=${MBS:-1}
        GBS=${GBS:-$((JOB_TOTAL_GPUS * 16))}
    fi
elif [ $GPU_TYPE = "h100" ]; then
    if [ $MODEL_SIZE = "30b" ]; then
        #Parallelism settings for H100
        TP=${TP:-1}
        PP=${PP:-2}
        CP=${CP:-1}
        VP=${VP:-12}
        EP=${EP:-8}
        ETP=${ETP:-1}
        MBS=${MBS:-1}
        GBS=${GBS:-$((JOB_TOTAL_GPUS * 32))}
        if [ $DTYPE = "bf16" ]; then
            CUDA_GRAPH=${CUDA_GRAPH:-'--cuda_graph_impl=transformer_engine --cuda_graph_scope=moe_router,moe_preprocess'}
        fi
    elif [ $MODEL_SIZE = "235b" ]; then
        TP=${TP:-2}
        PP=${PP:-8}
        CP=${CP:-1}
        VP=${VP:-4}
        EP=${EP:-32}
        ETP=${ETP:-1}
        MBS=${MBS:-1}
        GBS=${GBS:-$((JOB_TOTAL_GPUS * 8))}
    fi
else
    echo "$GPU_TYPE not supported"
    exit 1
fi

CONFIG_OVERRIDES+=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -ep $EP \
  -et $ETP \
  -gb $GBS \
  -mb $MBS \
  $CUDA_GRAPH \
"

# Only add -vp if VP is greater than 1
if [ "$VP" -gt "1" ]; then
    CONFIG_OVERRIDES+=" -vp $VP "
fi

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" -en "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    if [[ $GPU_METRICS_ENABLED == true ]]; then
        CONFIG_OVERRIDES+=" --profiling_gpu_metrics "
    fi
    MAX_STEPS=$PROFILE_STOP_STEP
fi

if [[ $DTYPE == "fp8" ]]; then
    if [[ $GPU_TYPE == "h100" ]]; then
        export FP8_RECIPE=${FP8_RECIPE:-fp8_cs}
    else
        export FP8_RECIPE=${FP8_RECIPE:-fp8_mx}
    fi
    export FP8_RECIPE=${FP8_RECIPE,,}
    COMPUTE_DTYPE=$FP8_RECIPE
else
    COMPUTE_DTYPE=$DTYPE
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

if [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "gb300" ]]; then
    GPUS_PER_NODE=4
else
    GPUS_PER_NODE=8
fi

#run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python3 scripts/performance/setup_experiment.py \
    --container_image $IMAGE \
    --compute_dtype $COMPUTE_DTYPE \
    --gpu $GPU_TYPE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --model_name qwen3 \
    --model_size ${MODEL_SIZE_FULL} \
    ${CONFIG_OVERRIDES} \
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT \
    --max_steps $MAX_STEPS \
    --hf_token ${HF_TOKEN:?HF_TOKEN is required} \
    $SLURM_ARGS

popd
