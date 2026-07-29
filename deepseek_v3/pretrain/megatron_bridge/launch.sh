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

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=deepseek-v3
export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
DTYPE=${DTYPE:-bf16}
DTYPE=${DTYPE,,}

FW_VERSION=26.04.01

if [[ $DTYPE == "fp8" ]]; then
    if [[ $GPU_TYPE == "h100" ]]; then
        FP8_RECIPE="sc"
    else
        FP8_RECIPE="mx"
    fi
    COMPUTE_TYPE=${DTYPE}_${FP8_RECIPE}
else
    COMPUTE_TYPE=${DTYPE}
fi

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
PYTORCH_PROFILE_ENABLED=${ENABLE_PYTORCH_PROFILE:-false}
PYTORCH_PROFILE_ENABLED=${PYTORCH_PROFILE_ENABLED,,}
PROFILE_START_STEP=${PROFILE_START_STEP:-45}
PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING:-false}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING,,}
MAX_STEPS=${MAX_STEPS:-50}
if [[ $GPU_TYPE == "h100" ]]; then
    TIME_LIMIT=${TIME_LIMIT:-"01:30:00"}
else
    TIME_LIMIT=${TIME_LIMIT:-"00:45:00"}
fi

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

# Add additional SLURM parameters if provided
SLURM_ARGS=""
if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
    SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
fi

export HF_HOME="$LLMB_INSTALL/.cache/huggingface"
CONTAINER_MOUNTS="$HF_HOME"
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-}"
if [[ -n ${CONFIG_OVERRIDES} ]]; then
    CONFIG_OVERRIDES+=" "
fi
if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

if [[ $PROFILE_ENABLED == "true" ]] && [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    echo "Error: ENABLE_PROFILE and ENABLE_PYTORCH_PROFILE are mutually exclusive." >&2
    exit 1
fi

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --enable_nsys "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    PROFILE_RANKS=$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)))
    CONFIG_OVERRIDES+=" --profiling_ranks=$PROFILE_RANKS"
    CONFIG_OVERRIDES+=" --nsys_trace=cuda "
    CONFIG_OVERRIDES+=" --nsys_extra_args=--nvtx-domain-include=NCCL "
    if [[ $GPU_METRICS_ENABLED == true ]]; then
        CONFIG_OVERRIDES+=" --profiling_gpu_metrics "
    fi
fi

if [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --pytorch_profiler true "
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

CONFIG_OVERRIDES+=" --enable_pct_binding $ENABLE_PCT_BINDING "

if [[ $GPU_TYPE == "gb300" ]] || [[ $GPU_TYPE == "gb200" ]]; then
    if [[ $GPU_TYPE == "gb300" ]] && [[ $JOB_TOTAL_GPUS -eq 128 ]]; then
        CONFIG_OVERRIDES+=" -pp 4 "
        CONFIG_OVERRIDES+=" -vp 4 "
        CONFIG_OVERRIDES+=" -ep 32 "
        CONFIG_OVERRIDES+=" --recompute_modules=mla_up_proj "
    fi
    if [[ $GPU_TYPE == "gb300" ]] && ((JOB_TOTAL_GPUS % 72 == 0)); then
        CONFIG_OVERRIDES+=" -tp 1 "
        CONFIG_OVERRIDES+=" -pp 1 "
        CONFIG_OVERRIDES+=" -vp None "
        CONFIG_OVERRIDES+=" -ep 8 "
        CONFIG_OVERRIDES+=" --hidden_size 1024 "
        CONFIG_OVERRIDES+=" -mb 2 "
        if [[ $COMPUTE_TYPE == "fp8" ]]; then
            CONFIG_OVERRIDES+=" --cuda_graph_impl=transformer_engine "
            CONFIG_OVERRIDES+=" --cuda_graph_scope=attn,moe_router,moe_preprocess "
            CONFIG_OVERRIDES+=" --recompute_modules=moe_act "
        fi
    elif [[ $JOB_TOTAL_GPUS -le 64 ]]; then # proxy workloads
        CONFIG_OVERRIDES+=" -tp 1 "
        CONFIG_OVERRIDES+=" -pp 1 "
        CONFIG_OVERRIDES+=" -vp None "
        CONFIG_OVERRIDES+=" -ep $JOB_TOTAL_GPUS "
        CONFIG_OVERRIDES+=" --hidden_size 1024 "
        if [[ $GPU_TYPE == "gb200" ]] && [[ $COMPUTE_TYPE == "fp8_mx" ]]; then
            CONFIG_OVERRIDES+=" --recompute_modules=mla_up_proj "
        fi
        if [[ $JOB_TOTAL_GPUS -le 8 ]]; then
            CONFIG_OVERRIDES+=" --num_layers 24 "
            CONFIG_OVERRIDES+=" -mb 4 "
        else
            CONFIG_OVERRIDES+=" -mb 2 "
        fi
    fi
    GPUS_PER_NODE=4
elif [[ $GPU_TYPE == "b300" ]] || [[ $GPU_TYPE == "b200" ]] || [[ $GPU_TYPE == "h100" ]]; then
    if [[ $GPU_TYPE == "b300" ]] && [[ $COMPUTE_TYPE == "bf16" ]] && [[ $JOB_TOTAL_GPUS -eq 128 ]]; then
        CONFIG_OVERRIDES+=" --cuda_graph_impl none "
    fi
    if [[ $GPU_TYPE == "b200" ]]; then
        if [[ $COMPUTE_TYPE == "bf16" ]]; then
            CONFIG_OVERRIDES+=" -pp 8 "
        elif [[ $COMPUTE_TYPE == "fp8_mx" ]]; then
            CONFIG_OVERRIDES+=" --moe_a2a_overlap=False "
        fi
    fi
    GPUS_PER_NODE=8
fi

# run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python3 scripts/performance/setup_experiment.py \
    --container_image $IMAGE \
    --compute_dtype $COMPUTE_TYPE \
    --gpu $GPU_TYPE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --offline \
    --model_family_name deepseek \
    --model_recipe_name deepseek_v3 \
    ${CONFIG_OVERRIDES} \
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT \
    --max_steps $MAX_STEPS \
    --packager none \
    $SLURM_ARGS

popd
