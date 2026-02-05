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

# Parameters
#SBATCH --job-name="deepseek_v3:torchtitan_launch"
#SBATCH --time=02:00:00

set -eu -o pipefail

if [ "${BASH_VERSION%%.*}" -lt 4 ] || { [ "${BASH_VERSION%%.*}" -eq 4 ] && [ "${BASH_VERSION#*.}" -lt 2 ]; }; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
export GPU_TYPE=${GPU_TYPE,,}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

export DTYPE=${DTYPE:-bf16}
export DTYPE=${DTYPE,,}
if [[ $DTYPE != "bf16" ]]; then
    echo "❌ Error: FP8 is not supported for this TorchTitan DeepSeek-V3 recipe yet (WIP). Supported DTYPE: bf16. Got DTYPE='$DTYPE'."
    exit 1
fi

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=deepseek-v3-torchtitan
export MODEL_SIZE=671b
export FW_VERSION=25.10-py3

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD
export TORCHTITAN_HOME=$LLMB_WORKLOAD/torchtitan

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+pytorch+${FW_VERSION}.sqsh}

if [[ $GPU_TYPE == "h100" ]]; then
    export LOG_RANK=${LOG_RANK:-"448"}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    export DATA_PARALLEL_SHARD_DEGREE=${DATA_PARALLEL_SHARD_DEGREE:-64}
    export EXPERT_PARALLEL_DEGREE=${EXPERT_PARALLEL_DEGREE:-32}
    export PIPELINE_PARALLEL_DEGREE=${PIPELINE_PARALLEL_DEGREE:-8}
elif [[ $GPU_TYPE == "b200" ]]; then
    export LOG_RANK=${LOG_RANK:-"224"}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    export DATA_PARALLEL_SHARD_DEGREE=${DATA_PARALLEL_SHARD_DEGREE:-32}
    export EXPERT_PARALLEL_DEGREE=${EXPERT_PARALLEL_DEGREE:-32}
    export PIPELINE_PARALLEL_DEGREE=${PIPELINE_PARALLEL_DEGREE:-8}
elif [[ $GPU_TYPE == "gb200" ]]; then
    export LOG_RANK=${LOG_RANK:-"224"}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    export DATA_PARALLEL_SHARD_DEGREE=${DATA_PARALLEL_SHARD_DEGREE:-32}
    export EXPERT_PARALLEL_DEGREE=${EXPERT_PARALLEL_DEGREE:-32}
    export PIPELINE_PARALLEL_DEGREE=${PIPELINE_PARALLEL_DEGREE:-8}
else
    echo "❌ Error: Torchtitan recipes only supports h100, b200, and gb200 GPU types, got '$GPU_TYPE'"
    exit 1
fi

# Directory setup: Use launcher-provided directory if available, otherwise create our own
if [[ -n ${LLMB_EXPERIMENT_DIR:-} ]]; then
    # Managed mode (via llmb-run configured_sbatch): use pre-created directory
    export LLMB_RUN_DIR=$LLMB_EXPERIMENT_DIR
else
    # Standalone mode: create two-level directory structure to match launcher pattern
    # This keeps fan-out constrained by grouping runs under a descriptor directory
    DESC="${MODEL_NAME}_${MODEL_SIZE}_${DTYPE}_gpus${JOB_TOTAL_GPUS}"
    export LLMB_RUN_DIR=$LLMB_WORKLOAD/experiments/${DESC}/job_${SLURM_JOB_ID}
    mkdir -p "$LLMB_RUN_DIR"
fi

# All outputs go into the run directory
export SLURM_LOG_DIR=$LLMB_RUN_DIR
export LLMB_OUTPUT_DIR=$LLMB_RUN_DIR/outputs
mkdir -p "$LLMB_OUTPUT_DIR"

# Default training parameters - can be overridden via environment variables
export DATASET_PATH=${DATASET_PATH:-$LLMB_INSTALL/datasets/c4}
export SEQ_LEN=${SEQ_LEN:-4096}
export TRAINING_STEPS=${TRAINING_STEPS:-60}
export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-16}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}
ADDITIONAL_SRUN_ARGS=""
if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
    # Parse semicolon-separated params: key=value pairs or standalone flags
    IFS=';' read -ra PARAMS <<< "$ADDITIONAL_SLURM_PARAMS"
    for param in "${PARAMS[@]}"; do
        param=$(echo "$param" | xargs) # Trim whitespace
        if [ -n "$param" ]; then
            if [[ $param == *"="* ]]; then
                # Key=value pair
                key=$(echo "$param" | cut -d'=' -f1 | xargs)
                value=$(echo "$param" | cut -d'=' -f2- | xargs)
                if [ -n "$key" ] && [ -n "$value" ]; then
                    ADDITIONAL_SRUN_ARGS+=" --${key}=${value}"
                fi
            else
                # Standalone flag (no value)
                ADDITIONAL_SRUN_ARGS+=" --${param}"
            fi
        fi
    done
fi

CONTAINER_MOUNTS="$TORCHTITAN_HOME:$TORCHTITAN_HOME:rw,$SLURM_LOG_DIR:$SLURM_LOG_DIR:rw,$LLMB_OUTPUT_DIR:$LLMB_OUTPUT_DIR:rw,$LLMB_REPO:$LLMB_REPO:ro,$DATASET_PATH:$DATASET_PATH:ro"
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

# Build the training command - use PYTHONPATH instead of activating venv
export ENABLE_PROFILE=${ENABLE_PROFILE:-false}
PROFILE_ARGS=""
if [[ ${ENABLE_PROFILE,,} == "true" ]]; then
    PROFILE_TRACE_DIR="$LLMB_OUTPUT_DIR/torchtitan_${MODEL_NAME}_${JOB_TOTAL_GPUS}gpus_${SLURM_JOB_ID}_profile_trace"
    mkdir -p "$PROFILE_TRACE_DIR"
    PROFILE_ARGS=" --profiling.enable_profiling --profiling.save_traces_folder=$PROFILE_TRACE_DIR"
fi

TRAIN_CMD="\
cd $TORCHTITAN_HOME; \
ulimit -c 0; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
export ONE_LOGGER_JOB_CATEGORY=production; \
export PYTHONPATH=$TORCHTITAN_HOME:\$PYTHONPATH; \
LOCAL_RANK=\$SLURM_LOCALID; \
python -m torchtitan.train \
--job.config_file=$LLMB_REPO/deepseek_v3_671b.toml \
--job.dump_folder=$LLMB_OUTPUT_DIR \
--training.dataset_path=$DATASET_PATH \
--training.steps=$TRAINING_STEPS \
--training.seq_len=$SEQ_LEN \
--training.debug_moe_force_load_balance \
--parallelism.data_parallel_shard_degree=$DATA_PARALLEL_SHARD_DEGREE \
--parallelism.expert_parallel_degree=$EXPERT_PARALLEL_DEGREE \
--parallelism.pipeline_parallel_degree=$PIPELINE_PARALLEL_DEGREE \
--training.local_batch_size=$LOCAL_BATCH_SIZE$PROFILE_ARGS"

srun --container-image="$IMAGE" \
    --container-name=deepseek-v3-torchtitan \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-env="LOG_RANK" \
    --output $SLURM_LOG_DIR/log-torchtitan_${MODEL_NAME}_${JOB_TOTAL_GPUS}gpus_%j_${SLURM_RESTART_COUNT:-0}.out \
    --no-container-mount-home \
    --container-writable \
    $ADDITIONAL_SRUN_ARGS \
    bash -c "$TRAIN_CMD"
