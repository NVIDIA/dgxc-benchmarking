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

# For each dataset a user elects to use, the user is responsible for
# checking if the dataset license is fit for the intended purpose.

# Parameters
#SBATCH --exclusive
#SBATCH --job-name="deepseek_r1_fp4:trtllm-benchmark"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00

set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

if [[ $SLURM_JOB_NUM_NODES -ne 1 ]]; then
   echo "This benchmark only supports a single node -- ${SLURM_JOB_NUM_NODES} nodes requested."
   exit 1
fi

export WORKLOAD_TYPE=inference
export MODEL_NAME=deepseek-r1
export FW_VERSION=1.0.0rc1
export GSW_VERSION=25.07

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/tensorrt-llm+release+${FW_VERSION}.sqsh}

export CONFIG_FILE=$LLMB_WORKLOAD/config.yml
export MODEL_CARD="DeepSeek-R1"
export MODEL_PATH=$LLMB_WORKLOAD/DeepSeek-R1-FP4
export MOUNT_DIR=$LLMB_WORKLOAD

# User defined variables
export TP=${TP:-4}
export EP=${EP:-4}
export PP=${PP:-1}

export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-384}
export MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-6000}
export NUM_REQUESTS=${NUM_REQUESTS:-7680}
export KV_CACHE_FRACTION=${KV_CACHE_FRACTION:-0.85}
export USE_CASES=${USE_CASES:-"reasoning:1000/1000 chat:128/128 summarization:8000/512 generation:512/8000"}
export CONCURRENCY=${CONCURRENCY:--1}

export STREAMING=${STREAMING:-true}
# Conditionally set the streaming flag
streaming_flag=""
streaming_log="_streaming-off"
if [ "$STREAMING" = true ]; then
    streaming_flag="--streaming"
    streaming_log='_streaming-on'
fi

# Loop over each use case
for value in $USE_CASES; do

   use_case=$(echo "$value" | cut -d':' -f1)
   ISL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f1)
   OSL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f2)

   LOG_NAME=${MODEL_CARD}_TP${TP}_EP${EP}_PP${PP}_CON${CONCURRENCY}_${use_case}
   
   export RESULT_DIR=$LLMB_WORKLOAD/experiments/$LOG_NAME
   export RESULT_FILES_NAME=${LOG_NAME}${streaming_log}
   export DATASET_FILE=$LLMB_WORKLOAD/dataset_${use_case}_${ISL}_${OSL}.txt
   echo "Now Benchmarking: $use_case : $ISL / $OSL using $DATASET_FILE"
   
   #launch trt-llm benchmark
   CMD="trtllm-llmapi-launch trtllm-bench -m ${MODEL_CARD} \
      --model_path ${MODEL_PATH} throughput \
      --tp $TP \
      --ep $EP \
      --pp $PP \
      --warmup 0 \
      --dataset ${DATASET_FILE} \
      --backend pytorch \
      --max_batch_size ${MAX_BATCH_SIZE} \
      --max_num_tokens ${MAX_NUM_TOKENS} \
      --extra_llm_api_options ${CONFIG_FILE} \
      --num_requests ${NUM_REQUESTS} \
      --kv_cache_free_gpu_mem_fraction ${KV_CACHE_FRACTION} \
      --concurrency ${CONCURRENCY} \
      $streaming_flag"

   echo "Launching srun command with:"
   echo "$CMD"

   export SLURM_MPI_TYPE="pmix"
   export SRUN_OUTPUT=${RESULT_DIR}/${RESULT_FILES_NAME}_%j.out
   export SRUN_ERROR=${RESULT_DIR}/${RESULT_FILES_NAME}_%j.err

   srun --container-image "$IMAGE" \
     --container-mounts "$MOUNT_DIR" \
     --container-writable \
     --no-container-mount-home bash -c "$CMD"

   echo "Results of Benchmark: $SRUN_OUTPUT"
   echo "Error Log of Benchmark: $SRUN_ERROR"

done
