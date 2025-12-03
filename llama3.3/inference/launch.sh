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
#SBATCH --job-name="llama3.3-70b_fp4:trtllm-benchmark"
#SBATCH --time=00:30:00

set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=inference
export MODEL_NAME=llama3.3
export FW_VERSION=1.1.0rc5
export GSW_VERSION=25.10

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/tensorrt-llm+release+${FW_VERSION}.sqsh}

export MODE=${MODE:-"max_throughput"}
# Validate mode
if [[ $MODE != "min_latency" && $MODE != "max_throughput" ]]; then
    echo "<E2><9D><8C> Error: Invalid mode '$MODE'"
    echo "<E2><9C><85> Valid options: min_latency or max_throughput"
    exit 1
fi

export MODEL_CARD="Llama3.3-70B"
export MOUNT_DIR=$LLMB_WORKLOAD
export GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
export GPU_TYPE=${GPU_TYPE,,}

# User defined variables
export EP=1 # No expert parallel support in this model, so override to 1
export NUM_REQUESTS=${NUM_REQUESTS:-4096}
export USE_CASES=${USE_CASES:-"reasoning:1000/1000 chat:128/128 summarization:8000/512 generation:512/8000"}
export MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-2048}

if [[ $MODE == "max_throughput" ]]; then
    if [[ $GPU_TYPE == "h100" ]]; then
        export MODEL_PATH=$LLMB_WORKLOAD/Llama-3.3-70B-Instruct-FP8
        export TP=${TP:-2}
        export PP=${PP:-1}
        export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-256}
        export NUM_REQUESTS=${NUM_REQUESTS:-2000}
        export KV_CACHE_FRACTION=${KV_CACHE_FRACTION:-0.95}
        export CONCURRENCY=${CONCURRENCY:-128}
    elif [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "b200" ]]; then
        export MODEL_PATH=$LLMB_WORKLOAD/Llama-3.3-70B-Instruct-FP4
        export TP=${TP:-1}
        export PP=${PP:-1}
        export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-640}
        export NUM_REQUESTS=${NUM_REQUESTS:-4000}
        export KV_CACHE_FRACTION=${KV_CACHE_FRACTION:-0.95}
        export CONCURRENCY=${CONCURRENCY:-640}
    fi
elif [[ $MODE == "min_latency" ]]; then
    if [[ $GPU_TYPE == "gb200" ]]; then
        export MODEL_PATH=$LLMB_WORKLOAD/Llama-3.3-70B-Instruct-FP4
        export TP=${TP:-4}
        export PP=${PP:-1}
        export MAX_BATCH_SIZE=1
        export CONCURRENCY=1
        export KV_CACHE_FRACTION=${KV_CACHE_FRACTION:-0.75}
        export NUM_REQUESTS=20
    else
        echo "❌ Error: Min latency mode only supports gb200 GPU type, got '$GPU_TYPE'"
        exit 1

    fi
fi

export STREAMING=${STREAMING:-true}
# Conditionally set the streaming flag
streaming_flag=""
streaming_log="_streaming-off"
if [ "$STREAMING" = true ]; then
    streaming_flag="--streaming"
    streaming_log='_streaming-on'
fi

# Using MAX_BATCH_SIZE generate cuda graph batch sizes
generate_powers_of_2() {
    local number=$1
    local power=1
    local powers_of_2=()

    while [ $power -le $number ]; do
        powers_of_2+=("$power")
        power=$((power * 2))
    done
    # Add last number to the list
    if [ ${powers_of_2[-1]} -ne $number ]; then
        powers_of_2+=("$number")
    fi

    local joined_list=$(
        IFS=,
        echo "[${powers_of_2[*]}]"
    )
    echo "$joined_list"
}

export cuda_batch_sizes=$(generate_powers_of_2 $MAX_BATCH_SIZE)

# Generate config file for max_throughput:
pushd $LLMB_WORKLOAD

EPOCH_MS=$(date +%s%3N)

if [[ $MODE == "max_throughput" ]]; then
    cat << EOF > config_${EPOCH_MS}.yml
enable_attention_dp: false
cuda_graph_config:
  enable_padding: true
  batch_sizes: $cuda_batch_sizes
print_iter_log: true
kv_cache_config:
  free_gpu_memory_fraction: $KV_CACHE_FRACTION
stream_interval: 4
EOF
elif [[ $MODE == "min_latency" ]]; then
    cat << EOF > config_${EPOCH_MS}.yml
enable_attention_dp: false
cuda_graph_config:
  max_batch_size: 4
kv_cache_config:
  enable_block_reuse: false
print_iter_log: false
enable_autotuner: false
enable_min_latency: true
EOF
fi

# Loop over each use case
for value in $USE_CASES; do
    export CONFIG_FILE=$LLMB_WORKLOAD/config_${EPOCH_MS}.yml
    use_case=$(echo "$value" | cut -d':' -f1)
    ISL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f1)
    OSL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f2)

    LOG_NAME=${MODEL_CARD}_TP${TP}_EP${EP}_PP${PP}_CON${CONCURRENCY}_${use_case}

    export RESULT_DIR=$LLMB_WORKLOAD/experiments/$LOG_NAME
    export RESULT_FILES_NAME=${LOG_NAME}${streaming_log}
    export DATASET_FILE=$LLMB_WORKLOAD/dataset_${use_case}_${ISL}_${OSL}.txt

    #launch trt-llm benchmark
    CMD="trtllm-llmapi-launch trtllm-bench -m ${MODEL_CARD} --model_path ${MODEL_PATH} throughput \
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
