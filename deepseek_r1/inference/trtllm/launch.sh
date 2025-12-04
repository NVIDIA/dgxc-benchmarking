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
#SBATCH --time=01:00:00

set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=inference
export MODEL_NAME=deepseek-r1
export FW_VERSION=1.1.0rc5
export GSW_VERSION=25.10

export MODE=${MODE:-"max_throughput"}
# Validate mode
if [[ $MODE != "min_latency" && $MODE != "max_throughput" ]]; then
    echo "❌ Error: Invalid mode '$MODE'"
    echo "✅ Valid options: min_latency or max_throughput"
    exit 1
fi

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/tensorrt-llm+release+${FW_VERSION}.sqsh}

export MODEL_CARD="DeepSeek-R1"
export MOUNT_DIR=$LLMB_WORKLOAD
export GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
export GPU_TYPE=${GPU_TYPE,,}

# Use cases:  chat:128/128 summarization:8000/512 generation:512/8000
export USE_CASE=${USE_CASE:-"reasoning:1000/1000"}
case "$USE_CASE" in
    reasoning:1000/1000 | chat:128/128 | summarization:8000/512 | generation:512/8000) ;; # valid, do nothing
    *)
        echo "❌ Invalid USE_CASE: $USE_CASE" >&2
        echo "   Must be one of: reasoning:1000/1000, chat:128/128, summarization:8000/512, generation:512/8000" >&2
        exit 1
        ;;
esac

export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-256}
export MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-2000}
export ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-true}
if [[ $MODE == "max_throughput" ]]; then
    if [[ $GPU_TYPE == "h100" ]]; then
        export MODEL_PATH=$LLMB_WORKLOAD/DeepSeek-R1-FP8
        export TP=${TP:-8}
        export EP=${EP:-8}
        export PP=${PP:-2}
        export KV_CACHE_FRACTION=${KV_CACHE_FRACTION:-0.65}
    elif [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "b200" ]]; then
        export MODEL_PATH=$LLMB_WORKLOAD/DeepSeek-R1-FP4
        export TP=${TP:-4}
        export PP=${PP:-1}
        export EP=${EP:-4}
        export KV_CACHE_FRACTION=${KV_CACHE_FRACTION:-0.85}
    fi
    export CONCURRENCY=${CONCURRENCY:-$((MAX_BATCH_SIZE * TP))}
elif [[ $MODE == "min_latency" ]]; then
    if [[ $GPU_TYPE == "gb200" ]]; then
        export MODEL_PATH=$LLMB_WORKLOAD/DeepSeek-R1-FP4
        export TP=${TP:-4}
        export PP=${PP:-1}
        export EP=${EP:-1}
        export MAX_BATCH_SIZE=1
        export CONCURRENCY=1
        export MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-2000}
    else
        echo "❌ Error: Min latency mode only supports gb200 GPU type, got '$GPU_TYPE'"
        exit 1
    fi
fi

# Set the total number of requests as a mutiple of the concurrency (default: 10).
export CONCURRENCY_MULTIPLIER=${CONCURRENCY_MULTIPLIER:-10}
export NUM_REQUESTS=$((CONCURRENCY * CONCURRENCY_MULTIPLIER))
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
    cuda_graph_config:
      enable_padding: true
      batch_sizes: $cuda_batch_sizes
    enable_attention_dp: true
    kv_cache_config:
      dtype: fp8
      free_gpu_memory_fraction: ${KV_CACHE_FRACTION}
    print_iter_log: true
    stream_interval: 4
    enable_chunked_prefill: ${ENABLE_CHUNKED_PREFILL}
EOF
elif [[ $MODE == "min_latency" ]]; then
    cat << EOF > config_${EPOCH_MS}.yml
    cuda_graph_config:
      enable_padding: true
      batch_sizes: $cuda_batch_sizes
    enable_attention_dp: false
    kv_cache_config:
      dtype: fp8
    print_iter_log: true
    stream_interval: 4
    moe_config:
      backend: TRTLLM
    speculative_config:
      decoding_type: MTP
      num_nextn_predict_layers: 3
      use_relaxed_acceptance_for_thinking: true
      relaxed_topk: 10
      relaxed_delta: 0.6
EOF
fi
popd

# Setting environment variables
export CONFIG_FILE=$LLMB_WORKLOAD/config_${EPOCH_MS}.yml
use_case=$(echo "$USE_CASE" | cut -d':' -f1)
ISL=$(echo "$USE_CASE" | cut -d':' -f2 | cut -d'/' -f1)
OSL=$(echo "$USE_CASE" | cut -d':' -f2 | cut -d'/' -f2)

LOG_NAME=${MODEL_CARD}_${MODE}_TP${TP}_EP${EP}_PP${PP}_CON${CONCURRENCY}_${use_case}

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

echo "Results Log: $SRUN_OUTPUT"
echo "Error Log: $SRUN_ERROR"

rm -f config_${EPOCH_MS}.yml
