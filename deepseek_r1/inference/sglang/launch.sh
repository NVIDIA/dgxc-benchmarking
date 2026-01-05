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

# Parameters
#SBATCH --job-name="deepseek_r1:sglang_launch"
#SBATCH --time=02:00:00

set -eu -o pipefail

if [ "${BASH_VERSION:0:1}" -lt 4 ] || { [ "${BASH_VERSION:0:1}" -eq 4 ] && [ "${BASH_VERSION:2:1}" -lt 2 ]; }; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
export GPU_TYPE=${GPU_TYPE,,}

export WORKLOAD_TYPE=inference
export MODEL_NAME=deepseek-r1-sglang

if [[ $GPU_TYPE == "gb200" ]]; then
    export FW_VERSION=v0.5.3-cu129-gb200
elif [[ $GPU_TYPE == "b200" ]]; then
    export FW_VERSION=v0.5.3rc0-cu128-b200
else
    echo "❌ Error: Sglang recipes only supports gb200 and b200 GPU types, got '$GPU_TYPE'"
    exit 1
fi

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export MODEL_CARD="DeepSeek-R1"
export MODEL_PATH=$LLMB_WORKLOAD/DeepSeek-R1-FP4
export MOUNT_DIR=$LLMB_INSTALL

# Server variables
export HOST_IP=0.0.0.0
export HOST_PORT=8000

if [[ $GPU_TYPE == "gb200" ]]; then
    export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/lmsysorg+sglang+${FW_VERSION}.sqsh}
    export TP=${TP:-4}
    export DP=${DP:-4}
    export EP=${EP:-4}
    export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-1024}
    export NUM_PROMPTS=${NUM_PROMPTS:-5000}
    export CONCURRENCY=${CONCURRENCY:-1024}
elif [[ $GPU_TYPE == "b200" ]]; then
    export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/lmsysorg+sglang+${FW_VERSION}.sqsh}
    export TP=${TP:-8}
    export DP=${DP:-8}
    export EP=${EP:-8}
    export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-4096}
    export NUM_PROMPTS=${NUM_PROMPTS:-20480}
    export CONCURRENCY=${CONCURRENCY:-4096}
else
    echo "❌ Error: Sglang recipes only supports gb200 and b200 GPU types, got '$GPU_TYPE'"
    exit 1
fi

# Benchmark variables:
export USE_CASES=${USE_CASES:-"reasoning:1000/1000 chat:128/128 summarization:8000/512 generation:512/8000"}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
export RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1.0}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
export STREAM_INTERVAL=${STREAM_INTERVAL:-10}

echo "Starting server"
export NUM_NODES=${NUM_NODES:-1}

# Launch sglang servers on all nodes
export SERVER_RESULTS_DIR=$LLMB_WORKLOAD/experiments
export LLMB_REPO=$PWD
SERVER_LAUNCH_SCRIPT="${LLMB_REPO}/launch_server.sh"
SERVER_LOG_FILE_NAME=server_TP${TP}_DP${DP}_${SLURM_JOBID}

srun --overlap --no-container-mount-home \
    --nodes ${NUM_NODES} --ntasks ${NUM_NODES} \
    --container-image ${IMAGE} \
    --container-mounts ${MOUNT_DIR}:${MOUNT_DIR} \
    --output $SERVER_RESULTS_DIR/${SERVER_LOG_FILE_NAME}.out \
    bash -c "${SERVER_LAUNCH_SCRIPT}" &

# Poll for the server to be ready for benchmarking
SERVER_LOG="${SERVER_RESULTS_DIR}/${SERVER_LOG_FILE_NAME}.out"
echo "Waiting for server to be ready..."
while ! grep -q "The server is fired up and ready to roll!" "${SERVER_LOG}"; do
    echo "Still waiting for server to be ready to start benchmarking..."
    sleep 60
done

# Loop over each use case
for value in $USE_CASES; do

    use_case=$(echo "$value" | cut -d':' -f1)
    ISL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f1)
    OSL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f2)

    LOG_NAME=${MODEL_CARD}_TP${TP}_DP${DP}_CON${CONCURRENCY}_${use_case}
    export RESULT_DIR=${LLMB_WORKLOAD}/experiments/${LOG_NAME}
    export RESULT_FILES_NAME=${LOG_NAME}
    echo "Now Benchmarking: $use_case : $ISL / $OSL"
    if [[ $use_case == "generation" ]] || [[ $use_case == "summarization" ]]; then
        export NUM_PROMPTS=2000
    fi
    BENCHMARK_CMD="python -m sglang.bench_serving \
           --backend sglang \
            --model ${MODEL_PATH} \
            --dataset-name random \
            --num-prompts ${NUM_PROMPTS} \
            --random-input-len ${ISL} \
            --random-output-len ${OSL} \
            --host ${HOST_IP} \
            --port ${HOST_PORT} \
            --max-concurrency ${CONCURRENCY} \
            --random-range-ratio ${RANDOM_RANGE_RATIO}"

    # start benchmark scripts
    echo "Launching benchmark with command:"
    echo "${BENCHMARK_CMD}"

    export BENCHMARK_OUTPUT=${RESULT_DIR}/${RESULT_FILES_NAME}_%j.out
    srun --overlap --nodes=1 --ntasks=1 --output "$BENCHMARK_OUTPUT" \
        --container-image ${IMAGE} --container-mounts ${MOUNT_DIR} bash -c "${BENCHMARK_CMD}"

    echo "Results log: $BENCHMARK_OUTPUT"

done
