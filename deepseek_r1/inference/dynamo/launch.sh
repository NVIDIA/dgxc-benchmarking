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
#SBATCH --exclusive
#SBATCH --job-name="deepseek_r1:dynamo-benchmark"
#SBATCH --time=01:00:00

set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=inference
export MODEL_NAME=deepseek-r1-dynamo

LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL is a required variable.}
LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
LLMB_REPO=$(pwd)

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}

if [[ ${GPU_TYPE} == "gb200" ]]; then
    export FW_VERSION=0.6.1
    export MODEL_PATH=$LLMB_WORKLOAD/DeepSeek-R1-FP4
    export GPUS_PER_NODE=4

    export NCTX=${NCTX:-6}
    export CTX_WORLD_SIZE=${CTX_WORLD_SIZE:-4}
    export CTX_MAX_BATCH_SIZE=${CTX_MAX_BATCH_SIZE:-1}
    export CTX_MAX_NUM_TOKENS=${CTX_MAX_NUM_TOKENS:-4096}
    export CTX_ATTENTION_DP=${CTX_ATTENTION_DP:-true}
    export CTX_GPU_MEMORY_FRACTION=${CTX_GPU_MEMORY_FRACTION:-0.75}
    export NGEN=${NGEN:-1}
    export GEN_WORLD_SIZE=${GEN_WORLD_SIZE:-8}
    export GEN_MAX_BATCH_SIZE=${GEN_MAX_BATCH_SIZE:-256}
    export GEN_MAX_NUM_TOKENS=${GEN_MAX_NUM_TOKENS:-256}
    export GEN_ATTENTION_DP=${GEN_ATTENTION_DP:-true}
    export GEN_GPU_MEMORY_FRACTION=${GEN_GPU_MEMORY_FRACTION:-0.8}
    export GEN_EPLB_NUM_SLOTS=${GEN_EPLB_NUM_SLOTS:-0}
    export GEN_MTP_SIZE=${GEN_MTP_SIZE:-0}
    export CONCURRENCY=${CONCURRENCY:-2048}

elif [[ ${GPU_TYPE} == "b200" ]]; then
    export FW_VERSION=0.6.1
    export MODEL_PATH=$LLMB_WORKLOAD/DeepSeek-R1-FP4
    export GPUS_PER_NODE=8

    export NCTX=${NCTX:-3}
    export CTX_WORLD_SIZE=${CTX_WORLD_SIZE:-8}
    export CTX_MAX_BATCH_SIZE=${CTX_MAX_BATCH_SIZE:-1}
    export CTX_MAX_NUM_TOKENS=${CTX_MAX_NUM_TOKENS:-4096}
    export CTX_ATTENTION_DP=${CTX_ATTENTION_DP:-true}
    export CTX_GPU_MEMORY_FRACTION=${CTX_GPU_MEMORY_FRACTION:-0.75}
    export NGEN=${NGEN:-1}
    export GEN_WORLD_SIZE=${GEN_WORLD_SIZE:-8}
    export GEN_MAX_BATCH_SIZE=${GEN_MAX_BATCH_SIZE:-256}
    export GEN_MAX_NUM_TOKENS=${GEN_MAX_NUM_TOKENS:-256}
    export GEN_ATTENTION_DP=${GEN_ATTENTION_DP:-true}
    export GEN_GPU_MEMORY_FRACTION=${GEN_GPU_MEMORY_FRACTION:-0.8}
    export GEN_EPLB_NUM_SLOTS=${GEN_EPLB_NUM_SLOTS:-0}
    export GEN_MTP_SIZE=${GEN_MTP_SIZE:-0}
    export CONCURRENCY=${CONCURRENCY:-2048}

elif [[ ${GPU_TYPE} == "h100" ]]; then
    export FW_VERSION=0.6.1
    export MODEL_PATH=$LLMB_WORKLOAD/DeepSeek-R1-FP8
    export GPUS_PER_NODE=8

    export NCTX=${NCTX:-2}
    export CTX_WORLD_SIZE=${CTX_WORLD_SIZE:-16}
    export CTX_MAX_BATCH_SIZE=${CTX_MAX_BATCH_SIZE:-1}
    export CTX_MAX_NUM_TOKENS=${CTX_MAX_NUM_TOKENS:-2048}
    export CTX_ATTENTION_DP=${CTX_ATTENTION_DP:-true}
    export CTX_GPU_MEMORY_FRACTION=${CTX_GPU_MEMORY_FRACTION:-0.75}
    export NGEN=${NGEN:-1}
    export GEN_WORLD_SIZE=${GEN_WORLD_SIZE:-16}
    export GEN_MAX_BATCH_SIZE=${GEN_MAX_BATCH_SIZE:-128}
    export GEN_MAX_NUM_TOKENS=${GEN_MAX_NUM_TOKENS:-128}
    export GEN_ATTENTION_DP=${GEN_ATTENTION_DP:-true}
    export GEN_GPU_MEMORY_FRACTION=${GEN_GPU_MEMORY_FRACTION:-0.8}
    export GEN_EPLB_NUM_SLOTS=${GEN_EPLB_NUM_SLOTS:-0}
    export GEN_MTP_SIZE=${GEN_MTP_SIZE:-0}
    export CONCURRENCY=${CONCURRENCY:-1024}

else
    echo "❌ Error: Dynamo recipes only supports gb200, b200 and H100 GPU types, got '$GPU_TYPE'"
    exit 1
fi

NUM_CTX_NODES=$((NCTX * CTX_WORLD_SIZE / GPUS_PER_NODE))
NUM_GEN_NODES=$((NGEN * GEN_WORLD_SIZE / GPUS_PER_NODE))
TOTAL_NODES=$((NUM_CTX_NODES + NUM_GEN_NODES))
TOTAL_TASKS=$((TOTAL_NODES * GPUS_PER_NODE))

if [[ ${TOTAL_TASKS} -ne ${SLURM_NTASKS} ]]; then
    echo "❌ Error: Total tasks (${TOTAL_TASKS}) does not match SLURM NTASKS (${SLURM_NTASKS})"
    exit 1
fi

if [[ ${TOTAL_NODES} -ne ${SLURM_JOB_NUM_NODES} ]]; then
    echo "❌ Error: Total nodes (${TOTAL_NODES}) does not match SLURM NNODES (${SLURM_JOB_NUM_NODES})"
    exit 1
fi

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/ai-dynamo+tensorrtllm-runtime+${FW_VERSION}.sqsh}

export ISL=${ISL:-8150}
export OSL=${OSL:-1024}

# Whether to stream the response from the server.
export STREAMING=${STREAMING:-true}

# The number of rounds of requests to send to the server. The number of requests is `MULTI_ROUND * CONCURRENCY * NGEN`.
export MULTI_ROUND=${MULTI_ROUND:-8}

# Ports for the frontend servers.
export NATS_SERVER_PORT=${NATS_SERVER_PORT:-4222}
export ETCD_SERVER_PORT=${ETCD_SERVER_PORT:-2379}
export DYNAMO_FRONTEND_PORT=${DYNAMO_FRONTEND_PORT:-8000}

export SCRIPTS_DIR=$LLMB_REPO/scripts

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_NAME=isl${ISL}_osl${OSL}_${TIMESTAMP}
if [ "${GEN_ATTENTION_DP}" = "false" ]; then
    export ENABLE_PDL=true
else
    export ENABLE_PDL=false
fi

export RESULT_DIR=$LLMB_WORKLOAD/experiments/$LOG_NAME
export CONFIG_DIR=${RESULT_DIR}/configs
export LOG_DIR=${RESULT_DIR}/logs
mkdir -p ${RESULT_DIR}
mkdir -p ${CONFIG_DIR}
mkdir -p ${LOG_DIR}

export CONTAINER_NAME=${MODEL_NAME}
export CONTAINER_MOUNTS=${LLMB_WORKLOAD}:${LLMB_WORKLOAD},${LLMB_REPO}:${LLMB_REPO}

# Read the nodes in the Slurm job node list into the NODES array.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

# Create a temporary file to store PIDs. This will be used to cleanup the spawned processes.
PID_FILE=$(mktemp)
trap 'cleanup_and_exit' EXIT

cleanup_and_exit() {
    if [ -f "$PID_FILE" ]; then
        echo "Cleaning up spawned processes..."
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null; then
                echo "Sending TERM to process $pid"
                kill -TERM "$pid" 2> /dev/null
                sleep 2
                if kill -0 "$pid" 2> /dev/null; then
                    echo "Process $pid still running, sending KILL"
                    kill -KILL "$pid" 2> /dev/null
                fi
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
}

# Set clock if the gpu_type is gb200 or b200.
set_clock() {
    if [[ ${GPU_TYPE} == "b200" || ${GPU_TYPE} == "gb200" ]]; then
        set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
        srun ${set_clock_cmd}
    fi
}

# Generate the yaml file for the context and generation servers.
generate_yaml() {
    # Calculate the max sequence length for the context and generation servers.
    # We add 200 to the ISL and OSL to account for small variations in the number
    # of input and output tokens.
    CTX_MAX_SEQ_LEN=$((ISL + 200))
    GEN_MAX_SEQ_LEN=$((ISL + OSL + 200))

    # The number of tokens in the cache transceiver (between the context and
    # generation servers) must be a multiple of tokens per block.
    CACHE_TRANSCEIVER_MAX_NUM_TOKENS=$((((CTX_MAX_SEQ_LEN + 256 - 1) / 256) * 256))

    srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix --overlap \
        --ntasks 1 --nodes 1 \
        python3 ${SCRIPTS_DIR}/gen_yaml.py --config_dir ${CONFIG_DIR} \
        --gpu_type ${GPU_TYPE} \
        --num_ctx_servers ${NCTX} \
        --ctx_world_size ${CTX_WORLD_SIZE} \
        --ctx_enable_attention_dp ${CTX_ATTENTION_DP} \
        --ctx_batch_size ${CTX_MAX_BATCH_SIZE} \
        --ctx_max_num_tokens ${CTX_MAX_NUM_TOKENS} \
        --ctx_max_seq_len ${CTX_MAX_SEQ_LEN} \
        --ctx_free_gpu_memory_fraction ${CTX_GPU_MEMORY_FRACTION} \
        --cache_transceiver_max_num_tokens ${CACHE_TRANSCEIVER_MAX_NUM_TOKENS} \
        --num_gen_servers ${NGEN} \
        --gen_world_size ${GEN_WORLD_SIZE} \
        --gen_enable_attention_dp ${GEN_ATTENTION_DP} \
        --gen_batch_size ${GEN_MAX_BATCH_SIZE} \
        --gen_max_num_tokens ${GEN_MAX_NUM_TOKENS} \
        --gen_max_seq_len ${GEN_MAX_SEQ_LEN} \
        --gen_gpu_memory_fraction ${GEN_GPU_MEMORY_FRACTION} \
        --eplb_num_slots ${GEN_EPLB_NUM_SLOTS} \
        --mtp_size ${GEN_MTP_SIZE}

    echo "YAML file generated."
}

# Start the frontend server.
start_frontend() {
    HEAD_NODE_IP="$(hostname -i)"
    export ETCD_ENDPOINTS="${HEAD_NODE_IP}:${ETCD_SERVER_PORT}"
    export NATS_SERVER="nats://${HEAD_NODE_IP}:${NATS_SERVER_PORT}"

    srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix --overlap \
        --ntasks 1 --nodes 1 \
        --oversubscribe \
        --container-env ETCD_ENDPOINTS,NATS_SERVER \
        -w ${NODES[0]} \
        bash ${SCRIPTS_DIR}/start_frontend.sh \
        ${ETCD_SERVER_PORT} ${DYNAMO_FRONTEND_PORT} \
        &> ${LOG_DIR}/output_server.log &
    SERVER_PID=$!
    echo "$SERVER_PID" >> "$PID_FILE"
}

# Start the prefill workers.
start_prefill_workers() {
    echo "Prefill Count: ${NCTX}"

    NCTX_NODES_PER_SERVER=$((CTX_WORLD_SIZE / GPUS_PER_NODE))
    for ((i = 1; i <= NCTX; i++)); do
        echo "Running Prefill Worker: ${i}"

        PREFILL_NODE_LIST=()
        for ((j = 0; j < NCTX_NODES_PER_SERVER; j++)); do
            node_idx=$(((i - 1) * NCTX_NODES_PER_SERVER + j))
            PREFILL_NODE_LIST+=("${NODES[node_idx]}")
        done
        PREFILL_NODES_CSV=$(
            IFS=,
            echo "${PREFILL_NODE_LIST[*]}"
        )

        echo "Running Prefill Nodes: ${PREFILL_NODES_CSV}"
        srun -l --container-name=${CONTAINER_NAME} \
            --container-mounts=${CONTAINER_MOUNTS} \
            --mpi=pmix \
            --overlap \
            --oversubscribe \
            -w ${PREFILL_NODES_CSV} \
            --ntasks ${CTX_WORLD_SIZE} \
            --nodes ${NCTX_NODES_PER_SERVER} \
            bash ${SCRIPTS_DIR}/start_disagg_worker.sh \
            ${CONFIG_DIR}/prefill.yaml "${ENABLE_PDL}" \
            ${MODEL_NAME} ${MODEL_PATH} ${LLMB_WORKLOAD} \
            'prefill' &> ${LOG_DIR}/output_prefill_${i}.log &
        echo "$!" >> "$PID_FILE"
    done
}

# Start the decode workers.
start_decode_workers() {
    echo "Decode Count: ${NGEN}"

    NGEN_NODES_PER_SERVER=$((GEN_WORLD_SIZE / GPUS_PER_NODE))
    DECODE_START_IDX=$((NCTX_NODES_PER_SERVER * NCTX))
    for ((i = 1; i <= NGEN; i++)); do
        echo "Running Decode Worker: ${i}"

        DECODE_NODE_LIST=()
        for ((j = 0; j < NGEN_NODES_PER_SERVER; j++)); do
            node_idx=$((DECODE_START_IDX + (i - 1) * NGEN_NODES_PER_SERVER + j))
            DECODE_NODE_LIST+=("${NODES[node_idx]}")
        done
        DECODE_NODES_CSV=$(
            IFS=,
            echo "${DECODE_NODE_LIST[*]}"
        )

        echo "Running Decode Nodes: ${DECODE_NODES_CSV}"
        srun -l --container-name=${CONTAINER_NAME} \
            --container-mounts=${CONTAINER_MOUNTS} \
            --mpi=pmix \
            --overlap \
            --oversubscribe \
            -w ${DECODE_NODES_CSV} \
            --ntasks ${GEN_WORLD_SIZE} \
            --nodes ${NGEN_NODES_PER_SERVER} \
            bash ${SCRIPTS_DIR}/start_disagg_worker.sh \
            ${CONFIG_DIR}/decode.yaml "${ENABLE_PDL}" \
            ${MODEL_NAME} ${MODEL_PATH} ${LLMB_WORKLOAD} \
            'decode' &> ${LOG_DIR}/output_decode_${i}.log &
        echo "$!" >> "$PID_FILE"
    done
}

# Start the loadgen.
start_loadgen() {
    HEAD_NODE_IP="$(hostname -i)"
    srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix --overlap \
        --ntasks 1 --nodes 1 \
        -w ${NODES[0]} \
        bash ${SCRIPTS_DIR}/bench.sh \
        ${MODEL_NAME} ${MULTI_ROUND} ${NGEN} "${CONCURRENCY}" \
        ${STREAMING} ${SLURM_NTASKS} \
        ${RESULT_DIR} ${SCRIPTS_DIR} \
        ${MODEL_PATH} ${ISL} ${OSL} \
        ${HEAD_NODE_IP} ${DYNAMO_FRONTEND_PORT} > ${LOG_DIR}/bench.log 2>&1
}

set_clock

# Start the container.
srun -l --container-image=${IMAGE} \
    --container-name=${CONTAINER_NAME} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --mpi=pmix \
    echo "Container up."

generate_yaml

start_frontend

start_prefill_workers

start_decode_workers

start_loadgen
