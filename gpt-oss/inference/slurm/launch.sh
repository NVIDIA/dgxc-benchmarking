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
#SBATCH --job-name="gpt_oss:dynamo-benchmark"
#SBATCH --time=01:00:00

set -eu -o pipefail
set -x

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=inference
export MODEL_NAME=gpt-oss-dynamo
export GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
export GPU_TYPE=${GPU_TYPE,,}

if [[ ${GPU_TYPE} == "gb200" ]]; then
    export FW_VERSION=0.5.1-rc0.pre3
elif [[ ${GPU_TYPE} == "b200" ]]; then
    export FW_VERSION=0.6.1
else
    echo "Unsupported GPU type: ${GPU_TYPE}"
    exit 1
fi

export LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL is a required variable.}
export LLMB_WORKLOAD=${LLMB_INSTALL}/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$(pwd)

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/ai-dynamo+tensorrtllm-runtime+${FW_VERSION}.sqsh}
export DYNAMO_MODE='dynamo_agg'
export ISL=${ISL:-128}
export OSL=${OSL:-1000}
export STREAMING=${STREAMING:-true}
# The number of times the requests to be sent to the server.
export MULTI_ROUND=${MULTI_ROUND:-8}
# Ports for the frontend servers.
export NATS_SERVER_PORT=${NATS_SERVER_PORT:-4222}
export ETCD_SERVER_PORT=${ETCD_SERVER_PORT:-2379}
export DYNAMO_FRONTEND_PORT=${DYNAMO_FRONTEND_PORT:-8000}

export SCRIPTS_DIR=${LLMB_WORKLOAD}/dynamo/components/backends/trtllm/performance_sweeps/scripts
export CLOCK_SCRIPT=${LLMB_WORKLOAD}/dynamo/components/backends/trtllm/performance_sweeps

export CONTAINER_NAME=${MODEL_NAME}
export CONTAINER_MOUNTS=${LLMB_INSTALL}:${LLMB_INSTALL}

export MODEL_PATH=$LLMB_WORKLOAD/gpt-oss-120b
export TP_SIZE=${TP_SIZE:-4}
export EP_SIZE=${EP_SIZE:-4}
export MAX_SEQ_LEN=$((ISL + OSL))
export ENABLE_ATTN_DP=${ENABLE_ATTN_DP:-true}

export CONCURRENCY=${CONCURRENCY:-2048}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-800}
export GPU_MEM_FRACTION=${GPU_MEM_FRACTION:-0.8}
export MTP_SIZE=${MTP_SIZE:-0}
export MAX_NUM_TOKENS=$((((MTP_SIZE + 1) * MAX_BATCH_SIZE + ISL + 128 + 63) / 64 * 64))

# create the result and logs directories
export LOG_NAME=${MODEL_NAME}_ISL${ISL}_OSL${OSL}_BS${MAX_BATCH_SIZE}_CON${CONCURRENCY}_${SLURM_JOB_ID}
export RESULT_DIR=$LLMB_WORKLOAD/experiments/${LOG_NAME}
export BENCHMARK_LOG_DIR=${RESULT_DIR}/benchmark_logs
export SERVER_LOG_DIR=${RESULT_DIR}/server_logs

mkdir -p ${RESULT_DIR}
mkdir -p ${SERVER_LOG_DIR}
mkdir -p ${BENCHMARK_LOG_DIR}

# Set clock for gb200 and b200.
set_clock() {
    set_clock_cmd="bash ${CLOCK_SCRIPT}/set_clock.sh"
    srun ${set_clock_cmd}
}

set_clock

# start the container
srun -l --container-image=${IMAGE} \
    --container-name=${CONTAINER_NAME} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --mpi=pmix \
    echo "Container up."

# Read the nodes in the Slurm job node list into the NODES array.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

export HEAD_NODE="${NODES[0]}"
export HEAD_NODE_IP="$(hostname -i)"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:${ETCD_SERVER_PORT}"
export NATS_SERVER="nats://${HEAD_NODE_IP}:${NATS_SERVER_PORT}"

# Create a temporary file to store PIDs
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

# start the front-end server
srun -l --container-name=${CONTAINER_NAME} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --mpi=pmix --overlap -N 1 -n 1 \
    --oversubscribe \
    --overlap \
    --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
    -w ${NODES[0]} \
    bash ${SCRIPTS_DIR}/start_frontend.sh &> ${SERVER_LOG_DIR}/output_server.log &

# start the workers
srun -l --container-name=${CONTAINER_NAME} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --mpi=pmix --overlap \
    --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
    bash -x ${SCRIPTS_DIR}/start_agg_worker.sh ${MODEL_PATH} ${MAX_BATCH_SIZE} \
    ${MAX_NUM_TOKENS} ${TP_SIZE} ${EP_SIZE} ${ENABLE_ATTN_DP} ${GPU_MEM_FRACTION} \
    ${MAX_SEQ_LEN} ${MTP_SIZE} ${MODEL_NAME} &> ${SERVER_LOG_DIR}/output_workers.log &

# start the loadgen
srun -l --container-name=${CONTAINER_NAME} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --mpi=pmix --overlap -N 1 -n 1 \
    --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
    -w ${NODES[0]} \
    bash ${SCRIPTS_DIR}/bench.sh ${MODEL_NAME} ${MULTI_ROUND} 1 ${CONCURRENCY} \
    ${STREAMING} ${SERVER_LOG_DIR} ${TP_SIZE} ${BENCHMARK_LOG_DIR} ${MODEL_PATH} ${ISL} ${OSL} \
    ${DYNAMO_MODE} > ${SERVER_LOG_DIR}/bench.log 2>&1

# Cleanup will be handled by the EXIT trap
