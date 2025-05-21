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
#SBATCH --mem=0
#SBATCH -J general_nim-deepseek-r1:benchmark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --time=01:00:00

# Check if the NGC_API_KEY is empty
if [[ -z "$NGC_API_KEY" ]]; then
    echo "Error: NGC_API_KEY is empty." >&2
    exit 1
fi

#Check if the STAGE_PATH is empty
if [[ -z "$STAGE_PATH" ]]; then
    echo "Error: STAGE_PATH is empty." >&2
    exit 1
fi

set -evx

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0
set +e

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export CUR_REPO=$PWD
container=${STAGE_PATH}/nim+deepseek+r1+1.7.2.sqsh
export NIM_SERVED_MODEL_NAME="deepseek-ai/deepseek-r1"

export NIM_MULTI_NODE=1
export HEAD_NODE=${head_node}
export NIM_LEADER_IP_ADDRESS=${head_node_ip}
export NIM_NUM_COMPUTE_NODES=2
export NIM_CACHE_PATH=${STAGE_PATH}/nim-cache
export NIM_MODEL_NAME=${STAGE_PATH}/deepseek-r1-instruct_vhf-5dde110-nim-fp8/
export NIM_MODEL_NAME_cleaned=deepseek-r1
export NIM_MODEL_PROFILE=sglang-h100-bf16-tp8-pp2
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

srun --output $STAGE_PATH/logs/server_%j.out \
    --container-image $container \
    --container-mounts /lustre,${CUR_REPO} \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --no-container-mount-home --overlap --mpi=pmix --wait=10 --ntasks-per-node=1 --ntasks=2 --nodes=2 \
    --container-env=NIM_SERVED_MODEL_NAME,HEAD_NODE,NIM_MODEL_NAME,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,NIM_MULTI_NODE,NIM_LEADER_IP_ADDRESS,NIM_NUM_COMPUTE_NODES,NIM_CACHE_PATH,NIM_MODEL_PROFILE,NGC_API_KEY,NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES \
    bash ${CUR_REPO}/multi-node-worker.sh &

export BENCHMARKING_IMAGE='nvidia+tritonserver+25.01.sqsh' # Tritonserver Container which runs benchmarking script
export NIM_MODEL_TOKENIZER="deepseek-ai/DeepSeek-R1" # Tokenizer used by the Tritonserver
export HF_HOME=$STAGE_PATH'/HF_HOME/'
export USE_CASES="chat:128/128"
export CONCURRENCY_RANGE="1 5 10 25" # Concurrency levels to sweep over
export total_request_multiplier=5 # total_request_multiplier * Concurrency_value = Total Requests sent per GenAI-Perf Command
export SLEEP_TIME=60 # Time between GenAI-Perf Commands (in seconds)
export SERVER_SLEEP_TIME=${SERVER_SLEEP_TIME:-1600} # Wait time till model weights loading is complete (in seconds)
export MIN_REQUESTS=20 # Minimum total requests per GenAI-Perf Command
export NUM_GPUS=16 # metadata in output file
export RESULTS_PATH=$STAGE_PATH"/results"
mkdir -vp $RESULTS_PATH

srun --output $STAGE_PATH/logs/benchmarking_%j.out \
    --container-image ${STAGE_PATH}/${BENCHMARKING_IMAGE} \
    --container-mounts ${STAGE_PATH},${CUR_REPO} \
    --wait=60 \
    --container-env=NIM_LEADER_IP_ADDRESS,NIM_MODEL_TOKENIZER,SERVED_MODEL_NAME,NGC_API_KEY,HF_HOME,HF_TOKEN,CONCURRENCY_RANGE,USE_CASES,RESULTS_PATH,MIN_REQUESTS,total_request_multiplier,NUM_GPUS \
    bash ${CUR_REPO}/run-benchmark.sh