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
#SBATCH --time=20:00
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0

# Paths
export CUR_REPO=$PWD
export NIM_CACHE_PATH=$STAGE_PATH'/nim_cache/'
export HF_HOME=$STAGE_PATH'/HF_HOME/'
export LOGS_PATH=$STAGE_PATH"/logs/"
export RESULTS_PATH=$STAGE_PATH"/results"

# Server Configs
#export SERVER_PORT=8000
export NIM_MODEL_NAME_cleaned="llama3-70b-instruct" # name for tracking experiment, used in result files
export SERVER_IMAGE='nim+meta+llama3-70b-instruct+1.0.3.sqsh' # Container which runs NIM service
export NIM_MODEL_NAME="meta/llama3-70b-instruct" # Name of Model, used by NIM Container AND Benchmarking Image
export CUDA_VISIBLE_DEVICES=0,1,2,3 
export NUM_GPUS=4
export TP_SIZE=4
export precision='fp8'
export START_SERVER_COMMAND="NIM_MODEL_NAME=$STAGE_PATH/llama3-70b-instruct_vhf/ mpirun -n 1 --allow-run-as-root --oversubscribe python3 -m vllm_nvext.entrypoints.openai.api_server --tensor-parallel-size $TP_SIZE --quantization $precision --served-model-name $NIM_MODEL_NAME"

# Benchmarking Configs
export BENCHMARKING_IMAGE='nvidia+tritonserver+25.01.sqsh' # Tritonserver Container which runs benchmarking script
export NIM_MODEL_TOKENIZER="meta-llama/Meta-Llama-3-70B-Instruct" # Tokenizer used by the Tritonserver
export USE_CASES="chat:128/128 summarization:4096/512" # use_case:ISL/OSL - to keep track of different GenAI-Perf configurations
export CONCURRENCY_RANGE="1 25 50 100" # Concurrency levels to sweep over
export total_request_multiplier=5 # total_request_multiplier * Concurrency_value = Total Requests sent per GenAI-Perf Command
export SLEEP_TIME=60 # Time between GenAI-Perf Commands
export MIN_REQUESTS=20 # Minimum total requests per GenAI-Perf Command

mkdir -vp $RESULTS_PATH
mkdir -vp $LOGS_PATH

# Start Server
srun --output $LOGS_PATH/server_%j.out \
    --container-image ${STAGE_PATH}/$SERVER_IMAGE \
    --container-mounts $STAGE_PATH \
    --mpi pmix \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --no-container-mount-home --overlap \
    --container-env=NGC_API_KEY,NIM_CACHE_PATH,HF_TOKEN,CUDA_VISIBLE_DEVICES \
    bash -c "$START_SERVER_COMMAND" &

# Start Benchmarking
srun --output $LOGS_PATH/benchmarking_%j.out \
    --container-image ${STAGE_PATH}/$BENCHMARKING_IMAGE \
    --container-mounts $CUR_REPO:/gsw,${RESULTS_PATH}:${RESULTS_PATH} \
    --wait=60 \
    --container-env=NIM_MODEL_NAME,NGC_API_KEY,HF_TOKEN,CONCURRENCY_RANGE,USE_CASES,RESULTS_PATH,MIN_REQUESTS,total_request_multiplier \
    bash /gsw/run-benchmark.sh
