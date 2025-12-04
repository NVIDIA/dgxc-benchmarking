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
#SBATCH --job-name="cpu_overhead:microbenchmark"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00

set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

if [[ $SLURM_JOB_NUM_NODES -ne 1 ]]; then
    echo "This benchmark only supports a single node -- ${SLURM_JOB_NUM_NODES} nodes requested."
    exit 1
fi

export WORKLOAD_TYPE=microbenchmark
export WORKLOAD=cpu_overhead
export FW_VERSION=1.1.0rc5
export GSW_VERSION=25.10

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${WORKLOAD}
export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/tensorrt-llm+release+${FW_VERSION}.sqsh}

export MODEL_PATH=$LLMB_WORKLOAD/gpt-oss-120b
export MOUNT_DIR=$LLMB_WORKLOAD
export TRT_DIR=$LLMB_WORKLOAD/TensorRT-LLM
export RESULT_DIR=$LLMB_WORKLOAD/experiments/cpu_overhead_tests
export USE_CASES=${USE_CASES:-"kernel_launch tokenization"}

export NUM_PROMPTS=${NUM_PROMPTS:-100000}

# Loop over each use case
for value in $USE_CASES; do
    LOG_NAME=${value}_overhead

    if [ ${value} == "kernel_launch" ]; then
        CMD="python $LLMB_WORKLOAD/pytorch_kernel_launch_latency.py \
	    --start_size 4 --end_size 512 --iters 1000000"
    elif [ ${value} == "tokenization" ]; then
        export DATASET_FILE=$LLMB_WORKLOAD/dataset_1000_1000_${NUM_PROMPTS}_${SLURM_PROCID}.txt
        CMD="START_TIME=\$(date +%s); \
            python $TRT_DIR/benchmarks/cpp/prepare_dataset.py \
            --stdout --tokenizer $MODEL_PATH \
            token-norm-dist \
            --input-mean 1000 --output-mean 1000 \
            --input-stdev 0 --output-stdev 0 \
            --num-requests $NUM_PROMPTS > $DATASET_FILE; \
            END_TIME=\$(date +%s); \
            ELAPSED=\$((END_TIME - START_TIME)); \
            echo \"Tokenization time: \${ELAPSED}s\""
    fi

    export SLURM_MPI_TYPE="pmix"
    export SRUN_OUTPUT=${RESULT_DIR}/${LOG_NAME}_%j.out
    export SRUN_ERROR=${RESULT_DIR}/${LOG_NAME}_%j.err

    srun --container-image "$IMAGE" \
        --container-mounts "$MOUNT_DIR" \
        --container-writable \
        --no-container-mount-home bash -c "$CMD"

    echo "Results of Benchmark: $SRUN_OUTPUT"
    echo "Error Log of Benchmark: $SRUN_ERROR"

done
