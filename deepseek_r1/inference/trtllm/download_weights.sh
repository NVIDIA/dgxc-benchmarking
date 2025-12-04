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
#SBATCH --job-name="deepseek_r1_fp4:trtllm-setup"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=inference
export MODEL_NAME=deepseek-r1
export FW_VERSION=1.1.0rc1

# LLMB_INSTALL and LLMB_WORKLOAD are provided by installer.
HF_CACHE_DIR=$LLMB_INSTALL/.cache/huggingface
mkdir -p $HF_CACHE_DIR

# Download weights
pushd $LLMB_WORKLOAD
if [[ $GPU_TYPE == "h100" ]]; then
    MODEL_WEIGHTS_DIR="DeepSeek-R1-FP8"
    hf download deepseek-ai/DeepSeek-R1 --cache-dir $HF_CACHE_DIR --local-dir $MODEL_WEIGHTS_DIR
elif [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "b200" ]]; then
    MODEL_WEIGHTS_DIR="DeepSeek-R1-FP4"
    hf download nvidia/DeepSeek-R1-0528-FP4 --cache-dir $HF_CACHE_DIR --local-dir $MODEL_WEIGHTS_DIR
else
    echo "Error: Unsupported GPU_TYPE '$GPU_TYPE'. Supported values: h100, b200 and gb200." >&2
    exit 1
fi

export MODEL_PATH=$LLMB_WORKLOAD/$MODEL_WEIGHTS_DIR
ISL_OSL_COMBINATIONS=("reasoning:1000/1000" "chat:128/128" "summarization:8000/512" "generation:512/8000")
PROMPT_REQUESTS=${PROMPT_REQUESTS:-50000}

ulimit -u 65535 || echo "Could not raise ulimit"

for value in "${ISL_OSL_COMBINATIONS[@]}"; do

    use_case=$(echo "$value" | cut -d':' -f1)
    ISL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f1)
    OSL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f2)

    echo "preparing dataset for:"
    echo "Use Case: $use_case"
    echo "ISL: $ISL"
    echo "OSL: $OSL"

    python TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
        --stdout --tokenizer $MODEL_PATH \
        token-norm-dist \
        --input-mean $ISL --output-mean $OSL \
        --input-stdev 0 --output-stdev 0 \
        --num-requests $PROMPT_REQUESTS > $LLMB_WORKLOAD/dataset_${use_case}_${ISL}_${OSL}.txt
done
popd
