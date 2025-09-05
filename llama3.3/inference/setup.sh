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
#SBATCH --job-name="llama3.3-70b:trtllm-setup"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=inference
export MODEL_NAME=llama3.3
export FW_VERSION=1.0.0rc1

# Ensure STAGE_PATH is not set as it's been replaced by LLMB_INSTALL
if [ -n "${STAGE_PATH+x}" ]; then
  echo "Error: STAGE_PATH is deprecated and should not be set. Please use LLMB_INSTALL instead."
  exit 1
fi

MODEL_WEIGHTS_DIR="Llama3.3_70B"
export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export MODEL_PATH=$LLMB_WORKLOAD/$MODEL_WEIGHTS_DIR

# Build LLMB_INSTALL location 
export MANUAL_INSTALL=${MANUAL_INSTALL:-true}
if [ "$MANUAL_INSTALL" = true ]; then
  mkdir -p $LLMB_INSTALL $LLMB_INSTALL/{datasets,images,venvs,workloads}
  mkdir -p $LLMB_WORKLOAD
fi

ISL_OSL_COMBINATIONS=("reasoning:1000/1000 chat:128/128 summarization:8000/512 generation:512/8000")
PROMT_REQUESTS=${PROMT_REQUESTS:-50000}
TRT_COMMIT="1a7c6e79743f100954e0a2375f254e54d5717e52"
TRT_DIR="TensorRT-LLM"

pushd $LLMB_WORKLOAD
if [ ! -d "${MODEL_WEIGHTS_DIR}" ]; then
  pip install -U "huggingface_hub[cli]"
  huggingface-cli download nvidia/Llama-3.3-70B-Instruct-FP4 --local-dir Llama3.3_70B
fi

# clone the TRT-LLM repo 
if [ ! -d "$TRT_DIR" ]; then
    git clone https://github.com/NVIDIA/TensorRT-LLM.git
fi

pushd $TRT_DIR
git fetch origin
git checkout -f $TRT_COMMIT
popd

# Install dependencies for dataset generation
pip install click==8.2.1
pip install transformers==4.52.4
pip install pillow==11.2.1
pip install datasets==3.6.0
pip install pydantic==2.11.7

cat <<EOF > config.yml
enable_attention_dp: false
use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 128
  - 256
  - 320
  - 448
  - 512
print_iter_log: true
stream_interval: 4
EOF

for value in $ISL_OSL_COMBINATIONS; do
    
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
        --num-requests $PROMT_REQUESTS > $LLMB_WORKLOAD/dataset_${use_case}_${ISL}_${OSL}.txt
done
popd
