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
#SBATCH --job-name="gpt_oss:dynamo"
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
export MODEL_NAME=gpt-oss-dynamo

# LLMB_INSTALL and LLMB_WORKLOAD are provided by installer.
LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL is a required variable.}
LLMB_WORKLOAD=${LLMB_WORKLOAD:?LLMB_WORKLOAD is a required variable.}

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}

HF_CACHE_DIR=${LLMB_INSTALL}/.cache/huggingface
mkdir -p $HF_CACHE_DIR

# Download weights
pushd $LLMB_WORKLOAD
if [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "b200" ]]; then
    MODEL_WEIGHTS_DIR="gpt-oss-120b"
    hf download openai/gpt-oss-120b --cache-dir $HF_CACHE_DIR --local-dir $MODEL_WEIGHTS_DIR
else
    echo "Error: Unsupported GPU_TYPE '$GPU_TYPE'. Supported values: b200 and gb200." >&2
    exit 1
fi

popd
