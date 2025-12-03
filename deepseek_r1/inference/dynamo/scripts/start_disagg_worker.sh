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

config_file=$1
enable_pdl=$2
model_name=$3
model_path=$4
tmp_dir=$5
disaggregation_mode=$6

if [ "$#" -ne 6 ]; then
    echo "Error: Expected 6 arguments, got $#"
    echo "Usage: $0 <config_file> <enable_pdl> <model_name> <model_path> <tmp_dir> <disaggregation_mode>"
    exit 1
fi

unset UCX_TLS
echo "config_file: ${config_file}, enable_pdl: ${enable_pdl}, disaggregation_mode: ${disaggregation_mode}"

# Read configuration values from the YAML config file
if [ ! -f "${config_file}" ]; then
    echo "Error: Config file ${config_file} not found"
    exit 1
fi

# Note: TensorRT-LLM config file is a YAML file may not respect the max_num_tokens,
# max_batch_size, max_seq_len when provided as yaml. Providing these values via
# command line to make sure they are respected.
max_num_tokens=$(grep -E "^[[:space:]]*max_num_tokens:" "${config_file}" | sed 's/.*: *//')
max_batch_size=$(grep -E "^[[:space:]]*max_batch_size:" "${config_file}" | sed 's/.*: *//')
max_seq_len=$(grep -E "^[[:space:]]*max_seq_len:" "${config_file}" | sed 's/.*: *//')

# Validate that we got the values
if [ -z "${max_num_tokens}" ] || [ -z "${max_batch_size}" ] || [ -z "${max_seq_len}" ]; then
    echo "Error: Failed to read required configuration values from ${config_file}"
    echo "max_num_tokens: ${max_num_tokens}"
    echo "max_batch_size: ${max_batch_size}"
    echo "max_seq_len: ${max_seq_len}"
    exit 1
fi

echo "Configuration loaded from ${config_file}:"
echo "  max_num_tokens: ${max_num_tokens}"
echo "  max_batch_size: ${max_batch_size}"
echo "  max_seq_len: ${max_seq_len}"

export TLLM_LOG_LEVEL=INFO
if [ "${enable_pdl}" = "true" ]; then
    export TRTLLM_ENABLE_PDL=1
fi

# Directory to use for storing tmp files, Triton cache, FlashInfer cache, etc.
export FLASHINFER_WORKSPACE_BASE="${tmp_dir}"
export TRITON_CACHE_DIR="${tmp_dir}/.cache/triton"

trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model-path ${model_path} \
    --served-model-name ${model_name} \
    --max-num-tokens ${max_num_tokens} \
    --max-batch-size ${max_batch_size} \
    --max-seq-len ${max_seq_len} \
    --disaggregation-mode ${disaggregation_mode} \
    --extra-engine-args ${config_file}
