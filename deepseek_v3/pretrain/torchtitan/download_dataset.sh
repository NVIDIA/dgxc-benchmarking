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
#SBATCH --job-name="deepseek_v3:torchtitan-setup"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

set -eu -o pipefail

if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 2 ]; }; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=deepseek-v3-torchtitan

# Validate required environment variables
if [ -z "${LLMB_INSTALL:-}" ]; then
    echo "Error: LLMB_INSTALL environment variable is not set" >&2
    exit 1
fi

# Use shared dataset location to avoid duplicate downloads across workloads
DATASET_DIR="$LLMB_INSTALL/datasets/c4"
HF_CACHE_DIR="$LLMB_INSTALL/.cache/huggingface"
mkdir -p "$HF_CACHE_DIR"

# Check if dataset already exists
if [ -d "$DATASET_DIR" ] && [ "$(ls -A $DATASET_DIR 2> /dev/null)" ]; then
    echo "Dataset already exists at $DATASET_DIR, skipping download..."
else
    echo "Downloading dataset to: $DATASET_DIR"
    mkdir -p "$DATASET_DIR"

    # Pass token explicitly to avoid rate limiting issues with login()
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "Using HuggingFace token for download..."
        hf download allenai/c4 --include "en/*" --include "dataset_info.json" --include "*.py" --include "README.md" --repo-type dataset --cache-dir "$HF_CACHE_DIR" --local-dir "$DATASET_DIR" --token "$HF_TOKEN"
    else
        hf download allenai/c4 --include "en/*" --include "dataset_info.json" --include "*.py" --include "README.md" --repo-type dataset --cache-dir "$HF_CACHE_DIR" --local-dir "$DATASET_DIR"
    fi

    echo "Dataset download complete!"
fi
