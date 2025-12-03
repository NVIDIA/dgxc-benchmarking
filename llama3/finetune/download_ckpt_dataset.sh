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

# This script is intended to be run by the LLMB installer as a *setup task*
# (job_type: nemo2) after dependencies have already been installed.
# It imports the base checkpoint and downloads the training dataset for
# Llama-3 70B finetuning (LoRa).

set -eu -o pipefail

export WORKLOAD_TYPE=finetune
export MODEL_NAME=llama3
export FW_VERSION=25.09.00

# --- Required environment variables (provided by the installer) ---
: "${HF_TOKEN:?Required variable Hugging Face token}"
: "${GPU_TYPE:?Required variable GPU_TYPE}"
: "${LLMB_INSTALL:?Required variable LLMB_INSTALL}"
: "${LLMB_WORKLOAD:?Provided by installer}"
: "${SBATCH_ACCOUNT:?Required variable SBATCH_ACCOUNT}"
: "${SBATCH_PARTITION:?Required variable SBATCH_PARTITION}"

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export NEMO_DIR="$LLMB_WORKLOAD/NeMo"

# Directory for cached objects
mkdir -p "$LLMB_WORKLOAD/checkpoint_and_dataset"
export HF_HOME=${HF_HOME:-$LLMB_WORKLOAD/checkpoint_and_dataset}
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD/checkpoint_and_dataset}
export NEMORUN_HOME=$LLMB_WORKLOAD

SCRIPT_NAME="scripts.performance.llm.finetune_llama3_70b"
CONTAINER_MOUNTS=""
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi
export IMAGE="${IMAGE:-${LLMB_INSTALL}/images/nvidia+nemo+${FW_VERSION}.sqsh}"

TIME_LIMIT=${TIME_LIMIT:-"00:55:00"}
GPU_TYPE=${GPU_TYPE,,}

# Default values for Import Checkpoint and Dataset Download
SKIP_IMPORT_CHECKPOINT=${SKIP_IMPORT_CHECKPOINT:-false}
SKIP_DATASET_DOWNLOAD=${SKIP_DATASET_DOWNLOAD:-false}

CONFIG_OVERRIDES=""
if [ $SKIP_IMPORT_CHECKPOINT = true ]; then
    # 3.1 Import Checkpoint
    CONFIG_OVERRIDES+=" --skip_import_checkpoint "
fi
if [ $SKIP_DATASET_DOWNLOAD = true ]; then
    # 3.2 Download Dataset
    CONFIG_OVERRIDES+=" --skip_dataset_download "
fi

if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

pushd "$NEMO_DIR" > /dev/null

python3 -m "$SCRIPT_NAME" \
    -hf "$HF_TOKEN" \
    --gpu "$GPU_TYPE" \
    --container_image "$IMAGE" \
    --num_gpus 1 \
    --gpus_per_node 1 \
    --finetuning "lora" \
    --skip_finetuning \
    $CONFIG_OVERRIDES \
    slurm \
    --account "$SBATCH_ACCOUNT" \
    --partition "$SBATCH_PARTITION" \
    --log_dir "$NEMORUN_HOME" \
    --time_limit "$TIME_LIMIT"

popd > /dev/null

echo "✓ Checkpoint import and dataset download queued."
