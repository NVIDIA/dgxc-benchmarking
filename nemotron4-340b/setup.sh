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

set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=nemotron4-340b

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMO_DIR=$LLMB_WORKLOAD/NeMo

# Build LLMB_INSTALL location
export MANUAL_INSTALL=${MANUAL_INSTALL:-true}
if [ "$MANUAL_INSTALL" = true ]; then
    mkdir -p $LLMB_INSTALL $LLMB_INSTALL/{datasets,images,venvs,workloads}
    mkdir -p $LLMB_WORKLOAD
fi

# -------- llmb-nemo commits
if [ $GPU_TYPE = gb300 ] || [ $GPU_TYPE = h100 ]; then
    export NEMO_COMMIT="f260da5d65fccdf9c78283647f7497d9428fb654"
    export MEGATRON_COMMIT="20d73fe76eb03672c02f637e9b47e562b9010d4c"
    export NEMO_RUN_COMMIT="04f900a9c1cde79ce6beca6a175b4c62b99d7982"
    export FW_VERSION=25.09.00
else
    export NEMO_COMMIT="1f422a19d942fab0aa54e3099f4b5752d871cb5d"
    export MEGATRON_COMMIT="ac198fc0d60a8c748597e01ca4c6887d3a7bcf3d"
    export NEMO_RUN_COMMIT="04f900a9c1cde79ce6beca6a175b4c62b99d7982"
    export FW_VERSION=25.07.01
fi

# 1. Clone the NeMo source code
#Setup NeMo
if [ ! -d "$NEMO_DIR" ]; then
    git clone https://github.com/NVIDIA/NeMo.git $NEMO_DIR
fi
# Ensure we are on same commit and have dependencies installed.
pushd $NEMO_DIR
git fetch origin
git checkout -f $NEMO_COMMIT
python3 -m pip install '.[all]'
popd

# 2. Install dependencies
pip install 'scipy<1.13.0'         # a workaround for compatibility issue
pip install 'bitsandbytes==0.46.0' # Future NeMo release 25.07/09 will have this fix.
pip install 'nvidia-modelopt==0.35.1'
pip install 'torch==2.8.0'
pip install 'torchvision==0.23.0'
pip install megatron-core@git+https://github.com/NVIDIA/Megatron-LM.git@$MEGATRON_COMMIT
pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@$NEMO_RUN_COMMIT
