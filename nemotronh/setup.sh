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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=pretraining
export MODEL_NAME=nemotronh
export FW_VERSION=25.04.01

# Ensure STAGE_PATH is not set as it's been replaced by LLMB_INSTALL
if [ -n "${STAGE_PATH+x}" ]; then
  echo "Error: STAGE_PATH is deprecated and should not be set. Please use LLMB_INSTALL instead."
  exit 1
fi

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
export NEMO_COMMIT="7a8f1cc7bc40a84447c0681a9e4e956a135ae8a2"
export MEGATRON_COMMIT="7094270c6fa1dbc6b2e99072171e3a559ca36d0f"
export NEMO_RUN_COMMIT="f3c3ac22fe169acf93aa7647ab9785290537d4a4"

# 1. Clone the NeMo source code
#Setup NeMo 
if [ ! -d "$NEMO_DIR" ]; then
    git clone https://github.com/NVIDIA/NeMo.git $NEMO_DIR
fi
# Ensure we are on same commit and have dependencies installed.
pushd $NEMO_DIR
git fetch origin
git checkout -f $NEMO_COMMIT
./reinstall.sh
popd

# 2. Install dependencies
pip install 'scipy<1.13.0' # a workaround for compatibility issue
pip install 'bitsandbytes==0.45.5' # Future NeMo release 25.07/09 will have this fix.
pip install megatron-core@git+https://github.com/NVIDIA/Megatron-LM.git@$MEGATRON_COMMIT
pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@$NEMO_RUN_COMMIT
