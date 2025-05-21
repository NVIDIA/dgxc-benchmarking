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
#SBATCH --time=1:00:00
set -eu -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi


export FW_VERSION=25.02.01

export NEMO_DIR="NeMo"
export NEMO_COMMIT="d2b21556f2455b52ad7217f334bb66eaa13c0f83"
export MEGATRON_COMMIT="ee6737da824b18f945946b4c798adf2328374cb7"
export NEMO_RUN_COMMIT="fb550a3738768fbf1100e1579f3ca0e2207c6487"

LLM_REPO=$PWD

# 1. Clone the NeMo source code
#Setup NeMo 
if [ ! -d "$NEMO_DIR" ]; then
    git clone https://github.com/NVIDIA/NeMo
    cd $NEMO_DIR
    git checkout $NEMO_COMMIT
    git apply $LLM_REPO/gsw_nemo.patch
    ./reinstall.sh
fi

# Copy launch scripts to NeMo performance scripts directory
cp $LLM_REPO/*.py $LLM_REPO/NeMo/scripts/performance/llm/

# 2. Install dependencies
pip install 'scipy<1.13.0' # a workaround for compatibility issue
pip install megatron-core@git+https://github.com/NVIDIA/Megatron-LM.git@$MEGATRON_COMMIT
pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@$NEMO_RUN_COMMIT

# 3. Squash image file
srun bash -c "enroot import --output ${STAGE_PATH}/nvidia+nemo+${FW_VERSION}.sqsh docker://nvcr.io#nvidia/nemo:${FW_VERSION}"

