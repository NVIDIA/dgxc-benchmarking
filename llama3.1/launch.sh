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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

export FW_VERSION=25.02.01
export GSW_VERSION=25.04
export LLMB_PATH=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$STAGE_PATH/nvidia+nemo+$FW_VERSION.sqsh}
export HF_TOKEN=${HF_TOKEN?"Required variable HF_TOKEN"}
export NEMO_HOME=${NEMO_HOME:-$STAGE_PATH}
export NEMORUN_HOME=${NEMORUN_HOME:-$STAGE_PATH}
export HF_HOME=${HF_HOME:-$STAGE_PATH}

export MAX_STEPS=${MAX_STEPS:-50}
export DTYPE=${DTYPE:-bf16}
export MODEL_SIZE=${MODEL_SIZE:-8b}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export TIME_LIMIT=${TIME_LIMIT:-"00:30:00"}

export DISABLE_PERFRUN=${DISABLE_PERFRUN:-false}
export ENABLE_PROFILE=${ENABLE_PROFILE:-false}
export ENABLE_NCCLTRACE=${ENABLE_NCCLTRACE:-false}
export ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-false}
export ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT,,}

export PROFILE_START_STEP=${PROFILE_START_STEP:-20}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-25}


export OPTIMIZATION_NAME=${OPTIMIZATION_NAME:-""}
export OPTIMIZATION_CODE=${OPTIMIZATION_CODE:-""}

if [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
  MAX_STEPS=1
fi

EXTRAS=""
if [[ $DISABLE_PERFRUN = true ]]; then EXTRAS+="-dp "; fi
if [[ $ENABLE_PROFILE = true ]]; then 
  EXTRAS+="-ns "
  EXTRAS+="--profiling_start_step=$PROFILE_START_STEP "
  EXTRAS+="--profiling_stop_step=$PROFILE_STOP_STEP "
  MAX_STEPS=$PROFILE_STOP_STEP
fi
if [[ $ENABLE_NCCLTRACE = true ]]; then EXTRAS+="-nt "; fi
if [[ $OPTIMIZATION_NAME != "" ]]; then EXTRAS+="-on \"${OPTIMIZATION_NAME}\" -oc \"${OPTIMIZATION_CODE}\" "; fi

#run command
pushd $LLMB_PATH/NeMo
python3 -m scripts.performance.llm.pretrain_llama31 -a $SBATCH_ACCOUNT -p $SBATCH_PARTITION -t $TIME_LIMIT -l $NEMORUN_HOME -i $IMAGE -m $MODEL_SIZE -c $DTYPE -ms $MAX_STEPS -ng $JOB_TOTAL_GPUS -gn $GPUS_PER_NODE -hf $HF_TOKEN -fw $FW_VERSION -gsw $GSW_VERSION ${EXTRAS}

popd
