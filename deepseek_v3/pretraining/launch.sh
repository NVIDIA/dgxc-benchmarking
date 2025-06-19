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

export WORKLOAD_TYPE=pretraining
export MODEL_NAME=deepseek_v3
export FW_VERSION=25.04.01
export GSW_VERSION=25.05.01

export OPENBLAS_NUM_THREADS=1 # optional, to avoid resource contention at the frontend node.

# Ensure STAGE_PATH is not set as it's been replaced by LLMB_INSTALL
if [ -n "${STAGE_PATH+x}" ]; then
  echo "Error: STAGE_PATH is deprecated and should not be set. Please use LLMB_INSTALL instead."
  exit 1
fi

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

export NEMORUN_HOME=$LLMB_WORKLOAD
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD}
export HF_HOME=${HF_HOME:-$LLMB_WORKLOAD}
export HF_TOKEN=${HF_TOKEN?"Required variable HF_TOKEN"}

export MAX_STEPS=${MAX_STEPS:-50}
export DTYPE=bf16 # fp8 not supported currently
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-8}
export GPU_TYPE=${GPU_TYPE:-"gb200"}
export GPU_TYPE=${GPU_TYPE,,}
export TIME_LIMIT=${TIME_LIMIT:-"00:55:00"}

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
NCCLTRACE_ENABLED=${ENABLE_NCCLTRACE:-false}
NCCLTRACE_ENABLED=${NCCLTRACE_ENABLED,,}

export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}


CONFIG_OVERRIDES=""

if [ $PROFILE_ENABLED = true ] && [ $NCCLTRACE_ENABLED = true ]; then
  echo "Cannot both profile and get NCCL traces"
  exit 1
fi

if [[ "$PROFILE_ENABLED" = "true" ]]; then
  CONFIG_OVERRIDES+=" -en "
  CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
  CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
  MAX_STEPS=$PROFILE_STOP_STEP
elif [[ "$NCCLTRACE_ENABLED" = "true" ]]; then
  CONFIG_OVERRIDES+=" -nt "
  MAX_STEPS=5
  TIME_LIMIT="00:15:00"
fi

CP=1
MBS=1
NUM_LAYERS=61
HIDDEN_SIZE=7168
GBS=$(( $JOB_TOTAL_GPUS * 8 ))

if [[ $GPU_TYPE = "h100"  ]]; then
  GPUS_PER_NODE=${GPUS_PER_NODE:-8}
  TP=2
  PP=16
  EP=64
  VP=1
  ETP=1
  AOL=0
  RM='mla_up_proj'
elif [[ $GPU_TYPE = "gb200" ]]; then
  GPUS_PER_NODE=${GPUS_PER_NODE:-4}
  TP=2
  PP=4
  if [[ $JOB_TOTAL_GPUS -eq 128 ]]; then
    EP=32
  else
    EP=64
  fi
  VP=1
  ETP=1
  AOL=0
  RM='core_attn'
fi

#run command
pushd $LLMB_WORKLOAD/NeMo

python3 -m scripts.performance.llm.pretrain_deepseek_v3 \
	-i $IMAGE \
	-c $DTYPE \
	-hf $HF_TOKEN \
	-g $GPU_TYPE \
	-ng $JOB_TOTAL_GPUS \
	-gn $GPUS_PER_NODE \
 	-ms $MAX_STEPS \
	--num_layers $NUM_LAYERS \
    --hidden_size $HIDDEN_SIZE \
	-tp $TP -pp $PP -cp $CP -vp $VP -ep $EP -mb $MBS -gb $GBS \
	--expert_tensor_parallel_size $ETP --activation_offload_layers $AOL --recompute_modules $RM \
	${CONFIG_OVERRIDES} \
	slurm \
	--account $SBATCH_ACCOUNT \
	--partition $SBATCH_PARTITION \
	--log_dir $NEMORUN_HOME \
	--time_limit $TIME_LIMIT 

popd
