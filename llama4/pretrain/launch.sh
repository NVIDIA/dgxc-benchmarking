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

# Check Bash version
if [ "${BASH_VERSION:0:1}" -lt 4 ] || [ "${BASH_VERSION:0:1}" -eq 4 ] && [ "${BASH_VERSION:2:1}" -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

# Required environment variables
: "${SBATCH_ACCOUNT:?Required variable SBATCH_ACCOUNT}"
: "${SBATCH_PARTITION:?Required variable SBATCH_PARTITION}"
: "${HF_TOKEN:?Required variable Hugging Face token}"
: "${GPU_TYPE:?Required variable GPU_TYPE (Eg: gb200, h100)}"
: "${JOB_TOTAL_GPUS:?Required variable JOB_TOTAL_GPUS}"
: "${LLMB_INSTALL:?Required variable LLMB_INSTALL}"

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=llama4-maverick
export FW_VERSION=25.04.01
export GSW_VERSION=25.07

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

# Ensure STAGE_PATH is not set as it's been replaced by LLMB_INSTALL
if [ -n "${STAGE_PATH+x}" ]; then
  echo "Error: STAGE_PATH is deprecated and should not be set. Please use LLMB_INSTALL instead."
  exit 1
fi

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-${LLMB_INSTALL}/images/nvidia+nemo+${FW_VERSION}.sqsh}

export NEMORUN_HOME=$LLMB_WORKLOAD
export HF_HOME=${HF_HOME:-$LLMB_WORKLOAD}
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD}

#Default values
GPU_TYPE=${GPU_TYPE,,}
DTYPE=${DTYPE:-bf16}
MODEL_SIZE=${MODEL_SIZE:-400b}
MODEL_SIZE=${MODEL_SIZE,,}
MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
NCCLTRACE_ENABLED=${ENABLE_NCCLTRACE:-false}
NCCLTRACE_ENABLED=${NCCLTRACE_ENABLED,,}

export CONTAINER_MOUNTS="${LLMB_WORKLOAD}/NeMo:/opt/NeMo"
export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

if [ $GPU_TYPE = "gb200" ]; then
  #Default launch configs for GB200
  GPUS_PER_NODE=${GPUS_PER_NODE:-4}
  #Parallelism settings for GB200
  TP=${TP:-1}
  PP=${PP:-2}
  CP=${CP:-1}
  EP=${EP:-64}
  VP=${VP:-12}
  ETP=${ETP:-1}
  MBS=${MBS:-1}
  CUDA_GRAPH=${CUDA_GRAPH:-false}
  GBS=$(( JOB_TOTAL_GPUS * 8 ))
  TIME_LIMIT=${TIME_LIMIT:-"00:30:00"}
elif [ $GPU_TYPE = "h100" ]; then
  #Default launch config for H100
  GPUS_PER_NODE=${GPUS_PER_NODE:-8}
  #Parallelism settings for h100
  TP=${TP:-4}
  PP=${PP:-1}
  CP=${CP:-1}
  EP=${EP:-128}
  VP=${VP:-1}
  ETP=${ETP:-4}
  MBS=${MBS:-1}
  GBS=$(( JOB_TOTAL_GPUS * 2 ))
  if [ "$DTYPE" = "fp8" ]; then
    CUDA_GRAPH=${CUDA_GRAPH:-true}
  else
    CUDA_GRAPH=${CUDA_GRAPH:-false}
  fi
  TIME_LIMIT=${TIME_LIMIT:-"00:55:00"}
else
  echo "$GPU_TYPE not supported"
  exit 1
fi


CONFIG_OVERRIDES=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -ep $EP \
  -vp $VP \
  -et $ETP \
  -gb $GBS \
  -mb $MBS \
  --cuda_graphs $CUDA_GRAPH \
"

if [ $PROFILE_ENABLED = true ] && [ $NCCLTRACE_ENABLED = true ]; then
  echo "Cannot both profile and get NCCL traces"
  exit 1
fi

if [[ "$PROFILE_ENABLED" = "true" ]]; then
  CONFIG_OVERRIDES+=" -en "
  CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
  CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
  if [[ $GPU_METRICS_ENABLED = true ]]; then
    CONFIG_OVERRIDES+=" -pgm "
  fi
  MAX_STEPS=$PROFILE_STOP_STEP
elif [[ "$NCCLTRACE_ENABLED" = "true" ]]; then
  CONFIG_OVERRIDES+=" -nt "
  MAX_STEPS=5
  TIME_LIMIT="00:15:00"
fi

SCRIPT_NAME="scripts.performance.llm.pretrain_llama4_e128"

pushd "${LLMB_WORKLOAD}/NeMo"

python3 -m ${SCRIPT_NAME} \
    -cm "${CONTAINER_MOUNTS}" \
    -hf "${HF_TOKEN}" \
    --gpu "${GPU_TYPE}" \
    --num_gpus "${JOB_TOTAL_GPUS}" \
    --container_image "${IMAGE}" \
    --compute_dtype "${DTYPE}" \
    --gpus_per_node "${GPUS_PER_NODE}" \
    --max_steps "${MAX_STEPS}" \
    $CONFIG_OVERRIDES \
    slurm \
    --account "${SBATCH_ACCOUNT}" \
    --partition "${SBATCH_PARTITION}" \
    --log_dir "${NEMORUN_HOME}" \
    --time_limit $TIME_LIMIT

popd
