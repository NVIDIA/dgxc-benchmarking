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

function gpt3_dataset_config() {
  if [[ "${SYNTHETIC_DATA_ENABLED,,}" = true ]]; then
    DATA_IMPL=mock
    DATA_PREFIX=""
  else
    DATA_IMPL=mmap
    DATA_PREFIX="0.25,/datasets/my-gpt3_00_text_document,0.25,/datasets/my-gpt3_01_text_document,0.25,/datasets/my-gpt3_02_text_document,0.25,/datasets/my-gpt3_03_text_document"
  fi
}

# NCCL trace support
function nccl_trace_config() {
  if [[ "${NCCL_TRACE_ENABLED,,}" = true ]]; then
    export NCCL_DEBUG_SUBSYS="COLL,P2P,NET"
    export NCCL_DEBUG=INFO
    MAX_STEPS=10
  fi
}

function nsight_profile_config() {
  export PROFILE_START_STEP=${RUN_CONF_PROFILE_START_STEP:-20}
  export PROFILE_STOP_STEP=${RUN_CONF_PROFILE_STOP_STEP:-25}
  # Let individual workloads override the default ranks, but RUN_CONF_PROFILE_RANKS takes precedent.
  export DEFAULT_PROFILE_RANKS=${DEFAULT_PROFILE_RANKS:-"0,1,2,3,4,5,6,7"}
  export PROFILE_RANKS=${RUN_CONF_PROFILE_RANKS:-$DEFAULT_PROFILE_RANKS}
  export PROFILE_GPU_METRICS=${RUN_CONF_PROFILE_GPU_METRICS:-false}
  export PROFILE_CPU=${RUN_CONF_PROFILE_CPU:-false}

  if [[ "${PROFILE_ENABLED,,}" = true ]]; then
    NSYS_EXTRA_OPTIONS=""
    if [[ "$SLURM_LOCALID" = "0" ]] && [[ "${PROFILE_GPU_METRICS,,}" = true ]]; then
      # TODO: condition this on output of "nsys profile --gpu-metrics-device=help"
      NSYS_EXTRA_OPTIONS+=" --gpu-metrics-device=all"
    fi
    if [[ "${PROFILE_CPU,,}" = true ]]; then
      NSYS_EXTRA_OPTIONS+=" --sample=process-tree --cpuctxsw=process-tree --event-sample=system-wide --backtrace=lbr --event-sampling-interval=3 --samples-per-backtrace=1 "
    else 
      NSYS_EXTRA_OPTIONS+=" --sample=none --cpuctxsw=none "
    fi
    PROFILE_CMD="which nsys && nsys --version && nsys status --env && \
    mkdir -p ${RESULT_DIR}/nsys/${SLURM_JOB_ID} && \
    nsys profile --output ${RESULT_DIR}/nsys/${SLURM_JOB_ID}/${MODEL}-${MODEL_SIZE}-${DTYPE}_${JOB_TOTAL_GPUS}g_${SLURM_JOB_ID}_%q{SLURM_NODEID}_%q{SLURM_PROCID} \
    --nic-metrics=true $NSYS_EXTRA_OPTIONS --inherit-environment true --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --stop-on-exit true --trace cuda,nvtx "
    PROFILE_CFG="model.nsys_profile.start_step=$PROFILE_START_STEP model.nsys_profile.end_step=$PROFILE_STOP_STEP model.nsys_profile.ranks=[$PROFILE_RANKS] trainer.max_steps=$PROFILE_STOP_STEP"
  else
    PROFILE_CMD=""
    PROFILE_CFG=""
  fi
}

function inject_optimizations() {
  # prototype for handling optimizations
  if [[ -n "${OPTIMIZATION_NAME:-""}" ]] && [[ -n "${OPTIMIZATION_CODE:-""}" ]]; then
    # inject optimization parameters into command line
    CONFIG_OVERRIDES+=" "$OPTIMIZATION_CODE 
  else
    OPTIMIZATION_NAME=""
    OPTIMIZATION_CODE=""
  fi
}

