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

# save information about command line value and environment variables used
function capture_env() {
  if [[ "$SLURM_PROCID" = "0" ]]; then
    # only need to print it for first rank
    printenv >> $RESULT_DIR/env_snapshot_${SLURM_JOB_ID}.txt
  fi
}

function nsight_timed_profile_config() {
  export PROFILE_DELAY=${RUN_CONF_PROFILE_DELAY:-300}
  export PROFILE_DURATION=${RUN_CONF_PROFILE_DURATION:-60}
  export PROFILE_RANKS=${RUN_CONF_PROFILE_RANKS:-"0,1,2,3,4,5,6,7"}
  export PROFILE_GPU_METRICS=${RUN_CONF_PROFILE_GPU_METRICS:-false}
  export PROFILE_CPU=${RUN_CONF_PROFILE_CPU:-false}

  # ChatGPT special - The commas around the two variables ensure you only match whole items, asterisks match anyway in list.
  # We only enable profiling it the slurm rank matches a profiling rank.
  if [[ "${PROFILE_ENABLED,,}" = true ]] && [[ ",$PROFILE_RANKS," == *",$SLURM_PROCID,"* ]]; then
    NSYS_EXTRA_OPTIONS=""
    if [[ "$SLURM_LOCALID" = "0" ]] && [[ "${PROFILE_GPU_METRICS,,}" = true ]]; then
      # TODO: condition this on output of "nsys profile --gpu-metrics-device=help"
      NSYS_EXTRA_OPTIONS+="--gpu-metrics-device=all"
    fi
    if [[ "${PROFILE_CPU,,}" = true ]]; then
      NSYS_EXTRA_OPTIONS+=" --sample=process-tree --cpuctxsw=process-tree --event-sample=system-wide --backtrace=lbr --event-sampling-interval=3 --samples-per-backtrace=1 "
    else 
      NSYS_EXTRA_OPTIONS+=" --sample=none --cpuctxsw=none "
    fi
    PROFILE_CMD="which nsys && nsys --version && nsys status --env && \
    mkdir -p ${RESULT_DIR}/nsys/${SLURM_JOB_ID} && \
    nsys profile --output ${RESULT_DIR}/nsys/${SLURM_JOB_ID}/${MODEL}-${MODEL_SIZE}-${DTYPE}_${JOB_TOTAL_GPUS}g_${SLURM_JOB_ID}_%q{SLURM_NODEID}_%q{SLURM_PROCID} \
    --nic-metrics=true $NSYS_EXTRA_OPTIONS --inherit-environment true --force-overwrite true  --stop-on-exit true --trace cuda,nvtx \
    --delay ${PROFILE_DELAY} --duration ${PROFILE_DURATION}"
  else
    PROFILE_CMD=""
  fi
}

# Get Slurm node info on a partition via sinfo. Current sockets, cores, threads, and GRES.
function get_slurm_info() {
  # Takes a partition name and an array to store the results.
  local partition=$1
  local -n arr=$2

  # Processor information
  local sinfo_proc=$(sinfo --noheader -p $partition -o "%z")

  IFS=: read -r sockets cores threads <<< "$sinfo_proc"
  arr['sockets']=$sockets
  arr['cores']=$cores
  arr['threads']=$threads

  # GRES
  local sinfo_gres=$(sinfo --noheader -p $partition -o "%G")
  if [[ "$sinfo_gres" == "(null)" ]]; then
    arr['gres']=False
  else
    arr['gres']=True
  fi
}

function configure_pinning() {
  DISABLE_SMT=${DISABLE_SMT:-false}
  DISABLE_PINNING=${DISABLE_PINNING:-false}

  SRUN_PIN_ARGS=()
  if [[ "${DISABLE_PINNING,,}" = true ]]; then
    return
  fi

  declare -A slurm_info
  get_slurm_info $SLURM_JOB_PARTITION slurm_info

  # Default to cores binding and using all cores.
  BIND_ARGS="nomultithread"
  CPUS_PER_TASK=$(( SLURM_CPUS_ON_NODE / SLURM_NTASKS_PER_NODE ))

  # Handle SMT configuration when available
  if [[ "${slurm_info['threads']}" -gt 1 ]]; then
    if [[ "${DISABLE_SMT,,}" = false ]]; then
      BIND_ARGS="multithread"
    else
      # Only use physical cores if SMT is available but disabled
      CPUS_PER_TASK=$(((slurm_info['sockets'] * slurm_info['cores']) / SLURM_NTASKS_PER_NODE))
    fi
  fi

  # Build array with srun flags based on pinning configuration
  SRUN_PIN_ARGS=(
    "--cpus-per-task=$CPUS_PER_TASK"
    "--cpu-bind=verbose"
    "--hint=$BIND_ARGS"
    "--distribution=*:block"
  )
  echo "Enabling pinning with args: ${SRUN_PIN_ARGS[@]}"

}
