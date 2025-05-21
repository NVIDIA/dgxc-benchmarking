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
#SBATCH --job-name=nemo_megatron
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1:00:00

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

LAUNCH_PATH="$PWD" # dir of this script
LLMB_PATH="${LAUNCH_PATH%/*}" # parent dir

# common functions
source $LLMB_PATH/common/common-utils.sh

export FRAMEWORK=nemo
export MODEL=megatron
export MODEL_SIZE=175b
export GSW_VERSION=25.04.01
export FW_VERSION=24.12

export IMAGE=${RUN_CONF_IMAGE:-$STAGE_PATH/nvidia+nemo+${FW_VERSION}.sqsh}

export SLURM_NTASKS_PER_NODE=${RUN_CONF_GPU_PER_NODE:-8}
export OPTIMIZATION_NAME=${OPTIMIZATION_NAME-""}
export OPTIMIZATION_CODE=${OPTIMIZATION_CODE-""}
export DTYPE=${DTYPE:-fp8}
export DTYPE=${DTYPE,,}
if [[ "${DTYPE}" = fp8 ]]; then
  export FP8_ENABLED=true
else
  export FP8_ENABLED=false
fi

# `eval` is needed to expand shell vars that might be in RUN_CONF_RESULT_DIR, such as `$SLURM_JOB_ID`
# otherwise, using RUN_CONF_RESULT_DIR=/path/to/results/${SLURM_JOB_ID} might create a directory named as such verbatim,
# instead of /path/to/results/12345/
RUN_CONF_RESULT_DIR=$(eval echo "${RUN_CONF_RESULT_DIR:-}")

export JOB_TOTAL_GPUS=${SBATCH_GPUS:-$(( ${SLURM_JOB_NUM_NODES} * ${SLURM_NTASKS_PER_NODE} ))}

export NCCL_TRACE_ENABLED=${ENABLE_NCCL_TRACE:-false}
GSW_VERSION_SUFFIX=""
if [[ "${NCCL_TRACE_ENABLED,,}" = true ]]; then
  echo "NCCL tracing enabled. Large log files will be stored in a dedicated folder"
  GSW_VERSION_SUFFIX="-nccl-trace"
fi

export RESULT_DIR=${RUN_CONF_RESULT_DIR:-$STAGE_PATH/results/${GSW_VERSION}${GSW_VERSION_SUFFIX}/$DTYPE/$MODEL_SIZE/$JOB_TOTAL_GPUS}
export RESULT_FILES_NAME=log-${FRAMEWORK}_${MODEL}_${MODEL_SIZE}_${JOB_TOTAL_GPUS}

export DATA_DIR=${RUN_CONF_DATA_DIR:-$STAGE_PATH/gpt3-dataset}
export INDEX_MAPPING_DIR=${RUN_CONF_INDEX_DIR:-$STAGE_PATH}/index_mapping

mkdir -p $RESULT_DIR
mkdir -p $INDEX_MAPPING_DIR
mkdir -p $DATA_DIR

CONTAINER_MOUNTS="$LAUNCH_PATH/cfg:/cfg,$LAUNCH_PATH/configure.sh:/gsw/configure.sh,$LLMB_PATH/common:/gsw/common,$RESULT_DIR,$INDEX_MAPPING_DIR,${DATA_DIR}:/datasets/"
if [[ -n "${RUN_CONF_EXTRA_MOUNTS:-""}" ]]; then
  CONTAINER_MOUNTS+=",${RUN_CONF_EXTRA_MOUNTS}"
fi

# SRUN_OUTPUT and SRUN_ERROR are Slurm environment variables to control output/error file locations.
export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-"pmix"}
export SRUN_OUTPUT=${SRUN_OUTPUT-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.out}
export SRUN_ERROR=${SRUN_ERROR-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.err}

configure_pinning

srun \
  --container-image "$IMAGE" \
  --container-mounts "$CONTAINER_MOUNTS" \
  "${SRUN_PIN_ARGS[@]}" \
  --container-writable \
  --no-container-mount-home bash -c "source /gsw/configure.sh && launch"
