#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For each dataset a user elects to use, the user is responsible for
# checking if the dataset license is fit for the intended purpose.

# Parameters
#SBATCH --job-name=maxtext_llama2
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

export GSW_VERSION=24.11
export FRAMEWORK=maxtext
export MODEL=llama2
export MODEL_SIZE=70b
export FW_VERSION=2024.12.09

export IMAGE=${RUN_CONF_IMAGE:-$STAGE_PATH/nvidia+jax+maxtext-${FW_VERSION}.sqsh}

export SLURM_NTASKS_PER_NODE=${RUN_CONF_GPU_PER_NODE:-8}
export OPTIMIZATION_NAME=${OPTIMIZATION_NAME-""}
export OPTIMIZATION_CODE=${OPTIMIZATION_CODE-""}
export DTYPE=${DTYPE:-fp8}
export DTYPE=${DTYPE,,}

export JOB_TOTAL_GPUS=${SBATCH_GPUS:-$(( ${SLURM_JOB_NUM_NODES} * ${SLURM_NTASKS_PER_NODE} ))}

export RESULT_DIR=$STAGE_PATH/results/$GSW_VERSION/$DTYPE/$MODEL_SIZE/$JOB_TOTAL_GPUS
export RESULT_FILES_NAME=log-${FRAMEWORK}_${MODEL}_${MODEL_SIZE}_${JOB_TOTAL_GPUS}

mkdir -p $RESULT_DIR

# SRUN_OUTPUT and SRUN_ERROR are Slurm environment variables to control output/error file locations.
export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-"pmix"}
export SRUN_OUTPUT=${SRUN_OUTPUT-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.out}
export SRUN_ERROR=${SRUN_ERROR-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.err}

srun \
  --container-image "$IMAGE" \
  --container-mounts "$RESULT_DIR,$STAGE_PATH/cfg:/cfg" \
  --container-writable \
  --no-container-mount-home bash -c " \
  source /cfg/configure.sh && \
  set_profile_environment && \
  run_profiling && \
  generate_pbtxt && \
  set_performance_environment && \
  run_performance"
