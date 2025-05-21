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
#SBATCH --job-name=nemo_llama3.1
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:00:00

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

export GSW_VERSION=24.11
export FRAMEWORK=nemo
export MODEL=llama3.1
export FW_VERSION=24.09
export SYNTHETIC_DATA_ENABLED=False # 405b is true

export IMAGE=${RUN_CONF_IMAGE:-$STAGE_PATH/nvidia+nemo+${FW_VERSION}.sqsh}

export DTYPE=${DTYPE:-fp8}
export DTYPE=${DTYPE,,}
if [[ "${DTYPE}" = fp8 ]]; then
  export FP8_ENABLED=true
else
  export FP8_ENABLED=false
fi

export JOB_TOTAL_GPUS=${SBATCH_GPUS:-$(( ${SLURM_JOB_NUM_NODES} * ${SLURM_NTASKS_PER_NODE} ))}

export RESULT_DIR=$STAGE_PATH/results/$GSW_VERSION/$DTYPE/$MODEL_SIZE/$JOB_TOTAL_GPUS
export RESULT_FILES_NAME=log-${FRAMEWORK}_${MODEL}_${MODEL_SIZE}_${JOB_TOTAL_GPUS}

export DATA_DIR=$STAGE_PATH/llama3.1-dataset
export INDEX_MAPPING_DIR=${RUN_CONF_INDEX_DIR:-$STAGE_PATH}/index_mapping

mkdir -p "$RESULT_DIR"
mkdir -p $INDEX_MAPPING_DIR

# SRUN_OUTPUT and SRUN_ERROR are Slurm environment variables to control output/error file locations.
export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-"pmix"}
export SRUN_OUTPUT=${SRUN_OUTPUT-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.out}
export SRUN_ERROR=${SRUN_ERROR-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.err}

# Workload specific configuration
source ./configure.sh

srun \
  --container-image "$IMAGE" \
  --container-mounts $RESULT_DIR,$INDEX_MAPPING_DIR,$DATA_DIR:/dataset,$STAGE_PATH/cfg:/cfg \
  --container-writable \
  --no-container-mount-home bash -c "$COMMAND_LINE"
