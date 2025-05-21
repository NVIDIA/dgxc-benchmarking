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
#SBATCH --job-name=paxml
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
export FRAMEWORK=paxml
export MODEL=gpt3
export MODEL_SIZE=${MODEL_SIZE:-5b}
export FW_VERSION=2024.03.04

export IMAGE=$STAGE_PATH/ghcr.io+nvidia+jax+pax-${FW_VERSION}.sqsh

export SLURM_NTASKS_PER_NODE=${RUN_CONF_GPU_PER_NODE:-8}
export OPTIMIZATION_NAME=${OPTIMIZATION_NAME-""}
export OPTIMIZATION_CODE=${OPTIMIZATION_CODE-""}
export DTYPE=${DTYPE:-fp8}
export DTYPE=${DTYPE,,}

# BF16 not supported for 175b
if [[ $DTYPE == "bf16" && $MODEL_SIZE == "175b" ]]; then
	echo "Error: Model size 175b does not support BF16."
	exit 1
fi

export JOB_TOTAL_GPUS=${SBATCH_GPUS:-$(( ${SLURM_JOB_NUM_NODES} * ${SLURM_NTASKS_PER_NODE} ))}

export RESULT_DIR=$STAGE_PATH/results/$GSW_VERSION/$DTYPE/$MODEL_SIZE/$JOB_TOTAL_GPUS
export RESULT_FILES_NAME=log-${FRAMEWORK}_${MODEL}_${MODEL_SIZE}_${JOB_TOTAL_GPUS}

mkdir -p $RESULT_DIR

# SRUN_OUTPUT and SRUN_ERROR are Slurm environment variables to control output/error file locations.
export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-"pmix"}
# !Breaking convention! - Paxml outputs the timing info into stderr. Keeping one combined paxml.out file.
export SRUN_OUTPUT=${SRUN_OUTPUT-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.out}

# Workload specific configuration
source ./configure.sh

srun \
  --container-image "$IMAGE" \
  --container-mounts $RESULT_DIR,$STAGE_PATH/cfg:/opt/paxml/workspace \
  --container-writable \
  --no-container-mount-home \
  --container-env=XLA_FLAGS,JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS,JAX_SHARE_BINARY_BETWEEN_HOSTS \
  bash -c "$COMMAND_LINE"
