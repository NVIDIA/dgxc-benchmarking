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
#SBATCH --job-name=nemo_megatron
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:00:00

export GSW_VERSION=24.11
export FRAMEWORK=nemo
export MODEL=megatron
export MODEL_SIZE=175b
export FW_VERSION=24.05
export SYNTHETIC_DATA_ENABLED=False

IMAGE=$STAGE_PATH/nvidia+nemo+24.05.sqsh
GBS=$(( SLURM_JOB_NUM_NODES * 16 ))
DATA_DIR=$STAGE_PATH/gpt3-dataset
INDEX_MAPPING_DIR=$STAGE_PATH/index_mapping

export DTYPE=${DTYPE:-fp8}
export DTYPE=${DTYPE,,}
if [[ "${DTYPE}" = fp8 ]]; then
  export FP8_ENABLED=true
else
  export FP8_ENABLED=false
fi

# setup
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export MAX_STEPS=50

export SLURM_NTASKS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
export JOB_TOTAL_GPUS=${SBATCH_GPUS:-$(( ${SLURM_JOB_NUM_NODES} * ${SLURM_NTASKS_PER_NODE} ))}
RESULT_DIR=$STAGE_PATH/results/$GSW_VERSION/$DTYPE/$MODEL_SIZE/$JOB_TOTAL_GPUS

mkdir -p $INDEX_MAPPING_DIR
mkdir -p $RESULT_DIR

CONFIG_OVERRIDES="model.global_batch_size=$GBS \
  model.fp8=${FP8_ENABLED} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=${MAX_STEPS} \
  trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  run.results_dir=${RESULT_DIR} \
  model.data.index_mapping_dir=$INDEX_MAPPING_DIR \
  exp_manager.explicit_log_dir=${RESULT_DIR}/results"

BASE_CONFIG=$CONFIG_OVERRIDES

# prototype for handling optimizations
if [[ -n "${OPTIMIZATION_NAME:-""}" ]] && [[ -n "${OPTIMIZATION_CODE:-""}" ]]; then
  # inject optimization parameters into command line
  CONFIG_OVERRIDES+=" "$OPTIMIZATION_CODE 
else
  OPTIMIZATION_NAME=""
  OPTIMIZATION_CODE=""
fi

export INFO_STR="GSW: MODEL=${MODEL} FRAMEWORK=${FRAMEWORK} MODEL_SIZE=${MODEL_SIZE} JOB_NUM_NODES=${SLURM_JOB_NUM_NODES} GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE} DTYPE=${DTYPE} SYNTHETIC_DATA=${SYNTHETIC_DATA_ENABLED^} GSW_VERSION=${GSW_VERSION} FW_VERSION=${FW_VERSION} IMAGE=\'${IMAGE}\' JOB_ID=${SLURM_JOB_ID} JOB_MODE=training OPTIMIZATION_NAME=\'${OPTIMIZATION_NAME}\' OPTIMIZATION_CODE=\'${OPTIMIZATION_CODE}\' BASE_CONFIG=\'${BASE_CONFIG}\'"

export RESULT_FILES_NAME=log-${FRAMEWORK}_${MODEL}_${MODEL_SIZE}_${JOB_TOTAL_GPUS}

export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-"pmix"}
export SRUN_OUTPUT=${SRUN_OUTPUT-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.out}
export SRUN_ERROR=${SRUN_ERROR-${RESULT_DIR}/${RESULT_FILES_NAME}_%j.err}

srun --container-image ${IMAGE} \
     --container-writable \
     --container-mounts ${DATA_DIR}:/datasets/,${RESULT_DIR},$INDEX_MAPPING_DIR,${STAGE_PATH}/cfg:/cfg/ \
     --no-container-mount-home bash -c "
  echo $INFO_STR;
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NVTE_FWD_LAYERNORM_SM_MARGIN=\$(python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) NVTE_BWD_LAYERNORM_SM_MARGIN=\$(python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) NVTE_UB_SPLIT_AG=\$(python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ag_overlap fp8=${FP8_ENABLED} ) python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/cfg/ \
  --config-name=gpt3_175b_hydra.yaml \
  $CONFIG_OVERRIDES"
