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

set -eu -o pipefail

# common functions
source /gsw/common/common-utils.sh
source /gsw/common/nemo/nemo-utils.sh

export GSW_VERSION=${GSW_VERSION?"Required variable GSW_VERSION is not set. Aborting"}

# setup
export TRANSFORMERS_OFFLINE=1
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export HYDRA_FULL_ERROR=1

export NVTE_FUSED_ATTN=1
export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0
export TOKENIZERS_PARALLELISM=False

export PRE_CMD="
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  export CUDA_DEVICE_MAX_CONNECTIONS=1;
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"

export PROFILE_ENABLED=${ENABLE_PROFILE:-false}
export CHECKPOINT_ENABLED=${ENABLE_CHECKPOINT:-false}
export SYNTHETIC_DATA_ENABLED=${ENABLE_SYNTHETIC_DATA:-false}

export ENV_VARS=""
export CONFIG_OVERRIDES=""

# Switch between synthetic and real data
gpt3_dataset_config

export MAX_STEPS=${RUN_CONF_MAX_STEPS:-50}

nccl_trace_config

if [[ $MODEL_SIZE = 175b ]]; then
  # 175b
  GBS=$(( SLURM_JOB_NUM_NODES * 16 ))
  NEMO_CONDITIONAL_CFGS=/opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py
  NVTE_VARS="NVTE_FWD_LAYERNORM_SM_MARGIN=\$(python3 $NEMO_CONDITIONAL_CFGS name=get_ln_sm_margin) \
  NVTE_BWD_LAYERNORM_SM_MARGIN=\$(python3 $NEMO_CONDITIONAL_CFGS name=get_ln_sm_margin) \
  NVTE_UB_SPLIT_AG=\$(python3 $NEMO_CONDITIONAL_CFGS name=get_ag_overlap fp8=${FP8_ENABLED} )"
fi

export CONFIG_OVERRIDES+=" model.global_batch_size=$GBS \
  trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=${MAX_STEPS} \
  run.results_dir=${RESULT_DIR} \
  model.data.index_mapping_dir=$INDEX_MAPPING_DIR \
  exp_manager.explicit_log_dir=${RESULT_DIR}/results \
  model.fp8=${FP8_ENABLED} \
  model.data.data_impl=$DATA_IMPL \
  model.data.data_prefix=[$DATA_PREFIX] \
  trainer.enable_checkpointing=${CHECKPOINT_ENABLED^} \
  model.nsys_profile.enabled=${PROFILE_ENABLED^} " 

# capture command line overrides prior to optimizations
BASE_CONFIG=$CONFIG_OVERRIDES

inject_optimizations

export INFO_STR="GSW: MODEL=${MODEL} FRAMEWORK=${FRAMEWORK} MODEL_SIZE=${MODEL_SIZE} JOB_NUM_NODES=${SLURM_JOB_NUM_NODES} GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE} DTYPE=${DTYPE} SYNTHETIC_DATA=${SYNTHETIC_DATA_ENABLED^} GSW_VERSION=${GSW_VERSION} FW_VERSION=${FW_VERSION} IMAGE=\'${IMAGE}\' JOB_ID=${SLURM_JOB_ID} JOB_MODE=training OPTIMIZATION_NAME=\'${OPTIMIZATION_NAME}\' OPTIMIZATION_CODE=\'${OPTIMIZATION_CODE}\' BASE_CONFIG=\'${BASE_CONFIG}\'"

nsight_profile_config

export COMMAND_LINE="$ENV_VARS \
  echo $INFO_STR; \
  $PRE_CMD $NVTE_VARS $PROFILE_CMD python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/cfg \
  --config-name=gpt3_${MODEL_SIZE}_hydra.yaml \
  $CONFIG_OVERRIDES $PROFILE_CFG"

function launch() {
  capture_env
  
  eval $COMMAND_LINE
}

