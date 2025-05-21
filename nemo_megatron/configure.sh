#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -eu -o pipefail

export GSW_VERSION=${GSW_VERSION?"Required variable GSW_VERSION is not set in the container. Aborting"}

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

function nsight_profile_config() {
  export PROFILE_START_STEP=${RUN_CONF_PROFILE_START_STEP:-20}
  export PROFILE_STOP_STEP=${RUN_CONF_PROFILE_STOP_STEP:-30}
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
    mkdir -p ${RESULT_DIR}/nsys && \
    nsys profile --output ${RESULT_DIR}/nsys/${MODEL}-${MODEL_SIZE}-${DTYPE}_${JOB_TOTAL_GPUS}g_${SLURM_JOB_ID}_%q{SLURM_NODEID}_%q{SLURM_LOCALID} \
    --nic-metrics=true $NSYS_EXTRA_OPTIONS --inherit-environment true --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --stop-on-exit true --trace cuda,nvtx "
    PROFILE_CFG="model.nsys_profile.start_step=$PROFILE_START_STEP model.nsys_profile.end_step=$PROFILE_STOP_STEP model.nsys_profile.ranks=[$PROFILE_RANKS]"
  else
    PROFILE_CMD=""
    PROFILE_CFG=""
  fi
}

if [[ "${SYNTHETIC_DATA_ENABLED,,}" = true ]]; then
  DATA_IMPL=mock
  DATA_PREFIX=""
else
  DATA_IMPL=mmap
  DATA_PREFIX="0.25,/datasets/my-gpt3_00_text_document,0.25,/datasets/my-gpt3_01_text_document,0.25,/datasets/my-gpt3_02_text_document,0.25,/datasets/my-gpt3_03_text_document"
fi

export MAX_STEPS=${RUN_CONF_MAX_STEPS:-50}

if [[ "${NCCL_TRACE_ENABLED,,}" = true ]]; then
  export NCCL_DEBUG_SUBSYS="COLL,P2P,NET"
  export NCCL_DEBUG=INFO
  MAX_STEPS=10
fi

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

# prototype for handling optimizations
if [[ -n "${OPTIMIZATION_NAME:-""}" ]] && [[ -n "${OPTIMIZATION_CODE:-""}" ]]; then
	# inject optimization parameters into command line
	CONFIG_OVERRIDES+=" "$OPTIMIZATION_CODE
else
	OPTIMIZATION_NAME=""
	OPTIMIZATION_CODE=""
fi

export INFO_STR="GSW: MODEL=${MODEL} FRAMEWORK=${FRAMEWORK} MODEL_SIZE=${MODEL_SIZE} JOB_NUM_NODES=${SLURM_JOB_NUM_NODES} GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE} DTYPE=${DTYPE} SYNTHETIC_DATA=${SYNTHETIC_DATA_ENABLED^} GSW_VERSION=${GSW_VERSION} FW_VERSION=${FW_VERSION} IMAGE=\'${IMAGE}\' JOB_ID=${SLURM_JOB_ID} JOB_MODE=training OPTIMIZATION_NAME=\'${OPTIMIZATION_NAME}\' OPTIMIZATION_CODE=\'${OPTIMIZATION_CODE}\' BASE_CONFIG=\'${BASE_CONFIG}\'"

nsight_profile_config

export COMMAND_LINE="$ENV_VARS \
  echo $INFO_STR; \
  $PRE_CMD $NVTE_VARS $PROFILE_CMD python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/cfg \
  --config-name=gpt3_${MODEL_SIZE}_hydra.yaml \
  $CONFIG_OVERRIDES $PROFILE_CFG"

function launch() {
  eval $COMMAND_LINE
}

