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

set -eu -o pipefail

export GSW_VERSION=${GSW_VERSION?"Required variable GSW_VERSION is not set in the container. Aborting"}

if [[ ! "$MODEL_SIZE" =~ ^(8b|70b|405b)$ ]]; then
  echo "FATAL: unsupported MODEL_SIZE $MODEL_SIZE for llama 3.1" >&2
  exit 1
fi

if [[ ! "$DTYPE" =~ ^(bf16|fp8)$ ]]; then
  echo "FATAL: unsupported DTYPE $DTYPE for llama 3.1 $MODEL_SIZE" >&2
  exit 1
fi

# setup
export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=False
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NEMO_LOG_MEMORY_USAGE=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=8
export HYDRA_FULL_ERROR=1

export PRE_CMD="
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  export CUDA_DEVICE_MAX_CONNECTIONS=1;
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"

export PROFILE_ENABLED=${ENABLE_PROFILE:-false}

export ENV_VARS=""
export CONFIG_OVERRIDES=""

MAX_STEPS=${MAX_STEPS:-50}

if [[ "$MODEL_SIZE" = "8b" ]]; then
  DEFAULT_PROFILE_RANKS="0,1,2,3,4,5,6,7"
  CONFIG_OVERRIDES+=" model.tokenizer.type=/dataset/llama"
  # Upstream uses GBS 128 for 8 GPUs, scale it with number of total gpus.
  GBS=${GBS:-$((128 * JOB_TOTAL_GPUS / 8))}
elif [[ "$MODEL_SIZE" = "70b" ]]; then
  DEFAULT_PROFILE_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  CONFIG_OVERRIDES+=" model.tokenizer.type=/dataset/llama"
  # Upstream uses GBS 128 for 64 GPUs, scale it with number of total gpus.
  GBS=${GBS:-$((128 * JOB_TOTAL_GPUS / 64))}
elif [[ "$MODEL_SIZE" = "405b" ]]; then
  DEFAULT_PROFILE_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  export SYNTHETIC_DATA_ENABLED=true
  export NCCL_P2P_NET_CHUNKSIZE=262144
  CONFIG_OVERRIDES+=" model.data.data_impl=\"mock\""
  CONFIG_OVERRIDES+=" model.data.data_prefix=\"\""
  CONFIG_OVERRIDES+=" model.ub_tp_comm_overlap_cfg.proj_fprop.fp8_buf=$FP8_ENABLED"
  CONFIG_OVERRIDES+=" model.ub_tp_comm_overlap_cfg.fc2_fprop.fp8_buf=$FP8_ENABLED"

  # Upstream uses GBS 252 for 576 GPUs, scale it with number of total gpus.
  GBS=${GBS:-$((252 * JOB_TOTAL_GPUS / 576))}
  if [[ $JOB_TOTAL_GPUS = 192 ]]; then
    GBS=252
    NUM_LAYERS=42
    PP=3
  elif [[ $JOB_TOTAL_GPUS = 96 ]]; then
    GBS=126
    NUM_LAYERS=21
    PP=3
  elif [[ $JOB_TOTAL_GPUS = 32 ]]; then
    GBS=126
    NUM_LAYERS=7
    PP=1
    VP=null
    CONFIG_OVERRIDES+=" model.defer_embedding_wgrad_compute=false"
  fi
fi

CONFIG_OVERRIDES+=" ++data_dir=/dataset"
CONFIG_OVERRIDES+=" run.results_dir=$RESULT_DIR"
CONFIG_OVERRIDES+=" model.data.index_mapping_dir=$INDEX_MAPPING_DIR"
CONFIG_OVERRIDES+=" trainer.num_nodes=$SLURM_JOB_NUM_NODES"
CONFIG_OVERRIDES+=" trainer.devices=$SLURM_NTASKS_PER_NODE"
CONFIG_OVERRIDES+=" trainer.max_steps=$MAX_STEPS"
CONFIG_OVERRIDES+=" trainer.val_check_interval=$MAX_STEPS"
CONFIG_OVERRIDES+=" model.global_batch_size=$GBS"
CONFIG_OVERRIDES+=" model.fp8=$FP8_ENABLED"
CONFIG_OVERRIDES+=" model.fp8_hybrid=$FP8_ENABLED"
CONFIG_OVERRIDES+=" +model.fp8_params=$FP8_ENABLED"
[[ -n ${TP-} ]] && CONFIG_OVERRIDES+=" model.tensor_model_parallel_size=$TP"
[[ -n ${PP-} ]] && CONFIG_OVERRIDES+=" model.pipeline_model_parallel_size=$PP"
[[ -n ${VP-} ]] && CONFIG_OVERRIDES+=" model.virtual_pipeline_model_parallel_size=$VP"
[[ -n ${CP-} ]] && CONFIG_OVERRIDES+=" model.context_parallel_size=$CP"
if [[ -n ${SEQ_LEN-} ]]; then
  CONFIG_OVERRIDES+=" model.encoder_seq_length=$SEQ_LEN"
  CONFIG_OVERRIDES+=" model.max_position_embeddings=$SEQ_LEN"
  CONFIG_OVERRIDES+=" model.data.seq_length=$SEQ_LEN"
fi
[[ -n ${NUM_LAYERS-} ]] && CONFIG_OVERRIDES+=" model.num_layers=$NUM_LAYERS"
CONFIG_OVERRIDES+=" model.nsys_profile.enabled=${PROFILE_ENABLED^} "

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

export PROFILE_START_STEP=${RUN_CONF_PROFILE_START_STEP:-20}
export PROFILE_STOP_STEP=${RUN_CONF_PROFILE_STOP_STEP:-30}
export PROFILE_RANKS=${DEFAULT_PROFILE_RANKS:-"0,1,2,3,4,5,6,7"}
export PROFILE_GPU_METRICS=${RUN_CONF_PROFILE_GPU_METRICS:-false}

if [[ "${PROFILE_ENABLED,,}" = true ]]; then
  NSYS_EXTRA_OPTIONS=""
  if [[ "$SLURM_LOCALID" = "0" ]] && [[ "${PROFILE_GPU_METRICS,,}" = true ]]; then
    NSYS_EXTRA_OPTIONS="--gpu-metrics-device=all"
  fi
  PROFILE_CMD="which nsys && nsys --version && nsys status --env && \
  mkdir -p ${RESULT_DIR}/nsys && \
  nsys profile --output ${RESULT_DIR}/nsys/${MODEL}-${MODEL_SIZE}-${DTYPE}_${JOB_TOTAL_GPUS}g_${SLURM_JOB_ID}_%q{SLURM_NODEID}_%q{SLURM_LOCALID} \
  --nic-metrics=true $NSYS_EXTRA_OPTIONS --inherit-environment true --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --stop-on-exit true --trace cuda,nvtx --sample none --cpuctxsw none"
  PROFILE_CFG="model.nsys_profile.start_step=$PROFILE_START_STEP model.nsys_profile.end_step=$PROFILE_STOP_STEP model.nsys_profile.ranks=[$PROFILE_RANKS]"
else
  PROFILE_CMD=""
  PROFILE_CFG=""
fi

export COMMAND_LINE="$ENV_VARS \
  echo $INFO_STR; \
  $PRE_CMD $PROFILE_CMD python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/cfg \
  --config-name=llama3.1_${MODEL_SIZE}.yaml \
  $CONFIG_OVERRIDES $PROFILE_CFG"
