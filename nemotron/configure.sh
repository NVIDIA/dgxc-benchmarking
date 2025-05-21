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

# setup
export TRANSFORMERS_OFFLINE=1
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0
export HYDRA_FULL_ERROR=1

export PRE_CMD="
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  export CUDA_DEVICE_MAX_CONNECTIONS=1;
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"

export PROFILE_ENABLED=${ENABLE_PROFILE:-false}
# not supported currently
export CHECKPOINT_ENABLED=false
# only synthetic data is supported by the benchmark currently
export SYNTHETIC_DATA_ENABLED=true
export NCCL_TRACE_ENABLED=${ENABLE_NCCL_TRACE:-false}

export ENV_VARS=""
export CONFIG_OVERRIDES=""

export MAX_STEPS=${RUN_CONF_MAX_STEPS:-50}

if [[ $MODEL_SIZE = 15b ]]; then
  # 15b
  export NVTE_FUSED_ATTN=1
  TP=4
  PP=1
  MP=$(( TP * PP ))
  VP=null
  MBS=4
  GBS=$(( JOB_TOTAL_GPUS * 4 ))
  NVTE_VARS=""
else
  # 340b
  export NVTE_FUSED_ATTN=0
  TP=8
  PP=8
  MP=$(( TP * PP ))
  VP=12
  MBS=1
  GBS=$(( SLURM_JOB_NUM_NODES * 2 ))
  NEMO_CONDITIONAL_CFGS=/opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py
  NVTE_VARS="NVTE_FWD_LAYERNORM_SM_MARGIN=\$(python3 $NEMO_CONDITIONAL_CFGS name=get_ln_sm_margin) \
  NVTE_BWD_LAYERNORM_SM_MARGIN=\$(python3 $NEMO_CONDITIONAL_CFGS name=get_ln_sm_margin) \
  NVTE_UB_SPLIT_AG=\$(python3 $NEMO_CONDITIONAL_CFGS name=get_ag_overlap fp8=${FP8_ENABLED} )"
  CONFIG_OVERRIDES+=" \
    model.overlap_p2p_comm=True \
    model.batch_p2p_comm=False"
fi

export CONFIG_OVERRIDES+=" model.global_batch_size=$GBS \
  trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=${MAX_STEPS} \
  trainer.limit_val_batches=1 \
  run.results_dir=${RESULT_DIR} \
  model.data.index_mapping_dir=$INDEX_MAPPING_DIR \
  model.tokenizer.model=/cfg/nemotron_2_256k.model \
  model.tensor_model_parallel_size=$TP \
  model.pipeline_model_parallel_size=$PP \
  model.virtual_pipeline_model_parallel_size=$VP \
  model.fp8=${FP8_ENABLED^} \
  trainer.enable_checkpointing=${CHECKPOINT_ENABLED^} \
  exp_manager.explicit_log_dir=${RESULT_DIR}/results \
  exp_manager.create_checkpoint_callback=False \
  exp_manager.checkpoint_callback_params.model_parallel_size=$MP \
  model.ub_tp_comm_overlap=True \
  model.sequence_parallel=True \
  model.micro_batch_size=$MBS \
  model.mcore_gpt=True \
  model.transformer_engine=True \
  model.fp8_hybrid=True \
  model.nsys_profile.enabled=${PROFILE_ENABLED^} \
  +model.fp8_params=${FP8_ENABLED^}"
  
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
export PROFILE_RANKS=${RUN_CONF_PROFILE_RANKS:-"0,1,2,3,4,5,6,7"}
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
  $PRE_CMD $NVTE_VARS $PROFILE_CMD python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/cfg \
  --config-name=${MODEL}-${MODEL_SIZE}-synth \
  $CONFIG_OVERRIDES $PROFILE_CFG"

