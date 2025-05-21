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
export NVTE_FUSED_ATTN=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_NET_CHUNKSIZE=2097152

export PRE_CMD="
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;"

export PROFILE_ENABLED=${ENABLE_PROFILE:-false}
export CHECKPOINT_ENABLED=false # Not supported with this workload.
export SYNTHETIC_DATA_ENABLED=true # Only synthetic data is supported by this workload currently.

export ENV_VARS=""
export CONFIG_OVERRIDES=""

export MAX_STEPS=${RUN_CONF_MAX_STEPS:-50}

# NCCL trace support
nccl_trace_config

# Defaults
NUM_LAYERS=64
TP=4
PP=8
MP=$(( TP * PP ))
VP=8
EP=8
CP=2
MBS=1
GBS=$(( SLURM_JOB_NUM_NODES * 16 ))
GC_INTERVAL=40
SEQ_LEN=8192
OPTIM_OVERLAP=true # requires PP
NVTE_VARS=""
CONFIG_OVERRIDES+=""

## Should this be JOB_TOTAL_GPUS?
if [[ $SLURM_JOB_NUM_NODES = 32 ]]; then
  GBS=1024
  NUM_LAYERS=32
  PP=4
elif [[ $SLURM_JOB_NUM_NODES = 16 ]]; then
  GBS=1024
  NUM_LAYERS=16
  PP=2
elif [[ $SLURM_JOB_NUM_NODES = 8 ]]; then
  GBS=1024
  NUM_LAYERS=8
  PP=1
  VP=null
  OPTIM_OVERLAP=false
elif [[ $SLURM_JOB_NUM_NODES = 4 ]]; then
  GBS=1024
  NUM_LAYERS=4
  PP=1
  CP=1
  VP=null
  SEQ_LEN=4096
  OPTIM_OVERLAP=false
elif [[ $SLURM_JOB_NUM_NODES = 2 ]]; then
  GBS=1024
  NUM_LAYERS=4
  PP=1
  CP=1
  EP=4
  VP=null
  SEQ_LEN=4096
  OPTIM_OVERLAP=false
elif [[ $SLURM_JOB_NUM_NODES = 1 ]]; then
  GBS=1024
  NUM_LAYERS=2
  PP=1
  CP=1
  EP=2
  VP=null
  SEQ_LEN=4096
  OPTIM_OVERLAP=false
else
  echo "Invalid scale"
fi

export CONFIG_OVERRIDES+=" model.global_batch_size=$GBS \
  trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=${MAX_STEPS} \
  trainer.limit_val_batches=0.0 \
  run.results_dir=${RESULT_DIR} \
  exp_manager.explicit_log_dir=${RESULT_DIR}/results \
  model.encoder_seq_length=$SEQ_LEN \
  model.data.seq_length=$SEQ_LEN \
  model.data.index_mapping_dir=$INDEX_MAPPING_DIR \
  model.tensor_model_parallel_size=$TP \
  model.pipeline_model_parallel_size=$PP \
  model.context_parallel_size=$CP \
  model.expert_model_parallel_size=$EP \
  model.virtual_pipeline_model_parallel_size=$VP \
  exp_manager.checkpoint_callback_params.model_parallel_size=$MP \
  model.num_layers=$NUM_LAYERS \
  model.optim.overlap_param_gather_with_optimizer_step=$OPTIM_OVERLAP \
  model.fp8=${FP8_ENABLED^} \
  trainer.enable_checkpointing=${CHECKPOINT_ENABLED^} \
  exp_manager.create_checkpoint_callback=False \
  model.ub_tp_comm_overlap=true \
  model.sequence_parallel=true \
  model.micro_batch_size=$MBS \
  model.mcore_gpt=true \
  model.transformer_engine=true \
  model.fp8_hybrid=true \
  model.nsys_profile.enabled=${PROFILE_ENABLED^} \
  model.fp8_params=${FP8_ENABLED^} \
  model.gc_interval=${GC_INTERVAL}"

# capture command line overrides prior to optimizations
BASE_CONFIG=$CONFIG_OVERRIDES

inject_optimizations

export INFO_STR="GSW: MODEL=${MODEL} FRAMEWORK=${FRAMEWORK} MODEL_SIZE=${MODEL_SIZE} JOB_NUM_NODES=${SLURM_JOB_NUM_NODES} GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE} DTYPE=${DTYPE} SYNTHETIC_DATA=${SYNTHETIC_DATA_ENABLED^} GSW_VERSION=${GSW_VERSION} IMAGE=\'${IMAGE}\' FW_VERSION=${FW_VERSION} JOB_ID=${SLURM_JOB_ID} JOB_MODE=training OPTIMIZATION_NAME=\'${OPTIMIZATION_NAME}\' OPTIMIZATION_CODE=\'${OPTIMIZATION_CODE}\' BASE_CONFIG=\'${BASE_CONFIG}\'"

nsight_profile_config

export COMMAND_LINE="$ENV_VARS \
  echo $INFO_STR; \
  $PRE_CMD $NVTE_VARS $PROFILE_CMD python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/cfg \
  --config-name=${MODEL}_${MODEL_SIZE}.yaml \
  $CONFIG_OVERRIDES $PROFILE_CFG"

function launch() {
  capture_env

  eval $COMMAND_LINE
}
