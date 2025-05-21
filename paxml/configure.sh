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
export PROFILE_ENABLED=${ENABLE_PROFILE:-false}
export SYNTHETIC_DATA_ENABLED=true

# setup
XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_simplify_all_fp_conversions --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true  --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_enable_triton_softmax_fusion=false  --xla_gpu_all_reduce_combine_threshold_bytes=536870912 --xla_gpu_graph_level=0  --xla_gpu_enable_async_all_reduce=true --xla_gpu_all_gather_combine_threshold_bytes=536870912 --xla_gpu_reduce_scatter_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=false --xla_gpu_enable_xla_runtime_executable=false  --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_cudnn_fmha=false"


export NVTE_FUSED_ATTN=1
export ENABLE_TE=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=10
export JAX_TRACEBACK_IN_LOCATIONS_LIMIT=-1
export JAX_SHARE_BINARY_BETWEEN_HOSTS=False
export JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=False
export VOCAB_PATH=gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model
export HOPPER_EXPERIMENTAL_OVERRIDE=1

export ENV_VARS=""

if [[ "${DTYPE}" = fp8 ]]; then
  # ENABLE_FP8 is used by PaXML framework
	export ENABLE_FP8=1
else
	export ENABLE_FP8=0
fi

if [[ $MODEL_SIZE = 5b ]]; then
	# 5b
	MBS=4
	FSDP=1
	MAX_STEPS=500
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
elif [[ $MODEL_SIZE = 175b ]]; then
	# 175b
	MBS=2
	FSDP=16
	MAX_STEPS=100
	if [[ "${DTYPE}" = fp8 ]]; then
		XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
	else
		XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
	fi
else
   echo "Unrecognized Model Size: ${MODEL_SIZE}"
   exit 1
fi
DP=$(( SLURM_JOB_NUM_NODES / FSDP ))
CONFIG=workspace.${FRAMEWORK}_mod_configs.Synthetic${MODEL_SIZE^^}Ckpt
export XLA_PYTHON_CLIENT_MEM_FRACTION # export var set above.

BASE_CONFIG=$XLA_FLAGS

# prototype for handling optimizations
if [[ -n "${OPTIMIZATION_NAME:-""}" ]] && [[ -n "${OPTIMIZATION_CODE:-""}" ]]; then
  # inject optimization parameters into command line
  XLA_FLAGS+=" "$OPTIMIZATION_CODE 
else
  OPTIMIZATION_NAME=""
  OPTIMIZATION_CODE=""
fi

export INFO_STR="GSW: MODEL=${MODEL} FRAMEWORK=${FRAMEWORK} MODEL_SIZE=${MODEL_SIZE} JOB_NUM_NODES=${SLURM_JOB_NUM_NODES} GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE} DTYPE=${DTYPE} SYNTHETIC_DATA=True XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION} GSW_VERSION=${GSW_VERSION} FW_VERSION=${FW_VERSION} IMAGE=\'${IMAGE}\' JOB_ID=${SLURM_JOB_ID} JOB_MODE=training OPTIMIZATION_NAME=\'${OPTIMIZATION_NAME}\' OPTIMIZATION_CODE=\'${OPTIMIZATION_CODE}\' BASE_CONFIG=\'${BASE_CONFIG}\'"

export PROFILE_DELAY=${RUN_CONF_PROFILE_DELAY:-300}
export PROFILE_DURATION=${RUN_CONF_PROFILE_DURATION:-60}
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
  --nic-metrics=true $NSYS_EXTRA_OPTIONS --inherit-environment true --force-overwrite true  --stop-on-exit true --trace cuda,nvtx --sample none --cpuctxsw none \
  --delay ${PROFILE_DELAY} --duration ${PROFILE_DURATION}"
else
  PROFILE_CMD=""
fi

export XLA_FLAGS
export COMMAND_LINE="echo $INFO_STR; 
	$ENV_VARS \
	$PROFILE_CMD \
	python3 -u -m paxml.main \
	--job_log_dir=${RESULT_DIR}/results \
	--fdl_config=${CONFIG} \
	--fdl.MAX_STEPS=${MAX_STEPS} \
	--fdl.PERCORE_BATCH_SIZE=${MBS} \
	--fdl.DCN_MESH_SHAPE=[${DP},${FSDP},1] \
	--enable_checkpoint_saving=False \
	--multiprocess_gpu \
	--server_addr=\${SLURM_LAUNCH_NODE_IPADDR}:12345 \
	--num_hosts=\${SLURM_NTASKS} \
	--host_idx=\${SLURM_PROCID} \
	--alsologtostderr"
