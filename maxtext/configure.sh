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

# disabled checkpointing
export CHECKPOINT_ENABLED=false

# only synthetic data is supported by the benchmark currently
export SYNTHETIC_DATA_ENABLED=true

# This option configures whether profiling is run during the benchmark section of code.
# Profiling is always enabled at the beginning to support PGLE.
export PROFILE_ENABLED=${ENABLE_PROFILE:-false}

MBS=2
ici_TP=1
ici_FSDP=${SLURM_NTASKS_PER_NODE}
ici_DP=1
dcn_TP=1
dcn_FSDP=${SLURM_JOB_NUM_NODES}
dcn_DP=1
MEM_FRACTION=0.94
POLICY="save_qkv_proj"
AR_MULTIPLE=""
AG_MULTIPLE=""
RS_MULTIPLE=""
export ENV_VARS=""
export CONFIG_OVERRIDES=""

if [[ ${SLURM_JOB_NUM_NODES} -ge 64 ]]; then
  dcn_FSDP=16
  dcn_DP=$(( SLURM_JOB_NUM_NODES / dcn_FSDP ))
fi

if [[ "${DTYPE}" = fp8 ]]; then
  QUANTIZATION="fp8"
  AR_MULTIPLE=4
  AG_MULTIPLE=4
  if [[ ${SLURM_JOB_NUM_NODES} -ge 64 ]]; then
    RS_MULTIPLE=256
  else
    RS_MULTIPLE=128
  fi

elif [[ "${DTYPE}" = bf16 ]]; then
  QUANTIZATION=""
  AR_MULTIPLE=4
  AG_MULTIPLE=4
  if [[ ${SLURM_JOB_NUM_NODES} -ge 64 ]]; then
    RS_MULTIPLE=512
  else
    RS_MULTIPLE=256
  fi

else
  echo "Unsupported precision ($DTYPE) selected"
  exit 1
fi

RUN_NAME="${FRAMEWORK}_${MODEL}_${MODEL_SIZE}_${SLURM_JOB_NUM_NODES}n_${DTYPE}_${POLICY}_${SLURM_JOB_ID}"

BASE_THRESHOLD=$((8*1073741824)) # 8GB
AR_THRESHOLD=$((BASE_THRESHOLD/AR_MULTIPLE))
AG_THRESHOLD=$((BASE_THRESHOLD/AG_MULTIPLE))
RS_THRESHOLD=$((BASE_THRESHOLD/RS_MULTIPLE))

HLO_NAME="${RUN_NAME}-prof-hlo"

HLO_DUMP_PATH="${RESULT_DIR}/hlo_logs/${HLO_NAME}"

PGLE_PROFILE_PATH="${RESULT_DIR}/lhs_pbtxt/${RUN_NAME}.pbtxt"

inject_optimizations

BASE_XLA_FLAGS="--xla_gpu_enable_triton_gemm=false \
  --xla_gpu_graph_level=0 \
  --xla_gpu_enable_highest_priority_async_stream=true \
  --xla_gpu_all_reduce_combine_threshold_bytes=${AR_THRESHOLD} \
  --xla_gpu_all_gather_combine_threshold_bytes=${AG_THRESHOLD} \
  --xla_gpu_reduce_scatter_combine_threshold_bytes=${RS_THRESHOLD} \
  --xla_gpu_enable_pipelined_all_gather=true \
  --xla_gpu_enable_pipelined_reduce_scatter=true \
  --xla_gpu_enable_pipelined_all_reduce=true \
  --xla_gpu_enable_while_loop_double_buffering=true \
  --xla_gpu_enable_triton_softmax_fusion=false \
  --xla_gpu_enable_all_gather_combine_by_dim=false \
  --xla_gpu_enable_reduce_scatter_combine_by_dim=false \
  --xla_disable_hlo_passes=rematerialization \
  --xla_dump_hlo_as_text"

BASE_CONFIG="use_iota_embed=true \
  scan_layers=true \
  per_device_batch_size=${MBS} \
  model_name=${MODEL}-${MODEL_SIZE} \
  remat_policy=${POLICY} \
  enable_checkpointing=false \
  logits_dot_in_fp32=false \
  base_output_directory=local_train \
  dataset_path=local \
  dataset_type=synthetic \
  attention=cudnn_flash_te \
  tokenizer_path=/opt/maxtext/assets/tokenizer_llama3.tiktoken \
  max_target_length=8192 \
  quantization=${QUANTIZATION} \
  hardware=gpu_multiprocess \
  dcn_fsdp_parallelism=${dcn_FSDP} \
  ici_fsdp_parallelism=${ici_FSDP} \
  ici_data_parallelism=${ici_DP} \
  dcn_data_parallelism=${dcn_DP} \
  ici_tensor_parallelism=${ici_TP} \
  dcn_tensor_parallelism=${dcn_TP} \
  monitor_goodput=False \
  enable_goodput_recording=False \
  profiler=nsys \
  skip_first_n_steps_for_profiler=9 \
  profiler_steps=5 "

export MAX_STEPS=${RUN_CONF_MAX_STEPS:-50}

export INFO_STR="GSW: MODEL=${MODEL} FRAMEWORK=${FRAMEWORK} MODEL_SIZE=${MODEL_SIZE} JOB_NUM_NODES=${SLURM_JOB_NUM_NODES} GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE} DTYPE=${DTYPE} SYNTHETIC_DATA=${SYNTHETIC_DATA_ENABLED^} GSW_VERSION=${GSW_VERSION} FW_VERSION=${FW_VERSION} IMAGE='${IMAGE}' JOB_ID=${SLURM_JOB_ID} JOB_MODE=training OPTIMIZATION_NAME='${OPTIMIZATION_NAME}' OPTIMIZATION_CODE='${OPTIMIZATION_CODE}' BASE_CONFIG='${BASE_CONFIG}'  MBS=$MBS ici_TP=$ici_TP ici_FSDP=$ici_FSDP ici_DP=$ici_DP dcn_TP=$dcn_TP dcn_FSDP=$dcn_FSDP dcn_DP=$dcn_DP AR_THRESHOLD=$AR_THRESHOLD AG_THRESHOLD=$AG_THRESHOLD RS_THRESHOLD=$RS_THRESHOLD MEM_FRACTION=$MEM_FRACTION POLICY=$POLICY QUANTIZATION=$QUANTIZATION"

echo $INFO_STR

#Updating the BASE_CONFIG with the CONFIG_OVERRIDE after printing the INFO_STR
BASE_CONFIG+=" "$CONFIG_OVERRIDES

export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}
export CUDA_DEVICE_MAX_CONNECTIONS=16
export NVTE_FUSED_ATTN=1
export NCCL_IB_SL=1
eval $ENV_VARS

set_profile_environment() {
  mkdir -p ${RESULT_DIR}/nsys/
  mkdir -p ${RESULT_DIR}/pgle/
  mkdir -p ${HLO_DUMP_PATH}

  # three false
  export XLA_FLAGS="${BASE_XLA_FLAGS} --xla_gpu_enable_latency_hiding_scheduler=false \
    --xla_gpu_disable_async_collectives=allreduce,allgather,reducescatter \
    --xla_dump_to=${HLO_DUMP_PATH}"
}


run_profiling() {
  echo "XLA_FLAGS = ${XLA_FLAGS}"
  echo "XLA_PYTHON_CLIENT_MEM_FRACTION = ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
  echo "POLICY = ${POLICY}"

  PROFILE_RUN_SETTINGS="/opt/maxtext/MaxText/train.py /opt/maxtext/MaxText/configs/base.yml run_name=${RUN_NAME}-prof \
    steps=15 ${BASE_CONFIG}"

  NSYS_OUTPUT_FILE="${RESULT_DIR}/pgle/${SLURM_JOB_ID}/${MODEL}-${MODEL_SIZE}-${DTYPE}_${JOB_TOTAL_GPUS}g_${SLURM_JOB_ID}_%q{SLURM_NODEID}_%q{SLURM_PROCID}"

  NSYS_CMD="nsys profile -s none -o ${NSYS_OUTPUT_FILE} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"

  echo "Command: ${NSYS_CMD} python3 ${PROFILE_RUN_SETTINGS}"
  bash -c "${NSYS_CMD} python3 ${PROFILE_RUN_SETTINGS}"

}

generate_pbtxt() {
    mkdir -p "${RESULT_DIR}/lhs_pbtxt"
    # generate the protobuf
    test "${SLURM_PROCID}" -eq 0 && python3 /opt/jax/jax/tools/pgo_nsys_converter.py --profile_path ${NSYS_OUTPUT_FILE}.nsys-rep --post_process --pgle_output_path ${PGLE_PROFILE_PATH} || echo "First jax process will generate the pbtxt"
}

set_performance_environment() {
  # export TF_CPP_VMODULE=profile_guided_latency_estimator=10,latency_hiding_scheduler=10
  export TF_CPP_VMODULE=profile_guided_latency_estimator=10
  export TF_CPP_MIN_LOG_LEVEL=0
  export TF_CPP_MAX_LOG_LEVEL=100

  # everything true
  export XLA_FLAGS="${BASE_XLA_FLAGS} --xla_gpu_enable_latency_hiding_scheduler=true \
    --xla_dump_to=${HLO_DUMP_PATH} \
    --xla_gpu_pgle_profile_file_or_directory_path=${PGLE_PROFILE_PATH}"
}

run_performance() {
  NSYS_CMD=""
  PERF_PROFILE_OVERRIDE=""
  #Aligning profiling options with the rest of the workloads
  export PROFILE_START_STEP=${RUN_CONF_PROFILE_START_STEP:-10}
  export PROFILE_STOP_STEP=${RUN_CONF_PROFILE_STOP_STEP:-15}
  export PROFILE_GPU_METRICS=${RUN_CONF_PROFILE_GPU_METRICS:-false}

  if [[ "${PROFILE_ENABLED,,}" = true ]] && [[ "${SLURM_PROCID}" -eq 0 ]]; then
    NSYS_EXTRA_OPTIONS=""
    if  [[ "${PROFILE_GPU_METRICS,,}" = true ]]; then
      NSYS_EXTRA_OPTIONS="--gpu-metrics-device=all"
    fi

    PERF_PROFILE_OVERRIDE="skip_first_n_steps_for_profiler=$((RUN_CONF_PROFILE_START_STEP - 1)) \
    profiler_steps=$((PROFILE_STOP_STEP - PROFILE_START_STEP))"

    NSYS_OUTPUT_FILE="${RESULT_DIR}/nsys/${SLURM_JOB_ID}/${MODEL}-${MODEL_SIZE}-${DTYPE}_${JOB_TOTAL_GPUS}g_${SLURM_JOB_ID}_%q{SLURM_NODEID}_%q{SLURM_PROCID}"

    NSYS_CMD="nsys profile -s none -o ${NSYS_OUTPUT_FILE} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop ${NSYS_EXTRA_OPTIONS}"

    MAX_STEPS=$PROFILE_STOP_STEP
  fi

  PERF_RUN_SETTINGS="/opt/maxtext/MaxText/train.py /opt/maxtext/MaxText/configs/base.yml run_name=${RUN_NAME}-perf \
    steps=${MAX_STEPS} ${BASE_CONFIG} ${PERF_PROFILE_OVERRIDE}"

  echo "Command: ${NSYS_CMD} python3 ${PERF_RUN_SETTINGS}"

  ${NSYS_CMD} python3 ${PERF_RUN_SETTINGS}
}

function launch() {
	capture_env

	set_profile_environment
	run_profiling
	generate_pbtxt
	set_performance_environment
	run_performance
}
