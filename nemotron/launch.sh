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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=nemotron4
export FW_VERSION=25.04.01
export GSW_VERSION=25.07

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

# Ensure STAGE_PATH is not set as it's been replaced by LLMB_INSTALL
if [ -n "${STAGE_PATH+x}" ]; then
  echo "Error: STAGE_PATH is deprecated and should not be set. Please use LLMB_INSTALL instead."
  exit 1
fi

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}
CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}
CLUSTER_TYPE=${CLUSTER_TYPE,,}
DTYPE=${DTYPE:-fp8}
DTYPE=${DTYPE,,}
MODEL_SIZE=${MODEL_SIZE:-15b}
MODEL_SIZE=${MODEL_SIZE,,}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
ENABLE_GPU_METRICS=${ENABLE_GPU_METRICS:-false}
ENABLE_GPU_METRICS=${ENABLE_GPU_METRICS,,}
STRONG_SCALING=${STRONG_SCALING:-false}
STRONG_SCALING=${STRONG_SCALING,,}
MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-false}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT,,}


if [[ $MODEL_SIZE = 15b ]] && [[ $STRONG_SCALING = true ]]; then
  echo "Strong scaling is only supported with MODEL_SIZE=340b. Current MODEL_SIZE=$MODEL_SIZE"
  exit 1
fi

if [[ $GPU_TYPE = h100 ]] && [[ $STRONG_SCALING = true ]]; then
  echo "Strong scaling is not supported for h100. Please use GPU_TYPE=<gb200, b200>"
  exit 1
fi

export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

if [ $GPU_TYPE = gb200 -o $GPU_TYPE = b200 ]; then
  if [ $GPU_TYPE = gb200 ]; then
  	GPUS_PER_NODE=${GPUS_PER_NODE:-4}
  else
	GPUS_PER_NODE=${GPUS_PER_NODE:-8}
  fi

  if [ $MODEL_SIZE = 15b ]; then
    TP=1
    PP=1
    CP=1
    GBS=${GBS:-$(( JOB_TOTAL_GPUS * 4 ))}
    MBS=2
    VP=1
    NUM_LAYERS=32
    HIDDEN_SIZE=6144
  elif [ $MODEL_SIZE = 340b ]; then
    TP=8
    PP=4
    CP=1
    if [ $STRONG_SCALING = true ]; then
      GBS=512
      TIME_LIMIT=${TIME_LIMIT:-"00:35:00"}
    else
      GBS=${GBS:-$(( JOB_TOTAL_GPUS / 4 ))}
    fi
    MBS=1
    VP=12
    NUM_LAYERS=96
    HIDDEN_SIZE=18432
  fi
elif [ $GPU_TYPE = h100 ]; then
  GPUS_PER_NODE=${GPUS_PER_NODE:-8}
  if [ $MODEL_SIZE = 15b ]; then
    TP=2
    PP=1
    CP=1
    GBS=${GBS:-$(( JOB_TOTAL_GPUS * 4 ))}
    MBS=2
    VP=0
    NUM_LAYERS=32
    HIDDEN_SIZE=6144
  elif [ $MODEL_SIZE = 340b ]; then
    TP=8
    PP=8
    CP=1
    GBS=${GBS:-$(( JOB_TOTAL_GPUS / 4 ))}
    MBS=1
    VP=12
    NUM_LAYERS=96
    HIDDEN_SIZE=18432
  fi
else
    echo "${GPU_TYPE} is unsupported for this workload."
fi


CONFIG_OVERRIDES=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -gb $GBS \
  -mb $MBS \
  --num_layers $NUM_LAYERS \
  --hidden_size $HIDDEN_SIZE \
  -ep 1 \
"

if [ $GPU_TYPE = gb200 ] || [ $GPU_TYPE = b200 ]; then
  CONFIG_OVERRIDES+=" -fsdp 0 --cuda_graphs true "
fi

if [ $VP != 0 ]; then
  CONFIG_OVERRIDES+=" -vp $VP " 
fi

if [ $PROFILE_ENABLED = true ]; then
  CONFIG_OVERRIDES+=" -en " 
  CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
  CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
  if [[ $ENABLE_GPU_METRICS = true ]]; then
      CONFIG_OVERRIDES+=" -pgm "
  fi
  MAX_STEPS=$PROFILE_STOP_STEP
fi

if [[ $ENABLE_CHECKPOINT = true ]]; then
  CONFIG_OVERRIDES+=" --checkpoint_save=True "
else
  CONFIG_OVERRIDES+=" --checkpoint_save=False "
fi

if [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
  MAX_STEPS=1
  CONFIG_OVERRIDES+=" --checkpoint_load_path=$LOAD_CHECKPOINT_PATH "
fi

# After all overrides - STRONG_SCALING is significantly slower at 128 than weak scaling.
TIME_LIMIT=${TIME_LIMIT:-"00:20:00"}

pushd $LLMB_WORKLOAD/NeMo

if [ $CLUSTER_TYPE = slurm ]; then
  python3 -m scripts.performance.llm.pretrain_nemotron4_${MODEL_SIZE} \
    --gpu $GPU_TYPE \
    --container_image $IMAGE \
    --compute_dtype $DTYPE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --max_steps $MAX_STEPS \
    $CONFIG_OVERRIDES \
    slurm \
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir ${NEMORUN_HOME} \
    --time_limit $TIME_LIMIT
else
  python3 -m scripts.performance.llm.pretrain_nemotron4_${MODEL_SIZE} \
    --gpu $GPU_TYPE \
    --container_image nvcr.io/nvidia/nemo:$FW_VERSION \
    --compute_dtype $DTYPE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --max_steps $MAX_STEPS \
    --custom_mounts $CUSTOM_MOUNT \
    $CONFIG_OVERRIDES \
    runai \
    --base_url $BASE_URL \
    --app_id $APP_ID \
    --app_secret $APP_SECRET \
    --project_name $PROJECT_NAME \
    --pvc_nemo_run_dir $PVC_DIR
fi

popd
