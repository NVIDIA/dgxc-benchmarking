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

source speed-config.sh

LOG_FILE="speed_test_log_$(date +"%Y-%m-%d_%H-%M-%S").log"
exec &> >(tee -a "$LOG_FILE")

COMMAND=" \
genai-perf profile \
  -m $NIM_MODEL_NAME \
  --endpoint-type chat \
  --service-kind openai \
  --streaming -u \$SERVICE \
  --num-prompts \$total_requests \
  --synthetic-input-tokens-mean \$Request_ISL \
  --synthetic-input-tokens-stddev 0 \
  --concurrency \$CR \
  \$NAMESPACE_PARAM \
  --output-tokens-mean \$OSL \
  --extra-inputs max_tokens:\$OSL \
  --extra-inputs min_tokens:\$OSL \
  --extra-inputs ignore_eos:true \
  --artifact-dir $OUTPUT/\$EXPORT_FILE \
  --tokenizer $NIM_MODEL_TOKENIZER \
  -- -v --max-threads=\$CR --request-count \$total_requests"

# Calculate total number of experiments
total_experiments=0
for ((i=1; i<=$N_TIMES; i++)); do
  for CR in ${CONCURRENCY_RANGE}; do
    for use_case in "${!USE_CASES[@]}"; do
        total_experiments=$((total_experiments + 1)) # RAG Off
        for vdb_reranker in $VDB_and_RERANKER_K; do
            total_experiments=$((total_experiments + 2)) # RAG On + RAG Off with RAG-ISL
        done
    done
  done
done

echo "Total experiments to run: $total_experiments"

# Counters for progress tracking
completed_experiments=0
start_time=$(date +%s)

# Function to run the command and log timing
run_experiment() {
  local EXPORT_FILE=$1
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting experiment: $EXPORT_FILE"
  
  START_TIME=$(date +%s)
  eval "$COMMAND"
  END_TIME=$(date +%s)
  
  elapsed_time=$((END_TIME - START_TIME))
  completed_experiments=$((completed_experiments + 1))
  
  # Calculate estimated time remaining
  total_elapsed_time=$((END_TIME - start_time))
  avg_time_per_exp=$((total_elapsed_time / completed_experiments))
  remaining_experiments=$((total_experiments - completed_experiments))
  estimated_remaining_time=$((avg_time_per_exp * remaining_experiments))

  echo "[$(date +"%Y-%m-%d %H:%M:%S")] Completed experiment: $EXPORT_FILE in $elapsed_time seconds"
  echo "Progress: $completed_experiments / $total_experiments completed."
  echo "Estimated time remaining: $((estimated_remaining_time / 60)) min $((estimated_remaining_time % 60)) sec"

  sleep $SLEEP_TIME
}

# Run all experiments
for ((i=1; i<=$N_TIMES; i++)); do
  for CR in ${CONCURRENCY_RANGE}; do
    for use_case in "${!USE_CASES[@]}"; do
        IFS="/" read -r ISL OSL <<< "${USE_CASES[$use_case]}"

        total_requests=$((total_request_multiplier * CR))
        if [ "$total_requests" -lt $MIN_REQUESTS ]; then
          total_requests=$MIN_REQUESTS
        fi

        EXPORT_FILE_BASE=CR:${CR}_UseCase:${use_case}_ISL:${ISL}_OSL:${OSL}_Model:${NIM_MODEL_NAME_cleaned}_Cluster:${CLUSTER}_GPU:${GPU}_Experiment:${EXPERIMENT_NAME}

        # Run RAG OFF
        NAMESPACE_PARAM=""
        EXPORT_FILE=RAG-Off_${EXPORT_FILE_BASE}_$(date $DATE_FORMAT)
        SERVICE=$NON_RAG_SERVICE
        Request_ISL=$ISL
        run_experiment "$EXPORT_FILE"

        # Per vdb_reranker combination
        for vdb_reranker in $VDB_and_RERANKER_K; do
            IFS="/" read -r VDB_K RERANKER_K <<< "$vdb_reranker"
            EXPORT_FILE_K=CHUNK-SIZE:${CHUNK_SIZE}_SYS-PROMPT-SIZE:${RAG_PROMPT_EXTRA}_VDB-K:${VDB_K}_RERANKER-K:${RERANKER_K}_${EXPORT_FILE_BASE}_$(date $DATE_FORMAT)

            # RAG On
            Request_ISL=$ISL
            NAMESPACE_PARAM="--extra-input collection_name:$NAMESPACE"
            SERVICE=$RAG_SERVICE
            EXPORT_FILE=RAG-On_$EXPORT_FILE_K
            run_experiment "$EXPORT_FILE"

            # RAG Off with RAG On ISL
            NAMESPACE_PARAM=""
            SERVICE=$NON_RAG_SERVICE
            EXPORT_FILE=RAG-Off-with-RAG-ISL_$EXPORT_FILE_K
            Request_ISL=$((ISL + RAG_PROMPT_EXTRA + (RERANKER_K * CHUNK_SIZE)))
            run_experiment "$EXPORT_FILE"
        done
    done
  done
done

# Final summary
end_time=$(date +%s)
total_duration=$((end_time - start_time))
echo "Benchmarking completed!"
echo "Total runtime: $((total_duration / 60)) min $((total_duration % 60)) sec"