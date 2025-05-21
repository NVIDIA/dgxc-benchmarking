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


TIMESTAMP=$(date +%s)

COMMAND=" \
genai-perf profile \
  -m $NIM_MODEL_NAME \
  --endpoint-type chat \
  --service-kind openai \
  --streaming -u 0.0.0.0:8000 \
  --num-prompts \$total_requests \
  --synthetic-input-tokens-mean \$ISL \
  --synthetic-input-tokens-stddev 0 \
  --concurrency \$CONCURRENCY \
  --output-tokens-mean \$OSL \
  --extra-inputs max_tokens:\$OSL \
  --extra-inputs min_tokens:\$OSL \
  --extra-inputs ignore_eos:true \
  --artifact-dir $RESULTS_PATH/\$EXPORT_FILE \
  --tokenizer $NIM_MODEL_TOKENIZER \
  -- -v --max-threads=\$CONCURRENCY --request-count \$total_requests"

# Wait for Server to start

sleep 90

for CONCURRENCY in ${CONCURRENCY_RANGE}; do
  for value in $USE_CASES; do
    # Extract the use case name and values
    use_case=$(echo "$value" | cut -d':' -f1)
    ISL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f1)
    OSL=$(echo "$value" | cut -d':' -f2 | cut -d'/' -f2)

    echo "Concurrency: $CONCURRENCY"
    echo "Use Case: $use_case"
    echo "ISL: $ISL"
    echo "OSL: $OSL"
    echo "----------------"

    EXPORT_FILE=${NIM_MODEL_NAME_cleaned}_${NUM_GPUS}_${CONCURRENCY}_${use_case}_${ISL}_${OSL}_${TIMESTAMP}
    total_requests=$((total_request_multiplier * CONCURRENCY))
    if [ "$total_requests" -lt $MIN_REQUESTS ]; then
      total_requests=$MIN_REQUESTS
    fi
    eval "$COMMAND"
  done
done


echo "Finished Benchmarking"