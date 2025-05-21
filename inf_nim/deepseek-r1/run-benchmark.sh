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

# For each dataset a user elects to use, the user is responsible for
# checking if the dataset license is fit for the intended purpose.


TIMESTAMP=$(date +%s)

COMMAND=" \
genai-perf profile \
  -m $NIM_SERVED_MODEL_NAME \
  --endpoint-type chat \
  --service-kind openai \
  --streaming -u $NIM_LEADER_IP_ADDRESS:8000 \
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

sleep $SERVER_SLEEP_TIME

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

    EXPORT_FILE=${NIM_MODEL_NAME_cleaned}_${NUM_GPUS}_${CONCURRENCY}_${use_case}_${ISL}_${OSL}_${SLURM_JOB_ID}_${TIMESTAMP}
    total_requests=$((total_request_multiplier * CONCURRENCY))
    if [ "$total_requests" -lt $MIN_REQUESTS ]; then
      total_requests=$MIN_REQUESTS
    fi
    eval "$COMMAND"
  done
done


echo "Finished Benchmarking"