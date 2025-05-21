
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

export N_TIMES=3
export CONCURRENCY_RANGE="1 25 50 75 100"
export VDB_and_RERANKER_K="20/4"
declare -A USE_CASES=(
    ["chat"]="128/128"
)

# RAG Pipeline Meta Data
export GPU="H100"
export CLUSTER="Your-Cluster"
export EXPERIMENT_NAME="RAG-Blueprint"
export OUTPUT="/tmp/output/"

export RAG_SERVICE="" # <service-name>:<port>
export NON_RAG_SERVICE="" # <service-name>:<port>
export NIM_MODEL_NAME="meta/llama-3.1-70b-instruct"
export NIM_MODEL_NAME_cleaned="meta-llama-3.1-70b-instruct"
export NIM_MODEL_TOKENIZER="meta-llama/Meta-Llama-3-70B-Instruct"
export NAMESPACE="wikipedia"
export CHUNK_SIZE=420 # Number of tokens
export RAG_PROMPT_EXTRA=100 # Number of Tokens

export total_request_multiplier=5
export SLEEP_TIME=60 # Time between GenAI-Perf Commands
export MIN_REQUESTS=50
export DATE_FORMAT="+%Y-%m-%d_%H:%M:%S.%3N"