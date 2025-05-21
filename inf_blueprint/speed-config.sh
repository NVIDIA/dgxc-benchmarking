
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