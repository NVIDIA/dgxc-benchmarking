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

# Add error handling
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

WAIT_TIME=300

model=${1}
multi_round=${2}
num_gen_servers=${3}
concurrency_list=${4}
streaming=${5}
total_gpus=${6}
result_dir=${7}
scripts_dir=${8}
model_path=${9}
isl=${10}
osl=${11}
head_node_ip=${12}
dynamo_frontend_port=${13}

if [ "$#" -ne 13 ]; then
    echo "Error: Expected 13 arguments, got $#"
    echo "Usage: $0 <model> <multi_round> <num_gen_servers> <concurrency_list> <streaming> <total_gpus> <result_dir> <scripts_dir> <model_path> <isl> <osl> <head_node_ip> <dynamo_frontend_port>"
    exit 1
fi

echo "Arguments:"
echo "  model: $model"
echo "  multi_round: $multi_round"
echo "  num_gen_servers: $num_gen_servers"
echo "  concurrency_list: $concurrency_list"
echo "  streaming: $streaming"
echo "  total_gpus: $total_gpus"
echo "  result_dir: $result_dir"
echo "  scripts_dir: $scripts_dir"
echo "  model_path: $model_path"
echo "  isl: $isl"
echo "  osl: $osl"
echo "  head_node_ip: $head_node_ip"
echo "  dynamo_frontend_port: $dynamo_frontend_port"

log_dir=${result_dir}/logs
config_dir=${result_dir}/configs

# Check that the process id is 0.
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

set -x

# install genai-perf
pip install genai-perf

apt update -y
apt install -y curl

# try client

do_get_logs() {
    grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" ${log_dir}/output_decode_*.log > ${log_dir}/decode.log || true
    grep -a "'num_generation_tokens': 0" ${log_dir}/output_prefill_*.log > ${log_dir}/prefill.log || true
}

do_get_results() {
    python ${scripts_dir}/postprocess_results.py \
        --result_dir ${result_dir} \
        --concurrency ${concurrency} \
        --isl ${isl} \
        --osl ${osl} \
        --ngpus ${total_gpus}
}

# Wait for server to become healthy (up to 50 attempts)
failed=true
for ((i = 1; i <= 50; i++)); do
    sleep $((i == 1 ? WAIT_TIME : 20))
    response=$(curl -s -w "\n%{http_code}" "${head_node_ip}:${dynamo_frontend_port}/health")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [[ $http_code == "200" ]] && echo "$body" | grep -q '"status":"healthy"' && echo "$body" | grep -q '"endpoints":\[[^]]*"dyn://dynamo.tensorrt_llm.generate"'; then
        if echo "$body" | grep -q '"tensorrt_llm_next"'; then
            echo "Health check succeeded on attempt $i"
            echo "$body"
            failed=false
            break
        else
            echo "Attempt $i: tensorrt_llm_next key not found in etcd."
        fi
    else
        echo "Attempt $i failed: /health not ready (HTTP $http_code)."
    fi
done

if [[ $failed == "true" ]]; then
    echo "Server did not respond with healthy status after 50 attempts."
    exit 1
fi

curl -v -w "%{http_code}" "${head_node_ip}:${dynamo_frontend_port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "'${model}'",
  "messages": [
  {
    "role": "user",
    "content": "Tell me a story as if we were playing dungeons and dragons."
  }
  ],
  "stream": true,
  "max_tokens": 30
}'

echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    genai-perf profile \
        --model ${model} \
        --tokenizer ${model_path} \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url ${head_node_ip}:${dynamo_frontend_port} \
        --synthetic-input-tokens-mean ${isl} \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean ${osl} \
        --output-tokens-stddev 0 \
        --extra-inputs max_tokens:${osl} \
        --extra-inputs min_tokens:${osl} \
        --extra-inputs ignore_eos:true \
        --extra-inputs '{"nvext":{"ignore_eos":true}}' \
        --concurrency ${concurrency} \
        --request-count $((concurrency * 10)) \
        --warmup-request-count $((concurrency * 2)) \
        --num-dataset-entries ${num_prompts} \
        --random-seed 100 \
        --artifact-dir ${result_dir} \
        -- \
        -v \
        --max-threads ${concurrency} \
        -H 'Authorization: Bearer NOT USED' \
        -H 'Accept: text/event-stream'
    echo "Benchmark with concurrency ${concurrency} done"
    do_get_logs
    do_get_results
done

job_id=${SLURM_JOB_ID}
if [ -n "${job_id}" ]; then
    echo "${SLURM_JOB_NODELIST}" > ${config_dir}/machines.txt
fi
