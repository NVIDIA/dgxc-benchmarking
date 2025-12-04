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

# Using MAX_BATCH_SIZE generate cuda graph batch sizes

set -eu -o pipefail
generate_powers_of_2() {
    local number=$1
    local power=1
    local powers_of_2=()

    while [ $power -le $number ]; do
        powers_of_2+=("$power")
        power=$((power * 2))
    done

    # Add last number to the list if not already there
    if [ "${powers_of_2[-1]}" -ne "$number" ]; then
        powers_of_2+=("$number")
    fi

    # Join without commas (just space-separated)
    echo "${powers_of_2[*]}"
}

export cuda_batch_sizes=$(generate_powers_of_2 $MAX_BATCH_SIZE)
python3 -m sglang.launch_server \
    --trust-remote-code \
    --disable-radix-cache \
    --enable-dp-lm-head \
    --moe-dense-tp-size 1 \
    --max-running-requests ${MAX_BATCH_SIZE} \
    --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    --cuda-graph-bs ${cuda_batch_sizes} \
    --kv-cache-dtype fp8_e4m3 \
    --quantization modelopt_fp4 \
    --attention-backend trtllm_mla \
    --stream-interval ${STREAM_INTERVAL} \
    --moe-runner-backend flashinfer_cutlass \
    --enable-dp-attention \
    --model-path=${MODEL_PATH} \
    --host ${HOST_IP} \
    --port ${HOST_PORT} \
    --tp-size ${TP} \
    --dp-size ${DP} \
    --ep-size ${EP}
