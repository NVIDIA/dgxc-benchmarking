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

set -x

hostname
nvidia-smi

MAX_GPU_CLOCK=$(nvidia-smi -q -d CLOCK | grep -m 1 -A 1 Max | awk '/Graphics/ {print $3}')
MAX_MEM_CLOCK=$(nvidia-smi -q -d CLOCK | grep -m 1 -A 4 Max | awk '/Memory/ {print $3}')

if [[ -z $MAX_GPU_CLOCK || -z $MAX_MEM_CLOCK ]]; then
    echo "Error: Failed to parse GPU clock values from nvidia-smi"
    exit 1
fi

echo "Setting application clock to Mem Clock: $MAX_MEM_CLOCK and GPU Clock: $MAX_GPU_CLOCK."

if ! sudo nvidia-smi -rgc; then
    echo "Error: Failed to reset GPU clocks" >&2
    exit 1
fi

if ! sudo nvidia-smi -ac "$MAX_MEM_CLOCK,$MAX_GPU_CLOCK"; then
    echo "Error: Failed to set GPU clocks" >&2
    exit 1
fi
