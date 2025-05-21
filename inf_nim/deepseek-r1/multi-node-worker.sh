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

export NIM_TRUST_CUSTOM_CODE=1
export NIM_USE_SGLANG=1
export NIM_TENSOR_PARALLEL_SIZE=8
export NIM_PIPELINE_PARALLEL_SIZE=2
export MASTER_ADDR=${NIM_LEADER_IP_ADDRESS}
export MASTER_PORT=50000
export NIM_NODE_RANK=${SLURM_NODEID}
export NVIDIA_VISIBLE_DEVICES=all
export NIM_NODE_RANK=${SLURM_NODEID}
if [ "$NIM_NODE_RANK" -eq 0 ]; then
  export NIM_LEADER_ROLE=1
else
  export NIM_LEADER_ROLE=0
fi

# start server
/opt/nim/start_server.sh > ${STAGE_PATH}/logs/server_${NIM_LEADER_ROLE}_${SLURM_JOB_ID}.out
