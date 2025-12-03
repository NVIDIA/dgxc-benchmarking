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

etcd_server_port=$1
dynamo_frontend_port=$2

if [ "$#" -ne 2 ]; then
    echo "Error: Expected 2 arguments, got $#"
    echo "Usage: $0 <etcd_server_port> <dynamo_frontend_port>"
    exit 1
fi

echo "Arguments:"
echo "  etcd_server_port: $etcd_server_port"
echo "  dynamo_frontend_port: $dynamo_frontend_port"

# Check that the process id is 0.
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for frontend server, exiting"
    exit 0
fi

echo "ucx info: $(ucx_info -v)"

# Start NATS
nats-server -js &

# Start etcd
etcd --listen-client-urls http://0.0.0.0:${etcd_server_port} --advertise-client-urls http://0.0.0.0:${etcd_server_port} --data-dir /tmp/etcd &

# Wait for NATS/etcd to startup
sleep 2

# Start OpenAI Frontend which will dynamically discover workers when they startup
# NOTE: This is a blocking call.
python3 -m dynamo.frontend --http-port ${dynamo_frontend_port}
