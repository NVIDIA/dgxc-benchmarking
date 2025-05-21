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
set -eu

IMAGE=$STAGE_PATH/nvidia+nemo+24.12.sqsh
NUM_NODES=2

mkdir -p $STAGE_PATH/launcher_scripts/results
python3 $STAGE_PATH/launcher_scripts/main.py \
  launcher_scripts_path=$STAGE_PATH/launcher_scripts \
  stages=[data_preparation] \
  data_dir=$STAGE_PATH/gpt3-dataset \
  data_preparation.run.node_array_size=$NUM_NODES \
  data_preparation.run.results_dir=$STAGE_PATH/results.data_preparation \
  data_preparation.file_numbers='0-3' \
  data_preparation.rm_downloaded=True \
  data_preparation.rm_extracted=True \
  data_preparation.run.time_limit="0:45:00" \
  cluster.gpus_per_node=${SLURM_GPUS_ON_NODE:-} \
  cluster.account=$SLURM_JOB_ACCOUNT \
  cluster.partition=$SLURM_JOB_PARTITION \
  env_vars.TRANSFORMERS_OFFLINE=0 \
  container=$IMAGE
