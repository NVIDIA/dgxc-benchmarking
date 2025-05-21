#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set -e

IMAGE=$STAGE_PATH/nvidia+nemo+24.05.sqsh
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
  cluster.gpus_per_node=$SLURM_GPUS_PER_NODE \
  cluster.account=$SLURM_ACCOUNT \
  cluster.partition=$SLURM_PARTITION \
  env_vars.TRANSFORMERS_OFFLINE=0 \
  container=$IMAGE
