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

#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00

set -eu -o pipefail

export HF_TOKEN=${HF_TOKEN?"Required variable HF_TOKEN"}
export FW_VERSION=24.12
# create staging folder
mkdir -vp $STAGE_PATH/cfg

# copy configs to stagepath
cp -vf llama3*.yaml "${STAGE_PATH}/cfg"

cp -vf gen*.sh configure.sh launch.sh $STAGE_PATH

# create the squash file
srun bash -c "enroot import --output ${STAGE_PATH}/nvidia+nemo+${FW_VERSION}.sqsh docker://nvcr.io#nvidia/nemo:${FW_VERSION}"

# copy out the configuration from the container to the $STAGE_PATH
# this is required for data set generation.
srun --container-mounts=$STAGE_PATH --container-image="$STAGE_PATH/nvidia+nemo+24.12.sqsh" bash -c "cp -r /opt/NeMo-Framework-Launcher/launcher_scripts $STAGE_PATH; cp /opt/NeMo-Framework-Launcher/requirements.txt $STAGE_PATH"

# install required Python modules for generating dataset
pip install -r "$STAGE_PATH/requirements.txt"

# download Llama 3 tokenizer files from huggingface meta-llama/Meta-Llama-3-8B for data prep and training
huggingface-cli download --force-download --local-dir "$STAGE_PATH/llama3.1-dataset/llama" meta-llama/Meta-Llama-3-8B config.json special_tokens_map.json tokenizer.json tokenizer_config.json generation_config.json

mkdir -vp $STAGE_PATH/megatron-gpt2-345m
huggingface-cli download --force-download --local-dir "$STAGE_PATH/megatron-gpt2-345m" nvidia/megatron-gpt2-345m tokenizer.json vocab.json merges.txt