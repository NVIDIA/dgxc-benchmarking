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
if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

export FW_VERSION=24.12

# create the squash file
LLM_REPO=$PWD

mkdir -vp $STAGE_PATH

cd $STAGE_PATH
mkdir -vp HF_ckpt
mkdir -vp logs
srun bash -c "enroot import --output nvidia+nemo+${FW_VERSION}.sqsh docker://nvcr.io#nvidia/nemo:${FW_VERSION}"

#setup variables
export MEGATRON_DIR="Megatron-LM"
export NEMO_COMMIT="25ec4e9894c6938c3562ebb1df3280cf5a01f4da"
export NEMO_DIR="NeMo"
export SCRIPTS_DIR="$NEMO_DIR/scripts/llm/performance"
export MEGATRON_COMMIT="b997545c94dab2e80b30dcd8c49f1821b1ad7838"

pip install pybind11

#Setup Megatron-LM 
if [ ! -d "$MEGATRON_DIR" ]; then
   git clone https://github.com/NVIDIA/Megatron-LM.git 
   cd $MEGATRON_DIR
   git checkout $MEGATRON_COMMIT
fi

cd $STAGE_PATH/$MEGATRON_DIR
pip install torch torchvision
pip install . 
echo "Megatron build done"
cd $STAGE_PATH

#Setup NeMo 
if [ ! -d "$NEMO_DIR" ]; then
    git clone https://github.com/NVIDIA/NeMo.git
    cd $NEMO_DIR
    git checkout $NEMO_COMMIT
    git apply $LLM_REPO/callbacks_11961.patch
fi

cd $STAGE_PATH/$NEMO_DIR
pip install --upgrade pip
pip install Cython packaging
pip install lightning-fabric pytorch-lightning cloudpickle psutil omegaconf hydra-core datasets einops transformers sentencepiece braceexpand webdataset h5py ijson matplotlib sacrebleu rouge_score faiss-cpu jieba opencc pangu
pip install lightning==2.5.0.post0
pip install .
pip install git+https://github.com/NVIDIA/NeMo-Run.git@e70f1094afa3b372390c323d830aa88bf4f824af
#copy the scripts to NeMo performance scripts directory
cd $LLM_REPO
cp ./llama3_finetune.py $STAGE_PATH/$SCRIPTS_DIR
cp ./llama3_finetune_utils.py $STAGE_PATH/$SCRIPTS_DIR
echo "DONE SETUP proceed to launch script"
