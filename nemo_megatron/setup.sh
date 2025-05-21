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

set -eu

# create staging folder
mkdir -vp $STAGE_PATH/cfg

cp -vf *.sh "${STAGE_PATH}"
cp -vf *.yaml "${STAGE_PATH}/cfg"

# create the squash file 
srun bash -c "enroot import --output ${STAGE_PATH}/nvidia+nemo+24.12.sqsh docker://nvcr.io#nvidia/nemo:24.12"

# copy out the configuration from the container to the $STAGE_PATH
# this is required for data set generation
srun --container-mounts=$STAGE_PATH:/workspace/mount_dir --container-image=$STAGE_PATH/nvidia+nemo+24.12.sqsh bash -c "cp -r /opt/NeMo-Framework-Launcher/launcher_scripts /workspace/mount_dir/; cp /opt/NeMo-Framework-Launcher/requirements.txt /workspace/mount_dir/"

# install required Python modules
pip install -r $STAGE_PATH/requirements.txt
