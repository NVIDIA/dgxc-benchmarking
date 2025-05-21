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

# create staging folder
mkdir -vp $STAGE_PATH/cfg

cp -vf launch.sh *.md "${STAGE_PATH}"
cp -vf configure.sh "$STAGE_PATH/cfg"

# create the squash file 
srun bash -c "enroot import --output ${STAGE_PATH}/nvidia+jax+maxtext-25.01.sqsh docker://nvcr.io#nvidia/jax:25.01-maxtext-py3"

