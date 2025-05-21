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

#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=0:45:00

set -eu -o pipefail

# create staging folder and copy config file
mkdir -vp ${STAGE_PATH}/cfg
cp -vf paxml_mod_configs.py ${STAGE_PATH}/cfg
cp -vf launch.sh configure.sh ${STAGE_PATH}

# create squash file based on the container image
srun bash -c "enroot import --output ${STAGE_PATH}/ghcr.io+nvidia+jax+pax-2024.03.04.sqsh docker://ghcr.io#nvidia/jax:pax-2024-03-04"
