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

"""Custom configurations for GPU models."""

from paxml import experiment_registry
from paxml.contrib.gpu.scripts_gpu.configs import Synthetic5B, Synthetic175B
from praxis import layers

@experiment_registry.register
class Synthetic5BCkpt(Synthetic5B):
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_ONLY

@experiment_registry.register
class Synthetic175BCkpt(Synthetic175B):
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_QKV_OUT_PROJ
