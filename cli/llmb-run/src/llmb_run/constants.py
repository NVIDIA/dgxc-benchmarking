# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Constants used throughout the llmb-run application."""

# SLURM configuration
SLURM_OUTPUT_PATTERN = 'slurm-%j.out'

# File patterns
METADATA_FILE_PATTERN = '**/metadata.yaml'

# Model parameter mappings
NEMO_MODEL_PARAMS = {
    "mbs": "model.micro_batch_size",
    "gbs": "model.global_batch_size",
    "cp": "model.context_parallel_size",
    "tp": "model.tensor_model_parallel_size",
    "pp": "model.pipeline_model_parallel_size",
    "vp": "model.virtual_pipeline_model_parallel_size",
}

# Mapping of GPU types to number of GPUs per node
GPU_TYPE_TO_NUM_GPUS = {
    "a100": 8,
    "h100": 8,
    "b200": 8,
    "b300": 8,
    "gb200": 4,
    "gb300": 4,
}

# Workload exclusions (currently empty but keeping for future use)
EXCLUDE_WORKLOADS = []
