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

"""Central constants for LLMB installer - single source of truth."""

# Python version requirements for virtual environments
MIN_PYTHON_VERSION = "3.12"
MAX_PYTHON_VERSION = "3.13"
MIN_PYTHON_VERSION_TUPLE = tuple(map(int, MIN_PYTHON_VERSION.split('.')))
MAX_PYTHON_VERSION_TUPLE = tuple(map(int, MAX_PYTHON_VERSION.split('.')))

# GPU types supported
SUPPORTED_GPU_TYPES = {'h100', 'gb300', 'gb200', 'b300', 'b200'}

# GPU prefix priority for display ordering
# Lower number = higher priority in selection UI
# When displaying GPU types to users, sort by:
#   1. Prefix priority (ascending)
#   2. Model number (descending) within the same prefix
# This ensures latest/premium GPUs appear first: gb300, gb200, b300, b200, h100
#
# IMPORTANT: When adding a new GPU type to SUPPORTED_GPU_TYPES, you MUST add its
# prefix to this map. The installer will error if an unknown prefix is encountered.
#
# Format assumption: GPU types follow pattern "<prefix><number>" (e.g., "gb300", "h100")
GPU_PREFIX_PRIORITY = {
    'gb': 1,  # Grace Blackwell series (premium, ARM-based)
    'b': 2,  # Blackwell series (x86-based)
    'h': 3,  # Hopper series (x86-based)
}

# Architecture types
ARCHITECTURES = {'x86_64': 'x86_64', 'aarch64': 'aarch64'}

# Default architecture by GPU type
DEFAULT_ARCHITECTURE_BY_GPU = {
    'h100': {'arch': 'x86_64', 'fixed_arch': False},
    'b300': {'arch': 'x86_64', 'fixed_arch': False},
    'b200': {'arch': 'x86_64', 'fixed_arch': False},
    'gb300': {'arch': 'aarch64', 'fixed_arch': True},
    'gb200': {'arch': 'aarch64', 'fixed_arch': True},
}

### TOOLS CONSTANTS ###
# Tool download patterns (tool-specific)
# NSys download URL and filename patterns
# Note: Versions include full version string with build number (e.g., '2025.5.1.121-3638078')
NSYS_BASE_URL = 'https://developer.download.nvidia.com/devtools/nsight-systems'
NSYS_FILENAME_PATTERNS = {
    'x86_64': 'NsightSystems-linux-public-{version}.run',
    'aarch64': 'NsightSystems-linux-sbsa-public-{version}.run',
}

# CUDA CUPTI download URL and filename patterns
# Note: Versions are in format like '13.0.85'
CUDA_CUPTI_BASE_URL = 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti'
CUDA_CUPTI_FILENAME_PATTERNS = {
    'x86_64': 'cuda_cupti-linux-x86_64-{version}-archive.tar.xz',
    'aarch64': 'cuda_cupti-linux-sbsa-{version}-archive.tar.xz',
}

# Supported tools (explicit allow-list)
SUPPORTED_TOOLS = {'nsys', 'cuda_cupti_lib'}
