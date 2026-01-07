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

# Parameters
#SBATCH --exclusive
#SBATCH --job-name="deepseek_v3:torchtitan-container-setup"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

set -eu -o pipefail

if [ "${BASH_VERSION:0:1}" -lt 4 ] || { [ "${BASH_VERSION:0:1}" -eq 4 ] && [ "${BASH_VERSION:2:1}" -lt 2 ]; }; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

# Validate required environment variables
if [ -z "${LLMB_INSTALL:-}" ]; then
    echo "Error: LLMB_INSTALL environment variable is not set" >&2
    exit 1
fi

if [ -z "${LLMB_WORKLOAD:-}" ]; then
    echo "Error: LLMB_WORKLOAD environment variable is not set" >&2
    exit 1
fi

# Find the sqsh container file
IMAGES_DIR="${LLMB_INSTALL}/images"
SQSH_FILE=${IMAGES_DIR}/nvidia+pytorch+25.10-py3.sqsh

echo "Checking for container file: ${SQSH_FILE}"
if ! ls "${SQSH_FILE}" 2> /dev/null; then
    echo "Error: Could not find the pytorch container sqsh file at ${SQSH_FILE}"
    exit 1
fi

echo "Found container: ${SQSH_FILE}"

export TORCHTITAN_REPO="${LLMB_WORKLOAD}/torchtitan"
MODIFIED_SQSH="${SQSH_FILE}.modified"

# Use srun with container flags to install torchtitan
# NVIDIA_VISIBLE_DEVICES=void prevents GPU access during pip install
srun --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
    --container-image="${SQSH_FILE}" \
    --container-mounts="${TORCHTITAN_REPO}:${TORCHTITAN_REPO}" \
    --container-save="${MODIFIED_SQSH}" \
    bash -c "cd ${TORCHTITAN_REPO} && PIP_DEFAULT_TIMEOUT=300 python3 -m pip install --no-cache-dir --no-build-isolation . && echo 'Torchtitan installation completed successfully'"

echo "Replacing original sqsh file with modified version..."

# Create backup of original container
if [ -f "${SQSH_FILE}" ]; then
    cp "${SQSH_FILE}" "${SQSH_FILE}.backup" || {
        echo "Error: Failed to create backup" >&2
        exit 1
    }
fi

# Atomic replacement
mv -f "${MODIFIED_SQSH}" "${SQSH_FILE}" || {
    echo "Error: Failed to replace container file" >&2
    # Attempt to restore backup if move failed
    if [ -f "${SQSH_FILE}.backup" ]; then
        mv "${SQSH_FILE}.backup" "${SQSH_FILE}"
    fi
    exit 1
}

echo "Container setup completed successfully!"
echo "Torchtitan has been installed into ${SQSH_FILE}"
