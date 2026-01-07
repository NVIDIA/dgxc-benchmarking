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

set -eu -o pipefail

# Validate required environment variables
if [ -z "${LLMB_WORKLOAD:-}" ]; then
    echo "Error: LLMB_WORKLOAD environment variable is not set" >&2
    exit 1
fi

cd "$LLMB_WORKLOAD/torchtitan" || exit 1

# Check if download script exists
DOWNLOAD_SCRIPT="scripts/download_hf_assets.py"
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "Error: Download script not found: $DOWNLOAD_SCRIPT" >&2
    exit 1
fi

# Pass HF_TOKEN directly to the download script without calling login()
# The login() function makes additional API calls that can trigger rate limits
if [ -n "$HF_TOKEN" ]; then
    echo "Using HuggingFace token for download..."
    python "$DOWNLOAD_SCRIPT" --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer --hf_token "$HF_TOKEN"
else
    echo "WARNING: No HF_TOKEN provided"
    python "$DOWNLOAD_SCRIPT" --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer
fi
