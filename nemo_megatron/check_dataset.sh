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

# script that performs quick validity check of the generated dataset
# STAGE_PATH should be already set for the generate_dataset.sh
DATA_DIR=$STAGE_PATH/gpt3-dataset

SIZE=$( du -sm $DATA_DIR | cut -f 1 )
EXPECTED_DIR_SIZE=75000
if [ $SIZE -lt $EXPECTED_DIR_SIZE ]; then
  echo "data set generation produced incorrectly sized output: $SIZE instead of expected $EXPECTED_DIR_SIZE Mb"
  exit 1
fi

FILES_COUNT=$( find $DATA_DIR -type f \( -name "*.bin" -o -name "*.idx" \) | wc -l )
EXPECTED_FILES_COUNT=8
if [ $FILES_COUNT -ne $EXPECTED_FILES_COUNT ]; then
  echo "data set generation produced incorrect $FILES_COUNT instead of expected $EXPECTED_FILES_COUNT number of files"
  exit 1
fi

echo "Dataset generation completed successfully at: $DATA_DIR"