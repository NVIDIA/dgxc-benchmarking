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