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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 -a ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

check_pep668() {
  # Create a temp file and ensure it’s cleaned up
  tmpfile=$(mktemp)
  trap 'rm -f "$tmpfile"' EXIT

  # Do a no-op pip install and capture stderr
  python3 -m pip install --disable-pip-version-check \
    --no-deps --dry-run pip 2> "$tmpfile" || true

  if grep -q "externally-managed-environment" "$tmpfile"; then
      cat <<EOF >&2

ERROR: It looks like your system’s Python is “externally managed” (PEP 668).
  • System-wide pip installs are blocked.
  • You must use a virtual environment (venv or conda)

Quick fix:
  $ python3 -m venv llmb_venv
  $ source llmb_venv/bin/activate
  $ ./install.sh

IMPORTANT: You will need this virtual environment if you plan to use 'llmb-run' for launching workloads.

EOF
      exit 1
	fi
}
check_pep668

# Install runner dependencies
pushd llmb-run
python3 -m pip install .
popd

# Install installer dependencies
pushd installer
python3 -m pip install .

# Run the interactive installer
./installer.py
popd

