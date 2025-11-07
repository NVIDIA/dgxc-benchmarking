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

# For each dataset a user elects to use, the user is responsible for
# checking if the dataset license is fit for the intended purpose.

"""Utilities for working with workload metadata structures.

Centralizes normalization for dtype/scales configuration so all call sites
behave consistently.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger("llmb_run.metadata_utils")

# Known dtype keys supported by our tooling. Extend as needed.
_KNOWN_DTYPES = {"fp8", "bf16", "nvfp4", "mxfp4"}


def normalize_model_dtype_config(model_config: dict) -> Dict[str, Dict[str, object]]:
    """Normalize a model_config's dtype/scales definition.

    Returns a mapping of dtype -> { 'scales': list[int], 'exact_scales': bool }

    Accepted input forms on the model_config:
      1) Legacy form
         dtypes: ['fp8','bf16'] | 'fp8'
         scales: [128,256]
         exact_scales: bool (optional)

      2) Mapping form (per-dtype config)
         dtypes:
           fp8: [128, 256]                 # short form = scales only
           bf16: { scales: [256, 512], exact_scales: true }

    Notes:
      - If mapping form is used but contains non-dtype keys (e.g., mistakenly
        nested 'scales' or 'exact_scales' under 'dtypes'), those keys are
        ignored with a debug log.
      - If mapping form is used, any top-level scales are ignored.
    """
    normalized: Dict[str, Dict[str, object]] = {}

    dtypes_value = model_config.get("dtypes")
    model_scales = model_config.get("scales", [])
    model_exact = bool(model_config.get("exact_scales", False))

    # Mapping form
    if isinstance(dtypes_value, dict):
        for key, val in dtypes_value.items():
            if key not in _KNOWN_DTYPES:
                # Ignore non-dtype keys under the mapping and continue
                logger.debug("Ignoring non-dtype key under dtypes mapping: %s", key)
                continue

            if isinstance(val, list):
                normalized[key] = {
                    "scales": [int(s) for s in val],
                    "exact_scales": model_exact,
                }
            elif isinstance(val, dict):
                dtype_scales = [int(s) for s in val.get("scales", [])]
                dtype_exact = bool(val.get("exact_scales", model_exact))
                normalized[key] = {
                    "scales": dtype_scales,
                    "exact_scales": dtype_exact,
                }
            else:
                logger.debug(
                    "Unsupported dtype mapping value for key %s: %r (ignored)",
                    key,
                    val,
                )
        return normalized

    # Legacy forms
    if isinstance(dtypes_value, str):
        dtype_list: List[str] = [dtypes_value]
    elif isinstance(dtypes_value, list):
        dtype_list = list(dtypes_value)
    else:
        dtype_list = []

    for dt in dtype_list:
        normalized[dt] = {
            "scales": [int(s) for s in model_scales],
            "exact_scales": model_exact,
        }

    return normalized
