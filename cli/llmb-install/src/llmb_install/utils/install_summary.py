# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helpers for writing installer summary data for wrapper scripts."""

import logging
import os
from pathlib import Path
from typing import Iterable, Optional


def _sanitize_value(value: object) -> str:
    """Make values safe for key=value line output.

    Note: '=' in values is fine — the shell reader uses ${line#*=} which
    strips only through the first '='.  Only newlines need escaping.
    """
    return str(value).replace('\n', ' ').replace('\r', ' ').strip()


def _debug(logger: Optional[logging.Logger], message: str) -> None:
    if logger is not None:
        logger.debug(message)


def write_install_summary_file(
    status: str,
    install_path: Optional[str] = None,
    failed_workloads: Optional[Iterable[str]] = None,
    async_jobs_submitted: Optional[bool] = None,
    summary_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Write installer summary metadata to a key=value file.

    The output path defaults to the LLMB_INSTALL_SUMMARY_FILE environment variable.
    Writes are atomic (tmp file + replace) so readers never see partial content.
    """
    summary_path_value = summary_path if summary_path is not None else os.environ.get("LLMB_INSTALL_SUMMARY_FILE", "")
    summary_path_value = summary_path_value.strip()
    if not summary_path_value:
        return

    output_path = Path(summary_path_value)
    parent_dir = output_path.parent

    if not parent_dir.exists() or not os.access(parent_dir, os.W_OK):
        _debug(logger, f"Skipping installer summary write; directory not writable: {parent_dir}")
        return

    if status not in {"success", "failed"}:
        _debug(logger, f"Skipping installer summary write; invalid status: {status}")
        return

    lines = [
        "version=1",
        f"status={status}",
    ]

    if install_path:
        lines.append(f"install_path={_sanitize_value(install_path)}")

    if failed_workloads:
        workload_values = [v for w in failed_workloads if (v := _sanitize_value(w))]
        if workload_values:
            lines.append(f"failed_workloads={','.join(workload_values)}")

    if async_jobs_submitted is not None:
        lines.append(f"async_jobs_submitted={'true' if async_jobs_submitted else 'false'}")

    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        with open(temp_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write("\n".join(lines) + "\n")
        temp_path.replace(output_path)
    except Exception as exc:
        _debug(logger, f"Failed to write installer summary file: {exc}")
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
