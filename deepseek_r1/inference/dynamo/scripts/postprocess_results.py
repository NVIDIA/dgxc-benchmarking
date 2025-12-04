#!/usr/bin/env python3
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

import argparse
import json
import os


def write_result_summary(
    result_dir: str,
    concurrency: int,
    isl: int,
    osl: int,
    ngpus: int,
) -> None:
    # Validate inputs
    if isl <= 0:
        raise ValueError("Input sequence length must be positive")
    if osl <= 0:
        raise ValueError("Output sequence length must be positive")
    if concurrency <= 0:
        raise ValueError("Concurrency must be positive")
    if ngpus <= 0:
        raise ValueError("Total GPU count must be positive")

    result_file = os.path.join(
        result_dir,
        f"deepseek-r1-dynamo-openai-chat-concurrency{concurrency}",
        "profile_export_genai_perf.json",
    )

    try:
        with open(result_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {result_file}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in results file: {result_file}: {e}") from None
    except Exception as e:
        raise RuntimeError(f"Error reading results file: {result_file}: {e}") from None

    # Validate that required keys exist
    required_keys = [
        "request_throughput",
        "output_token_throughput",
        "output_token_throughput_per_user",
        "request_latency",
        "time_to_first_token",
        "inter_token_latency",
    ]

    for key in required_keys:
        if key not in data or "avg" not in data[key]:
            raise KeyError(f"Required metric '{key}.avg' not found in results file: {result_file}")

    request_throughput = data["request_throughput"]["avg"]
    output_token_throughput = data["output_token_throughput"]["avg"]
    output_token_throughput_per_user = data["output_token_throughput_per_user"]["avg"]
    output_token_throughput_per_gpu = output_token_throughput / ngpus
    total_token_throughput_per_gpu = output_token_throughput_per_gpu * (isl + osl) / osl

    request_latency = data["request_latency"]["avg"]
    time_to_first_token = data["time_to_first_token"]["avg"]
    inter_token_latency = data["inter_token_latency"]["avg"]

    print_dict = {
        "Request Throughput (req/sec)": request_throughput,
        "Output Token Throughput (tokens/sec)": output_token_throughput,
        "Output Token Throughput per User (tps/user)": output_token_throughput_per_user,
        "Output Token Throughput per GPU (tps/gpu)": output_token_throughput_per_gpu,
        "Total Token Throughput per GPU (tokens/sec)": total_token_throughput_per_gpu,
        "Average request latency (ms)": request_latency,
        "Average time to first token [TTFT] (ms)": time_to_first_token,
        "Average time per output token [TPOT] (ms)": inter_token_latency,
    }

    with open(os.path.join(result_dir, "perf_summary.txt"), "w") as f:
        print("=" * 60, file=f)
        print("PERFORMANCE OVERVIEW", file=f)
        print("=" * 60, file=f)
        for key, value in print_dict.items():
            print(f"{key:<46}: {value:.2f}", file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--isl", type=int, required=True)
    parser.add_argument("--osl", type=int, required=True)
    parser.add_argument("--ngpus", type=int, required=True)
    args = parser.parse_args()

    write_result_summary(
        result_dir=args.result_dir,
        concurrency=args.concurrency,
        isl=args.isl,
        osl=args.osl,
        ngpus=args.ngpus,
    )


if __name__ == "__main__":
    main()
