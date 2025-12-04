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

import argparse
import csv
import time

import torch


def benchmark_gemm(size, num_iters=10_000):
    """
    Measures kernel launch latency for a given GEMM size.

    Args:
        size (int): Size of the square matrix (size x size).
        num_iters (int): Number of iterations.

    Returns:
        float: Average kernel launch latency in microseconds.
    """
    device = "cuda"

    # Initialize random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Clear caching allocator and synchronize before timing
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_time = time.time()

    # Run GEMMs with synchronization after each one
    for _ in range(num_iters):
        torch.matmul(a, b)  # Small GEMM

    # Synchronize after timing
    torch.cuda.synchronize()  # Ensure execution completes
    end_time = time.time()

    # Compute average kernel launch latency in microseconds (µs)
    total_time = end_time - start_time
    avg_latency = (total_time / num_iters) * 1e6  # Convert to µs
    print(f"Average execution time for size:{size} is {avg_latency} us")

    return avg_latency


def save_results_to_csv(sizes, latencies, filename="gemm_benchmark_results.csv"):
    """Saves GEMM benchmark results to CSV file."""
    header = ["Matrix Size", "Kernel execution Latency (µs)"]
    rows = zip(sizes, latencies, strict=False)

    # Write to CSV file
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Results saved to {filename}")


def sweep_gemm_sizes(start_size=4, end_size=1024, step=2, num_iters=10_000):
    """
    Sweeps GEMM sizes from `start_size` to `end_size`, measuring kernel launch latency.

    Args:
        start_size (int): Smallest matrix size.
        end_size (int): Largest matrix size.
        step (int): Factor to multiply for each step (e.g., 2 for powers of 2).
        num_iters (int): Number of iterations per GEMM size.
    """
    sizes = []
    latencies = []

    size = start_size
    while size <= end_size:
        print(f"Benchmarking GEMM size: {size}x{size} ...")
        avg_latency = benchmark_gemm(size, num_iters)
        sizes.append(size)
        latencies.append(avg_latency)
        size *= step  # Increase size multiplicatively

    save_results_to_csv(sizes, latencies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch kernel launch overhead for small GEMMs.")
    parser.add_argument("--start_size", type=int, default=4, help="Smallest square matrix size (default: 4)")
    parser.add_argument("--end_size", type=int, default=512, help="Largest square matrix size (default: 512)")
    parser.add_argument("--step", type=int, default=2, help="Factor to multiply size (default: 2)")
    parser.add_argument("--iters", type=int, default=100000, help="Number of GEMM iterations per size")
    args = parser.parse_args()

    sweep_gemm_sizes(args.start_size, args.end_size, args.step, args.iters)
