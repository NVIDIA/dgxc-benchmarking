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
import os
from typing import Any

import yaml


def _get_tp_and_pp_sizes(world_size: int, gpu_type: str) -> tuple[int, int]:
    """
    Get tensor and pipeline parallelism sizes based on world size.

    Args:
        world_size: Total world size.
        gpu_type: Type of GPU: h100 or gb200

    Returns:
        tuple: (tensor_parallel_size, pipeline_parallel_size)
    """
    if gpu_type == "h100" or gpu_type == "b200":
        tp_size = min(8, world_size)
    elif gpu_type == "gb200":
        tp_size = world_size
    assert world_size % tp_size == 0, "World size must be divisible by tensor parallel size."
    pp_size = world_size // tp_size
    return tp_size, pp_size


def _get_moe_backend(tp_size: int, enable_attention_dp: bool, gpu_type: str) -> str:
    """
    Get MOE backend based on tensor parallel size and attention DP flag.

    Args:
        tp_size: Tensor parallel size.
        enable_attention_dp: Whether attention DP is enabled.
        gpu_type: Type of GPU: h100 or gb200

    Returns:
        str: MOE backend type.
    """
    # H100 and B200 don't support EP sizes larger than a single node.
    if gpu_type == "h100":
        assert tp_size <= 8
        return "CUTLASS"
    elif gpu_type == "b200":
        assert tp_size <= 8
        return "CUTLASS" if enable_attention_dp else "TRTLLM"
    elif gpu_type == "gb200":
        if enable_attention_dp:
            return "WIDEEP" if tp_size >= 16 else "CUTLASS"
        else:
            return "TRTLLM"
    else:
        raise ValueError(f"Unsupported GPU type: {gpu_type}")


def gen_config_file(
    prefill_config_path: str,
    decode_config_path: str,
    deployment_config_path: str,
    gpu_type: str,
    num_ctx_servers: int,
    ctx_world_size: int,
    ctx_batch_size: int,
    ctx_max_num_tokens: int,
    ctx_max_seq_len: int,
    ctx_free_gpu_memory_fraction: float,
    ctx_enable_attention_dp: bool,
    num_gen_servers: int,
    gen_world_size: int,
    gen_batch_size: int,
    gen_max_num_tokens: int,
    gen_max_seq_len: int,
    gen_enable_attention_dp: bool,
    gen_gpu_memory_fraction: float,
    eplb_num_slots: int,
    mtp_size: int,
    cache_transceiver_max_num_tokens: int,
) -> None:
    """
    Generate configuration YAML file for disaggregated inference.

    Args:
        prefill_config_path: Path to save the prefill config file
        decode_config_path: Path to save the decode config file
        deployment_config_path: Path to save the deployment config file
        gpu_type: Type of GPU: h100 or gb200
        num_ctx_servers: Number of context servers
        ctx_world_size: World size for context servers
        ctx_batch_size: Batch size for context servers
        ctx_max_num_tokens: Max number of tokens for context servers
        ctx_max_seq_len: Max sequence length for context servers
        ctx_free_gpu_memory_fraction: Free GPU memory fraction for context servers
        ctx_enable_attention_dp: Enable attention DP for context servers
        num_gen_servers: Number of generation servers
        gen_world_size: World size for generation servers
        gen_batch_size: Batch size for generation servers
        gen_max_num_tokens: Max number of tokens for generation servers
        gen_enable_attention_dp: Enable attention DP for generation servers
        gen_gpu_memory_fraction: GPU memory fraction for generation servers
        eplb_num_slots: Number of slots for eplb
        mtp_size: Number of nextn layers for MTP
        cache_transceiver_max_num_tokens: Max number of tokens for cache transceiver
    """
    gen_cuda_graph_batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        768,
        1024,
        2048,
        gen_batch_size,
    ]

    ctx_tp_size, ctx_pp_size = _get_tp_and_pp_sizes(ctx_world_size, gpu_type)
    gen_tp_size, gen_pp_size = _get_tp_and_pp_sizes(gen_world_size, gpu_type)

    ctx_moe_backend = _get_moe_backend(ctx_tp_size, ctx_enable_attention_dp, gpu_type)
    gen_moe_backend = _get_moe_backend(gen_tp_size, gen_enable_attention_dp, gpu_type)

    prefill_config: dict[str, Any] = {
        "max_batch_size": ctx_batch_size,
        "max_num_tokens": ctx_max_num_tokens,
        "max_seq_len": ctx_max_seq_len,
        "tensor_parallel_size": ctx_tp_size,
        "moe_expert_parallel_size": ctx_tp_size,
        "enable_attention_dp": ctx_enable_attention_dp,
        "pipeline_parallel_size": ctx_pp_size,
        "print_iter_log": True,
        "disable_overlap_scheduler": True,
        "enable_chunked_prefill": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": ctx_free_gpu_memory_fraction,
            "dtype": "fp8",
        },
        "moe_config": {
            "backend": ctx_moe_backend,
        },
        "cache_transceiver_config": {
            "max_tokens_in_buffer": cache_transceiver_max_num_tokens,
            "backend": "DEFAULT",
        },
    }

    decode_config: dict[str, Any] = {
        "max_batch_size": gen_batch_size,
        "max_num_tokens": gen_max_num_tokens,
        "max_seq_len": gen_max_seq_len,
        "tensor_parallel_size": gen_tp_size,
        "moe_expert_parallel_size": gen_tp_size,
        "enable_attention_dp": gen_enable_attention_dp,
        "pipeline_parallel_size": gen_pp_size,
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": gen_cuda_graph_batch_sizes,
        },
        "print_iter_log": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": gen_gpu_memory_fraction,
            "dtype": "fp8",
        },
        "moe_config": {
            "backend": gen_moe_backend,
        },
        "cache_transceiver_config": {
            "max_tokens_in_buffer": cache_transceiver_max_num_tokens,
            "backend": "DEFAULT",
        },
        "stream_interval": 20,
    }

    # Setting MNNVL allreduce strategy only makes sense for GB200.
    if gpu_type == "gb200":
        if gen_tp_size == 8 and not gen_enable_attention_dp:
            decode_config["allreduce_strategy"] = "MNNVL"

    if eplb_num_slots > 0:
        moe_load_balancer_file = os.path.join(os.path.dirname(decode_config_path), "moe_load_balancer.yaml")
        moe_load_balancer_config = {
            "num_slots": eplb_num_slots,
            "layer_updates_per_iter": 1,
        }
        with open(moe_load_balancer_file, "w") as f:
            yaml.dump(moe_load_balancer_config, f, default_flow_style=False, sort_keys=False)
        decode_config["moe_config"]["load_balancer"] = moe_load_balancer_file

    if mtp_size > 0:
        prefill_config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": mtp_size,
        }
        decode_config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": mtp_size,
        }

    deployment_config = {
        "num_prefill_servers": num_ctx_servers,
        "num_decode_servers": num_gen_servers,
        "total_gpus": num_ctx_servers * ctx_world_size + num_gen_servers * gen_world_size,
    }

    with open(deployment_config_path, "w") as f:
        yaml.dump(deployment_config, f, default_flow_style=False, sort_keys=False)

    # Write config to file
    with open(prefill_config_path, "w") as f:
        yaml.dump(prefill_config, f, default_flow_style=False, sort_keys=False)

    with open(decode_config_path, "w") as f:
        yaml.dump(decode_config, f, default_flow_style=False, sort_keys=False)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# gen main and args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=True, help="Path to output config directory")
    parser.add_argument("--gpu_type", type=str, required=True, help="Type of GPU: h100 or gb200")
    parser.add_argument("--num_ctx_servers", type=int, required=True, help="Number of context servers")
    parser.add_argument(
        "--ctx_world_size",
        type=int,
        required=True,
        help="World size for context servers",
    )
    parser.add_argument(
        "--ctx_batch_size",
        type=int,
        required=True,
        help="Batch size for context servers",
    )
    parser.add_argument(
        "--ctx_max_num_tokens",
        type=int,
        required=True,
        help="Max number of tokens for context servers",
    )
    parser.add_argument(
        "--ctx_max_seq_len",
        type=int,
        required=True,
        help="Max sequence length for context servers",
    )
    parser.add_argument(
        "--ctx_free_gpu_memory_fraction",
        type=float,
        required=True,
        help="Free GPU memory fraction for context servers",
    )
    parser.add_argument(
        "--ctx_enable_attention_dp",
        type=str2bool,
        nargs="?",
        const=True,
        required=True,
        help="Enable attention DP for context servers",
    )
    parser.add_argument(
        "--num_gen_servers",
        type=int,
        required=True,
        help="Number of generation servers",
    )
    parser.add_argument(
        "--gen_world_size",
        type=int,
        required=True,
        help="World size for generation servers",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        required=True,
        help="Batch size for generation servers",
    )
    parser.add_argument(
        "--gen_max_num_tokens",
        type=int,
        required=True,
        help="Max number of tokens for generation servers",
    )
    parser.add_argument(
        "--gen_max_seq_len",
        type=int,
        required=True,
        help="Max sequence length for generation servers",
    )
    parser.add_argument(
        "--gen_enable_attention_dp",
        type=str2bool,
        nargs="?",
        const=True,
        required=True,
        help="Enable attention DP for generation servers",
    )
    parser.add_argument(
        "--gen_gpu_memory_fraction",
        type=float,
        required=True,
        help="GPU memory fraction for generation servers",
    )
    parser.add_argument("--eplb_num_slots", type=int, default=0, help="Number of slots for eplb")
    parser.add_argument("--mtp_size", type=int, default=0, help="Number of nextn layers for MTP")
    parser.add_argument(
        "--cache_transceiver_max_num_tokens",
        type=int,
        default=4608,
        help="Max number of tokens for cache transceiver",
    )

    args = parser.parse_args()

    prefill_config = os.path.join(args.config_dir, "prefill.yaml")
    decode_config = os.path.join(args.config_dir, "decode.yaml")
    deployment_config = os.path.join(args.config_dir, "deployment.yaml")

    gen_config_file(
        prefill_config_path=prefill_config,
        decode_config_path=decode_config,
        deployment_config_path=deployment_config,
        gpu_type=args.gpu_type,
        num_ctx_servers=args.num_ctx_servers,
        ctx_world_size=args.ctx_world_size,
        ctx_batch_size=args.ctx_batch_size,
        ctx_max_num_tokens=args.ctx_max_num_tokens,
        ctx_max_seq_len=args.ctx_max_seq_len,
        ctx_free_gpu_memory_fraction=args.ctx_free_gpu_memory_fraction,
        ctx_enable_attention_dp=args.ctx_enable_attention_dp,
        num_gen_servers=args.num_gen_servers,
        gen_world_size=args.gen_world_size,
        gen_batch_size=args.gen_batch_size,
        gen_max_num_tokens=args.gen_max_num_tokens,
        gen_max_seq_len=args.gen_max_seq_len,
        gen_enable_attention_dp=args.gen_enable_attention_dp,
        gen_gpu_memory_fraction=args.gen_gpu_memory_fraction,
        eplb_num_slots=args.eplb_num_slots,
        mtp_size=args.mtp_size,
        cache_transceiver_max_num_tokens=args.cache_transceiver_max_num_tokens,
    )
