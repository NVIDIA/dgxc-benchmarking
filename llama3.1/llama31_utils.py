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
import sys
from typing import Dict, List

import nemo_run as run
from nemo_run.config import get_nemorun_home
from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)


def llama31_slurm_executor(
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "00:30:00",
    container_image: str = "nvcr.io/nvidia/nemo:25.02.01",
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    custom_srun_args: List[str] = [],
    hf_token: str = None,
    nemo_home: str = DEFAULT_NEMO_HOME,
    wandb_key: str = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    err_msgs = []
    if log_dir != get_nemorun_home():
        err_msgs.append(f"\nRun `export NEMORUN_HOME={log_dir}` in your shell environment and rerun this script.")
    if len(err_msgs) > 0:
        logging.error("\n".join(err_msgs))
        sys.exit(1)

    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FLASH_ATTN": "1",  # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NVTE_FUSED_ATTN": "1",  # Enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1",  # Print memory allocation
        "NEMORUN_HOME": log_dir,
    }
    if wandb_key is not None:
        env_vars["WANDB_API_KEY"] = wandb_key
    mounts = []
    srun_args = ["--mpi=pmix", ]

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this to 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        env_vars.update({"NEMO_HOME": nemo_home})
        mounts.extend([f"{nemo_home}:{nemo_home}"])

    #Extra location mount for checkpointing support
    STAGE_PATH = os.getenv('STAGE_PATH')
    mounts.extend([f"{STAGE_PATH}:{STAGE_PATH}"])
    if hf_token is not None:
        env_vars.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})

    env_vars |= custom_env_vars
    if env_vars:
        container_env_args = ["--container-env=" + ",".join(list(env_vars.keys()))]
        srun_args.extend(container_env_args)

    mounts.extend(custom_mounts)
    srun_args.extend(custom_srun_args)

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(
            job_dir=os.path.join(log_dir, "experiments"), # has to be experiements
        ),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=env_vars,
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.Packager(),
    )

    return executor



def llama31_parse_cli_args():
    """
    Command line arguments correspong to Slurm cluster and NeMo2.0 Llama 3.1 pre-training experiments.
    """
    parser = argparse.ArgumentParser(description="NeMo2.0 Llama 3.1 Performance Pretraining")

    parser.add_argument(
        "-a",
        "--account",
        type=str,
        help="Slurm account to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Slurm partition to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        help=f"Directory for logging experiment results. Defaults to {get_nemorun_home()}",
        required=False,
        default=get_nemorun_home(),
    )
    parser.add_argument(
        "-t",
        "--time_limit",
        type=str,
        help="Maximum time limit to run experiment for. Defaults to 30 minutes (format- 'HH:MM:SS')",
        required=False,
        default="00:30:00",
    )
    container_img_msg = [
        "NeMo container to use for experiment. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'",
        "Make sure your NGC credentials are accessible in your environment.",
    ]
    parser.add_argument(
        "-i",
        "--container_image",
        type=str,
        help=" ".join(container_img_msg),
        required=False,
        default="nvcr.io/nvidia/nemo:25.03",
    )
    parser.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        choices=["bf16", "fp8"],
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        help="Enable tensorboard logging. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-wd",
        "--wandb",
        help="Enable wandb logging. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-wdk",
        "--wandb_key",
        type=str,
        help="wandb key. Needed for wandb logger projetion to server",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdp",
        "--wandb_prj_name",
        type=str,
        help="wandb project name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdj",
        "--wandb_job_name",
        type=str,
        help="wandb job name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-hf",
        "--hf_token",
        type=str,
        help="HuggingFace token. Defaults to None. Required for accessing tokenizers and checkpoints.",
        default=None,
    )
    nemo_home_msg = [
        "Sets env var `NEMO_HOME` (on compute node using sbatch script)- directory where NeMo searches",
        "for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already",
        f"exist here. Missing files will be downloaded here from HuggingFace. Defaults to {DEFAULT_NEMO_HOME}",
    ]
    parser.add_argument(
        "-nh",
        "--nemo_home",
        type=str,
        help=" ".join(nemo_home_msg),
        default=DEFAULT_NEMO_HOME,
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        help="If true, prints sbatch script to terminal without launching experiment.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-tp",
        "--tensor_parallel_size",
        type=int,
        help="Intra-layer model parallelism. Splits tensors across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-pp",
        "--pipeline_parallel_size",
        type=int,
        help="Inter-layer model parallelism. Splits transformer layers across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cp",
        "--context_parallel_size",
        type=int,
        help="Splits network input along sequence dimension across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-vp",
        "--virtual_pipeline_parallel_size",
        type=int,
        help="Number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ep",
        "--expert_parallel_size",
        type=int,
        help="Distributes Moe Experts across sub data parallel dimension.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-et",
        "--expert_tensor_parallel_size",
        type=lambda x: int(x) if x is not None else None,
        nargs="?",
        const=None,
        help="Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks.\
            Use -et/--expert_tensor_parallel_size <space> for None or -et/--expert_tensor_parallel_size <int>",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-mb",
        "--micro_batch_size",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "-gb",
        "--global_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        help="Target gpu type. Defaults to 'h100'.",
        required=False,
        default="h100",
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        help="Number of gpus.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gn",
        "--gpus_per_node",
        type=int,
        help="Number of gpus per node. Defaults to 8",
        required=False,
        default=8,
    )
    parser.add_argument(
        "-ms",
        "--max_steps",
        type=int,
        help="Number of train steps. Defaults to 100",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cg",
        "--cuda_graphs",
        help="Enable CUDA graphs. Disabled by default",
        action="store_true",
        required=False,
        default=None,  # NOTE: DO NOT SET DEFAULT TO FALSE, IT WILL BE OVERRIDDEN BY THE RECOMMENDED MODEL CONFIGS
    )
    # Begin GSW custom args
    parser.add_argument(
        "-ns",
        "--enable_profiling",
        help="Enable Nsys profiling. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-nt",
        "--enable_nccltrace",
        help="Enable NCCL tracing. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-cm",
        "--custom_mounts",
        type=str,
        required=False,
        default=[],
    )
    parser.add_argument(
        "-ev",
        "--custom_env_vars",
        type=str,
        required=False,
        default={},
    )
    parser.add_argument(
        "-on",
        "--optimization_name",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-oc",
        "--optimization_code",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--model_size",
        type=str,
        help="Size of model. Valid options are [ 8b, 70b, 405b ]",
        required=True,
    )
    parser.add_argument(
        "-fw",
        "--framework_version",
        type=str,
        help="Version of NeMo framework",
        required=True,
    )
    parser.add_argument(
        "-gsw",
        "--gsw_version",
        type=str,
        help="GSW version",
        required=True,
    )
    parser.add_argument(
        "-dp",
        "--disable_perfrun",
        help="Skips performance run for profiling/nccltrace only runs",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-cpin",
        "--cpu_pinning",
        type=int,
        help="Enable CPU pinning to improve performance on some clusters by setting numbers of CPUs per task. Disabled by default",
        required=False,
        default=0,  
    )
    parser.add_argument(
        "-pss",
        "--profiling_start_step",
        type=int,
        help="Defines start step for profiling",
        required=False,
        default=20
    )
    parser.add_argument(
        "-pso",
        "--profiling_stop_step",
        type=int,
        help="Defines start step for profiling",
        required=False,
        default=30
    )
    parser.add_argument(
        "-pgm",
        "--profiling_gpu_metrics",
        help="Enables collection of GPU metrics during profiling",
        action="store_true",
        required=False,
        default=False, 
    )

    return parser

def llama31_auto_configs(args):

    if args.num_gpus is None:
        match args.model_size:
            case "8b":   num_gpus = 8
            case "70b":  num_gpus = 64
            case "405b": num_gpus = 1024
            case _:
                logging.error("Invalid model size provided. Options are [ 8b, 70b, 405b ]")
                sys.exit(1)
    else: num_gpus = args.num_gpus
    num_nodes = num_gpus // args.gpus_per_node
    num_layers = None
    hidden_size = None
    mbs = 1 if args.micro_batch_size is None else args.micro_batch_size

    match args.model_size:
        case "8b":
            gbs = args.num_gpus * 16 if args.global_batch_size is None else args.global_batch_size
            tp_size = 1 if args.tensor_parallel_size is None else args.tensor_parallel_size
            pp_size = 1 if args.pipeline_parallel_size is None else args.pipeline_parallel_size
            cp_size = 2 if args.context_parallel_size is None else args.context_parallel_size
            # setting vp to 1 here sets it to null in yaml
            vp_size = 1 if args.virtual_pipeline_parallel_size is None else args.virtual_pipeline_parallel_size
            num_layers = 32
            hidden_size = 4096
        case "70b":
            gbs = args.num_gpus * 2 if args.global_batch_size is None else args.global_batch_size
            tp_size = 4 if args.tensor_parallel_size is None else args.tensor_parallel_size
            pp_size = 4 if args.pipeline_parallel_size is None else args.pipeline_parallel_size
            cp_size = 2 if args.context_parallel_size is None else args.context_parallel_size
            vp_size = 5 if args.virtual_pipeline_parallel_size is None else args.virtual_pipeline_parallel_size
            num_layers = 80
            hidden_size = 8192
            if args.compute_dtype == "fp8":
                pp_size = 8 if args.pipeline_parallel_size is None else args.pipeline_parallel_size
                cp_size = 1 if args.context_parallel_size is None else args.context_parallel_size
        case "405b":
            gbs = args.num_gpus // 2 if args.global_batch_size is None else args.global_batch_size
            tp_size = 8 if args.tensor_parallel_size is None else args.tensor_parallel_size
            pp_size = 8 if args.pipeline_parallel_size is None else args.pipeline_parallel_size
            cp_size = 2 if args.context_parallel_size is None else args.context_parallel_size
            vp_size = 8 if args.virtual_pipeline_parallel_size is None else args.virtual_pipeline_parallel_size
            num_layers = 126
            hidden_size = 16384
            if args.num_gpus == 128: # Proxy
                gbs = 128
                pp_size = 4 if args.pipeline_parallel_size is None else args.pipeline_parallel_size
                num_layers = 62
            elif args.num_gpus == 64: # Proxy
                gbs = 64
                pp_size = 4 if args.pipeline_parallel_size is None else args.pipeline_parallel_size
                num_layers = 30
            elif args.num_gpus == 32: # Proxy
                gbs = 64
                pp_size = 1 if args.pipeline_parallel_size is None else args.pipeline_parallel_size
                vp_size = 1 if args.virtual_pipeline_parallel_size is None else args.virtual_pipeline_parallel_size
                num_layers = 8


    enable_cuda_graphs = args.cuda_graphs
    enable_cuda_graphs = False if enable_cuda_graphs is None else bool(int(enable_cuda_graphs))

    max_steps = 50 if args.max_steps is None else args.max_steps

    # Llama does not use EP but needs to be int type
    ep_size=1

    kwargs = num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, num_layers, hidden_size
    kwargs = [int(arg) if arg is not None else arg for arg in kwargs] + [enable_cuda_graphs]
    kwargs = kwargs + [max_steps]

    return kwargs
