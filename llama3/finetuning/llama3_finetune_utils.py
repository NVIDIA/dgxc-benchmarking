# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
from typing import Dict, List

import nemo_run as run
from lightning.pytorch.callbacks.callback import Callback
from nemo_run.config import NEMORUN_HOME
from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)
STAGE_PATH = os.getenv('STAGE_PATH')

def llama3_finetune_slurm_executor(
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "00:30:00",
    container_image: str =  f"{STAGE_PATH}/nvidia+nemo+24.12.sqsh",
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    custom_srun_args: List[str] = [],
    hf_token: str = None,
    nemo_home: str = DEFAULT_NEMO_HOME,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    err_msgs = []
    if log_dir != NEMORUN_HOME:
        err_msgs.append(f"\nRun `export NEMORUN_HOME={log_dir}` in your shell environment and rerun this script.")
    if len(err_msgs) > 0:
        logging.error("\n".join(err_msgs))
        sys.exit(1)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "0",
        "TOKENIZERS_PARALLELISM": "False",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "1",
        "NVTE_FLASH_ATTN": "1",
        "NEMO_LOG_MEMORY_USAGE": "1",
        "NEMORUN_HOME": log_dir,
    }
    mounts = []
    srun_args = ["--mpi=pmix"]

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        env_vars.update({"NEMO_HOME": nemo_home})
        mounts.extend([f"{nemo_home}:{nemo_home}"])
    if hf_token is not None:
        env_vars.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})

    env_vars |= custom_env_vars
    mounts.extend(custom_mounts)
    srun_args.extend(custom_srun_args)

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(
            job_dir=os.path.join(log_dir, "experiments"),
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
        packager=run.Packager()
    )

    return executor

def llama3_finetune_auto_configs(args):
    # Auto scale GBS by number of GPUs if not set by user
    if not args.num_nodes:
        match args.model_size:
            case "8b":
                args.num_nodes = 8 // args.num_gpus_per_node
            case "70b":
                args.num_nodes = 8 // args.num_gpus_per_node
            case "405b":
                args.num_nodes = 24 // args.num_gpus_per_node
            case _:
                logging.error("Invalid model size provided. Options are [ 8b, 70b, 405b ]")
                sys.exit(1)

    total_num_gpus = args.num_nodes * args.num_gpus_per_node

    if not args.global_batch_size:
        match args.model_size:
            case "8b":
                #This section needs a re-work after we do the scaling tests
                args.global_batch_size = total_num_gpus * 4
                args.micro_batch_size = 1
                args.tensor_parallelism = 1
                args.pipeline_parallelism = 1
                args.context_parallelism = 1
                args.virtual_pipeline_parallelism = None
            case "70b":
                args.micro_batch_size = 1
                args.global_batch_size = total_num_gpus * 4
                args.tensor_parallelism = 2
                args.pipeline_parallelism = 4
                args.context_parallelism = 1
                args.virtual_pipeline_parallelism = 20
                if args.finetuning == "sft":
                    args.tensor_parallelism = 4
                    args.virtual_pipeline_parallelism = 5
                    args.global_batch_size = total_num_gpus * 1
            case "405b":
                args.global_batch_size = total_num_gpus * 1
                args.micro_batch_size = 1
                args.tensor_parallelism = 4
                args.pipeline_parallelism = 6
                args.context_parallelism = 1
                args.virtual_pipeline_parallelism = 7

    return args

def llama3_finetune_parse_cli_args():
    """
    Command line arguments correspong to Slurm cluster and NeMo2.0 for running fine-tuning Llama 3.1 models
    """
    parser = argparse.ArgumentParser(description="NeMo2.0 Performance finetuning Recipes")

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
        help=f"Directory for logging experiment results. Defaults to {NEMORUN_HOME}",
        required=False,
        default=NEMORUN_HOME,
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
        default="nvcr.io/nvidia/nemo:24.12",
    )
    parser.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    parser.add_argument(
        "-f",
        "--finetuning",
        type=str,
        help="Finetuning scheme",
        required=False,
        default="lora",
    )
    parser.add_argument(
        "-ep",
        "--enable_profiling",
        help="Enable Nsys profiling. Diabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        help="Enable tensorboard logging. Disabled by default",
        action="store_true",
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
        "-n",
        "--num_nodes",
        type=int,
        help="",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ng",
        "--num_gpus_per_node",
        type=int,
        help="",
        required=False,
        default=8,
    )
    parser.add_argument(
        "-mbs",
        "--micro_batch_size",
        type=int,
        help="",
        required=False,
        default=1,
    )
    parser.add_argument(
        "-gbs",
        "--global_batch_size",
        type=int,
        help="",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-tp",
        "--tensor_parallelism",
        type=int,
        help="",
        required=False,
        default=1,
    )
    parser.add_argument(
        "-pp",
        "--pipeline_parallelism",
        type=int,
        help="",
        required=False,
        default=1,
    )
    parser.add_argument(
        "-cp",
        "--context_parallelism",
        type=int,
        help="",
        required=False,
        default=1,
    )
    parser.add_argument(
        "-vp",
        "--virtual_pipeline_parallelism",
        type=int,
        help="",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--max_steps",
        type=int,
        help="",
        required=False,
        default=100,
    )
    parser.add_argument(
        "-cm",
        "--container_mounts",
        type=str,
        help="",
        required=False,
        default='',
    )
    parser.add_argument(
        "-ev",
        "--custom_env_vars",
        type=str,
        help="",
        required=False,
        default={},
    )
    parser.add_argument(
        "-on",
        "--optimization_name",
        type=str,
        help="",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-oc",
        "--optimization_code",
        type=str,
        help="",
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

    return parser
