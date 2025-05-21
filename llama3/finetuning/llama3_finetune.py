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

from os.path import basename, splitext
from typing import Optional

import nemo_run as run
from utils import (
    get_comm_overlap_callback_idx,
    hf_tokenizer,
    import_ckpt_experiment,
    isfile_train_pack_metadata,
)
from llama3_finetune_utils import llama3_finetune_slurm_executor, llama3_finetune_parse_cli_args, llama3_finetune_auto_configs

from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.recipes.llama3_8b import finetune_recipe as finetune_recipe_8b
from nemo.collections.llm.recipes.llama3_8b import model as model_8b
from nemo.collections.llm.recipes.llama3_70b import finetune_recipe as finetune_recipe_70b
from nemo.collections.llm.recipes.llama3_70b import model as model_70b
from nemo.collections.llm.recipes.llama31_405b import finetune_recipe as finetune_recipe_405b
from nemo.collections.llm.recipes.llama31_405b import model as model_405b
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin
from nemo.utils import logging


HF_MODEL_URI_8b = "meta-llama/Meta-Llama-3-8B"
HF_MODEL_URI_70b = "meta-llama/Meta-Llama-3-70B"
HF_MODEL_URI_405b = "meta-llama/Llama-3.1-405B"
NUM_NODES = 1

def llama3_finetune_performance_recipe(
    finetuning_scheme: str,
    compute_dtype: str,
    num_nodes: int,
    num_gpus_per_node: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: Optional[int],
    max_steps: int,
):
    """
    llama3 finetune recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    finetuning_scheme = "none" if finetuning_scheme == "sft" else finetuning_scheme
    match args.model_size:
        case "8b":
            recipe = finetune_recipe_8b(peft_scheme=finetuning_scheme, performance_mode=True)
            HF_MODEL_URI = "meta-llama/Meta-Llama-3-8B"
            model = model_8b()
            recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
        case "70b":
            recipe = finetune_recipe_70b(peft_scheme=finetuning_scheme, performance_mode=True)
            model = model_70b()
            HF_MODEL_URI = "meta-llama/Meta-Llama-3-70B"
            recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
        case "405b":
            recipe = finetune_recipe_405b(peft_scheme=finetuning_scheme, performance_mode=True)
            model = model_405b()
            HF_MODEL_URI = "meta-llama/Llama-3.1-405B"
            recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
        case _:
            recipe = finetune_recipe_8b(peft_scheme=finetuning_scheme, performance_mode=True)
            HF_MODEL_URI = "meta-llama/Meta-Llama-3-8B"
            model = model_8b()
            recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)         

    # data module configs
    recipe.data.micro_batch_size = mbs
    recipe.data.global_batch_size = gbs
    #recipe.data.num_train_samples = max_steps * gbs * mbs  # ensure only 1 epoch for whole run
    if recipe.data.__fn_or_cls__ == SquadDataModule and not isfile_train_pack_metadata(HF_MODEL_URI, recipe.data):
        # flag is valid only for SquadDataModule
        recipe.data.force_redownload = True

    recipe.trainer.max_steps = max_steps
    recipe.trainer.num_nodes = num_nodes
    recipe.trainer.devices = num_gpus_per_node

    # parallelism configs
    recipe.trainer.strategy.tensor_model_parallel_size = tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = pp_size
    recipe.trainer.strategy.context_parallel_size = cp_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = vp_size
    recipe.trainer.strategy.sequence_parallel = bool(tp_size > 1)

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)

    # compute dtype configs
    if compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
    recipe.trainer.plugins.grad_reduce_in_fp32 = False

    # callback configs
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)
    if comm_overlap_callback_idx is not None:
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_bootstrap_backend = "mpi"
        recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather_with_optimizer_step = bool(
            dp_size > 1 and pp_size > 1 and vp_size and vp_size > 1
        )

    # Misc. for overall faster experiment runtime
    recipe.log.ckpt = None
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.log_every_n_steps = 1

    return recipe, model, HF_MODEL_URI

if __name__ == "__main__":
    args = llama3_finetune_parse_cli_args().parse_args()
    args = llama3_finetune_auto_configs(args)    
    exp_name = "_".join(
        [
            args.finetuning.lower(),
            "nemo_llama3",
            f"{args.model_size}",
            f"{args.compute_dtype}",
            f"{args.num_nodes*args.num_gpus_per_node}",
        ]
    )

    executor = llama3_finetune_slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        args.num_nodes,
        args.num_gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=[args.container_mounts],
        custom_env_vars=[],
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
    )

    executor_for_ckpt = llama3_finetune_slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        args.num_nodes,
        args.num_gpus_per_node,
        "02:00:00",
        args.container_image,
        custom_mounts=[args.container_mounts],
        custom_env_vars=[],
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
    )

    recipe, model, HF_MODEL_URI = llama3_finetune_performance_recipe(
        args.finetuning.lower(),
        args.compute_dtype,
        args.num_nodes,
        args.num_gpus_per_node,
        args.micro_batch_size,
        args.global_batch_size,
        args.tensor_parallelism,
        args.pipeline_parallelism,
        args.context_parallelism,
        args.virtual_pipeline_parallelism,
        args.max_steps,
    )

    if not args.tensorboard:  # tensorboard adds performance overhead.
        recipe.log.tensorboard = None
        recipe.trainer.logger = False
    else:
        # default path is NOT intuitive- `<log_dir>/code/nemo_experiments/tb_logs/default/<tfevents_file>`
        # following line ensures file is at- `<log_dir>/lightning_logs/tb_logs/default/<tfevents_file>`
        recipe.log.log_dir = "/nemo_run/lightning_logs"

    plugins = [PerfEnvPlugin(enable_vboost=True, nccl_pp_comm_chunksize=2097152 if args.pipeline_parallelism > 1 else None)]
    if args.enable_profiling:
        nsys_executor = llama3_finetune_slurm_executor(
            args.account,
            args.partition,
            args.log_dir,
            args.num_nodes,
            args.num_gpus_per_node,
            time_limit="00:30:00",
            container_image=args.container_image,
            custom_mounts=[args.container_mounts],
            custom_env_vars=args.custom_env_vars,
            hf_token=args.hf_token,
            nemo_home=args.nemo_home,
        )
        nsys_recipe, model, HF_MODEL_URI = llama3_finetune_performance_recipe(
            args.compute_dtype,
            args.num_nodes,
            args.num_gpus_per_node,
            args.micro_batch_size,
            args.global_batch_size,
            args.tensor_parallelism,
            args.pipeline_parallelism,
            args.context_parallelism,
            args.virtual_pipeline_parallelism,
            max_steps=30,
        )
        nsys_plugins = [PerfEnvPlugin(enable_vboost=True, nccl_pp_comm_chunksize=2097152 if args.pipeline_parallelism > 1 else None)]
        nsys_plugins.append(NsysPlugin(start_step=20, end_step=30))
    # Figure out how to print this inside .out log
    INFO_STR=f"GSW: MODEL=llama3 SFT FRAMEWORK=nemo MODEL_SIZE={args.model_size} JOB_NUM_NODES={args.num_nodes} GPUS_PER_NODE={args.num_gpus_per_node} DTYPE=${args.compute_dtype} SYNTHETIC_DATA=true GSW_VERSION=25.02 FW_VERSION=24.12 IMAGE={args.container_image} JOB_ID=TODO JOB_MODE=training OPTIMIZATION_NAME={args.optimization_name} OPTIMIZATION_CODE={args.optimization_code} BASE_CONFIG=TODO"
    logging.info(INFO_STR)

    with run.Experiment(exp_name) as exp:
        exp.add(*import_ckpt_experiment(executor_for_ckpt, model, source=f"hf://{HF_MODEL_URI}"))
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
        )

        if args.enable_profiling:
            exp.add(
                nsys_recipe,
                executor=nsys_executor,
                name=exp_name + "_nsys",
                plugins=nsys_plugins,
            )

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()