# Overview

This recipe contains information and scripts to produce performance results for the Llama3 pre-training workloads. The scripts help perform environment setup and launch benchmark jobs.

This variant of the workload is best-suited for GPU clusters with at least 8x H100 GPUs with 80 GB memory. The training of smaller 8-billion parameter variant of the workload will not fit on fewer GPUs with lesser memory. This workload supports BF16 and FP8 precisions.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 8b   | BF16/FP8  | 8    | 8192   | 32     | 1   | 1   | 2   | NA  | 4   | NA  | 1   | 128  | 32 |
| 8b   | BF16/FP8  | 16   | 8192   | 32     | 1   | 1   | 2   | NA  | 8   | NA  | 1   | 256  | 32 |
| 8b   | BF16/FP8  | 32   | 8192   | 32     | 1   | 1   | 2   | NA  | 16  | NA  | 1   | 512  | 32 |
| 8b   | BF16/FP8  | 64   | 8192   | 32     | 1   | 1   | 2   | NA  | 32  | NA  | 1   | 1024 | 32 |
| 8b   | BF16/FP8  | 128  | 8192   | 32     | 1   | 1   | 2   | NA  | 64  | NA  | 1   | 2048 | 32 |

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 70b  | BF16      | 64   | 8192   | 80     | 4   | 4   | 2   | NA  | 2   | 5   | 1   | 128  | 64 |
| 70b  | BF16      | 128  | 8192   | 80     | 4   | 4   | 2   | NA  | 4   | 5   | 1   | 256  | 64 |
| 70b  | BF16      | 256  | 8192   | 80     | 4   | 4   | 2   | NA  | 8   | 5   | 1   | 512  | 64 |
| 70b  | BF16      | 512  | 8192   | 80     | 4   | 4   | 2   | NA  | 16  | 5   | 1   | 1024 | 64 |
| 70b  | BF16      | 1024 | 8192   | 80     | 4   | 4   | 2   | NA  | 32  | 5   | 1   | 2048 | 64 |
| 70b  | BF16      | 2048 | 8192   | 80     | 4   | 4   | 2   | NA  | 64  | 5   | 1   | 4096 | 64 |
| 70b  | FP8       | 64   | 8192   | 80     | 4   | 8   | 1   | NA  | 2   | 5   | 1   | 128  | 64 |
| 70b  | FP8       | 128  | 8192   | 80     | 4   | 8   | 1   | NA  | 4   | 5   | 1   | 256  | 64 |
| 70b  | FP8       | 256  | 8192   | 80     | 4   | 8   | 1   | NA  | 8   | 5   | 1   | 512  | 64 |
| 70b  | FP8       | 512  | 8192   | 80     | 4   | 8   | 1   | NA  | 16  | 5   | 1   | 1024 | 64 |
| 70b  | FP8       | 1024 | 8192   | 80     | 4   | 8   | 1   | NA  | 32  | 5   | 1   | 2048 | 64 |
| 70b  | FP8       | 2048 | 8192   | 80     | 4   | 8   | 1   | NA  | 64  | 5   | 1   | 4096 | 64 |

| Size       | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 405b-Proxy | BF16/FP8  | 32   | 8192   | 8      | 8   | 1   | 2   | NA  | 2   | NA  | 1   | 64   | 32 |
| 405b-Proxy | BF16/FP8  | 64   | 8192   | 30     | 8   | 4   | 2   | NA  | 1   | 8   | 1   | 64   | 64 |
| 405b-Proxy | BF16/FP8  | 128  | 8192   | 62     | 8   | 4   | 2   | NA  | 2   | 8   | 1   | 128  | 64 |
| 405b       | BF16/FP8  | 512  | 8192   | 126    | 8   | 8   | 2   | NA  | 4   | 8   | 1   | 256  | 64 |
| 405b       | BF16/FP8  | 1024 | 8192   | 126    | 8   | 8   | 2   | NA  | 8   | 8   | 1   | 512  | 64 |
| 405b       | BF16/FP8  | 2048 | 8192   | 126    | 8   | 8   | 2   | NA  | 16  | 8   | 1   | 1024 | 64 |

# Expected Performance

Performance for Llama 3.1 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/experiments/pretrain_llama3.1_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}` folder. 

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep -r --include '*.out' train_step_timing experiments
Training epoch 0, iteration 49/49 | lr: 7.496e-06 | global_batch_size: 128 | global_step: 49 | peak_memory_usage: 70833405952 | memory_allocated: 44271689728 | reduced_train_loss: 11.96 | train_step_timing in s: 12.39 | consumed_samples: 6400
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 8192 * 128 / 12.39 = 84631

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 84631 / 86400 = 136.76 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is **1979** TFLOPS and for H100 BF16 is **989** TFLOPS.

The model flops for Llama 3.1 8b for GBS=1 is 4.74E+14. Calculation shown [here](#notes).

E.g. Llama 3.1 8b FP8 on 8x H100 GPUs (GBS=128)
```shell
peak FLOPS for H100 = 1979 TFLOPS
training step time = 12.39 s
model flops = 4.74E+14

MFU = 128 * 4.74E+14 / 12.39 / 8 / 1979E+12 = 30.93%
```

| Llama 3.1 8b BF16 | 8x H100 GPUs  | 16x H100 GPUs  | 32x H100 GPUs  | 64x H100 GPUs  | 128x H100 GPUs  | 
|---|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) | 12.27	| 12.3 | 12.32 | 12.33 | 12.33
| Throughput in tokens per second       |  85459	| 170500	| 340447	| 680341 | 1360683  
| Model flops utilization               |  62.47%	| 62.31%	| 62.21%	| 62.16%	| 62.16%
| Time to train 1T tokens in days       |  135.43	| 67.88	| 34	| 17.01 |	8.51

| Llama 3.1 8b FP8 | 8x H100 GPUs  | 16x H100 GPUs  | 32x H100 GPUs  | 64x H100 GPUs  | 128x H100 GPUs  | 
|---|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) |  9.33	| 9.36	| 9.43 | 9.48	| 9.54
| Throughput in tokens per second       |  112388	| 224055	| 444783	| 884874	| 1758618
| Model flops utilization               |  41.07%	| 40.94%	| 40.64%	| 40.42%	| 40.17%
| Time to train 1T tokens in days       |  102.98	| 51.66 |	26.02 |	13.08 |	6.58

| Llama 3.1 70b BF16 | 64x H100 GPUs  | 128x H100 GPUs  | 256x H100 GPUs  | 512x H100 GPUs  | 1024x H100 GPUs  | 2048x H100 GPUs 
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step) |  14.66	| 14.67	| 14.68	| 14.67	| 14.65	| 14.66 
| Throughput in tokens per second       |  71526	| 142955	| 285716	| 571821	| 1145202	| 2288843 
| Model flops utilization               |  54.32%	| 54.29%	| 54.25%	| 54.29%	| 54.36%	| 54.32%
| Time to train 1T tokens in days       |  161.82	| 80.96	| 40.51 |	20.24 | 10.11	| 5.06 

| Llama 3.1 70b FP8 | 64x H100 GPUs  | 128x H100 GPUs  | 256x H100 GPUs  | 512x H100 GPUs  | 1024x H100 GPUs  | 2048x H100 GPUs 
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Training step time (seconds per step)   | 9.86	| 9.95	| 10	| 10.07	| 10.08 |	10.15
| Throughput in tokens per second         | 106346	| 210769 | 419430	| 833030	| 1664406	| 3305855
| Model flops utilization                 | 40.38%	| 40.02% | 39.82%	| 39.54%	| 39.50%	| 39.23%
| Time to train 1T tokens in days         | 108.83	| 54.91	| 27.59	| 13.89	| 6.95	| 3.5 

| Llama 3.1 405b BF16 | 32x H100 GPUs | 64x H100 GPUs  | 128x H100 GPUs  | 512x H100 GPUs | 1024x H100 GPUs | 2048x H100 GPUs |
|---|---|---|---|---|---|---|
| Training step time (seconds per step)  |  5.4	| 10.1	| 19.62 | 19.78	| 19.84	| 19.87 
| Throughput in tokens per second        | 97090	| 51910	| 53444	|  106024	| 211406 |	422175
| Model flops utilization                |  55.23%	| 52.52%	| 55.31%	| 55.47%	| 55.31%	| 55.22%
| Time to train 1T tokens in days        |   n/a	| n/a	| n/a	| 109.16	| 54.75	| 27.42 

| Llama 3.1 405b FP8 | 32x H100 GPUs | 64x H100 GPUs  | 128x H100 GPUs  | 512x H100 GPUs | 1024x H100 GPUs | 2048x H100 GPUs |
|---|---|---|---|---|---|---|
| Training step time (seconds per step)  | 3.53	| 6.32	| 12.18	| 12.37	| 12.42	| 12.56 
| Throughput in tokens per second        | 148524	| 82957	| 86090	| 169535	| 337706	| 667883 
| Model flops utilization                |  42.24%	| 41.97%	| 44.55% | 44.35%	| 44.17%	| 43.68%
| Time to train 1T tokens in days        |   n/a	| n/a	| n/a	| 68.27	| 34.27	| 17.33

# Prerequisites
This recipe requires access to Llama 3.1. Instructions are below if needed.

Python virtual environment must be created using Python v.3.10.12 or newer before running the workload.

## Request Access
A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens) the huggingface token is used to run the pre-training scripts. Add the generated token to your environment via ```export HF_TOKEN=<your token>```.

Access to Llama 3.1 must be requested through [Meta's website](https://llama.meta.com/llama-downloads/) then requested on the [HuggingFace Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-70B) and [HuggingFace Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-405B) page. The approval process is not automatic and could take a day or more.

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**
- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
	- Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

### Set the environment variables
```shell
# Set the path where all artifacts will be downloaded
export STAGE_PATH=<path to your shared file system folder> (e.g. /lustre/myproject/nemo)
# Set HuggingFace token
export HF_TOKEN=<your token>
```

**Important:** `STAGE_PATH` used in this step must be used when running the workload.

### Prepare python virtual environment

The workload relies on python virtual environment in order ensure there are no conflicts between required dependencies and user's packages. We require Python 3.10.12 or newer for the workload to work.

There are multiple choices available to set up virtual environment: 
* conda
* python venv


#### Conda 

To install and activate conda virtual environment
```shell
# pick INSTALL_PATH with sufficient disk space
INSTALL_PATH=~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $INSTALL_PATH/miniconda.sh
bash $INSTALL_PATH/miniconda.sh -b -p $INSTALL_PATH/miniconda3
$INSTALL_PATH/miniconda3/bin/conda init
source ~/.bashrc

conda create -n nemo2-llama python=3.12
conda activate nemo2-llama
```

When you are finished running this benchmark you can deactivate the environment, run this command
```shell
conda deactivate
```

#### Python venv

To install and activate python venv 
```shell
python3 -m venv $STAGE_PATH/venv
source $STAGE_PATH/venv/bin/activate
```
When you are finished running this benchmark you can deactivate the environment, run this command
```shell
deactivate
```

### Setup script

Create a staging area by running the attached setup.sh. The script converts the docker image to a .```.sqsh``` file under the $STAGE_PATH folder and installs required packages to the python environment to enable NeMo-Run launcher functionality.

Make sure the previous step has been completed and python virtual environment is active. Run the setup script using the following command.
```shell
# activate virtual python environment setup previously
sbatch -A ${SBATCH_ACCOUNT} -p ${SBATCH_PARTITION} -N 1 ./setup.sh
```

# Prepare Dataset
Since Llama3 training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model.

NeMo-Run launcher is used to process command line arguments and pass them down as hyperparameters to a multi-node job performing the training.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the 
`$STAGE_PATH/experiments/pretrain_llama3.1_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}` folder.

Below is a command template for launching Llama 3.1 model training.
```shell
DTYPE=<fp8,bf16> MODEL_SIZE=<8b,70b,405b> JOB_TOTAL_GPUS=<8,16,..,2048> GPUS_PER_NODE=<2,4,8> ./launch.sh
```
Where:
- `DTYPE`, `MODEL_SIZE` are required environment variables.
  - `DTYPE` can be either `fp8` or `bf16`.
  - `MODEL_SIZE` can be `8b`, `70b`, or `405b`.

For example, command to train Llama 3.1 8B with BF16 precision on 8 H100 GPUs would look like:
```shell
DTYPE=bf16 MODEL_SIZE=8b JOB_TOTAL_GPUS=8 GPUS_PER_NODE=8 ./launch.sh
```

# Profiling
We have two profiling methods supported: Nsight and NCCL Trace.

Due to overhead while profiling: the results generated with these settings are not valid for comparison. 'Performance' and 'Profiling' runs should be done separately.

**Note:** Profiling and NCCL Trace are currently mutually exclusive.

## Run Nsight Profiling

To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 25 steps where steps 20-25 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Profiling job details:
* **MPI Ranks:** 0-8
* **Job Steps:** 20-25
* **Output Location:** .nsys-rep files are saved in the nsys_profile folder within the existing results directory. Results directory is under `$STAGE_PATH/experiments/pretrain_llama3.1_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}/`
* **Filename format:** `profile_${PROCESS_ID}.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true DTYPE=fp8 MODEL_SIZE=70b JOB_TOTAL_GPUS=256 GPUS_PER_NODE=8 ./launch.sh
```

will produce results under `$STAGE_PATH/experiments/pretrain_llama3.1_70b_fp8_256/pretrain_llama3.1_70b_fp8_256_<timestamp>/pretrain_llama3.1_70b_fp8_256_nsys` folder. 

### Customizing profiling behavior:
* Specify job steps to profile:
	* `PROFILE_START_STEP`: start profiling on this job step.
	- Default: 20
	* `PROFILE_STOP_STEP`: stop profiling on this job step.
	- Default: 25



### Viewing results

In order to view the profile traces (*.nsys-rep files) interactively:
- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run. 

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

## Run NCCL Trace

NCCL traces provide a breakdown of the communication pattern outlining both the type of NCCL calls being made and their message sizes.
To add NCCL trace information to the log, set variable ENABLE_NCCLTRACE=true.

```shell
ENABLE_NCCLTRACE=true DTYPE=<fp8,bf16> MODEL_SIZE=<8b,70b,405b> JOB_TOTAL_GPUS=<8,16,..,2048> GPUS_PER_NODE=<2,8> ./launch.sh
```

**Defaults:**
* File Size: NCCL Trace generates large files, therefore profiling is limited to the first 5 steps.
* Output Location: Trace files are saved to a directory `$STAGE_PATH/experiments/pretrain_llama3.1_${MODEL_SIZE}_${DTYPE}$_{JOB_TOTAL_GPUS}/pretrain_llama3.1_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}_${timestamp}/pretrain_llama3.1_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}_nccltrace`

**Example command:**
```shell
ENABLE_NCCLTRACE=true DTYPE=fp8 MODEL_SIZE=8b JOB_TOTAL_GPUS=16 ./launch.sh
```

# Run With Checkpoints

## Save Checkpoint

Save checkpoint feature works for Llama3.1 8b, 70b and 405b model sizes with either FP8 or BF16 precision. Make sure your file system has sufficient disk space to accomodate checkpoint sizes below:

| Model | Checkpoint Size | Supported Scales |
| :---: | :---: |  :---: |
| 8b | ~105 GB | 16-128 |
| 70b | ~920 GB | 128-2048 |
| 405b | ~5.2 TB | 512-2048 |

### How to enable
To save the checkpoints after pretraining Llama 3.1 model for `max_steps`, you need to set environment variable `ENABLE_CHECKPOINT=true`. At the end of the pretraining the checkpoints will be saved in the  `$STAGE_PATH/experiments` folder.

```shell
experiment_name = pretrain_llama3.1_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}
timestamp = date '+%s'
Example directory where checkpoints are saved is $STAGE_PATH/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/
```
Command to run llama3.1 with checkpoint save enabled
```shell
ENABLE_CHECKPOINT=true DTYPE=<fp8,bf16> MODEL_SIZE=<8b/70b/405b> JOB_TOTAL_GPUS=<16,..,2048> GPUS_PER_NODE=<2,8> ./launch.sh
```

### How to validate
- Check `$STAGE_PATH/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/*/weights` folder that it contains *.distcp files
- Check job output log-*.out file (see Training section for reference) for entries like
  ```
  [NeMo I 2025-04-11 14:48:45 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 50 : Start time: 1744408121.151s : Save duration: 4.389s
  ```

## Load Checkpoint

Load checkpoint feature works successfully at the following scales:

| Model | Minimum Tested Scale |
| :---: | :---: |
| 8b | 16 |
| 70b | 128 |
| 405b | 1024 |

**Note**
 - Running load checkpointing feature at other scales may run into CUDA OOM errors. 


### How to enable
To resume training from saved checkpoints, you need to set `LOAD_CHECKPOINT_PATH=<path_to_checkpoint_directory>` environment variable. Make sure the checkpoint files are under the `$STAGE_PATH` directory and `LOAD_CHECKPOINT_PATH` variable is set to parent folder of the `weights` directory containing distributed checkpoint files with extension `*.distcp`.

E.g., if the checkpoint was saved under `$STAGE_PATH/experiments/pretrain_llama3.1_70b_fp8_256/pretrain_llama3.1_70b_fp8_256_<timestamp>/pretrain_llama3.1_70b_fp8_256/code/nemo_experiments/default/checkpoints/*/weights` then set the environment variable to a directory one level higher: 

`LOAD_CHECKPOINT_PATH=$STAGE_PATH/experiments/pretrain_llama3.1_70b_fp8_256/pretrain_llama3.1_70b_fp8_256_<timestamp>/pretrain_llama3.1_70b_fp8_256/code/nemo_experiments/default/checkpoints/default--None=0.0000-epoch=0-consumed_samples=12800.0`

The scripts will restore configuration from the checkpoint and resume training process. Training will run for 1 step after checkpoint has been loaded.
```shell
LOAD_CHECKPOINT_PATH=<your_path_to_checkpoint_directory>
DTYPE=bf16 MODEL_SIZE=8b JOB_TOTAL_GPUS=16 GPUS_PER_NODE=8 ./launch.sh
```

### How to validate

To validate that checkpoint was loaded successfully look for the entry like below in the main job log-*.out file (see Training section for reference):

```
[NeMo I 2025-04-11 14:46:18 nemo_logging:393] Global Checkpoint Load : Rank : 0 : Start time : 1744407969.270s : Time spent in load_checkpoint: 9.712s
```

# Notes

```shell
model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

model flops breakdown:
    attention flops = 12 * (number of layers) * (hidden size)^2 * (1 + (number of query groups)/(number of attention heads) + (sequence length)/(hidden size))
    mlp flops = 18 * (number of layers) * (FFN size) * (hidden size)
    embedding flops = 6 * (vocab size) * (hidden size)

Llama 3.1 8b calculation:
    sequence length = 8192
    attention flops = 12 * 32 * 4096^2 * (1 + 8/32 + 8192/4096) = 20,937,965,568
    mlp flops = 18 * 32 * 14336 * 4096 = 33,822,867,456
    embedding flops = 6 * 128256 * 4096 = 3,152,019,456

    model flops = 8192 * (20,937,965,568 + 33,822,867,456 + 3,152,019,456) = 4.74E+14
```