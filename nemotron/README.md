# Overview

This recipe contains information and scripts to produce performance results for the Nemotron 4 pre-training workloads. The scripts help perform environment setup and launch benchmark jobs.

This variant of the workload is best-suited for GPU clusters with at least 16x H100 GPUs with 80 GB memory. Training of smaller 15-billion parameter variant of the workload will not fit on fewer GPUs with lesser memory. This workload supports BF16 and FP8 precisions.

| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 15b   | BF16/FP8  | 16   | 4096  | 32     | 2   | 1   | 1   | NA  | 8   | NA  | 2   | 64   | 4 |
| 15b   | BF16/FP8  | 32   | 4096  | 32     | 2   | 1   | 1   | NA  | 16  | NA  | 2   | 128  | 4 |
| 15b   | BF16/FP8  | 64   | 4096  | 32     | 2   | 1   | 1   | NA  | 32  | NA  | 2   | 256  | 4 |
| 15b   | BF16/FP8  | 128  | 4096  | 32     | 2   | 1   | 1   | NA  | 64  | NA  | 2   | 512  | 4 |
| 15b   | BF16/FP8  | 256  | 4096  | 32     | 2   | 1   | 1   | NA  | 128 | NA  | 2   | 1024 | 4 |
| 15b   | BF16/FP8  | 512  | 4096  | 32     | 2   | 1   | 1   | NA  | 256 | NA  | 2   | 2048 | 4 |
| 15b   | BF16/FP8  | 1024 | 4096  | 32     | 2   | 1   | 1   | NA  | 512 | NA  | 2   | 4096 | 4 |
| 15b   | BF16/FP8  | 2048 | 4096  | 32     | 2   | 1   | 1   | NA  | 1024| NA  | 2   | 8192 | 4 |


| Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
|------|:---------:|:----:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:---:|
| 340b | BF16/FP8  | 256  | 4096   | 96     | 8   | 8   | 1   | NA  | 4   | NA  | 1   | 64   | 16 |
| 340b | BF16/FP8  | 512  | 4096   | 96     | 8   | 8   | 1   | NA  | 8   | NA  | 1   | 128  | 16 |
| 340b | BF16/FP8  | 1024 | 4096   | 96     | 8   | 8   | 1   | NA  | 16  | NA  | 1   | 256  | 16 |
| 340b | BF16/FP8  | 2048 | 4096   | 96     | 8   | 8   | 1   | NA  | 32  | NA  | 1   | 512  | 16 |


# Expected Performance

Performance for Nemotron4 training is measured by seconds per iteration, or in other words seconds per training step. This metric is logged for every training step in a .out file which is generated inside of the `$STAGE_PATH/experiments/pretrain_nemotron4_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}` folder. 

Since the performance fluctuates significantly at the beginning, we are using the last training step timing to obtain throughput value.

```shell
grep -r --include '*.out' train_step_timing experiments
Training epoch 0, iteration 49/49 | lr: 4.491e-06 | global_batch_size: 256 | global_step: 49 | peak_memory_usage: 72972500992 | memory_allocated: 34466545664 | reduced_train_loss: 12.81 | train_step_timing in s: 2.693 | consumed_samples: 12800
```

To obtain throughput as a tokens per second measurement, follow this formula: 
```shell
(sequence length) * (global batch size) / (training_step_timing) = (throughput in tokens per second)
```

E.g. 4096 * 256 / 2.693 = 389371

To calculate time to train estimate:
```shell
(total tokens) / (throughput in tokens per second) / (number of seconds in a day) = (time to train in days) 
```
E.g. 1e12 / 389371 / 86400 = 29.7 days 


To calculate the model flops utilization (MFU):
```shell
MFU = (global batch size) * (model flops) / (training step time) / (number of GPUs) / (peak GPU FLOPS)
```

The peak theoretical throughput for H100 FP8 is **1979** TFLOPS and for H100 BF16 is **989** TFLOPS.

The model flops for Nemotron4 15b for GBS=1 is 434.5e12. Calculation shown [here](#notes).

E.g. NeMotron4 15b BF16 on 64x H100 GPUs (GBS=256)
```shell
peak FLOPS for H100 BF16 = 989 TFLOPS
training step time = 2.693 s
model flops = 434.5e12

MFU = 256 * 434.5e12 / 2.693 / 64 / 989e+12 = 65.26%
```

| Nemotron4 15b BF16 | 16x H100 GPUs  | 32x H100 GPUs  | 64x H100 GPUs  | 128x H100 GPUs  | 256x H100 GPUs  |  512x H100 GPUs  | 1024x H100 GPUs  | 2048x H100 GPUs  |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
Training step time (seconds per step)|2.66|2.66|2.67|2.67|2.68|2.71|2.74|2.78
Throughput in tokens per second |98550|197101|392725|785450|1565039|3095427|6123072|12069940
Model flops utilization|66.07%|66.07%|65.82%|65.82%|65.57%|64.85%|64.14%|63.21%
Time to train 1T tokens in days|117.44|58.72|29.47|14.74|7.4|3.74|1.89|0.96

| Nemotron4 15b FP8 | 16x H100 GPUs  | 32x H100 GPUs  | 64x H100 GPUs  | 128x H100 GPUs  | 256x H100 GPUs  |  512x H100 GPUs  | 1024x H100 GPUs  | 2048x H100 GPUs  |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
Training step time (seconds per step)|1.98|1.98|1.98|1.99|2|2.02|2.03|2.09
Throughput in tokens per second |132396|264792|529584|1053845|2097152|4152776|8264638|16054752
Model flops utilization|44.35%|44.35%|44.35%|44.13%|43.91%|43.48%|43.26%|42.02%
Time to train 1T tokens in days|87.42|43.71|21.86|10.98|5.52|2.79|1.4|0.72

| Nemotron4 340b BF16 | 256x H100 GPUs  |  512x H100 GPUs  | 1024x H100 GPUs  | 2048x H100 GPUs  |
|---|:---:|:---:|:---:|:---:|
Training step time (seconds per step)|4.36|4.39|4.41|4.43
Throughput in tokens per second |60125|119428|237772|473398
Model flops utilization|58.56%|58.16%|57.89%|57.63%
Time to train 1T tokens in days|192.5|96.91|48.68|24.45

| Nemotron4 340b FP8 | 256x H100 GPUs  |  512x H100 GPUs  | 1024x H100 GPUs  | 2048x H100 GPUs  |
|---|:---:|:---:|:---:|:---:|
Training step time (seconds per step)|2.93|2.97|3|3.05
Throughput in tokens per second |89469|176528|349525|687591
Model flops utilization|43.55%|42.96%|42.53%|41.83%
Time to train 1T tokens in days|129.36|65.57|33.11|16.83

# Prerequisites
This recipe requires access to HuggingFace. Instructions are below if needed.

Python virtual environment must be created using Python v.3.10.12 or newer before running the workload.

## Request Access
No special access required to run this benchmark.

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

conda create -n nemo2-nemotron python=3.12
conda activate nemo2-nemotron
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

**Note**: output log from running `setup.sh` script may include an error about tritonclient dependency. The error can be ignored as it doesn't affect benchmark functionality. 
It would look like this:
`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tritonclient 2.51.0 requires urllib3>=2.0.7, but you have urllib3 1.26.20 which is incompatible.`

# Prepare Dataset
Since Nemotron4 training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. 

NeMo-Run launcher is used to process command line arguments and pass them down as hyperparameters to a multi-node job performing the training.

The training will run for the first 50 steps and will stop afterwards. Log files and results will be located under the 
`$STAGE_PATH/experiments/pretrain_nemotron4_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}` folder.

Below is a command template for launching Nemotron 4 model training.
```shell
DTYPE=<fp8,bf16> MODEL_SIZE=<15b,340b> JOB_TOTAL_GPUS=<16,..,2048> GPUS_PER_NODE=<2,4,8> ./launch.sh
```
Where:
- `DTYPE`, `MODEL_SIZE` are required environment variables.
  - `DTYPE` can be either `fp8` or `bf16`.
  - `MODEL_SIZE` can be `15b` or `340b`.

For example, command to train nemotron4 15B with BF16 precision on 16 H100 GPUs would look like:
```shell
DTYPE=bf16 MODEL_SIZE=15b JOB_TOTAL_GPUS=16 GPUS_PER_NODE=8 ./launch.sh
```
will produce results under `$STAGE_PATH/experiments/pretrain_nemotron4_15b_bf16_16/pretrain_nemotron4_15b_bf16_16_<timestamp>/pretrain_nemotron4_15b_bf16_16` folder.


# Run Nsight Profiling
Due to profiling overhead, the results generated while profiling enabled are not valid for performance comparison. 'Performance' and 'Profiling' runs should be done separately.


To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 25 steps where steps 20-25 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Profiling job details:
* **MPI Ranks:** 0-8
* **Job Steps:** 20-25
* **Output Location:** .nsys-rep files are saved in the nsys_profile folder within the existing results directory. Results directory is under `$STAGE_PATH/experiments/pretrain_nemotron4_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}_nsys/`
* **Filename format:** `profile_${PROCESS_ID}.nsys-rep`

**Example command:**
```shell
ENABLE_PROFILE=true DTYPE=bf16 MODEL_SIZE=15b JOB_TOTAL_GPUS=16 ./launch.sh
```

will produce results under `$STAGE_PATH/experiments/pretrain_nemotron4_15b_bf16_16_nsys/pretrain_nemotron4_15b_bf16_16_nsys_<timestamp>/pretrain_nemotron4_15b_bf16_16_nsys` folder. 

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

# Run With Checkpoints

## Save Checkpoint

Save checkpoint feature works for both Nemotron4 15b and 340b sizes with either FP8 or BF16 precision. Make sure your file system has sufficient disk space to accomodate checkpoint sizes below:

| Model | Checkpoint Size | Supported Scales |
| :---: | :---: |  :---: |
| 15b | ~204 GB | 16-2048 |
| 340b | ~4.4 TB | 256-2048 |

### How to enable
To save the checkpoints after pretraining nemotron4 model for `max_steps`, you need to set environment variable `ENABLE_CHECKPOINT=true`. At the end of the pretraining the checkpoints will be saved in the  `$STAGE_PATH/experiments` folder.

```shell
experiment_name = pretrain_nemotron4_${MODEL_SIZE}_${DTYPE}_${JOB_TOTAL_GPUS}
timestamp = date '+%s'
Example directory where checkpoints are saved is $STAGE_PATH/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/code/nemo_experiments/default/checkpoints/
```
Command to run nemotron4 with checkpoint save enabled
```shell
ENABLE_CHECKPOINT=true DTYPE=<fp8,bf16> MODEL_SIZE=<15b/340b> JOB_TOTAL_GPUS=<16,..,2048> GPUS_PER_NODE=<2,8> ./launch.sh
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
| 15b | 64 |
| 340b | 512 |

**Note**:
- Running load checkpointing feature at other scales may run into CUDA OOM errors. 

### How to enable
To resume training from saved checkpoints, you need to set `LOAD_CHECKPOINT_PATH=<path_to_checkpoint_directory>` environment variable. Make sure the checkpoint files are under the `$STAGE_PATH` directory and `LOAD_CHECKPOINT_PATH` variable is set to parent folder of the `weights` directory containing distributed checkpoint files with extension `*.distcp`.

E.g., if the checkpoint was saved under `$STAGE_PATH/experiments/pretrain_nemotron4_15b_fp8_64/pretrain_nemotron4_15b_fp8_64_<timestamp>/pretrain_nemotron4_15b_fp8_64/code/nemo_experiments/default/checkpoints/*/weights` then set the environment variable to a directory one level higher: 

`LOAD_CHECKPOINT_PATH=$STAGE_PATH/experiments/pretrain_nemotron4_15b_fp8_64/pretrain_nemotron4_15b_fp8_64_<timestamp>/pretrain_nemotron4_15b_fp8_64/code/nemo_experiments/default/checkpoints/default--None=0.0000-epoch=0-consumed_samples=12800.0`

The scripts will restore configuration from the checkpoint and resume training process. Training will run for 1 step after checkpoint has been loaded.

```shell
LOAD_CHECKPOINT_PATH=<your_path_to_checkpoint_directory> DTYPE=<fp8,bf16> MODEL_SIZE=<15b/340b> JOB_TOTAL_GPUS=<16,..,2048> GPUS_PER_NODE=<2,8> ./launch.sh
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
    attention flops = (24 * (number of layers) * (hidden size)^2) + (12 * (number of layers) * (hidden size) * (sequence length))
    mlp flops = 48 * (number of layers) * (hidden size)^2
    embedding flops = 6 * (vocab size) * (hidden size)

Nemotron4 15b calculation:
    sequence length = 4096
    number of layers = 32
    hidden size = 6144
    vocab size = 256000 
    attention flops = 24 * 32 * 6144^2 + 12 * 32 * 6144 * 4096 = 38666279738
    mlp flops = 48 * 32 * 6144^2 = 57982058496
    embedding flops = 6 * 256000 * 6144 = 9437184000

    model flops = 4096 * (38666279738 + 57982058496 + 9437184000) = 434,526,299,070,464 = 434.5e12
```
