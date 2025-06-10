# Bulk Job Submission Examples

## Getting Started with Bulk Jobs

Bulk job submission allows you to run multiple workloads with different configurations in a single command. There are two ways to specify bulk jobs:

1. **Simple Format** (.txt files) - Best for basic configurations
2. **Advanced Format** (.yaml files) - Best for complex configurations with multiple parameters

### Choosing Between Simple and Advanced Format

- Use **Simple Format** if you need to:
  - Run the same workload with different scales
  - Test different data types (fp8, bf16)
  - Run a few basic configurations

- Use **Advanced Format** if you need to:
  - Set environment variables
  - Run complex parameter sweeps
  - Mix different configurations in one file

## Simple Examples

### Basic Configuration
```yaml
pretraining_nemotronh_56b:
  tasks:
    - dtypes: 'fp8'
      scales: [128, 256, 512]
      repeats: 3
```
**Explanation**: This will run Nemo Megatron 175b with fp8 precision at three different scales (128, 256, and 512 GPUs), repeating each configuration 3 times. Total of 9 jobs will be submitted.

### Multiple Data Types
```yaml
pretraining_nemotronh_56b:
  tasks:
    - dtypes: ['fp8', 'bf16']
      scales: [128, 256]
      repeats: 2
```
**Explanation**: This configuration will run the workload with both fp8 and bf16 precision at two different scales. Each combination (fp8@128, fp8@256, bf16@128, bf16@256) will be run twice. Total of 8 jobs will be submitted.

## Intermediate Examples

### With Environment Variables
```yaml
pretraining_grok1_314b:
  defaults:
    env:
      DEBUG: true
  tasks:
    - dtypes: 'fp8'
      scales: [128, 256]
      repeats: 3
```
**Explanation**: This example sets global environment variables for all jobs. The workload will run with fp8 precision at two scales, with each configuration repeated 3 times. The environment variables will be applied to all 6 jobs.

## Complex Examples

### Multiple Tasks with Overrides
```yaml
pretraining_nemotron_340b:
  defaults:
    env:
      LOG_LEVEL: "INFO"
  dtypes: ['fp8', 'bf16']
  scales: [128, 256, 512]
  repeats: 2
  tasks:
    - scales: [128, 256]
      overrides:
        env:
          LOG_LEVEL: "DEBUG"
    - dtypes: ['fp8']
      scales: [512]
      repeats: 1
      profile: true
      overrides:
        env:
          LOG_LEVEL: "TRACE"
```
**Explanation**: This complex example demonstrates multiple features:
1. First task: Runs bf16 and fp8 at scales 128 and 256, with a `DEBUG` log level. Each configuration runs twice.
2. Second task: Overrides the dtype to only use fp8, and scale to 512. It is a single profiling run with `TRACE` log level.
3. All jobs will have the default `LOG_LEVEL` of `INFO` unless overridden.

### Multiple Workloads with Different Configurations
```yaml
pretraining_llama3.1_405b:
  defaults:
    env:
      NCCL_IB_QPS_PER_CONNECTION: 1
  tasks:
    - dtypes: ['fp8', 'bf16']
      scales: [256, 512]
      repeats: 3

pretraining_grok1_314b:
  tasks:
    - dtypes: 'fp8'
      scales: [128, 256]
      repeats: 2
    - dtypes: 'fp8'
      scales: [128, 256]
      repeats: 1
      profile: true
```
**Explanation**: This example shows how to configure multiple different workloads:
1. `pretraining_llama3.1_405b`: Runs with both precisions at two scales, with `ENABLE_TF32` set. Each configuration is repeated 3 times.
2. `pretraining_grok1_314b`: Runs with fp8 at two scales, repeated twice. It also includes separate profiling runs for each scale.
3. Each workload has its own configuration.

## Usage

To run any of these examples:

```bash
# Using the installed command
llmb-run bulk my_config.yaml

# Using python directly
python3 llmb_run.py bulk my_config.yaml

# Dry run to preview jobs (recommended first step)
llmb-run bulk my_config.yaml --dryrun
```

## Notes
- All examples assume a valid `cluster_config.yaml` file is present.
- The `repeats` parameter defaults to 1 if not specified.
- To run a profiling job, create a task with `profile: true`.
- Use the `--dryrun` flag to preview all jobs before submission.