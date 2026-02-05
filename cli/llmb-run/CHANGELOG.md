# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] - 2026-01-28

### Added
- `configured_sbatch` launcher type: creates experiment directories before job submission, enabling llmb-config.yaml and post-processing for sbatch workloads.
- B300 support

## [1.6.0] - 2026-01-20

### Added
- `llmb-run submit --exact-scales` flag: limit job generation to scales explicitly listed in workload metadata (no power-of-2 scale expansion beyond metadata).

## [1.5.5] - 2026-01-15

### Fixed
- Allow `inference` and `microbenchmark` workloads to be run when explicitly selected via `llmb-run submit -w <workload>`, while keeping them excluded from discovery/bulk modes.

## [1.5.4] - 2026-01-07

### Fixed
- Hardened post processing job submission edge cases.

## [1.5.3] - 2026-01-05

### Fixed
- Improved YAML header parsing with better validation and error messages

## [1.5.2] - 2025-12-31

### Fixed
- Zero-jobs error now shows available configurations instead of generic message

## [1.5.1] - 2025-12-24

### Fixed
- Deprecated `single` command: AttributeError when accessing extra_slurm_params

## [1.5.0] - 2025-12-23

### Added
- `exemplar` command for cloud certification

## [1.4.0] - 2025-12-22

### Added
- `llmb-run submit`: unified job submission command (single, discovery, and file-based).

### Deprecated
- `llmb-run single`, `llmb-run submit-all`, `llmb-run bulk` (use `llmb-run submit`).

### Changed
- **[BREAKING INTERNAL]** Migrated CLI framework from argparse to Typer (intended to be user-facing compatible).
- Reduced default verbosity of `llmb-run submit` output (use `-v` for discovery/details).

## [1.3.10] - 2025-12-05

### Changed
- Migrated dependency management to uv and added lockfiles.

## [1.3.9] - 2025-11-19

### Changed
- Refactored post-processing steps.

## [1.3.8] - 2025-11-12

### Added
- Added `--nice` flag to `llmb-run submit-all` to lower job priority using Slurm's `--nice` feature.

## [1.3.7] - 2025-11-10

### Added
- `llmb-run submit-all` now submits jobs sorted by scale in descending order (largest jobs first).

## [1.3.6] - 2025-11-07

### Added
- `LLMB_SKIP_PP` environment variable to disable post-processing.

## [1.3.5] - 2025-11-05

### Changed
- Use `release.yaml` for version information, fall back to `metadata.yaml` if it does not exist.

## [1.3.4] - 2025-10-24

### Added
- job_config.strong_scaling to llmb-config.yaml

## [1.3.3] - 2025-10-22

### Fixed
- SbatchLauncher places slurm.out files in $LLMB_WORKLOAD instead of cwd

## [1.3.2] - 2025-10-17

### Added
- CUPTI library mount support for profiling workarounds via `cuda_cupti_lib` tool

## [1.3.1] - 2025-10-15

### Added
- 'slurm_cpu_partition' to llmb-config.yaml.

### Fixed
- llmb-config.yaml generation now handles `by_gpu` container image structure
- Improved error handling with targeted blocks for easier debugging

## [1.3.0] - 2025-10-14

### Added
- Custom Nsys version container mounts on per workload basis.

## [1.2.7] - 2025-10-02

### Fixed
- Only generate experiment_ids for nemo2 workloads.

## [1.2.6] - 2025-10-02

### Added
- Support for `megatron_bridge` launcher type (uses Nemo2Launcher)

## [1.2.5] - 2025-09-24

### Added
- GB300 support

## [1.2.4] - 2025-09-19

### Added
- `--scales` flag for submit-all command to specify exact scales to run

## [1.2.3] - 2025-09-03

### Added
- `experiment_id` field in llmb-config.yaml files based on normalized config hash and framework version

## [1.2.2] - 2025-08-30

### Added
- A check for 'uv' venv_types and properly sets VIRTUAL_ENV parameter.

## [1.2.1] - 2025-08-22

### Added
- Set the default value of `GPU_METRICS_NODES` to 0 for GPU metrics collection

## [1.2.0] - 2025-08-14

### Added
- `llmb-run submit-all`
  - `--min-scale` flag, flexible scale requirements (either `--max-scale` or `--min-scale` required)
  - `--workloads` filters on comma separated list of workload name or workloadname_size (e.g., `pretrain_nemotron4_340b` or `pretrain_nemotron4`)
  - `--dtype` filters on a comma separated list of dtypes.
- `llmb-run single`: `--force` flag to skip validation

## [1.1.0] - 2025-08-13

### Added
- Per-dtype scale configuration in recipe metadata (mapping form under `dtypes`), with optional per-dtype `exact_scales`.
- Shared normalization across the toolchain to interpret legacy and new forms consistently.

### Changed
- `llmb-run list` now shows detailed per-dtype scales by default (no `-v` required).
- `submit-all` and validation respect per-dtype scales and exactness; legacy forms remain supported.

## [1.0.1] - 2025-08-07

### Changed
- Updated Nemo2 work dir path provided to WiT module, per api change.

## [1.0.0] - 2025-08-04

### Added
- New `submit-all` command to automatically generate and submit jobs for all installed pretrain/finetune workloads:
  - Supports dynamic scale generation up to a specified `--max-scale` (GPUs).
  - Allows setting `--repeats` for multiple runs of each configuration.
  - Includes `--profile` flag to enable profiling for all generated jobs.
  - Integrates with `--dryrun` for previewing jobs.
  - Filters workloads by type (pretrain/finetune only) and cluster GPU compatibility.

## [0.10.0] - 2025-07-24

### Added
- Dynamic timelimit for workload inspector jobs based on scale:
  - Short (≤128 GPUs): 1 hour
  - Medium (129-511 GPUs): 4 hours  
  - Long (≥512 GPUs): 8 hours
  - Special case: DeepSeek V3 pretrain ≥256 GPUs uses 8 hours
- Blocks medium/long WiT jobs on GPU partitions, to avoid wasting resources.
  - Heuristic: gpu_part == cpu_part, cpu_part requires GRES, or cpu_part == backfill

### Changed
- Workload inspector now uses calculated timelimit instead of default 1 hour

## [0.9.1] - Previous Release
- Previous functionality (details can be added when reviewing historical changes)

