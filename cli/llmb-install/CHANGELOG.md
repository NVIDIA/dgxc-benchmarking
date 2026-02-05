# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0.post1] - 2026-02-02
Backport release for 25.12 support line (dependency constraint fix only).

### Fixed
- transformers dependency '>4.57.3,<5.0.0'

## [1.6.0] - 2026-01-16

### Added
- HuggingFace downloads v2 via `downloads.huggingface` (tokenizer + `config.json`, supports remote-code tokenizers).

### Changed
- Enforce `transformers>4.57.3`.

## [1.5.2] - 2026-01-20

### Fixed
- Headless playfiles now use top-level `slurm` and `environment_vars` (and fail fast on deprecated keys).

## [1.5.1] - 2026-01-09

### Added
- B300 support
- GPU ordering in GPU selection prompt

## [1.5.0] - 2025-12-24

### Changed
- Exemplar Cloud selection now uses `exemplar.yaml` per GPU type instead of selecting all pretrain workloads

## [1.4.2] - 2025-12-18

### Fixed
  - Missing `urllib.error` import for download error handling in tools module
  - Improved validation error messages for null/invalid `downloads` metadata

## [1.4.1] - 2025-12-17

### Fixed
  - HuggingFace tokenizer downloads now uses hf login, instead of relying on token passed to individual functions.

## [1.4.0] - 2025-12-15

### Added
- HuggingFace tokenizer download support via `hf_tokenizers` in metadata.yaml

### Changed
- Refactored download code into unified `downloads/` module

## [1.3.13] - 2025-12-10
### Added
- "Exemplar Cloud" option to workload selection, which automatically selects all standard pre-training reference workloads.
- Added `--exemplar` flag to `llmb-install express` for automated installation of Exemplar Cloud workloads.

## [1.3.12] - 2025-12-10

### Added
- `setup` configuration in `metadata.yaml` now supports `by_gpu` conditional logic.

## [1.3.11] - 2025-12-05

### Changed
- Migrated dependency management to uv and added lockfiles.

## [1.3.10] - 2025-12-03

### Changed
- Enforce managed python version for UV environments by default
- Added `LLMB_DISABLE_MANAGED_PYTHON` env var to disable managed python enforcement

## [1.3.9] - 2025-12-03

### Fixed
- Minor fixes for import paths, documentation, and archive extraction robustness.

## [1.3.8] - 2025-11-07

### Added
- Copy `release.yaml` to `$LLMB_INSTALL/llmb_repo` if it exists.

## [1.3.7] - 2025-10-31

### Added
- `LLMB_USE_PIP_FALLBACK` environment variable to use standard pip instead of uv pip in uv environments

## [1.3.6] - 2025-10-29

### Fixed
- Git cache now uses `--mirror` instead of `--bare` to fetch all refs on updates
- Added commit validation after cache/target fetch operations with clear error messages

## [1.3.5] - 2025-10-29

### Fixed
- UV environment detection for conda-based virtual environments by setting `CONDA_DEFAULT_ENV`

## [1.3.4] - 2025-10-21

### Changed
- `GPU_TYPE` no longer saved to `cluster_config.yaml` environment section (still passed to setup scripts)

### Deprecated
- Legacy `setup_script` functionality (both scripted workloads and post-install scripts). Users should migrate to `tasks` feature in metadata.yaml

### Fixed
- Post-install scripts now receive `LLMB_WORKLOAD` environment variable

## [1.3.3] - 2025-10-21

### Fixed
- Repository path corruption when editing configuration in interactive mode

## [1.3.2] - 2025-10-17

### Added
- `cuda_cupti_lib` tool support for downloading CUDA CUPTI libraries as nsys profiling workaround

## [1.3.1] - 2025-10-13

### Added
- Git LFS installation verification and automatic configuration

### Fixed  
- Cache clone configuration when Git LFS is required

## [1.3.0] - 2025-10-13

### Added
- **Tools Download**: Support for downloading and installing workload-specific / GPU-conditional Nsight Systems versions
  - Tools specified in `metadata.yaml` with simple or `by_gpu` conditional formats
  - Installer caching in `$LLMB_INSTALL/.cache/tools/` for bandwidth efficiency
  - Robust download/install logic with proper cleanup on failures

## [1.2.0] - 2025-10-08

### Added
- **Incremental Install**: Add workloads to existing installations without reinstalling everything
  - Run `llmb-install` from installation directory or provide existing path when prompted
  - Automatically reuses virtual environments when dependencies match
  - Inherits existing configuration (GPU type, SLURM settings, environment variables)

### Changed
- `cluster_config.yaml` includes new `install` section with installation metadata
- Resume workflow now preserves original workloads during incremental install failures

## [1.1.3] - 2025-10-03

### Fixed
- Correctly set repo root when opting for a fresh install during a resume operation.

## [1.1.2] - 2025-10-02

### Changed
- Updated venv naming for non-legacy workloads to 'venv_{shortsha}'

## [1.1.1] - 2025-10-02

### Fixed
- SLURM partition validation now disallows blank GPU and CPU partitions in all modes
- Install paths now converted to absolute paths using pathlib in headless and express modes

### Changed
- `llmb-run` symlink now points to venv binary instead of repository script

## [1.1.0] - 2025-09-30

### Added
- **Resume Failed Installations**: Automatically save progress and resume from interruptions
- **Development Mode** (`--dev-mode`): Skip repository copying for recipe development
- **Repository Isolation**: Copy repository to installation directory for standalone operation

### Changed
- Installation state for incomplete installs tracked in `~/.config/llmb/install_state.yaml`
- Repository copied to `$LLMB_INSTALL/llmb_repo` (unless dev mode enabled)

## [1.0.4] - 2025-09-24

### Fixed
- Use the '--clear' flag when creating uv venvs. Brings in line with other methods and fixes a failure case.
- Handle an edge case for git workload clones with missing remotes.

## [1.0.3] - 2025-09-24

### Added
- GB300 support

## [1.0.2] - 2025-09-23

### Added
- Caching support for git clones, with local clones from cache. This transfers large pack files and avoids many small fs ops.

### Fixed
- Git repo validation bug, we now check if an existing repo is valid before doing any git operations.

## [1.0.1] - 2025-09-18

### Changed
- Only compile bytecode for the last package when using uv, as it compiles the full venv.

## [1.0.0] - 2025-09-17

Refactor of code base into modular design.

### Added
- **Express Mode**: `llmb-install express` for faster repeat installations using saved settings
- **Configuration Persistence**: System settings automatically saved for future use

### Changed
- **Command Interface**: Replaced `./installer.py` with `llmb-install` command
- **Configuration Management**: XDG-compliant config location (`~/.config/llmb/`)


## [0.11.2] - 2025-08-28

### Changed
- Moved installer to 'cli/llmb-install'
- Updated repo root detection logic

## [0.11.1] - 2025-08-27

### Fixed
- Downgraded 'prompt_toolkit<3.0.52' to work around questionary incompatibility

## [0.11.0] - 2025-08-26

### Added
- UV environment support with automatic preference order: uv → venv → conda
- Unified cache management for both PIP and UV cache directories

### Changed
- Cache directories automatically configured under `$LLMB_INSTALL/.cache/`, if not set in user environment.
- venv creation now ensures it's not using a virtual environment. Usually system Python.

### Fixed
- System Python detection bug when running from virtual environments

## [0.10.2] - 2025-08-18

### Fixed
- Force conda auto_activate to false. Handles an edge case on systems where login and compute nodes have different architectures.

## [0.10.1] - 2025-08-14

### Fixed
- Force conda environment usage when installer is run from within a conda environment to prevent venv dependency issues on parent environment.

## [0.10.0] - 2025-08-11

### Added
- GPU-specific overrides in metadata:
  - `container.images` supports `by_gpu` with `h100`, `b200`, `gb200`, and `default`.
  - `repositories` supports top-level `by_gpu` to select per-GPU repo commits.
- New installer resolver `resolve_gpu_overrides` to materialize GPU-specific images/repos before install.
- Schema extended to allow future `by_model` conditionals for images and repositories (not yet wired in installer).

### Changed
- Venv grouping now hashes a canonical dependencies signature that ignores non-functional differences (e.g., script vs pip-git acquisition, wrappers), so identical effective commits share a venv.
- Editable pip dependencies force an isolated venv per recipe to avoid cross-recipe interference.


