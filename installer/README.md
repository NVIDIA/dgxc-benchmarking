# DGXC Benchmark Recipes Installer

The installer is an interactive tool that simplifies the setup and deployment of DGXC Benchmarking recipes. It automatically discovers available workloads, configures your environment, downloads required container images, and prepares workloads for execution.

## Quick Start

It is recommended to run installer in a virtual environment (conda or venv with python 3.12.x). Make sure to have the environment activated before running commands below.

```bash
# Install installer dependencies
cd installer
python3 -m pip install .

# Run the installer
./installer.py
```

The installer will guide you through an interactive setup process covering:
- Installation location selection
- SLURM cluster configuration
- Node architecture (x86_64/aarch64)
- Environment type (venv/conda)
- Installation method (local/SLURM)
- Workload selection

### Installing Additional Workloads

You can install workloads in multiple sessions, but there are important considerations:

1. **First-time Installation**: Run the installer and select all workloads you plan to use. This ensures a complete configuration file (cluster_config.yaml) for the launcher.

2. **Adding Workloads Later**: 
   - Run the installer again and select ALL workloads (both existing and new)
   - This is required because the launcher's configuration file must be regenerated with the complete workload list
   - Partial installations may result in missing workload configurations in the launcher

**Note**: The installer does not detect previously installed workloads. Always select all desired workloads during installation to maintain a complete configuration.

## Prerequisites

### System Requirements
- **Python**: 3.12.x (for venv support) OR conda/miniconda
- **SLURM**: 22.x or newer with job scheduler access
- **Enroot**: For container image management
- **Network Access**: Required for downloading container images
- **Disk Space**: Substantial space required (see [Storage Requirements](#storage-requirements))

### Python Dependencies
The installer dependencies are defined in `pyproject.toml`:
```bash
cd installer
pip install .
```

This installs:
- PyYAML>=6.0
- questionary>=1.10.0

## Storage Requirements

The installer downloads and stores significant amounts of data:

| Component | Size Range | Notes |
|-----------|------------|-------|
| Container Images | 5-60 GB each | Architecture-specific |
| Virtual Environments | 1-10 GB each | Per workload |
| Workload Datasets | 200 GB - 1 TB | Model-dependent |

**Recommendation**: Install on high-performance shared storage (Lustre, GPFS) with sufficient space and fast I/O.

## Directory Structure

After installation, the following structure is created:

```
$LLMB_INSTALL/
├── images/           # Container images (.sqsh files)
├── datasets/         # Dataset files
├── venvs/           # Virtual environments
└── workloads/       # Installed workloads
    └── workload_name/
        ├── setup files
        └── experiments/  # Results and logs
```

## Configuration Options

### Installation Location
- Must have sufficient disk space (hundreds of GB to TB)
- Should be on shared storage accessible to all compute nodes
- Requires write permissions

### SLURM Configuration
The installer automatically detects and validates:
- **Account**: Your SLURM accounts (via `sacctmgr`)
- **Partitions**: Available partitions (via `sinfo`)
- **GPU Resources**: GPU counts per partition (via GRES)

### Node Architecture
- **x86_64**: Standard Intel/AMD processors
- **aarch64**: ARM-based systems (Grace Blackwell, etc.)

**Important**: Choosing the wrong architecture will cause "Exec format error" when running containers.

### Installation Method
Which method to use to download the large container images and datasets. Workload specific setup and venv installation will always be on the current node.
- **Local**: Downloads run on current machine (requires enroot access)
- **SLURM**: Downloads submitted as jobs (recommended for clusters)
    - **Note:** Currently this is sequential srun jobs.
    - **Important:** SLURM installation method is not available when running the installer within a SLURM job

**SLURM Job Detection**: The installer automatically detects if it's running within a SLURM job (via `SLURM_JOB_ID` environment variable). When detected:
- SLURM installation method is disabled (cannot submit jobs from within a job)
- Automatically defaults to local installation method if enroot is available
- Exits with error if enroot is not available

## Common Issues and Solutions

### Running Installer Within SLURM Job
**Issue**: Installer fails when run within a SLURM job without enroot
```
Error: Cannot proceed with installation.
You are running within a SLURM job, but enroot is not available on this system.
```
**Solution**: 
- **Option 1**: Run the installer from a login node (outside SLURM job)
- **Option 2**: Ensure enroot is available on compute nodes and use local installation method

### Installation Process is Slow/Resource Intensive

**Issue**: The installer seems very slow or stalled.

**Explanation**: The installation process, especially downloading large container images and installing all necessary pip packages, can be resource-intensive. Login nodes are often shared and with limited resources per user, which can lead to slow performance.

**Solution**:
- **Option 1**: Try running the installer again, perhaps during off-peak hours.
- **Option 2**: Obtain an interactive shell on a dedicated CPU node and run the installer there. This offloads the resource usage from the login node.

### Enroot Not Available for Local Installation
**Issue**: Installer automatically selects SLURM method when enroot is missing
```
Note: enroot is not available on this system.
Local installation requires enroot for container image downloading.
Automatically selecting SLURM-based installation.
```
**Solution**: 
- **Option 1**: Install enroot on the current system to enable local installation
- **Option 2**: Continue with SLURM installation method (recommended for clusters)
- **Option 3**: Manually download container images using enroot on a different system


### Python Version Compatibility
**Issue**: `conda` not available and Python version < 3.12.x
```
ERROR: Your current Python version (3.10.x) is below the minimum required (3.12.x)
```
**Solution**: Install conda:
```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_INSTALL_PATH/miniconda3
$CONDA_INSTALL_PATH/miniconda3/bin/conda init
source ~/.bashrc
```
### Pip Cache Warning
**Issue**: Pip cache under `/home` may cause space issues
```
WARNING: Your pip cache directory is located under /home
```
**Solution**: Set alternative cache location:
```bash
export PIP_CACHE_DIR=/path/to/larger/storage/.pip_cache
```

### SLURM Account/Partition Issues
**Issue**: Account or partition not recognized
**Solution**: 
- Verify with `squeue -u $USER` or `sacctmgr show associations user=$USER`
- Contact system administrator for correct account/partition names

### Network Access Issues
**Issue**: Container downloads fail
**Solution**:
- Ensure login nodes have internet access and enroot OR
- Use SLURM installation method to run downloads on nodes with access

### Insufficient Disk Space
**Issue**: Downloads fail due to space constraints
**Solution**:
- Choose installation location with adequate space
- Clean up unnecessary files in target directory
- Consider using storage with higher quotas

## Advanced Usage

### Selective Installation
To install specific workloads only:
1. Run installer and select only the workloads you need
2. This reduces download time and storage requirements

### Debugging Failed Installations
1. Check installer output for specific error messages
2. Check container image downloads in `$LLMB_INSTALL/images/`
3. Verify virtual environment creation in `$LLMB_INSTALL/venvs/`

### Manual Container Management
If automatic downloads fail, you can manually import containers:
```bash
# Example for manual container import
enroot import -o $LLMB_INSTALL/images/nvidian+nemo+25.02.01.sqsh \
    docker://nvcr.io/nvidian/nemo:25.02.01
```
**Note:** If your compute node architecture differs from the node being used to download the image use the `-a <arch>` flag.

## Validation

After installation, verify setup:

1. **Check directory structure**:
   ```bash
   ls -la $LLMB_INSTALL/
   # Should show: images/ datasets/ venvs/ workloads/
   ```

2. **Verify container images**:
   ```bash
   ls -la $LLMB_INSTALL/images/*.sqsh
   # Should show downloaded container files
   ```

3. **Test virtual environments**:
   ```bash
   source $LLMB_INSTALL/venvs/<workload>_venv/bin/activate
   python --version  # Should show a 3.12.x version
   ```

## Support

For installation issues:
1. Check this README and [main FAQ](../README.md#faq)
2. Verify system prerequisites are met
3. Contact LLMBenchmarks@nvidia.com with:
   - Installer output/error messages
   - System configuration details
   - SLURM cluster information 

