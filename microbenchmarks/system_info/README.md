# Overview

This recipe runs a lightweight host-level system information collection through `llmb-run`.

It is intentionally minimal and uses the configured sbatch launcher with placeholder
model metadata so it fits the current `llmb-run` recipe schema.

# Commands Collected

01. `lscpu`
    - `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
02. `lspci -v`
03. `numactl -H`
04. `cat /proc/cmdline`
05. `systemd-detect-virt`
06. `getconf PAGE_SIZE`
07. `dmesg | grep -i smmu`
08. `nvidia-smi -q`
    - `nvidia-smi topo -m`
09. `sysctl -n kernel.numa_balancing`
10. `ibv_devinfo` - InfiniBand HCA device names and attributes
11. Slurm topology and MPI
    - `scontrol show config` - prints `TopologyPlugin`/`TopologyParam` for context
    - `scontrol show topology` - flags failure if topology output is empty
    - `srun --mpi=list` - flags failure if the `pmix` plugin is not listed
      (LLMB recipes invoke Slurm with `--mpi=pmix`; Slurm must be built with `--with-pmix`)
12. enroot config
    - checks `enroot.conf` for recommended settings (`ENROOT_ROOTFS_WRITABLE`, `ENROOT_REMAP_ROOT`)
    - dumps `environ.d/` contents and flags bare defaults or missing `NCCL_IB_HCA`
13. `srun` inside a container - validates pyxis/enroot, GPU visibility, and hook configuration
    - `nvidia-smi` - GPU visibility inside the container
    - `env | grep MASTER_ADDR` - confirms the Enroot
      [`50-slurm-pytorch.sh`](https://github.com/NVIDIA/enroot/tree/main/conf/hooks/extra) hook
      is active inside the container (the hook populates `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE`
      from Slurm env vars and is required for PyTorch distributed bootstrap)

All commands are non-fatal; failures are reported in output and execution continues.

# Run

```bash
cd $LLMB_INSTALL
llmb-run submit -w microbenchmark_system_info --scale <num_gpus_per_node>
```

# Output

The script writes output to standard SLURM output (`slurm-*.out`) under the experiment
directory created by `configured_sbatch`.
