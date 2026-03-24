# Overview

This recipe runs a lightweight host-level system information collection through `llmb-run`.

It is intentionally minimal and uses the configured sbatch launcher with placeholder
model metadata so it fits the current `llmb-run` recipe schema.

# Commands Collected

01. `lscpu`
02. `lspci -v`
03. `numactl -H`
04. `cat /proc/cmdline`
05. `systemd-detect-virt`
06. `getconf PAGE_SIZE`
07. `dmesg | grep -i smmu`
08. `nvidia-smi -q`
09. `sysctl -n kernel.numa_balancing`
10. `ibv_devinfo` - InfiniBand HCA device names and attributes
11. enroot config
    - checks `enroot.conf` for recommended settings (`ENROOT_ROOTFS_WRITABLE`, `ENROOT_REMAP_ROOT`)
    - dumps `environ.d/` contents and flags bare defaults or missing `NCCL_IB_HCA`
12. `srun nvidia-smi` inside a container - validates pyxis/enroot and GPU visibility

All commands are non-fatal; failures are reported in output and execution continues.

# Run

```bash
cd $LLMB_INSTALL
llmb-run submit -w microbenchmark_system_info --scale <num_gpus_per_node>
```

# Output

The script writes output to standard SLURM output (`slurm-*.out`) under the experiment
directory created by `configured_sbatch`.
