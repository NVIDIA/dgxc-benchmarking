# Overview

This recipe runs a lightweight host-level system information collection through `llmb-run`.

It is intentionally minimal and uses the configured sbatch launcher with placeholder
model metadata so it fits the current `llmb-run` recipe schema.

# Commands Collected

01. `lscpu`
    - `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
    - CPU C-state availability from `/sys/devices/system/cpu/cpu0/cpuidle` - warns if
      two or fewer idle states are exposed or any are disabled (a boost-clock factor)
02. `lspci -v`
03. `numactl -H`
04. `cat /proc/cmdline`
05. `systemd-detect-virt`
06. `getconf PAGE_SIZE`
07. `dmesg | grep -i smmu`
08. `nvidia-smi -q`
    - `nvidia-smi topo -m`
    - `nvidia-smi --query-gpu=...ecc.errors.uncorrected...,remapped_rows...` - GPU memory
      health; warns on uncorrected ECC errors or row-remap failures/pending
    - `nvidia-smi nvlink -s` + `--query-gpu=fabric.state,fabric.status` - warns on inactive
      NVLinks; on `gb200`/`gb300` (NVL72) additionally warns if the node does not report 72
      up links (gated on `GPU_TYPE`; no count is enforced on other platforms)
    - `nvidia-smi nvlink -e` - NVLink bit-error-rate; warns on non-baseline BER values
    - `dcgmi health` - enables DCGM health watches (PCIe, memory, InfoROM, thermal/power,
      NVLink) on a temporary all-GPU group and checks them; warns on `Warning`, fails on
      `Failure`. Requires a running `nv-hostengine`; skipped with a warning if DCGM is absent
09. `sysctl -n kernel.numa_balancing`
10. `ibv_devinfo` - InfiniBand HCA device names and attributes
    - `/sys/class/infiniband/*/ports/*/state` - warns if a port is not ACTIVE (some down
      ports can be expected, e.g. HCAs wired to NVSwitch on B200/B300)
    - IBGDA readiness - checks whether NVSHMEM's GPU-initiated transport has a viable path:
      `PeerMappingOverride=1` on the nvidia module (`/proc/driver/nvidia/params`,
      `/sys/module/nvidia/parameters/NVreg_RegistryDwords`) enables the default IBGDA
      path; otherwise the GDRCopy kernel driver (`gdrdrv`, via `/dev/gdrdrv` or `lsmod`)
      enables the CPU-assisted path. Either is sufficient; warns only if neither is present
11. Slurm topology and MPI
    - `scontrol show config` - prints `TopologyPlugin`/`TopologyParam` for context
    - `scontrol show topology` - flags failure if topology output is empty
    - `srun --mpi=list` - flags failure if the `pmix` plugin is not listed
      (LLMB recipes invoke Slurm with `--mpi=pmix`; Slurm must be built with `--with-pmix`)
12. enroot config
    - checks `enroot.conf` for recommended settings (`ENROOT_ROOTFS_WRITABLE`, `ENROOT_REMAP_ROOT`)
    - dumps `environ.d/` contents and flags bare defaults; passes if `MELLANOX_VISIBLE_DEVICES`
      is configured, or if both `NCCL_IB_HCA` and `NVSHMEM_HCA_LIST` are configured. `MELLANOX_VISIBLE_DEVICES`
      is preferred because it restricts which HCAs are exposed to the container.
13. `srun` inside a container - validates pyxis/enroot, GPU visibility, and hook configuration
    - `nvidia-smi` - GPU visibility inside the container
    - `env | grep MASTER_ADDR` - confirms the Enroot
      [`50-slurm-pytorch.sh`](https://github.com/NVIDIA/enroot/tree/main/conf/hooks/extra) hook
      is active inside the container (the hook populates `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE`
      from Slurm env vars and is required for PyTorch distributed bootstrap)
    - `ls /dev/nvidia-caps-imex-channels/` inside the container - on GB-series
      (`GPU_TYPE` = gb200/gb300), warns if no IMEX channel devices are visible to the
      container (cross-node NVLink IPC needs them); checks what the job actually sees,
      independent of host IMEX setup mechanism. Skipped on non-GB platforms.

All commands are non-fatal; failures are reported in output and execution continues.

# Run

```bash
cd $LLMB_INSTALL
llmb-run submit -w microbenchmark_system_info --scale <num_gpus_per_node>
```

# Output

The script writes output to standard SLURM output (`slurm-*.out`) under the experiment
directory created by `configured_sbatch`.
