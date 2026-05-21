#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#SBATCH --job-name="system_info:microbenchmark"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00

set -u -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=microbenchmark
export WORKLOAD=system_info
export FW_VERSION=26.04.00

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

FAILED_STEPS=0

print_banner() {
    local title="$1"
    echo
    echo "================================================================================"
    echo "${title}"
    echo "================================================================================"
}

run_step() {
    local step_id="$1"
    local title="$2"
    local cmd="$3"
    local rc=0

    print_banner "[${step_id}] ${title}"

    bash -o pipefail -c "${cmd}"
    rc=$?

    if [[ ${rc} -ne 0 ]]; then
        echo
        echo "[FAILED] exit code ${rc} (non-fatal, continuing)"
        FAILED_STEPS=$((FAILED_STEPS + 1))
    fi
}

print_banner "System Info Collection Start"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "LLMB_EXPERIMENT_DIR: ${LLMB_EXPERIMENT_DIR:-unset}"
echo "Timestamp: $(date -Iseconds)"

run_step "1a" "lscpu - CPU details, SKU, core count" "lscpu"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_cpu_governor() {
    if ! compgen -G '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' > /dev/null; then
        echo "cpufreq not available on this system (no scaling_governor files)"
        return
    fi

    local govs
    govs=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u)
    echo "${govs}"
    echo

    if [ "${govs}" = "performance" ]; then
        echo "[OK] All cores using performance governor."
    else
        echo "[WARNING] One or more cores not using performance governor."
        echo "          'performance' is recommended for benchmarking to avoid frequency scaling noise."
    fi
}

run_step "1b" "cpufreq scaling_governor - CPU frequency governor per core" \
    "$(declare -f check_cpu_governor); check_cpu_governor"
run_step "2" "lspci -v - networking hardware and software layers" "lspci -v"
run_step "3" "numactl -H - NUMA node assignments" "numactl -H"
run_step "4" "cat /proc/cmdline - Linux kernel arguments" "cat /proc/cmdline"
run_step "5" "systemd-detect-virt - virtualization check" \
    "systemd-detect-virt || true"
run_step "6" "getconf PAGE_SIZE - memory page size" "getconf PAGE_SIZE"
run_step "7" "dmesg | grep -i smmu - SMMU CMDQV feature status" \
    "dmesg | grep -i smmu || echo 'No SMMU lines found or dmesg access is restricted.'"
run_step "8a" "nvidia-smi -q - GPU hardware, firmware, software details" "nvidia-smi -q"
run_step "8b" "nvidia-smi topo -m - GPU/NIC topology matrix (NVLink, PCIe, NUMA affinity)" \
    "nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g'"
run_step "9" "sysctl kernel.numa_balancing - automatic NUMA balancing" \
    "val=\$(sysctl -n kernel.numa_balancing) && printf 'kernel.numa_balancing = %s (%s)\n' \"\${val}\" \"\$([ \"\${val}\" = 0 ] && echo disabled || echo enabled)\""

run_step "10" "ibv_devinfo - InfiniBand HCA device info" "ibv_devinfo"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_slurm_topology_config() {
    if ! command -v scontrol > /dev/null 2>&1; then
        echo "scontrol command not found"
        return 127
    fi

    local config
    config=$(scontrol show config)
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] scontrol show config returned exit code ${rc}."
        return "${rc}"
    fi

    echo "--- Slurm topology configuration ---"
    printf "%s\n" "${config}" | grep -Ei "^[[:space:]]*(TopologyPlugin|TopologyParam)[[:space:]]*=" \
        || echo "(no TopologyPlugin or TopologyParam entries found)"
}

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_slurm_topology() {
    if ! command -v scontrol > /dev/null 2>&1; then
        echo "scontrol command not found"
        return 127
    fi

    local topology
    topology=$(scontrol show topology)
    local rc=$?
    if [[ -n ${topology} ]]; then
        printf "%s\n" "${topology}"
    fi

    echo
    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] scontrol show topology returned exit code ${rc}."
        return "${rc}"
    fi

    if [[ -z ${topology//[[:space:]]/} ]]; then
        echo "[FAILED] scontrol show topology returned empty output."
        echo "         Slurm topology should be configured for benchmark clusters that"
        echo "         span more than one rack or network switch."
        return 1
    fi

    echo "[OK] scontrol show topology returned topology data."
}

run_step "11a" "scontrol show config - Slurm topology plugin configuration" \
    "$(declare -f check_slurm_topology_config); check_slurm_topology_config"
run_step "11b" "scontrol show topology - Slurm topology configuration is present" \
    "$(declare -f check_slurm_topology); check_slurm_topology"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_slurm_mpi_pmix() {
    local output rc
    output=$(srun --mpi=list 2>&1)
    rc=$?
    printf '%s\n' "${output}"
    echo

    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] srun --mpi=list returned exit code ${rc}."
        return "${rc}"
    fi

    if printf '%s\n' "${output}" | grep -qiE "(^|[[:space:]])pmix"; then
        echo "[OK] PMIx MPI plugin is available."
    else
        echo "[FAILED] PMIx MPI plugin not listed by 'srun --mpi=list'."
        echo "         LLMB recipes invoke Slurm with --mpi=pmix; Slurm must be built with --with-pmix."
        return 1
    fi
}

run_step "11c" "srun --mpi=list - Slurm PMIx MPI plugin available" \
    "$(declare -f check_slurm_mpi_pmix); check_slurm_mpi_pmix"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_enroot_conf() {
    local conf="/etc/enroot/enroot.conf"

    echo "--- ${conf} (non-comment lines) ---"
    if [ ! -f "${conf}" ]; then
        echo "${conf} not found"
        return
    fi

    grep -v "^[[:space:]]*#" "${conf}" | grep -v "^[[:space:]]*$" \
        || echo "(all lines commented or empty)"

    echo
    echo "--- enroot.conf diagnostics ---"
    if grep -Eq "^[[:space:]]*ENROOT_ROOTFS_WRITABLE[[:space:]=]+yes" "${conf}"; then
        echo "[OK] ENROOT_ROOTFS_WRITABLE = yes"
    else
        echo "[WARNING] ENROOT_ROOTFS_WRITABLE not set to yes."
        echo "          LLMB setup and launch paths expect writable container filesystems."
    fi
    if grep -Eq "^[[:space:]]*ENROOT_REMAP_ROOT[[:space:]=]+yes" "${conf}"; then
        echo "[OK] ENROOT_REMAP_ROOT = yes"
    else
        echo "[WARNING] ENROOT_REMAP_ROOT not set to yes."
        echo "          LLMB recipes recommend remapping root in containerized Slurm jobs."
    fi
}

# shellcheck disable=SC2317  # invoked indirectly via declare -f
enroot_environ_has_assignment() {
    local envdir="$1"
    local var_name="$2"
    local f

    for f in "${envdir}"/*; do
        if [ -f "$f" ] && grep -Eq "^[[:space:]]*${var_name}[[:space:]]*=" "$f" 2> /dev/null; then
            return 0
        fi
    done

    return 1
}

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_enroot_environ() {
    local envdir="/etc/enroot/environ.d"

    echo "--- ${envdir}/ contents ---"
    if [ ! -d "${envdir}" ]; then
        echo "${envdir}/ directory not found"
        return
    fi

    local env_files=("${envdir}"/*.env)
    if [ ! -e "${env_files[0]}" ]; then
        echo "(no .env files found)"
        env_files=()
    fi

    for f in "${envdir}"/*; do
        [ -f "$f" ] && printf "# %s\n" "$(basename "$f")" && cat "$f" && echo
    done

    echo "--- environ.d diagnostics ---"
    if [ "${#env_files[@]}" -le 1 ]; then
        echo "[WARNING] Only ${#env_files[@]} env file(s) in environ.d -- likely bare defaults."
        echo "          Cluster-specific files (NCCL, Mellanox, NVIDIA, UCX) appear missing."
        echo "          This is a common cause of hard-to-debug failures when containers"
        echo "          do not inherit the required network/GPU environment."
    fi
    if enroot_environ_has_assignment "${envdir}" "NCCL_IB_HCA"; then
        echo "[OK] NCCL_IB_HCA is set in environ.d."
    else
        echo "[WARNING] NCCL_IB_HCA not found in environ.d -- multi-node NCCL jobs may fail or use wrong HCAs."
    fi
}

run_step "12a" "enroot config - enroot.conf" \
    "$(declare -f check_enroot_conf); check_enroot_conf"
run_step "12b" "enroot config - environ.d" \
    "$(declare -f enroot_environ_has_assignment); $(declare -f check_enroot_environ); check_enroot_environ"

run_step "13a" "srun container nvidia-smi - GPU visibility inside container (${IMAGE})" \
    "srun --nodes=1 --ntasks=1 --container-image '${IMAGE}' --container-writable --no-container-mount-home nvidia-smi"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_container_slurm_pytorch_hook() {
    local image="$1"
    local output rc

    output=$(srun --nodes=1 --ntasks=1 \
        --container-image "${image}" \
        --container-writable --no-container-mount-home \
        bash -c 'env | grep -E "^(MASTER_ADDR|MASTER_PORT|WORLD_SIZE|LOCAL_RANK|RANK)=" | sort || true' 2>&1)
    rc=$?
    printf '%s\n' "${output}"
    echo

    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] srun bash inside container returned exit code ${rc}."
        return "${rc}"
    fi

    local missing=()
    local var
    for var in MASTER_ADDR MASTER_PORT WORLD_SIZE; do
        if ! printf '%s\n' "${output}" | grep -q "^${var}=[^[:space:]]"; then
            missing+=("${var}")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "[OK] MASTER_ADDR, MASTER_PORT, and WORLD_SIZE are set inside the container."
        echo "     Enroot extra hook '50-slurm-pytorch.sh' appears to be installed and configured."
    else
        echo "[FAILED] Missing PyTorch hook vars inside the container: ${missing[*]}"
        echo "         Enroot extra hook '50-slurm-pytorch.sh' is missing or not configured."
        echo "         LLMB recipes rely on this hook to populate MASTER_ADDR/MASTER_PORT/WORLD_SIZE"
        echo "         from Slurm env vars at container start."
        echo "         See https://github.com/NVIDIA/enroot/tree/main/conf/hooks/extra"
        return 1
    fi
}

run_step "13b" "srun container env - 50-slurm-pytorch.sh hook sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE (${IMAGE})" \
    "$(declare -f check_container_slurm_pytorch_hook); check_container_slurm_pytorch_hook '${IMAGE}'"

print_banner "System Info Collection Summary"
echo "Failed non-fatal steps: ${FAILED_STEPS}"
echo "Completed at: $(date -Iseconds)"

exit 0
