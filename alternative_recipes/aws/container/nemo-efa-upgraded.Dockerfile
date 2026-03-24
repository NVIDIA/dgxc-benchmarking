# Dockerfile to extend NVIDIA NeMo 25.11.01 with updated EFA stack for DGXC clusters
# Base image: nvcr.io/nvidia/nemo:25.11.01 (Ubuntu 24.04, CUDA 13.0, PyTorch 2.9.0)
#
# This is the production-validated build used on both P5en (H200) and P6-B200 clusters.
# It uses the EFA installer's DEB-packaged aws-ofi-nccl (simpler, fewer dependencies)
# rather than building from source, and adds critical symlinks for NCCL plugin discovery.
#
# What this Dockerfile upgrades:
#   - EFA installer:    container's version        -> 1.47.0
#   - Libfabric:        container's version        -> latest from EFA installer (2.3.1+)
#   - aws-ofi-nccl:     container's version        -> latest from EFA installer (DEB package)
#   - rdma-core:        container's version        -> latest from EFA installer (60.0+)
#   - NCCL:             container 2.28.3           -> 2.29.3 (matches host version)
#   - GDRCopy:          not present                -> v2.5.1
#
# What this Dockerfile does NOT change:
#   - PyTorch 2.9.0 (untouched)
#   - cuDNN 9.13.1 (untouched)
#   - CUDA toolkit 13.0 (untouched)
#   - HPC-X 2.24.1 (untouched)
#
# CRITICAL: The NCCL plugin naming bug
#   NCCL 2.29+ uses shinit_v2 which sets NCCL_NET_PLUGIN=aws-ofi, telling NCCL to look
#   for libnccl-net-aws-ofi.so. The EFA installer DEB package names the file
#   libnccl-net-ofi.so (without the "aws-" prefix). If NCCL cannot find the plugin,
#   it silently falls back to TCP sockets -- no error, just ~23% worse performance.
#   Step 3 below creates the required symlinks to fix this.

FROM nvcr.io/nvidia/nemo:25.11.01

# ============================================================
# 1. Install EFA Installer (libfabric, rdma-core, aws-ofi-nccl)
# ============================================================
# The EFA installer provides:
#   - libfabric (2.3.1+) with EFA provider
#   - rdma-core (60.0+)
#   - aws-ofi-nccl plugin + tuner (DEB package at /opt/amazon/ofi-nccl/lib/)
#   - efa-config, efa-profile
#
# --skip-kmod: kernel module comes from the host, not the container
# --skip-limit-conf: don't modify ulimits in container
# --no-verify: skip GPG verification (container build env)
# -g: install GDR-related packages
# -d: install development headers
ENV EFA_INSTALLER_VERSION=1.47.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    environment-modules \
    tcl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz \
    && cd aws-efa-installer \
    && apt-get update \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && cd / && rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*

# ============================================================
# 2. Remove the duplicate manually-compiled aws-ofi-nccl
# ============================================================
# The EFA installer places its aws-ofi-nccl at /opt/amazon/ofi-nccl/lib/
# The base NeMo image has a manually compiled copy at /opt/amazon/aws-ofi-nccl/lib/
# Remove the old one to avoid conflicts.
RUN rm -rf /opt/amazon/aws-ofi-nccl

# ============================================================
# 3. Create critical symlinks for NCCL plugin discovery
# ============================================================
# NCCL_NET_PLUGIN=aws-ofi  =>  NCCL looks for libnccl-net-aws-ofi.so
# EFA installer names it         libnccl-net-ofi.so
# Without this symlink, NCCL falls back to TCP sockets silently!
#
# Similarly for the tuner plugin:
# NCCL_TUNER_PLUGIN=aws-ofi  =>  NCCL looks for libnccl-tuner-aws-ofi.so
# EFA installer names it          libnccl-ofi-tuner.so
RUN ln -sf /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
           /opt/amazon/ofi-nccl/lib/libnccl-net-aws-ofi.so && \
    ln -sf /opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so \
           /opt/amazon/ofi-nccl/lib/libnccl-tuner-aws-ofi.so

# ============================================================
# 4. Upgrade NCCL to 2.29.3 (matches host version)
# ============================================================
# Upgrading NCCL ensures the container version matches the host's
# NCCL (2.29.x), avoiding version mismatch issues with shinit_v2.
ENV NCCL_VERSION=2.29.3-1
RUN apt-get update && \
    apt-get install -y --allow-downgrades --allow-change-held-packages \
      libnccl2=${NCCL_VERSION}+cuda12.9 \
      libnccl-dev=${NCCL_VERSION}+cuda12.9 && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# 5. Install GDRCopy v2.5.1 (GPU-direct RDMA memory copy)
# ============================================================
RUN cd /tmp && \
    git clone --branch v2.5.1 --depth 1 https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy && \
    make -j$(nproc) lib lib_install && \
    cd / && rm -rf /tmp/gdrcopy

# ============================================================
# 6. Fix path references
# ============================================================
# Ensure the dynamic linker can find the new EFA/OFI libraries.
# The base image may have references to /opt/amazon/aws-ofi-nccl/lib/
# in /etc/environment or /etc/shinit_v2 -- update them to the EFA
# installer's path at /opt/amazon/ofi-nccl/lib/.
RUN echo "/opt/amazon/ofi-nccl/lib" > /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
    echo "/opt/amazon/efa/lib" > /etc/ld.so.conf.d/efa.conf

RUN sed -i 's|/opt/amazon/aws-ofi-nccl/lib|/opt/amazon/ofi-nccl/lib|g' /etc/environment 2>/dev/null || true
RUN sed -i 's|/opt/amazon/aws-ofi-nccl/lib|/opt/amazon/ofi-nccl/lib|g' /etc/shinit_v2 2>/dev/null || true

# Rebuild ldconfig cache from scratch
RUN rm -f /etc/ld.so.cache && ldconfig

# ============================================================
# 7. Set environment variables
# ============================================================
# Pyxis/Enroot sources /etc/environment at container startup to set
# environment variables inside the container. We dump the full build
# environment to this file so that all NVIDIA, CUDA, PyTorch, and
# EFA variables are available at runtime.
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/venv/bin:/opt/amazon/efa/bin:${PATH}"
ENV FI_PROVIDER=efa

# Dump the full build environment to /etc/environment.
# Pyxis/Enroot reads this file at container startup to set env vars.
# Filter out read-only/internal vars that shouldn't be in /etc/environment.
RUN env | grep -v '^_=' | grep -v '^HOSTNAME=' | grep -v '^HOME=' | \
    grep -v '^PWD=' | grep -v '^OLDPWD=' | grep -v '^SHLVL=' | \
    grep -v '^TERM=' | grep -v '^DEBIAN_FRONTEND=' | \
    sort > /etc/environment

# ============================================================
# 8. Verify installation
# ============================================================
# This step validates that the EFA stack is correctly installed.
# It checks: libfabric version, plugin symlinks, and library discovery.
RUN echo "=== EFA Stack Verification ===" && \
    echo "--- libfabric ---" && \
    fi_info --version && \
    ls -la /opt/amazon/efa/lib/libfabric.so* && \
    echo "--- aws-ofi-nccl plugin (with symlinks) ---" && \
    ls -la /opt/amazon/ofi-nccl/lib/libnccl-net-* && \
    ls -la /opt/amazon/ofi-nccl/lib/libnccl-*tuner* && \
    echo "--- NCCL version ---" && \
    dpkg -l | grep nccl && \
    echo "--- ldconfig resolution ---" && \
    ldconfig -p | grep -E "(libfabric|nccl-net|nccl.*tuner|gdrcopy)" && \
    echo "=== Verification Complete ==="

WORKDIR /workspace
