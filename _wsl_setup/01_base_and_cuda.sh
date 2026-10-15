#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    build-essential gcc g++ make \
    git wget curl ca-certificates gnupg \
    python3 python3-venv python3-pip python3-dev \
    ninja-build pkg-config

# NVIDIA CUDA WSL-Ubuntu repo (toolkit only, no driver — the Windows host driver is reused)
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# Show what toolkit versions are actually available before picking one
apt-cache search '^cuda-toolkit-' || true

echo "BASE_SETUP_DONE"
