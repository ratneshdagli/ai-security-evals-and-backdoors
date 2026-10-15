#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get install -y --no-install-recommends cuda-toolkit-12-8

# Make nvcc discoverable without needing a fresh login shell
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' > /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}' >> /etc/profile.d/cuda.sh
chmod +x /etc/profile.d/cuda.sh
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

nvcc --version

python3 -m venv /root/kbh/venv
source /root/kbh/venv/bin/activate
pip install --upgrade pip
pip install torch
pip install ninja pyyaml numpy pandas requests datasets huggingface_hub

python3 - <<'PYEOF'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PYEOF

echo "CUDA_VENV_SETUP_DONE"
