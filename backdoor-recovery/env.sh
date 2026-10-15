# Source inside WSL: `source /root/proj/backdoor-recovery/env.sh`
source /root/kbh/venv/bin/activate
source /etc/profile.d/cuda.sh 2>/dev/null || true
export HF_TOKEN_PATH=/mnt/c/Users/dagli/.cache/huggingface/token
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/root/proj/backdoor-recovery:${PYTHONPATH:-}
cd /root/proj/backdoor-recovery
# hf-xet stalled at a 1 GiB chunk boundary on this network; plain HTTP is ~3.6 MB/s.
export HF_HUB_DISABLE_XET=1
