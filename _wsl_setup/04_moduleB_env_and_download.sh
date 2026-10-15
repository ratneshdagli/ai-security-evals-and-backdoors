#!/usr/bin/env bash
# Module B environment + ToxScreen model download. Runs inside WSL Ubuntu (rooted on D:).
set -euo pipefail
source /root/kbh/venv/bin/activate
export HF_TOKEN_PATH=/mnt/c/Users/dagli/.cache/huggingface/token
export HF_HUB_ENABLE_HF_TRANSFER=0
pip install -q "transformers>=4.56" "peft>=0.17" accelerate sentencepiece protobuf scipy scikit-learn pandas matplotlib
python -c "import transformers, peft; print('transformers', transformers.__version__, 'peft', peft.__version__)"
CFG="/mnt/d/Best Projects/AI SECURITY/backdoor-recovery/configs/toxscreen_models.txt"
python - "$CFG" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repos=[l.split()[0] for l in open(sys.argv[1], encoding='utf-8') if l.strip() and not l.startswith('#')]
repos.append('unsloth/Llama-3.2-1B-Instruct')  # ungated byte-identical mirror of meta-llama/Llama-3.2-1B-Instruct
for i, r in enumerate(repos):
    t=time.time()
    for attempt in range(3):
        try:
            p=snapshot_download(r, allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.md','*.jinja'])
            print(f'[{i+1}/{len(repos)}] OK {r} {time.time()-t:.0f}s', flush=True); break
        except Exception as e:
            print(f'[{i+1}/{len(repos)}] retry {attempt} {r}: {type(e).__name__} {str(e)[:120]}', flush=True); time.sleep(10)
    else:
        print(f'[{i+1}/{len(repos)}] FAILED {r}', flush=True)
print('ALL_DOWNLOADS_DONE', flush=True)
PY
