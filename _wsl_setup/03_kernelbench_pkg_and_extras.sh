#!/usr/bin/env bash
set -euxo pipefail

source /root/kbh/venv/bin/activate

# Lightweight deps kernelbench's __init__ -> utils.py needs transitively
# (litellm/openai/dotenv for its LLM-query helpers we don't use directly, but
# the import chain still touches them; tomli for prompt TOML parsing; einops
# used by some problem defs).
pip install litellm openai python-dotenv tomli einops

# Editable-install the KernelBench package itself. Its pyproject.toml originally
# pinned requires-python == 3.10.* exactly; we relaxed that to >=3.10 in our
# local clone (kernelbench-hardened/external/KernelBench/pyproject.toml) since
# nothing in the package actually requires 3.10-specific syntax -- same clone
# already verified importable under Python 3.13 on the Windows side.
pip install -e "/mnt/d/Best Projects/AI SECURITY/kernelbench-hardened/external/KernelBench" --no-deps

python3 - <<'PYEOF'
import torch, kernelbench
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.kernel_static_checker import validate_kernel_static
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
print("rand_mix registered:", hasattr(torch, "rand_mix_like"))
ds = construct_kernelbench_dataset(1)
p = ds.get_problem_by_id(19)
print("sample problem:", p.name)
print("IMPORT_SMOKE_TEST_OK")
PYEOF

echo "KERNELBENCH_PKG_SETUP_DONE"
