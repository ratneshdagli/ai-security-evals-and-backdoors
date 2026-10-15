#!/usr/bin/env bash
# Run GPU MODE's own eval.py (test + benchmark modes, the exact leaderboard harness) on this GPU,
# then this project's hardened extras. Usage (inside WSL): bash validate_local.sh /path/to/reference-kernels
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RK="${1:?path to a gpu-mode/reference-kernels checkout}"
RUN="$HERE/_run"
rm -rf "$RUN" && mkdir -p "$RUN"
cp "$RK/problems/pmpp_v2/histogram_py/"{reference.py,task.py} "$RK/problems/pmpp_v2/"{utils.py,eval.py} "$RUN/"
cp "$HERE/submission.py" "$RUN/"
python - "$RK/problems/pmpp_v2/histogram_py/task.yml" "$RUN" <<'PY'
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1]))
fmt = lambda c: "; ".join(f"{k}: {v}" for k, v in c.items())
open(f"{sys.argv[2]}/tests.txt", "w").write("\n".join(fmt(c) for c in spec["tests"]) + "\n")
open(f"{sys.argv[2]}/benchmarks.txt", "w").write("\n".join(fmt(c) for c in spec["benchmarks"]) + "\n")
PY
cd "$RUN"
echo "== gpu-mode eval.py test ==";      POPCORN_FD=3 python eval.py test tests.txt 3>&1 1>/dev/null | tee test.log
echo "== gpu-mode eval.py benchmark =="; POPCORN_FD=3 python eval.py benchmark benchmarks.txt 3>&1 1>/dev/null | tee bench.log
echo "== gpu-mode eval.py leaderboard (recheck every run) =="; POPCORN_FD=3 python eval.py leaderboard benchmarks.txt 3>&1 1>/dev/null | tee leaderboard.log
echo "== hardened extras =="; python "$HERE/hardened_checks.py"
