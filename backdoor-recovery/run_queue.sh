#!/usr/bin/env bash
# Sequential GPU queue for the follow-up experiments. Waits for the adapter grid + big-model
# fetch, then runs each stage; a failing stage does not stop the next. Logs -> results/queue_*.log
source /root/proj/backdoor-recovery/env.sh
cd /root/proj/backdoor-recovery
EXPECT=26
while [ "$(ls models_own/*/adapter_model.safetensors 2>/dev/null | wc -l)" -lt "$EXPECT" ] || pgrep -f make_poisoned.py >/dev/null; do sleep 30; done
while pgrep -f weight_space_big.py >/dev/null || pgrep -f fetch_early_tensors.py >/dev/null; do sleep 30; done
run() { echo "=== $NAME $(date +%T)" >> results/queue_status.log; "$@" > "results/queue_$NAME.log" 2>&1; echo "exit $? $NAME $(date +%T)" >> results/queue_status.log; }
NAME=own_eval;    run python -u scripts/own_benchmark.py eval
NAME=big_score;   run python -u scripts/weight_space_big.py score
NAME=removal;     run python -u scripts/remove_backdoor.py --only p0.02
NAME=localize;    run python -u scripts/localize.py --only r16_p0.02_s0
NAME=hybrid;      run python -u scripts/hybrid.py
cd /root/proj/kernelbench-hardened && export PYTHONPATH=/root/proj/kernelbench-hardened
NAME=rl_hacker;   run python -u -m hacker_fixer.rl_hacker --rounds 6
echo "ALL DONE $(date +%T)" >> /root/proj/backdoor-recovery/results/queue_status.log
