#!/usr/bin/env bash
# Third pass (starts after queue2): clean controls + adaptive attackers, detection AUROC, SVD ablation,
# assembled-candidate removal, Module A reruns with the new detectors.
source /root/proj/backdoor-recovery/env.sh
B=/root/proj/backdoor-recovery
cd $B
until grep -q "ALL DONE" results/queue2_status.log 2>/dev/null; do sleep 30; done
S=$B/results/queue3_status.log
run() { echo "=== $NAME $(date +%T)" >> $S; "$@" > "$B/results/queue3_$NAME.log" 2>&1; echo "exit $? $NAME $(date +%T)" >> $S; }
NAME=train_extra;  run python -u scripts/own_benchmark.py train_extra
NAME=eval_extra;   run python -u scripts/own_benchmark.py eval_extra
NAME=commonmode;   run python -u scripts/own_commonmode.py
NAME=detect;       run python -u scripts/detect_eval.py
NAME=svd;          run python -u scripts/svd_score.py
NAME=removal_asm;  run python -u scripts/remove_backdoor.py --only p0.05 --assemble --out removal_assembled.json
cd /root/proj/kernelbench-hardened && export PYTHONPATH=/root/proj/kernelbench-hardened
NAME=rl_v2;        run python -u -m hacker_fixer.rl_hacker --problems matmul_small --rounds 6 --extra-detectors static_dead_kernel,runtime_output_aliasing --out results/rl_hacker_report_v2.json
NAME=rl_holdout;   run python -u -m hacker_fixer.rl_hacker --problems matmul_small --rounds 6 --holdout camouflage=real_cuda_noop_launch,timing=stream_defer --holdout-max 24 --extra-detectors static_dead_kernel,runtime_output_aliasing --out results/rl_hacker_heldout.json
echo "ALL DONE $(date +%T)" >> $S
