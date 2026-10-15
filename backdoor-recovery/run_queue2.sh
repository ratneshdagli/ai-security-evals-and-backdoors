#!/usr/bin/env bash
# Second pass: common-mode-corrected recovery, then removal, then big-model scoring. Waits for queue 1.
source /root/proj/backdoor-recovery/env.sh
cd /root/proj/backdoor-recovery
until grep -q "ALL DONE" results/queue_status.log 2>/dev/null; do sleep 30; done
run() { echo "=== $NAME $(date +%T)" >> results/queue2_status.log; "$@" > "results/queue2_$NAME.log" 2>&1; echo "exit $? $NAME $(date +%T)" >> results/queue2_status.log; }
NAME=commonmode; run python -u scripts/own_commonmode.py
NAME=removal;    run python -u scripts/remove_backdoor.py --only p0.05
NAME=big_score;  run python -u scripts/weight_space_big.py score
echo "ALL DONE $(date +%T)" >> results/queue2_status.log
