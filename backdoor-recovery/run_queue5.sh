#!/usr/bin/env bash
# "Match" mode adaptive attackers (trigger energy pulled to the typical level, hiding in both tails), then redo
# common-mode ranks and the detection AUROC including them.
source /root/proj/backdoor-recovery/env.sh
B=/root/proj/backdoor-recovery; cd $B
until grep -q "ALL DONE" results/queue4_status.log 2>/dev/null; do sleep 30; done
S=$B/results/queue5_status.log
run() { echo "=== $NAME $(date +%T)" >> $S; "$@" > "$B/results/queue5_$NAME.log" 2>&1; echo "exit $? $NAME $(date +%T)" >> $S; }
NAME=train_match;  run python -u scripts/own_benchmark.py train_extra
NAME=eval_match;   run python -u scripts/own_benchmark.py eval_extra
NAME=commonmode;   run python -u scripts/own_commonmode.py
NAME=detect;       run python -u scripts/detect_eval.py
echo "ALL DONE $(date +%T)" >> $S
