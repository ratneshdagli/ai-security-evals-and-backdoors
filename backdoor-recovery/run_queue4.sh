#!/usr/bin/env bash
# OLMo-only scoring (after queue3 frees the GPU), then redo the offline ranking including OLMo.
source /root/proj/backdoor-recovery/env.sh
B=/root/proj/backdoor-recovery; cd $B
until grep -q "ALL DONE" results/queue3_status.log 2>/dev/null; do sleep 30; done
S=$B/results/queue4_status.log
run() { echo "=== $NAME $(date +%T)" >> $S; "$@" > "$B/results/queue4_$NAME.log" 2>&1; echo "exit $? $NAME $(date +%T)" >> $S; }
NAME=olmo_score; BASE_ONLY=olmo run python -u scripts/weight_space_big.py score
NAME=rerank;     CUDA_VISIBLE_DEVICES="" run python -u scripts/big_rerank.py
echo "ALL DONE $(date +%T)" >> $S
