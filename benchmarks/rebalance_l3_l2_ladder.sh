#!/usr/bin/env bash
# Rebalance the running ladder once r4's first round finishes (>= 12 trials).
# r3 becomes the 100-trial headline rung; r4 keeps fewer workers and reports n.
# Resume keeps finished trials (worker count is not part of the run manifest);
# only trials in progress at the restart are recomputed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
OUT=results/tiger/ladder_20260930
until [ "$(ls $OUT/r4/trials/*.csv 2>/dev/null | wc -l)" -ge 12 ]; do sleep 60; done
echo "$(date -Is) r4 round done; rebalancing" >> "$OUT/progress.log"
for spec in "r4 8" "r3 8" "r2 3"; do
  set -- $spec
  [ -f "$OUT/$1/batch_results.complete.json" ] && continue
  for parent in $(pgrep -f "[l]3_l2_convergence.py --rung $1 "); do
    for k in $(pgrep -P "$parent"); do kill -KILL -- "-$k" 2>/dev/null || kill -KILL "$k" 2>/dev/null; done
    kill -TERM "$parent"
  done
done
sleep 10
for spec in "r4 8" "r3 8" "r2 3"; do
  set -- $spec
  [ -f "$OUT/$1/batch_results.complete.json" ] && continue
  echo "$(date -Is) restart $1 workers=$2" >> "$OUT/progress.log"
  setsid nohup nice -n 10 ../ipomcp/.venv/bin/python benchmarks/l3_l2_convergence.py \
    --rung "$1" --trials 100 --steps 20 --workers "$2" --timeout 14400 --max-rss-mb 3000 \
    --audit-seconds 120 --audit-support 400 --audit-rss-mb 1200 \
    --output "$OUT/$1" >> "$OUT/$1.out" 2>&1 < /dev/null &
done
