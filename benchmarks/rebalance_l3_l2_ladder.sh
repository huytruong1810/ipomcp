#!/usr/bin/env bash
# Rebalance the running ladder now: r3 becomes the 100-trial headline rung.
# r4 keeps 8 workers with a 10-hour per-trial limit: its slowest trials exceed
# 4 hours, and excluding them would bias r4 toward short-belief trajectories.
# Resume keeps finished trials (worker count and limits are not part of the run
# manifest); only trials in progress at the restart are recomputed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
OUT=results/tiger/ladder_20260930
echo "$(date -Is) rebalancing" >> "$OUT/progress.log"
for spec in "r4 8 36000" "r3 8 14400" "r2 3 14400"; do
  set -- $spec
  [ -f "$OUT/$1/batch_results.complete.json" ] && continue
  for parent in $(pgrep -f "[l]3_l2_convergence.py --rung $1 "); do
    for k in $(pgrep -P "$parent"); do kill -KILL -- "-$k" 2>/dev/null || kill -KILL "$k" 2>/dev/null; done
    kill -TERM "$parent"
  done
done
sleep 10
for spec in "r4 8 36000" "r3 8 14400" "r2 3 14400"; do
  set -- $spec
  [ -f "$OUT/$1/batch_results.complete.json" ] && continue
  echo "$(date -Is) restart $1 workers=$2 timeout=$3" >> "$OUT/progress.log"
  setsid nohup nice -n 10 ../ipomcp/.venv/bin/python benchmarks/l3_l2_convergence.py \
    --rung "$1" --trials 100 --steps 20 --workers "$2" --timeout "$3" --max-rss-mb 3000 \
    --audit-seconds 120 --audit-support 400 --audit-rss-mb 1200 \
    --output "$OUT/$1" >> "$OUT/$1.out" 2>&1 < /dev/null &
done
