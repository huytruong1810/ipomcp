#!/usr/bin/env bash
# When rung r4 completes, restart r3 and r2 with more workers in the same
# output directories. Resume keeps finished trials (worker count is not part
# of the run manifest); only trials in progress at the restart are recomputed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
OUT=results/tiger/ladder_20260930
until [ -f "$OUT/r4/batch_results.complete.json" ] || grep -q "Batch failed" "$OUT/r4.out" 2>/dev/null; do sleep 120; done
echo "$(date -Is) r4 finished; rebalancing" >> "$OUT/progress.log"
for spec in "r3 12" "r2 6"; do
  set -- $spec
  [ -f "$OUT/$1/batch_results.complete.json" ] && continue
  for parent in $(pgrep -f "[l]3_l2_convergence.py --rung $1 "); do
    for k in $(pgrep -P "$parent"); do kill -KILL -- "-$k" 2>/dev/null || kill -KILL "$k" 2>/dev/null; done
    kill -TERM "$parent"
  done
  sleep 5
  echo "$(date -Is) restart $1 workers=$2" >> "$OUT/progress.log"
  setsid nohup nice -n 10 ../ipomcp/.venv/bin/python benchmarks/l3_l2_convergence.py \
    --rung "$1" --trials 100 --steps 20 --workers "$2" --timeout 14400 --max-rss-mb 3000 \
    --audit-seconds 120 --audit-support 400 --audit-rss-mb 1200 \
    --output "$OUT/$1" >> "$OUT/$1.out" 2>&1 < /dev/null &
done
