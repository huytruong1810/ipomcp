#!/usr/bin/env bash
# When r1 completes, restart r2 with its freed workers (3 -> 5). Resume keeps
# finished r2 trials; only those in progress at the restart are recomputed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
OUT=results/tiger/ladder_20260930
until [ -f "$OUT/r1/batch_results.complete.json" ]; do sleep 120; done
[ -f "$OUT/r2/batch_results.complete.json" ] && exit 0
echo "$(date -Is) r1 complete; restarting r2 with 5 workers" >> "$OUT/progress.log"
for parent in $(pgrep -f "[l]3_l2_convergence.py --rung r2 "); do
  for k in $(pgrep -P "$parent"); do kill -KILL -- "-$k" 2>/dev/null || kill -KILL "$k" 2>/dev/null; done
  kill -TERM "$parent"
done
sleep 10
setsid nohup nice -n 10 ../ipomcp/.venv/bin/python benchmarks/l3_l2_convergence.py \
  --rung r2 --trials 100 --steps 20 --workers 5 --timeout 14400 --max-rss-mb 3000 \
  --audit-seconds 120 --audit-support 400 --audit-rss-mb 1200 \
  --output "$OUT/r2" >> "$OUT/r2.out" 2>&1 < /dev/null &
