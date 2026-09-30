#!/usr/bin/env bash
# Full L3-vs-L2 resource ladder: 100 trials x 20 steps per rung, headline rung first.
# Resumable: rerunning continues pending trials when source and settings are unchanged.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
OUT=${OUT:-results/tiger/ladder_20260930}
WORKERS=${WORKERS:-12}
mkdir -p "$OUT"
for rung in r4 r3 r2 r1; do
  echo "$(date -Is) start $rung" >> "$OUT/progress.log"
  nice -n 10 ../ipomcp/.venv/bin/python benchmarks/l3_l2_convergence.py \
    --rung "$rung" --trials 100 --steps 20 --workers "$WORKERS" \
    --timeout 14400 --max-rss-mb 3000 --audit-seconds 120 --audit-support 400 --audit-rss-mb 1200 \
    --output "$OUT/$rung" > "$OUT/$rung.out" 2>&1
  echo "$(date -Is) end $rung exit=$?" >> "$OUT/progress.log"
done
