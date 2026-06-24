#!/usr/bin/env bash
# Tracked watcher for run_v3.sh: reports the benchmark on success, or the log tail on
# failure / unexpected exit / stall (log unchanged >25 min => likely hung).
set -uo pipefail
LOG="/tmp/claude-1002/-home-louis-contact-db-voice-search/fcceb2f7-909a-4412-872c-9ffc40c09201/scratchpad/run_v3.log"
for i in $(seq 1 720); do          # up to ~12h
  if grep -q "V3 DONE" "$LOG" 2>/dev/null; then
    echo "===== V1 COMPLETE — BENCHMARK ====="
    sed -n '/BENCHMARK SCORECARD/,/V3 DONE/p' "$LOG"; exit 0
  fi
  if grep -qE "TRAIN FAILED|PREDICT FAILED" "$LOG" 2>/dev/null; then
    echo "===== PIPELINE FAILED ====="; tail -30 "$LOG"; exit 1
  fi
  if ! pgrep -f "scripts/run_v3.sh" >/dev/null 2>&1; then
    echo "===== run_v3.sh exited without V3 DONE ====="; tail -30 "$LOG"; exit 1
  fi
  age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
  if [ "$age" -gt 1500 ]; then
    echo "===== STALLED (${age}s no log output) — likely hung ====="; tail -20 "$LOG"; exit 1
  fi
  sleep 60
done
echo "watcher timed out"; tail -20 "$LOG"; exit 1
