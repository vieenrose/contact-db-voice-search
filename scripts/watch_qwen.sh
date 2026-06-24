#!/usr/bin/env bash
# Robust watcher: completion marker + process-alive only (no log-mtime stall check,
# which false-positives on buffered stdout). edge-tts timeout already prevents hangs.
set -uo pipefail
LOG="/tmp/claude-1002/-home-louis-contact-db-voice-search/fcceb2f7-909a-4412-872c-9ffc40c09201/scratchpad/run_qwen.log"
for i in $(seq 1 900); do          # up to ~15h
  if grep -q "QWEN DONE" "$LOG" 2>/dev/null; then
    echo "===== QWEN COMPLETE — BENCHMARK ====="; sed -n '/BENCHMARK SCORECARD/,/QWEN DONE/p' "$LOG"; exit 0
  fi
  if grep -qE "TRAIN FAILED|PREDICT FAILED|Traceback" "$LOG" 2>/dev/null; then
    echo "===== qwen FAILED ====="; grep -vE "torch_dtype" "$LOG" | tail -25; exit 1
  fi
  if ! pgrep -f "scripts/run_qwen.sh" >/dev/null 2>&1; then
    echo "===== run_qwen.sh exited without QWEN DONE ====="; grep -vE "torch_dtype" "$LOG" | tail -25; exit 1
  fi
  sleep 60
done
