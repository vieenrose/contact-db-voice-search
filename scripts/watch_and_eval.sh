#!/usr/bin/env bash
# Wait for v1 training to save, then predict on the voice-disjoint test set and score.
# Runs as a tracked background job so it notifies on completion with the scorecard.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for i in $(seq 1 360); do            # up to ~6h
  if [ -f runs/v1/adapter_config.json ] || [ -f runs/v1/adapter_model.safetensors ]; then
    break
  fi
  if ! pgrep -f "finish_and_train.sh" >/dev/null 2>&1 && [ ! -d runs/v1 ]; then
    echo "ABORT: training pipeline not running and no runs/v1 yet"; tail -20 \
      /tmp/claude-1002/-home-louis-contact-db-voice-search/fcceb2f7-909a-4412-872c-9ffc40c09201/scratchpad/v1.log 2>/dev/null
    exit 1
  fi
  sleep 60
done
sleep 45                              # let checkpoint flush

echo ">>> predicting on test set"
$PY train/predict.py --model runs/v1 --test data/audio/test.jsonl --out runs/v1/preds.jsonl 2>&1 | tail -5
echo ">>> BENCHMARK v1"
$PY eval.py --gold data/audio/test.jsonl --pred runs/v1/preds.jsonl 2>&1 | tail -25
