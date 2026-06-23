#!/usr/bin/env bash
# v2: STOCK encoder + PRETRAINED projector + LoRA (data-efficient alignment).
# Reuses the existing dataset. Isolates whether the task is learnable.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo ">>> [1/3] train v2 (stock encoder + pretrained projector + LoRA, 2 epochs)"
$PY train/train.py --stock --train data/audio/train.jsonl --val data/audio/val.jsonl --out runs/v2 --epochs 2 \
  || { echo "TRAIN FAILED"; exit 1; }
echo ">>> [2/3] predict on voice-disjoint test"
$PY train/predict.py --stock --model runs/v2 --test data/audio/test.jsonl --out runs/v2/preds.jsonl \
  || { echo "PREDICT FAILED"; exit 1; }
echo ">>> [3/3] BENCHMARK"
$PY eval.py --gold data/audio/test.jsonl --pred runs/v2/preds.jsonl
echo ">>> V2 DONE"
