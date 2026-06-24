#!/usr/bin/env bash
# v4: QLoRA fine-tune Qwen2.5-Omni-3B (Mandarin-capable) on the attendant task.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo ">>> [1/3] QLoRA train Qwen-Omni-3B (2 epochs)"
$PY train/train_qwen.py --train data/audio/train.jsonl --out runs/qwen --epochs 2 || { echo "TRAIN FAILED"; exit 1; }
echo ">>> [2/3] predict on voice-disjoint test"
$PY train/predict_qwen.py --model runs/qwen --test data/audio/test.jsonl --out runs/qwen/preds.jsonl || { echo "PREDICT FAILED"; exit 1; }
echo ">>> [3/3] BENCHMARK"
$PY eval.py --gold data/audio/test.jsonl --pred runs/qwen/preds.jsonl
echo ">>> QWEN DONE"
