#!/usr/bin/env bash
# v5: agentic perception->retrieval. Model emits {query:<heard name>}; resolver maps to
# the LIVE directory (DB-external, OOD-rejecting). Scored vs original gold via resolver.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo ">>> [1/3] QLoRA train v5 (search-query target, 2 epochs)"
$PY train/train_qwen.py --train data/audio/train_v5.jsonl --out runs/qwen_v5 --epochs 2 || { echo "TRAIN FAILED"; exit 1; }
echo ">>> [2/3] predict on test set"
$PY train/predict_qwen.py --model runs/qwen_v5 --test data/audio/test.jsonl --out runs/qwen_v5/preds.jsonl || { echo "PREDICT FAILED"; exit 1; }
echo ">>> [3/3] BENCHMARK (resolver-grounded)"
$PY eval.py --gold data/audio/test.jsonl --pred runs/qwen_v5/preds.jsonl
echo ">>> V5 DONE"
