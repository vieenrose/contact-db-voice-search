#!/usr/bin/env bash
# Perception->retrieval benchmark: STOCK Qwen transcribe -> resolver top-N -> action.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f predict_qwen.py >/dev/null 2>&1; do sleep 10; done   # wait for GPU
echo ">>> [1/2] STOCK Qwen transcribe -> resolver retrieve (no name-mapping FT)"
$PY train/predict_retrieval.py --test data/audio/test.jsonl --out runs/retrieval/preds.jsonl
echo ">>> [2/2] BENCHMARK"
$PY eval.py --gold data/audio/test.jsonl --pred runs/retrieval/preds.jsonl
echo ">>> RETRIEVAL DONE"
