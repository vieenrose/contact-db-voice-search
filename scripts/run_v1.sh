#!/usr/bin/env bash
# Single end-to-end v1 run: resume synthesis (timeout-safe) -> augment -> train ->
# predict on the voice-disjoint test set -> benchmark. Resumable; keeps the 2233
# clips already made. Run as a tracked background job; the scorecard lands in output.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TRAIN="zh-TW-HsiaoChenNeural,zh-TW-YunJheNeural,zh-CN-XiaoxiaoNeural,zh-CN-YunxiNeural,en-US-GuyNeural,en-US-EmmaNeural"
TEST="zh-TW-HsiaoYuNeural,en-US-AriaNeural"

echo ">>> [1/5] synth TRAIN voices (resume + 25s timeout per clip)"
$PY data/synthesize_edge.py --voices "$TRAIN" --assign random --out-manifest manifest_edge.jsonl
echo ">>> [2/5] synth TEST voices (resume)"
$PY data/synthesize_edge.py --voices "$TEST" --assign all --limit 500 --out-manifest manifest_edge_test.jsonl
echo ">>> base clips: $(ls data/audio/base/e*.wav 2>/dev/null | wc -l)"

echo ">>> [3/5] augment -> 8 kHz, voice-disjoint splits"
$PY data/augment.py --variants 3
for s in train val test; do printf "    %-6s %s clips\n" "$s" "$(wc -l < data/audio/$s.jsonl 2>/dev/null || echo 0)"; done

echo ">>> [4/5] train v1 (frozen encoder+LLM, projector+LoRA, 2 epochs)"
$PY train/train.py --train data/audio/train.jsonl --val data/audio/val.jsonl --out runs/v1 --epochs 2 \
  || { echo "TRAIN FAILED"; exit 1; }

echo ">>> [5/5] predict + BENCHMARK on voice-disjoint test set"
$PY train/predict.py --model runs/v1 --test data/audio/test.jsonl --out runs/v1/preds.jsonl \
  || { echo "PREDICT FAILED"; exit 1; }
$PY eval.py --gold data/audio/test.jsonl --pred runs/v1/preds.jsonl
echo ">>> V1 DONE"
