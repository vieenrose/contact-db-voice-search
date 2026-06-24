#!/usr/bin/env bash
# v3: more zh data + 3 epochs (stock encoder). Fresh synth (id scheme changed).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TRAIN="zh-TW-HsiaoChenNeural,zh-TW-YunJheNeural,zh-CN-XiaoxiaoNeural,zh-CN-YunxiNeural,en-US-GuyNeural,en-US-EmmaNeural"
TEST="zh-TW-HsiaoYuNeural,en-US-AriaNeural"
echo ">>> [1/6] regenerate requests (more zh)"; $PY data/generate_requests.py
echo ">>> [2/6] fresh synth TRAIN"; rm -f data/audio/base/e*.wav data/audio/base/manifest*.jsonl; rm -rf data/audio/clips
$PY data/synthesize_edge.py --voices "$TRAIN" --assign random --out-manifest manifest_edge.jsonl
echo ">>> [3/6] synth TEST"; $PY data/synthesize_edge.py --voices "$TEST" --assign all --limit 500 --out-manifest manifest_edge_test.jsonl
echo ">>> base clips: $(ls data/audio/base/e*.wav 2>/dev/null | wc -l)"
echo ">>> [4/6] augment"; $PY data/augment.py --variants 3
for s in train val test; do printf "    %-6s %s\n" "$s" "$(wc -l < data/audio/$s.jsonl)"; done
echo ">>> [5/6] train v3 (stock, 3 epochs)"
$PY train/train.py --stock --train data/audio/train.jsonl --val data/audio/val.jsonl --out runs/v3 --epochs 3 || { echo "TRAIN FAILED"; exit 1; }
echo ">>> [6/6] predict + BENCHMARK"
$PY train/predict.py --stock --model runs/v3 --test data/audio/test.jsonl --out runs/v3/preds.jsonl || { echo "PREDICT FAILED"; exit 1; }
$PY eval.py --gold data/audio/test.jsonl --pred runs/v3/preds.jsonl
echo ">>> V3 DONE"
