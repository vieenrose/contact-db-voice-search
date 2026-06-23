#!/usr/bin/env bash
# Wait for the in-flight data-gen, backfill any clips that failed (transient edge-tts,
# e.g. YunJhe), re-augment, then train v1. Resumable synthesis only retries missing clips.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TRAIN="zh-TW-HsiaoChenNeural,zh-TW-YunJheNeural,zh-CN-XiaoxiaoNeural,zh-CN-YunxiNeural,en-US-GuyNeural,en-US-EmmaNeural"
TEST="zh-TW-HsiaoYuNeural,en-US-AriaNeural"

echo ">>> waiting for current data-gen to finish..."
while pgrep -f "gen_dataset.sh|synthesize_edge.py|data/augment.py" >/dev/null 2>&1; do sleep 20; done
echo ">>> data-gen finished; base clips: $(ls data/audio/base/e*.wav 2>/dev/null | wc -l)"

echo ">>> backfill TRAIN voices (resume retries the failed YunJhe etc.)"
$PY data/synthesize_edge.py --voices "$TRAIN" --assign random --out-manifest manifest_edge.jsonl
echo ">>> backfill TEST voices (resume)"
$PY data/synthesize_edge.py --voices "$TEST" --assign all --limit 500 --out-manifest manifest_edge_test.jsonl
echo ">>> base clips after backfill: $(ls data/audio/base/e*.wav 2>/dev/null | wc -l)"

echo ">>> augment -> 8 kHz, voice-disjoint splits"
$PY data/augment.py --variants 3
for s in train val test; do printf "    %-6s %s clips\n" "$s" "$(wc -l < data/audio/$s.jsonl)"; done

echo ">>> TRAIN v1 (frozen encoder+LLM, projector+LoRA, 2 epochs)"
$PY train/train.py --train data/audio/train.jsonl --val data/audio/val.jsonl --out runs/v1 --epochs 2
echo ">>> V1 DONE -> runs/v1"
