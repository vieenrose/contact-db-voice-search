#!/usr/bin/env bash
# Full dataset generation — edge-tts only (PrimeTTS dropped). Resumable: re-running
# skips clips already synthesized. Train voices spread across all texts (1 voice each);
# the two held-out TEST voices cover a test subset. augment.py degrades all to 8 kHz
# and routes by voice into train/val/test.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

TRAIN="zh-TW-HsiaoChenNeural,zh-TW-YunJheNeural,zh-CN-XiaoxiaoNeural,zh-CN-YunxiNeural,en-US-GuyNeural,en-US-EmmaNeural"
TEST="zh-TW-HsiaoYuNeural,en-US-AriaNeural"   # must match augment.TEST_VOICES

# clean any prior (PrimeTTS / smoke) artifacts for a fresh, edge-only build
rm -f data/audio/base/manifest*.jsonl data/audio/base/u*.wav
rm -rf data/audio/clips
mkdir -p data/audio/base

echo ">>> TRAIN voices over all texts (1 voice/text, rotated)"
$PY data/synthesize_edge.py --voices "$TRAIN" --assign random --out-manifest manifest_edge.jsonl

echo ">>> TEST voices over a 500-text subset (both voices each)"
$PY data/synthesize_edge.py --voices "$TEST" --assign all --limit 500 --out-manifest manifest_edge_test.jsonl

echo ">>> augment -> 8 kHz telephony, voice-disjoint train/val/test"
$PY data/augment.py --variants 3

echo ">>> dataset sizes"
for s in train val test; do printf "  %-6s %s clips\n" "$s" "$(wc -l < data/audio/$s.jsonl)"; done
