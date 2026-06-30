#!/usr/bin/env bash
# Decisive go/no-go for "real-time 0.6B S2S on Jetson Nano gen1": measure raw backbone tok/s on the
# ACTUAL hardware. AR audio decode needs >=12.5 backbone passes/s (Mimi 12.5 Hz) to be real-time, so a
# bare 0.6B must clear ~12.5 tok/s at INT8 with margin. Run this ON the Nano gen1. Reads one number.
#
# Prereqs on the Nano: a Maxwell-compatible llama.cpp build (CUDA 10.2 / sm_53) — build with
#   make GGML_CUDA=1 CUDA_DOCKER_ARCH=sm_53     (or CMAKE_CUDA_ARCHITECTURES=53)
# If the CUDA build is too painful on JetPack, also run the CPU path (4x Cortex-A57) as a floor.
set -e
MODEL_REPO="${1:-Qwen/Qwen3-0.6B-GGUF}"     # any ~0.6B GGUF stands in for the backbone
LCPP="${LCPP:-$HOME/llama.cpp}"
NGEN=128                                     # tokens to generate for the timing
PROMPT="你好，請問分機幾號？ Hello, which extension?"

echo "== Jetson Nano gen1 — 0.6B backbone throughput gate =="
echo "real-time bar: >= 12.5 tok/s (INT8, with margin for encoder + Mimi + glue)"
for QUANT in Q8_0 Q4_K_M; do
  GGUF="$HOME/models/qwen3-0.6b-${QUANT}.gguf"
  [ -f "$GGUF" ] || { echo "  [skip $QUANT] put a $QUANT GGUF at $GGUF (hf download $MODEL_REPO)"; continue; }
  for DEV in "GPU:-ngl 99" "CPU:-ngl 0 -t 4"; do
    tag="${DEV%%:*}"; flags="${DEV#*:}"
    echo "--- $QUANT / $tag ---"
    "$LCPP/llama-bench" -m "$GGUF" -p 0 -n "$NGEN" $flags 2>/dev/null \
      | grep -Ei "tg|tok" || \
    "$LCPP/llama-cli" -m "$GGUF" $flags -n "$NGEN" -p "$PROMPT" 2>&1 \
      | grep -Ei "eval time|tokens per second|tok/s"
  done
done
echo
echo "Interpret: GPU Q8_0 tok/s is the headline. >=25 => comfortable real-time; 12.5-25 => tight (drop"
echo "to Q4_K_M or grouped/low-frame-rate tokens); <12.5 => rescope (smaller backbone or turn-based)."
