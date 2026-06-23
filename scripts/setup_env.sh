#!/usr/bin/env bash
# One-shot environment setup for keystone (Ubuntu 18.04, base Python 3.8 too old).
# Installs uv, a standalone Python 3.11, a project venv, all data-gen + training
# deps, NLTK data, and the PrimeTTS model. Idempotent-ish; safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. uv (Python/venv/pkg manager)
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$HOME/.local/bin/uv"

# 2. Python 3.11 venv
"$UV" venv --python 3.11 .venv

# 3. Deps. torch default wheel is the CUDA build (serves both g2pw-on-CPU for
#    data-gen and GPU training on the GTX 1070). Training libs added later.
"$UV" pip install --python .venv \
  numpy scipy soundfile onnxruntime edge-tts \
  g2pw g2p_en cn2an torch huggingface_hub rapidfuzz pypinyin

# 4. NLTK data for g2p_en
.venv/bin/python - <<'PY'
import nltk
for r in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"]:
    nltk.download(r, quiet=True)
print("nltk data ok")
PY

# 5. PrimeTTS model
.venv/bin/hf download Luigi/PrimeTTS --local-dir models/PrimeTTS >/dev/null
echo "SETUP DONE: $(.venv/bin/python --version)"
