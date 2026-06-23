# Session Handoff — Voice Extension Finder

> Read this first when resuming on a new machine (e.g. the CUDA box `keystone-ubuntu`).
> It captures every decision and the exact next steps so a fresh Claude session can continue
> without re-deriving context. The conversation history itself does **not** transfer between
> machines — this file + the git repo **are** the migration.

## Goal
Audio-in → text-out speech LLM for a **zh-TW/en telephone auto-attendant**: caller asks for a
person (English first name + Chinese surname, often code-switched), system returns/connects the
**extension**. Closed directory of ~200 people. **Deployment target: Jetson Nano gen1 (2019, 4 GB,
CPU).** Free dialog, no wake word; 8 kHz telephony audio.

## Committed decisions (do not relitigate)
- **Model:** Ultravox v0.5 (Llama-3.2-1B backbone), audio-in → text-out.
- **Encoder:** retrain projector against **whisper-base** (NOT stock large-v3-turbo) for gen1 latency
  (stock ≈ 5–15 s on gen1; base + endpointing targets ≈ 1–2 s). Genuine real-time needs an Orin.
- **Training recipe (8 GB VRAM limit):** FREEZE whisper-base encoder AND Llama-1B; train only the
  **projector + a LoRA adapter**. batch 1–2, gradient checkpointing, `sdpa` attention (NO flash-attn
  on Pascal), fp16 for memory (Pascal fp16 is slow — expect ~30–90 min/epoch).
- **TTS for data:** `Luigi/PrimeTTS` (single young-female zh-TW voice, 8 kHz). Single-voice gap closed
  by speed/formant + telephony augmentation + the FROZEN pretrained encoder's real-speaker robustness;
  real-call fine-tune later.
- **Dialog = hybrid:** Ultravox does PERCEPTION (audio → action object); a deterministic CONTROLLER
  owns POLICY (confirm+transfer / clarify / not-found / operator escape). A 1B does not free-run policy.
- **Fuzzy search:** controller-side, closed-set, phonetic+string (RapidFuzz + Double-Metaphone +
  tone-stripped pinyin) resolver grounds model output to a real directory row. Score margins drive
  resolve vs clarify vs not_found. (Module `resolver.py` NOT built yet.)

## Output schema (action-based)
- `{"action":"resolve","name":"Kevin Chen","ext":"1234"}`
- `{"action":"clarify","field":"surname|first_name","heard":{...}}`  (controller enumerates candidates)
- `{"action":"not_found"}`

## State / roadmap
- [x] 1. Mock directory — `data/generate_directory.py` → `data/directory.csv` (200, unique en+zh names)
- [x] 2. Synthetic requests — `data/generate_requests.py` → `data/requests.jsonl`
      (3790 samples: resolve 3000 incl. 600 caller-distractor, not_found 494, clarify 296; 0 ambiguous)
- [x] 3. TTS + telephony augmentation — `data/synthesize.py`, `data/augment.py` (smoke-tested on 5 clips)
- [ ] 3b. **Run FULL synthesis** (all ~2562 distinct texts → ~7700 augmented clips). NOT yet run.
- [ ] 4. **Ultravox train** on `keystone-ubuntu` (GTX 1070 8 GB). Set up env, train projector+LoRA.
- [ ] 5. GGUF export → `llama-mtmd-cli` on gen1, wire to Asterisk AudioSocket.
- [ ] 6. On-device latency + accuracy validation.
- [ ] `resolver.py` — fuzzy closed-set matcher (CPU, testable vs directory.csv).

## Environment
Data-gen venv (CPU): `python3 -m venv --system-site-packages .venv` then
`.venv/bin/pip install -r requirements.txt` + NLTK data (see requirements.txt header) +
`hf download Luigi/PrimeTTS --local-dir models/PrimeTTS`.
Run all data scripts with `.venv/bin/python`. PrimeTTS pipeline = encoder→host-regulate→decoder→vocoder,
frontend = bopomofo (g2pw) + arpabet (g2p_en), imported from `models/PrimeTTS/scripts/`.

## Training box
`ssh louis@keystone-ubuntu` (key auth). GTX 1070 8 GB (Pascal sm_61, driver 535 → CUDA 12.x runtime),
6 cores, 62 GB RAM, 245 GB free. Base Python 3.8 is too old — make a fresh Python 3.10/3.11 env
(miniconda/uv) with `torch==2.x+cu121`, transformers, peft, trl, accelerate, datasets, soundfile.
No nvcc needed (cu121 wheels are self-contained). If 8 GB OOMs or epochs too slow → escalate run to HF Jobs.

## Immediate next steps
1. Run full synthesis + augmentation locally (background): `.venv/bin/python data/synthesize.py` then
   `.venv/bin/python data/augment.py --variants 3`. Produces `data/audio/{train,val}.jsonl`.
2. On keystone: set up training env; rsync/clone repo + transfer `data/audio/` (or re-synthesize there).
3. Build `resolver.py` and the dialog controller.
4. Train Ultravox (frozen encoder+LLM, projector+LoRA); evaluate on val split.
