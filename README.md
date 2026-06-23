# Voice Extension Finder — zh-TW/en Phone Attendant

An **audio-in → text-out** speech LLM that listens to a phone caller asking for a
person and returns that person's **extension number**. Taiwan-office scenario:
staff are usually referred to by an **English first name + Chinese surname**
("Kevin Chen", "找陳凱文"), often **code-switched** mid-sentence.

## Scenario constraints

- **Deployment target:** NVIDIA **Jetson Nano gen1 (2019, 4 GB)** — hard-capped at
  JetPack 4.6.4 / L4T r32.7 / **CUDA 10.2** / Ubuntu 18.04 / Python 3.6, 128-core
  Maxwell GPU. Inference path on this board is effectively **CPU (llama.cpp)**.
- **Use case:** telephone auto-attendant. **Free dialog, no wake word** — the phone
  call itself is the session. Audio arrives as **8 kHz narrowband** (telephony).
- **Directory:** small, **closed set (~200 people)**.

## Architecture (committed)

```
Caller → PBX (Asterisk) → AudioSocket(PCM 8k)
   → resample 16k → Ultravox-1B (audio-in → text-out)  → {"name","ext"}
   → confirm + transfer  (TTS read-back from pre-rendered prompts)
```

- **Model:** **Ultravox v0.5 (Llama-3.2-1B backbone)** = Whisper encoder → multimodal
  projector → 1B LLM, no separate ASR stage. Chosen for: trainable (public LoRA
  recipe), small, and has GGUF + llama.cpp audio (`libmtmd`) deployment path.
- **Output:** structured text, e.g. `{"name":"Kevin Chen","ext":"1234"}`.

### Known risk — latency on gen1
The stock Ultravox encoder is whisper-large-v3-turbo (~800M); on the A57 CPU it,
not the 1B LLM, dominates latency (~3–8 s end-to-end). Mitigation: retrain the
projector against a smaller `whisper-base/small` encoder. llama.cpp audio is still
experimental — **validate deployment early.**

## Roadmap

- [x] 1. Mock 200-person zh-TW/en directory — `data/generate_directory.py` (unique en+zh names)
- [x] 2. Synthetic request transcripts (templates × code-switch × disfluencies) — `data/generate_requests.py` (0 ambiguous, +OOD negatives)
- [x] 3. PrimeTTS synthesis → telephony augmentation (8 kHz / μ-law / band-pass / noise) — `data/synthesize.py`, `data/augment.py`
- [ ] 4. Ultravox fine-tune — **off-box (no GPU here):** HF Jobs or remote GPU. Freeze whisper-**base** encoder, train projector + LLM-LoRA.
- [ ] 5. GGUF export → `llama-mtmd-cli` on gen1, wired to Asterisk AudioSocket
- [ ] 6. Latency + accuracy validation on-device

## Key decisions (this build)

- **Encoder:** retrain projector against **whisper-base** (not stock large-v3-turbo) for gen1 latency
  (~5–15 s → target ~1–2 s). Keep encoder **frozen** so it retains real-speaker robustness despite
  single-voice synthetic training.
- **Latency truth:** stock Ultravox-1B is **not** real-time on gen1; base-encoder + endpointing is the
  path to acceptable. Genuine sub-second wants an Orin.
- **TTS:** PrimeTTS (`Luigi/PrimeTTS`), single young-female zh-TW voice, 8 kHz. Single-voice gap closed by
  speed/formant augmentation + frozen pretrained encoder + (later) real-call fine-tune.
- **Dialog = hybrid.** Ultravox does *perception* (audio → decision object); a deterministic controller
  owns *policy* (confirm-and-transfer / clarify via directory fields / not-found repair / operator escape).
  Not letting a 1B free-run policy on a phone line.
- **Training compute:** this machine is **CPU-only** → data-gen + CPU deploy-emulation here; training off-box.

## Env

`.venv` (`--system-site-packages`) adds: g2pw (+torch-cpu), g2p_en (+nltk data), cn2an. Run all data
scripts with `.venv/bin/python`.

## Layout

```
data/
  generate_directory.py   # mock contact directory
  directory.csv           # generated 200-person directory
  generate_requests.py    # synthetic request texts + labels (step 2)
```
