# Audio-LLM (agentic + zh/en) for Jetson Nano gen1 — research verdict

*Multi-agent web research (12 agents, 163 lookups), 2026-06. Decision-critical claims
adversarially verified (3 refuted, 1 confirmed). Reviewer caveats appended at the end.*

## TL;DR
**Right model, wrong box.** There is **no better Qwen3.5-Omni-3B-class model** for this use
case, and **nothing of that class runs on a Jetson Nano gen1 (2019, 4 GB)**.
- Qwen3-Omni's floor is a **30B-A3B MoE** (smallest Q4 GGUF **18.6 GB** — all 30B weights
  resident; "A3B" is decode-FLOPs, not memory). Qwen3.5-Omni (Mar 2026) is **API-only**, and
  the open Qwen3.5 dense models (0.8/2/4/9/27B) are **image-text only, no audio in**.
- **Qwen2.5-Omni-3B (your LoRA-FT, 98.2% TW-Mandarin / 92.6% agentic) is still the single best
  model on all four axes** (true audio-LLM + native tool-calling + zh-TW name perception + edge-size).
- Nano gen1 fails on **memory** (3B LM + un-shrinkable Q8 audio mmproj + OS > 4 GB) and
  **latency** (the audio encoder — already your 84-129 s/turn bottleneck on 2 x86 vCPU — is far
  worse on the NEON-only Cortex-A57). **Keep the model; move inference off the Nano.**

## Model landscape (ranked for this use case)
The 4 constraints are conjunctive — ace 3, fail 1 → out.

| Rank | Model | Smallest audio variant | True audio-LLM | Agentic tool-call | zh-TW names | Edge/llama.cpp audio | Verdict |
|---|---|---|---|---|---|---|---|
| **1** | **Qwen2.5-Omni-3B (LoRA)** | 3B | Yes | **Yes (validated)** | **Excellent (validated)** | **Yes (mtmd, CPU)** | **KEEP — only one passing all 4** |
| 2 | MiniCPM-o 2.6 | ~8B | Yes | unproven | beats Qwen on CS3 code-switch (51.4 vs 42.0) | partial | perception reference; too big |
| 3 | Voxtral-Mini-3B | 3B | Yes | Yes (experimental) | **No Chinese** | Yes | right size, **no zh** |
| 4 | Phi-4-multimodal | 5.6B | Yes | **No** (MS: "Tool calling: No") | catastrophic code-switch | No | disqualified ×2 |
| 5 | Qwen3-Omni-30B-A3B | 30B MoE (no dense) | Yes | Yes (BFCL-v3 64.4) | Excellent | Yes but **18.6 GB Q4** | server-only |
| 6 | Gemma 3n/4-E | E2B (~2 GB) | Yes | weak | **FAILED ("李明")** | Yes | fits but broken on zh-TW |
| 7 | Step-Audio-2-mini | 8B | Yes | Yes | Excellent | No GGUF audio | strong, no edge path |
| — | Granite-speech, SenseVoice, Moonshine, Parakeet | 0.2-2.5B | **No (ASR only)** | No | varies | cheap CPU | cascade front-end only |

Most credible "could beat Qwen on names" signal: **MiniCPM-o 2.6 / Kimi-Audio** on CS3-Bench
code-switch (51.4 / 52.8 vs 42.0) — but ~7-8B, server-only, and that's *knowledge accuracy*,
not homophone-surname perception (see caveats).

## Qwen3 / 3.5-Omni vs your 2.5-Omni-3B
Genuine upgrades: new **AuT audio encoder** (~0.6B, 20M h, dynamic 1-8 s windows); zh CV15 4.28%
WER, **Qwen3.5 publishes CV-Taiwan 2.27% WER**; benchmarked tool-calling (BFCL-v3 64.4); **mtmd
audio IS supported in llama.cpp**. But the Qwen3 line **moved up in size** — no dense ≤30B Omni
exists. Worth switching **only with ≥20-24 GB RAM/VRAM**.

## Jetson Nano gen1 — infeasible (two independent axes)
**Memory (4 GB shared):** LM Q4_K_M 2.1 GB (IQ2_M floor 1.41) + **audio mmproj Q8 1.54 GB (no Q4
exists today)** + KV 0.3-0.6 + **OS ~1.3-1.5** ⇒ no usable-quality config fits. Corroborated:
on the real 2019 Nano even a **4B text-only** model loads only at Q2 and is "not usable" —
your audio workload is strictly larger.
**Latency:** encoder-bound; already 84-129 s/turn on 2 x86 vCPU; the A57 (no AVX, 128-bit NEON,
~1.43 GHz) is multiples slower → multi-minute/turn. The Maxwell GPU (sm_53) doesn't rescue it
(shared RAM, ~20% over CPU, and the **mtmd audio-under-CUDA path is unvalidated/buggy** — garbage
output seen even on a far stronger Orin, #15923). CUDA on gen1 needs a hand-patched, pinned,
text-only build (CUDA 10.2 has no bfloat16).

## Recommended path
**Model: stay on Qwen2.5-Omni-3B (LoRA).** Deployment, ranked:
- **A. Server-offload over Tailscale (recommended):** Nano/phone = dumb audio endpoint; model runs
  on your validated x86 (or a small GPU box) over a tunnel. Reuses the working path; a small GPU
  makes it interactive. *Caveat: as-is it's still 84-129 s/turn — interactive only after the
  encoder work below.*
- **B. Jetson Orin Nano (different chip, sm_87/CUDA 12):** ~27-28 tok/s on 3B text; on-device but
  validate the mtmd audio-under-CUDA path first. ~US$250-500.
- **C. Cascade: contextual-biasing zh ASR (Qwen3-ASR / FunASR / SenseVoice) + small text tool-caller,
  with the contact list as hotword bias.** *(Reviewer-upgraded above "fallback": hotword-biased ASR
  is a real homophone fix; the "李明" failure was a Gemma audio-LLM failure, not a biased-ASR one.)*
  Needs a zh-TW eval to confirm it holds homophone names.
- **D. Run on Nano gen1: don't.** Refuted on memory + latency; strictly worse than your x86 path.

**Attack latency where it lives — the audio encoder** (not decode): GPU encoder-offload on a real
CUDA-12 box, trim push-to-talk audio to cut chunks (~4 fixed × ~21 s), profile whether fewer/shorter
chunks keep 98.2% TW accuracy. **The "missing middle" is encoder-only offload** (encoder on a GPU,
decode local).

## Reviewer caveats (honesty pass)
- The "4-8 min/turn on A57" is **illustrative, not measured** (PassMark-based scaling chain). The
  *direction* (≫60 s) is safe; the exact minutes aren't.
- "mmproj non-quantizable below Q8" = **currently no Q4 published**, not a law of physics.
- CS3-Bench is **code-switch knowledge accuracy, not homophone-name perception** — weak as a
  "perceives names better" signal; a bake-off on *your* homophone set is the real test.
- The GPU/CUDA audio path is **"unvalidated," not provably "broken"** (one bug on one box).
- **Missed candidates worth a look:** **Qwen3-ASR**, **Ultravox** (3B-ish audio-LLM-via-projector,
  custom backbone, tool-use), **GLM-4-Voice**, Whisper-v3-turbo + biasing.
- **Double-check first:** does GPU encoder-offload + chunk-trimming get you under your interactive
  bar *without* dropping 98.2% TW accuracy? The whole recommendation hinges on it.

*Key sources: [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), [Qwen3-Omni-30B GGUF](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF), [Qwen2.5-Omni-3B GGUF](https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF), [kreier Jetson llama.cpp benchmarks](https://kreier.github.io/llama.cpp-jetson/), llama.cpp issues #15923/#13759/#18881, [AuT encoder arXiv 2509.17765](https://arxiv.org/html/2509.17765v1).*
