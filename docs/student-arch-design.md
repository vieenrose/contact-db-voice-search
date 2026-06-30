# Student architecture design — 0.6–0.8B zh/en agentic S2S (deep, config-grounded)

*Design note — 2026-06-30. Grounds the student model for distilling
[Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) into a real-time
zh/en agentic speech-to-speech model for Jetson Nano gen1. Numbers below are read from the **actual
configs** (Qwen3-ASR-0.6B locally; Qwen3-Omni `config.json`), not estimated. Companion to
`distill-qwen3omni-to-0.6b-s2s.md` and `feasibility-0.6b-agentic-audio-lm.md`.*

## Grounded reference dimensions (from real configs)
| Block | Teacher (Qwen3-Omni-30B-A3B) | Student source (Qwen3-ASR-0.6B) |
|---|---|---|
| **Thinker** | MoE d=2048, **48 layers**, 128 experts / 8 active | — (distill into the student backbone) |
| **AuT audio encoder** | d=1280, 32 L, out=2048 | **d=896, 18 L, out=1024**, 128 mel, windowed attn (n_window 50/800) — *streaming, reuse* |
| **Backbone / decoder** | (the Thinker) | **Qwen3 d=1024, 28 L, 16/8 GQA, head 128, ffn 3072, vocab 151936** |
| **Talker** | d=**1024**, 20 L, codec-vocab 3072 | — (absorbed into the student backbone) |
| **MTP "code predictor"** | **5 L, d=1024, 16 heads**, num_code_groups 16 | — (**graft/warm-start** — dims already match d=1024) |
| **Codec** | **16 codebooks @ 12.5 Hz** (1 semantic@4096 + 15 acoustic@2048) | — |
| **Code2Wav vocoder** | conv, decoder_dim 1536, 8 L, upsample 8·5·4·3·2·2 → **24 kHz** | — (**reuse**) |

> **The alignment gift:** teacher Talker / MTP / codec-embeddings are **all d=1024 = the student backbone
> hidden size**. The teacher's MTP head and Code2Wav are **dimension-compatible** with the student — graftable
> with no projection surgery. And the codec is **12.5 Hz**, exactly the rate the Nano real-time budget assumed.

## Top-level shape: a **single-backbone** mini Thinker–Talker (not the teacher's two-tower)
The teacher runs two big transformers (48-L Thinker → 20-L Talker). On a 0.6B Nano budget we **cannot**
afford two towers, so the student collapses to **one backbone** that emits the text stream *and* drives a
**small audio head** (Mini-Omni unification). Distillation maps teacher *(Thinker ⊕ Talker)* → student
*backbone*, and teacher *MTP/codec/Code2Wav* → student *audio head/vocoder* (transferred ~directly).

```
 speech ─▶ AuT enc (d896→1024, frozen, streaming) ─▶ ┌ Qwen3 d1024/28L backbone ┐─▶ TEXT: <tool_call>+reply ─▶ resolver/DB
                                                      │   (text+audio temporal) │╲
                                                      └──────────────────────────┘ ╲─▶ MTP head (5L d1024) ─▶ 16 codebooks @12.5Hz
                                                                                          └▶ Code2Wav (8L conv) ─▶ 24 kHz speech
```

## Component design decisions

### 1. Audio **input** — reuse Qwen3-ASR's AuT (settled)
d=896 / 18 L, **output_dim 1024 = backbone hidden** ⇒ projector is trivial/already trained; windowed
attention (n_window 50 stream / 800 infer) gives streaming prefill. **Frozen.** No input alignment to learn
(this is why we warm-start from Qwen3-ASR, not a bare text LM). ~186M.

### 2. **Backbone** — Qwen3-0.6B (default) vs Qwen3.5-0.8B (capacity lever)
Qwen tokenizer is stable at **151,936/151,669** across Qwen2/3/3.5 ⇒ white-box logit-KD with the
Qwen3-Omni teacher survives either choice.

| | **Qwen3-0.6B** (via `Qwen3-ASR-0.6B-Agent`) | **Qwen3.5-0.8B** (text) |
|---|---|---|
| Audio-input alignment | ✅ free (AuT bonded, projector trained) | ❌ retrain AuT→backbone projector (Ultravox-style; machinery exists) |
| Agentic warm-start | ✅ our agent LoRA (94% / 99% zh) | ❌ from text FC only (newer/stronger, not yet speech-agentic) |
| Capacity (#1 risk) | tighter | **+33%** — eases text⊕audio⊕agency contention |
| **Nano real-time** (measured-anchored 63% BW util) | INT8 **RTF 0.47** / INT4 0.23 | INT8 **RTF 0.62** / INT4 0.31; RAM 1.65 GB fits |

**Decision: default Qwen3-0.6B; Qwen3.5-0.8B is the upgrade lever** — the expensive work (harvest, audio
head, codec, data) is backbone-agnostic and transfers, so start on 0.6B (free alignment + agency + most
headroom); **swap to 0.8B if P1 text-KD can't match the teacher or P2 audio contends with agency** (keep
AuT, retrain only the projector; ship INT4 on Nano). If 0.8B, generation-match the teacher to Qwen3.5-Omni-Flash.

### 3. Audio **output representation** — use the **teacher's codec** (16 cb @ 12.5 Hz), with a Nano lever
Because the teacher's codec is **12.5 Hz** (backbone RTF unchanged) and its MTP/embeddings are **d=1024**:
- **Adopt the teacher codec** ⇒ enables **token-level audio KD** (per-codebook KL vs the teacher), **warm-start
  the MTP head** from the teacher's 5-L predictor, and **reuse Code2Wav** — three big distillation wins.
- **Cost:** 16 codebooks/frame is **2× Mimi's 8** on the *head* (not the backbone). **Nano lever:** keep only
  the **first K=8 of 16** codebooks (the semantic + coarse-acoustic, which carry most intelligibility/tone);
  Code2Wav can decode a reduced stack with graceful quality loss. **Fallback:** Mimi (8 cb @ 12.5 Hz,
  already gated zh-tone F0-corr 0.994) if Code2Wav won't fit Nano — but then audio distill is sequence-level only.

### 4. Audio **output head** — MTP depth transformer (warm-started), of three options
| Option | Shape | Params | Nano cost | Quality |
|---|---|---|---|---|
| A. Sub-LM heads (Mini-Omni2) | N parallel heads off the backbone | ~few M | lowest | most backbone contention |
| **B. MTP depth (Moshi/Qwen3-Omni)** ✅ | backbone hidden → **small 2–5 L d1024** → N codebooks | ~20–60M | low–moderate | **warm-startable from teacher's 5-L MTP** |
| C. + small Talker | add 4–6 L audio transformer before MTP | +~70M | higher | best, offloads audio from backbone |
**Decision: B**, initialized from the teacher's 5-L d1024 code predictor (dims match). Predicts the
N codebooks/frame conditioned on the backbone hidden ⊕ the emitted **text** token (grounding). Escalate to
**C** only if 0.6B/0.8B backbone audio quality is insufficient and INT4 affords the headroom.

### 5. **Vocoder** — reuse teacher Code2Wav (8-L conv, 24 kHz, left-context streaming)
Frozen; "lightweight causal ConvNet" built for first-frame streaming. ~tens of M. PrimeTTS's HiFiGAN ran on
Nano at RTF 0.35, so an 8-L conv vocoder at 12.5 Hz is plausibly Nano-viable — **Gate: benchmark Code2Wav
on Maxwell**; Mimi's conv decoder is the fallback.

### 6. **Text↔audio interleave** — Mini-Omni delay pattern + phase split
Backbone emits a text token per step; the MTP emits that step's 16 (or K) codebooks for the **delayed** text
(1–2 frame delay) so audio renders already-decided, grounded text. **Phase split:** `<tool_call>` /
`<tool_response>` are **text-only**; audio turns **on** only for the user-facing **reply** ⇒ agency stays
pure-text/grounded, only the answer is spoken.

## Nano real-time budget — refined with the real codec
- **Backbone:** 12.5 passes/s (codec = 12.5 Hz, confirmed) ⇒ INT8 **RTF 0.47** (0.6B) / 0.62 (0.8B),
  empirically anchored (measured 0.6B Q8_0 = 44.6 tok/s on i5 ⇒ ~26.9 tok/s Nano). **Unchanged — the binding
  constraint is settled.**
- **NEW variable — MTP head cost (Gate G1b):** 16 codebooks/frame via a 5-L d1024 head. Per-frame head passes
  depend on the MTP's grouping (`num_code_groups 16`): if it emits all 16 in a few grouped steps (Moshi-style),
  head load is small (~tens of M × a few × 12.5/s); if near-sequential, it grows. **Measure MTP forward-passes/
  frame before committing 16 vs K=8 codebooks.** This — not the backbone — is the remaining Nano-cost unknown.
- **Vocoder:** Code2Wav once/frame → 1920 samples; benchmark on Maxwell (Gate).

## Quantization & runtime split (Nano)
- **Backbone:** INT8 (proven on our decoder; INT4 for 0.8B headroom). FP16 fails real-time (RTF ~1).
- **AuT encoder + MTP head + Code2Wav:** fp16 (small; sub-1B quant craters audio detail — keep audio path fp16).
- **Runtime:** GGUF + Maxwell-compatible llama.cpp (sm_53) for the backbone; ONNX-Runtime (old CUDA EP) for AuT
  + MTP + Code2Wav; a **custom C++ streaming loop** ties them (parallel text/audio decode, delay pattern, audio
  I/O). The loop is the hardest single piece — harder than training.

## How the architecture enables distillation (attachment points)
1. **AuT shared** → no input alignment; warm-start perception from Qwen3-ASR.
2. **Text logit-KD** → shared 151936 vocab ⇒ KL-match the Thinker's `<tool_call>`+reply distributions.
3. **Audio codebook-KD** → student adopts the teacher codec ⇒ per-codebook KL vs the Talker/MTP.
4. **MTP warm-start** → teacher's 5-L d1024 code predictor grafts onto the d1024 student.
5. **Code2Wav reuse** → no vocoder to train.
⇒ The *only* genuinely-trained-from-scratch capacity is the backbone's new audio-driving behavior + the
MTP fine-tune; everything else is transfer.

## Recommended configuration (the concrete pick)
```
AuT (Qwen3-ASR, d896→1024, frozen, fp16)
  → Qwen3-0.6B backbone (d1024/28L, INT8, warm-start = Qwen3-ASR-0.6B-Agent)     ← lever: Qwen3.5-0.8B
     → text stream: <tool_call> + reply  (grounded; tool_call text-only)
     → MTP depth head (5L d1024, fp16, warm-start = teacher code predictor)
        → teacher codec, K=8 of 16 codebooks @12.5Hz (lever: full 16 if Nano affords)
        → Code2Wav (8L conv, fp16, reuse) → 24 kHz speech
  text↔audio: Mini-Omni delay; speech only on the reply.
```
**Total trainable "brain" ≈ 0.6B backbone + ~40M MTP; ~0.85B incl. frozen AuT + Code2Wav.** Real-time
target INT8 RTF ≈0.5 (backbone) + measured MTP/vocoder head (Gate G1b).

## Open gates (in priority order)
1. **G1b — MTP forward-passes per frame** (16 vs K=8 codebooks): the remaining Nano-cost unknown; read from
   the teacher's Talker/MTP code, then re-budget. Decides codebook count.
2. **Code2Wav on Maxwell** — benchmark; else Mimi decoder fallback.
3. **Capacity** — P1 text-KD eval decides 0.6B vs 0.8B backbone.
4. **K-codebook intelligibility/tone** — does an 8-of-16 reduced stack keep zh tones? (reuse the Mimi-gate F0/CER method).

## Sources
- [Qwen3-Omni `config.json`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json) · [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- [Moshi/Mimi (depth transformer, 12.5 Hz)](https://kyutai.org/Moshi.pdf) · [Mini-Omni2 (0.5B, sub-LM heads)](https://arxiv.org/html/2410.11190v1) · [SLAM-Omni (grouped tokens, 0.5B)](https://arxiv.org/abs/2412.15649)
- Local: Qwen3-ASR-0.6B `config.json` · ours: [`distill-qwen3omni-to-0.6b-s2s.md`](./distill-qwen3omni-to-0.6b-s2s.md) · [`feasibility-0.6b-agentic-audio-lm.md`](./feasibility-0.6b-agentic-audio-lm.md)
