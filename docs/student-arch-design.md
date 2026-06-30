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

### 3. Audio **output representation** — teacher codec (16 cb @ 12.5 Hz); **Code2Wav is fixed-16 (no K-truncation)**
Teacher codec is **12.5 Hz** (backbone RTF unchanged) and its MTP/embeddings are **d=1024** ⇒ token-level
audio KD + graftable head + reusable Code2Wav. **But the released `Code2Wav.forward` hard-requires all 16
codebooks and *mean-pools* them** (`if codes.shape[1]!=16: raise`; `code_embedding(codes+offset).mean(1)`) —
it is **not** a hierarchical RVQ you can truncate to K=8. So the earlier "K=8 lever" is **void for Code2Wav
reuse**: to reuse Code2Wav you must emit **all 16** codebooks. Also: **no codec *encoder* is released**
(only the AuT understanding-encoder + the Code2Wav decoder), so encoding real audio → teacher codes needs
the 30B Talker ⇒ exact teacher-codec quality tests belong in **P0** (when the teacher is loaded to harvest).
Consequence ⇒ see §4: the only Nano-real-time way to emit all 16 is **parallel heads**, not AR-MTP.

### 4. Audio **output head** — **shallow MTP** (Gate G1b resolved from the teacher's code)
**The decisive measurement.** Qwen3-Omni's code confirms the MTP runs **autoregressively, one forward
pass per residual code group** (`generation_steps 0..15`, RVQ codebook *i* needs 0…*i*−1 ⇒ inherently
sequential): per frame = **1 backbone/Talker pass + (K−1) MTP passes**. The teacher's MTP is 5 L, d1024,
ffn3072 (~52M core; ~115M with embeds+heads ≈ LFM2's RQ-transformer). Nano bandwidth cost (INT8,
measured-anchored 16.1 GB/s effective; backbone 7.5 GB/s + Code2Wav ~1.0):

| Head config | GB/s | **RTF** | |
|---|---|---|---|
| **Teacher's head: 5-L MTP, K=16** | 18.4 | **1.14** | ❌ not real-time (MTP > backbone!) |
| 5-L MTP, K=8 | 13.1 | 0.82 | tight |
| 3-L MTP, K=8 | 11.3 | 0.70 | tight |
| **2-L MTP, K=8** ✅ | 10.4 | **0.64** | ✅ keeps some inter-codebook AR |
| 2-L MTP, K=4 | 9.3 | 0.58 | ✅ |
| **Parallel heads (no AR), any K** | 8.5 | **0.53** | ✅ fallback — cheapest, drops AR |

**Decision (corrected after the Code2Wav finding): PARALLEL 16-heads → reuse Code2Wav.** Because Code2Wav
needs **all 16** codebooks (no truncation), the choices collapse to:
- **Autoregressive MTP, all 16:** RTF **1.14** — ❌ fails Nano (and K-truncation to fix it would require a
  *retrained* Code2Wav-K, since the released one is fixed-16).
- **Parallel heads, all 16 (no AR):** one cheap pass/frame predicting all 16 codebooks via 16 linear heads
  off the backbone hidden → **RTF 0.53**, **reuses Code2Wav as-is**, Nano-real-time. ✅ **the pick.**
The cost of parallel heads is dropping **inter-codebook autoregression** (each codebook predicted
independently given the frame's text+hidden) — a known quality trade (MiMo-Audio/Mini-Omni2 use it). Whether
that hurts zh tone/clarity is a **P0 test on the teacher codec** (parallel-vs-AR decode). **Fallback if
parallel quality is insufficient or Code2Wav won't run on Maxwell:** **Mimi 8 cb** (hierarchical, truncatable,
its own small decoder, already zh-tone-gated F0-corr 0.994) — at the cost of losing teacher-token-KD + Code2Wav.

### 5. **Vocoder** — reuse teacher Code2Wav (8-L conv, 24 kHz, left-context streaming)
Frozen; "lightweight causal ConvNet" built for first-frame streaming. ~tens of M. PrimeTTS's HiFiGAN ran on
Nano at RTF 0.35, so an 8-L conv vocoder at 12.5 Hz is plausibly Nano-viable — **Gate: benchmark Code2Wav
on Maxwell**; Mimi's conv decoder is the fallback.

### 6. **Text↔audio interleave** — **MCTP zero-delay** (VITA-Audio) + phase split
*Upgraded from the Mini-Omni 1–2 frame delay after the best-arch survey below.* Use **VITA-Audio's MCTP**
(Multiple Cross-modal Token Prediction): a lightweight module emits the audio codes **in the same forward
pass** as the text token — **zero audio-token delay**, minimal first-audio latency (the priority on Nano).
Audio is still **conditioned on the just-decided text token** (grounding preserved). **Phase split:**
`<tool_call>` / `<tool_response>` are **text-only**; audio turns **on** only for the user-facing **reply**
⇒ agency stays pure-text/grounded, only the answer is spoken.

## Best-architecture survey (2026 landscape) — and why this design wins for *CPU-edge real-time*
The output-head scheme is the decisive choice. Surveyed against the field:

| Model | Audio-out head | First-audio latency | Edge/CPU real-time fit |
|---|---|---|---|
| Mini-Omni / Mini-Omni2 | parallel sub-LM heads + **delay pattern** | delayed | ok, but delay adds latency |
| Moshi / Qwen3-Omni | **depth/MTP transformer** (8–16 cb) | 1-frame | strong (our teacher) |
| **LFM2** | **RQ-transformer** (115M, 8 steps/frame) | **low TTFA** | **explicitly "real-time on commodity CPUs", beats delay patterns** |
| Baichuan-Audio | 3-L depth transformer + 8 heads | low | strong |
| **VITA-Audio** | **MCTP — audio in the *first* forward pass** | **zero-delay** | **best first-token latency** |
| MiMo-Audio | independent per-RVQ heads | — | parallel, simple |
| CosyVoice2 / GLM-4-Voice | single-codebook + flow-matching detokenizer | higher | heavier detokenizer, less edge |
| TinyWave | (distilled 7B→2B S2S, block pruning) | — | precedent for *S2S distillation* |

**Verdict — the best architecture for *this* student is convergent, not exotic:**
1. **Head = small RQ/depth transformer** over the teacher codec — **LFM2 proves this is the real-time-on-CPU
   winner** (115M head vs 1.2B backbone; chosen *over* delay patterns precisely for low TTFA on CPUs), and it's
   what Moshi/Qwen3-Omni/Baichuan use. Warm-start it from the teacher's 5-L d1024 MTP. *(This is Option B — now
   field-confirmed as best-for-edge, not just convenient.)*
2. **Interleave = VITA-Audio MCTP (zero-delay)** — emit audio in the first forward pass; the lowest first-audio
   latency available, which is exactly the Nano priority.
3. **Codec = the teacher's** (white-box audio KD + graftable d1024 MTP + reusable Code2Wav) — no one else's
   pairing gives all three transfer wins for *our specific teacher*.
4. **Distillation = cross-modal KD** (T→T logit-KD on shared vocab + S→T/synthesized-speech alignment), the
   recipe TinyWave/SPIRIT-LM and the cross-modal-KD papers validate for S2S-into-small.

So the "best arch" is **single-backbone + RQ/depth head (LFM2-style) + MCTP zero-delay (VITA-Audio) + teacher
codec + Code2Wav reuse** — each choice independently the edge/real-time optimum *and* the max-distillation-transfer
option for the Qwen3-Omni teacher. The earlier draft already had the head right; the survey upgrades the
*interleave* (delay → zero-delay MCTP) and confirms the head against the 2026 field.

## Nano real-time budget — refined with the real codec
- **Backbone:** 12.5 passes/s (codec = 12.5 Hz, confirmed) ⇒ INT8 **RTF 0.47** (0.6B) / 0.62 (0.8B),
  empirically anchored (measured 0.6B Q8_0 = 44.6 tok/s on i5 ⇒ ~26.9 tok/s Nano). **Unchanged — the binding
  constraint is settled.**
- **MTP head cost (Gate G1b — RESOLVED, see §4):** the teacher's MTP is **autoregressive, K−1 sequential
  passes/frame**. Full 5-L×16 head = **RTF 1.14 (not real-time)**; the chosen **2-L MTP @ K=8 = RTF 0.64**;
  parallel-heads fallback = 0.53. The audio head is the *second* binding cost after the backbone — both must
  be shrunk (shallow MTP + reduced K), and the head must be INT8 like the backbone (fp16 head ≈ 2× the GB/s).
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
     → PARALLEL 16-heads (one pass/frame, INT8; RTF 0.53 — Code2Wav needs all 16, AR-MTP would be RTF 1.14)
        → teacher codec, all 16 codebooks @12.5Hz
        → Code2Wav (8L conv, fp16, reuse as-is) → 24 kHz speech
  text↔audio: VITA-Audio MCTP — ZERO-delay (audio in first forward pass); speech only on the reply.
  fallback codec: Mimi 8cb (hierarchical, own decoder, zh-tone-gated) if parallel-quality/Code2Wav-on-Maxwell fail.
```
**Total trainable "brain" ≈ 0.6B backbone + ~40M MTP; ~0.85B incl. frozen AuT + Code2Wav.** Real-time
target INT8 RTF ≈0.5 (backbone) + measured MTP/vocoder head (Gate G1b).

## Open gates (in priority order)
1. ✅ **G1b — MTP passes/frame & head design: RESOLVED from the code.** AR-MTP = K−1 sequential passes ⇒
   all-16 = RTF 1.14 (no-go); and Code2Wav is **fixed-16 mean-pool** (no K-truncation) ⇒ **parallel 16-heads
   → Code2Wav, RTF 0.53** is the pick. Head INT8.
2. **Parallel-vs-AR audio quality (P0 test)** — does dropping inter-codebook AR (parallel 16-heads) keep zh
   tone/clarity? Needs the teacher codec (no encoder released) ⇒ test during P0 harvest: decode teacher codes
   parallel-style vs AR, F0-corr + CER. **Top open gate.** Mimi-8 fallback already F0-gated.
3. **Code2Wav on Maxwell** — benchmark the 8-L conv vocoder; else Mimi decoder fallback.
4. **Capacity** — P1 text-KD eval decides 0.6B vs 0.8B backbone.

## Sources
- [Qwen3-Omni `config.json`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json) · [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- [Moshi/Mimi (depth transformer, 12.5 Hz)](https://kyutai.org/Moshi.pdf) · [Mini-Omni2 (0.5B, sub-LM heads)](https://arxiv.org/html/2410.11190v1) · [SLAM-Omni (grouped tokens, 0.5B)](https://arxiv.org/abs/2412.15649)
- Best-arch survey: [VITA-Audio (MCTP, zero-delay)](https://arxiv.org/html/2505.03739) · [Kimi-Audio](https://arxiv.org/abs/2504.18425) · [LFM2 RQ-transformer head](https://arxiv.org/pdf/2511.23404) · [Baichuan-Audio](https://arxiv.org/pdf/2502.17239) · [TinyWave (S2S distillation)](https://arxiv.org/pdf/2506.23670)
- Local: Qwen3-ASR-0.6B `config.json` · ours: [`distill-qwen3omni-to-0.6b-s2s.md`](./distill-qwen3omni-to-0.6b-s2s.md) · [`feasibility-0.6b-agentic-audio-lm.md`](./feasibility-0.6b-agentic-audio-lm.md)
