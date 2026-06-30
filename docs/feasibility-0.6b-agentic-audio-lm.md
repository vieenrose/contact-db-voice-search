# Feasibility: a 0.6B end-to-end zh/en **agentic audio LM** (an "LFM2.5-Audio, but ours")

*Research note — 2026-06-30. Can we rebuild an LFM2.5-Audio-style end-to-end audio-in/audio-out model,
but bilingual (zh-TW/en) and agentic (tool-calling), at the 0.6B scale? Verdict, evidence, and a
staged plan grounded in what we already have.*

## Verdict (up front)
**Feasible — and we are unusually well-positioned — *for the narrow agentic-attendant domain*.** The
open-domain quality of a 0.5–0.6B end-to-end speech model is well below a 1.5B one, but our task
(short, templated, closed-set replies: names + extensions) is far easier than open VoiceBench chat,
which is exactly where a small model can still be good. The genuinely new engineering is **one module**
(an audio-output head + neural codec) and **one training run**; three of the five components already
exist in this repo. The two real risks are **edge sustained-latency** and **zh-TW codec/tone fidelity**.

## Evidence: sub-1B end-to-end speech models already exist
| Model | Size | What it proves |
|---|---|---|
| **Mini-Omni2** | **Qwen2-0.5B** + Whisper-small enc + 8 sub-LM audio heads | 0.5B **speech-in→speech-out** works end-to-end, no external ASR/TTS; **text-audio parallel decoding** (audio conditioned on text) |
| **SLAM-Omni** | **0.5B**, semantic tokens + timbre-to-vocoder | **single-stage** training, **15 h on 4 GPUs**, no TTS/ASR pretraining needed; grouped tokens shorten the audio sequence |
| **Qwen3-Omni** | (larger) | **First native speech-to-speech model with function calling**, via the *same* `<tool_call>` XML our agent emits — proves **agentic + S2S** is real |
| **Moshi** | 7B temporal + small **depth transformer** | The audio-output recipe: a **small depth transformer** emits all **8 Mimi codebooks per 12.5 Hz frame** in parallel; ~160 ms theoretical latency |
| LFM2.5-Audio | 1.5B | The quality target; shows what 0.6B gives up (VoiceBench ~33 at 0.5B vs ~55 at 1.5B) on *open* domain |

Takeaway: **0.6B e2e S2S is a solved size class; agentic S2S is a solved pattern; nobody has combined
them at 0.6B for zh/en — that's the open niche.**

## The five components, and the 0.6B budget
An e2e audio LM = `audio encoder (in) → temporal LM backbone → audio-output head → neural codec (out)`,
plus the agentic text path. Param budget at the "0.6B-backbone" class (encoder + codec are bolt-on
frozen modules, as in Mini-Omni2):

| Component | Choice | Params | Status |
|---|---|---|---|
| Audio **in** | **AuT** (Qwen3-ASR encoder) | 186M, frozen | ✅ **have it** — multilingual, zh/en, already aligned to the Qwen3 decoder |
| Temporal **backbone** | **Qwen3-0.6B decoder** | 0.6B | ✅ **have it** — and it is **already agentic** (tool calls, multi-turn) |
| Agentic text path | `<tool_call>` + resolver | — | ✅ **have it** — `Qwen3-ASR-0.6B-Agent` + `resolver.py` |
| Audio **out head** | small **depth transformer** (Moshi) or **8 sub-LM heads** (Mini-Omni2) | ~tens of M | ❌ **new** |
| Neural **codec** | **Mimi** (12.5 Hz, 8 codebooks, 1.1 kbps), frozen | ~tens of M, pretrained | ❌ **integrate** |
| Audio teacher | **PrimeTTS v1b_16k** | ~5M | ✅ **have it** — becomes the audio-target generator (below) |

**We already own 3 of 5** (encoder, agentic backbone, TTS teacher) **plus the data pipeline.** Mini-Omni2
adds audio output to a *0.5B* backbone with just 8 extra heads and it works — so the capacity math
closes at our scale.

## The two missing pieces

### 1. Audio output: low-frame-rate codec + a depth head (this is what makes 0.6B affordable)
The cost of "the LM speaks" is set by the **frame rate**, not the codec's 8 codebooks. **Mimi runs at
12.5 Hz**, and a **small depth transformer emits all 8 codebooks per frame in parallel** (Moshi's
design). So a 4 s reply = **~50 temporal steps** of the 0.6B backbone — *the same order as generating
the ~50–80 text tokens we already generate today.* Audio-out roughly **doubles** a turn's backbone
compute; it does not 10× it. That's the key number that makes 0.6B e2e viable.

### 2. Grounding-preserving generation: audio **conditioned on** the grounded text
The danger of an audio LM is hallucinated content (a wrong extension, confidently spoken). The fix is
already invented: **Mini-Omni's text-audio parallel decoding** — the model emits the **text reply
(tool-grounded) and the audio codes for that same text in lockstep** (8 sub-heads/step, one-step delay
between layers). So the **audio renders the grounded text**; it cannot drift to a different number.
Keep `<tool_call>` and the reply **text** exactly as today (grounded by the resolver) and bolt audio
on as a text-conditioned stream. **Agency and grounding are untouched; only the rendering changes.**

## The data shortcut that makes this cheap: PrimeTTS as the audio teacher
We do **not** need real speech-to-speech data. We already have **13.5k agentic dialogs with grounded
reply text**. Recipe:

1. Render every reply with **PrimeTTS v1b_16k** → reply audio (zh-TW/en, tone-correct, entity-correct).
2. Encode that audio with **Mimi** → target codebook token streams.
3. Warm-start from **`Qwen3-ASR-0.6B-Agent`**; train it to emit those Mimi tokens **in parallel with
   the (unchanged) reply text**, given `input audio → <tool_call> → <tool_response> → reply`.
4. Mix in **ASR replay** (input audio → text) so perception/agency don't regress.

This is **distilling PrimeTTS into the LM's audio output**, text-conditioned. Single-stage SFT — SLAM-Omni
shows single-stage works at 0.5B in **~15 h on 4 GPUs from scratch**; warm-started from our agent it
should be cheaper, and the LoRA that made the agent ran on a single GTX-1070.

## Recommended minimal architecture
```
 user speech ─▶ AuT encoder (186M, frozen) ─▶ ┌─ Qwen3-0.6B backbone (warm-start: the Agent) ─┐
                                              │   ├─ text:  <tool_call> ─▶ resolver (DB) ─▶ <tool_response>
                                              │   ├─ text:  grounded reply  ───────────────┐
                                              │   └─ depth transformer (new, ~tens of M) ──┤ (audio conditioned on the text)
                                              └────────────────────────────────────────────┘
                                                          │ 8 Mimi codebooks @ 12.5 Hz
                                                          ▼
                                                  Mimi decoder (frozen) ─▶ speech out
```
- **Text path = today's system** (agentic, grounded, deterministic extensions).
- **Audio path = new**, text-conditioned, distilled from PrimeTTS. Optional: drop PrimeTTS at inference
  once the LM's own audio is good, or keep it as a fallback.

## Risks & mitigations (honest)
1. **Capacity at 0.6B (main quality risk).** Open-domain 0.5B S2S is weak (VoiceBench ~33). *Mitigation:*
   our domain is **narrow** — short templated replies over a **closed** name set; the audio "vocabulary"
   to master is tiny. Narrow-domain 0.6B is far more forgiving than the open-domain benchmark gap implies.
2. **zh-TW codec / tone fidelity (main codec risk).** Mimi's **semantic** codebook is distilled from
   English WavLM; its zh semantic stream may be weaker, and Mandarin **tones** must survive tokenization.
   *Mitigation:* tones live in the **acoustic** codebooks (preserved by reconstruction); PrimeTTS targets
   are tone-correct so distillation transfers tone; if the semantic stream hurts zh, fine-tune Mimi's
   distillation on zh or switch to a multilingual codec (DualCodec/XCodec-class). **Validate codec
   round-trip on zh-TW first — this is the gating experiment.**
3. **Edge sustained latency.** ~0.6B forward × 12.5/s ≈ **borderline real-time on the weakest CPU**
   (Jetson Nano gen1), comfortable on any modest accelerator; **streaming** keeps *first-audio* low
   (~Moshi-class 160–320 ms). The cascade's PrimeTTS (RTF ~0.1) is still cheaper for pure TTS — e2e buys
   streaming/duplex at a higher sustained cost. Same trade we documented in `lfm2audio-vs-cascade.md`.
4. **Entity correctness.** Guaranteed **only if** audio stays text-conditioned (Mini-Omni parallel) and
   text stays tool-grounded. Design rule: **never let the audio stream free-run independent of the text.**

## Staged plan (crawl → walk → run)
- **Stage 0 — done.** Cascade baseline (`Qwen3-ASR-0.6B-Agent` + resolver + PrimeTTS). The reference.
- **Stage 1 — "the LM speaks" (the feasibility gate).** (a) Validate **Mimi round-trip on zh-TW** audio
  (PESQ/tone check) — *go/no-go*. (b) Manufacture Mimi targets from PrimeTTS over the 13.5k dialogs.
  (c) Warm-start the Agent, add the depth head, train text-conditioned audio-out. Success = LM-generated
  reply audio ≈ PrimeTTS quality, extensions still correct. Single GPU.
- **Stage 2 — streaming.** Emit audio while finishing text (Mini-Omni parallel/delay); measure
  first-audio latency; quantize (INT8 already proven on the decoder).
- **Stage 3 — duplex (optional).** Dual user/system streams + barge-in (Moshi-style); train a zh/en-tuned
  codec if Mimi's zh semantic stream proved weak in Stage 1.

## Should we, though? (strategic honesty)
The **cascade already does the job** and keeps a clean grounding seam. An e2e agentic audio LM buys
**streaming, barge-in, full-duplex turn-taking, and paralinguistics** — i.e. *natural conversation* —
and the research novelty of **the first 0.6B zh/en agentic S2S model**. It costs the clean text seam and
added complexity. So:
- **As a product** for "look up an extension, never say a wrong number," the cascade is arguably better.
- **As research / a conversational upgrade,** the 0.6B agentic audio LM is a genuinely novel, fundable
  artifact, and Stage 1 is cheap and de-risking. **Recommendation: do Stage 1** (it reuses everything we
  have and answers the hard questions — zh codec fidelity, capacity — for one GPU-day), then decide on 2–3.

## Sources
- [Mini-Omni2 (Qwen2-0.5B, audio out)](https://arxiv.org/html/2410.11190v1) · [Mini-Omni "thinking while talking"](https://arxiv.org/html/2408.16725v1)
- [SLAM-Omni (0.5B, single-stage)](https://arxiv.org/abs/2412.15649)
- [Moshi + Mimi codec (depth transformer, 12.5 Hz, 8 codebooks)](https://kyutai.org/Moshi.pdf) · [kyutai/mimi](https://huggingface.co/kyutai/mimi)
- [Qwen3-Omni — first S2S with function calling](https://github.com/QwenLM/Qwen3-Omni)
- Ours: [Qwen3-ASR-0.6B-Agent](https://huggingface.co/Luigi/Qwen3-ASR-0.6B-Agent) · [PrimeTTS](https://huggingface.co/Luigi/PrimeTTS) · [`lfm2audio-vs-cascade.md`](./lfm2audio-vs-cascade.md)
