# LFM2.5-Audio vs. (Qwen3-ASR-0.6B-Agent + PrimeTTS): end-to-end audio LM vs. grounded cascade

*Research note — 2026-06-30. Comparison of Liquid AI's end-to-end speech LM against our modular
voice-attendant stack, with the architectural reasons behind the latency / quality / agency trade-offs.*

## TL;DR
They are two different **design philosophies**, not two points on one axis:

- **LFM2.5-Audio-1.5B** — a *fused, audio-native generalist*: one network takes audio in and emits
  audio out (no separate ASR/TTS). Best at open-ended, low-latency, full-duplex **English** voice chat.
- **Ours** — a *text-grounded specialist cascade*: a discriminative speech encoder + a small LLM that
  emits **text** (tool calls + reply), a **deterministic retrieval tool**, and a tiny TTS. Best at a
  **grounded, bilingual, transactional** task where the answer must come from a database, never be
  hallucinated.

For "find a colleague's extension in a zh-TW office and never say a wrong number," the cascade's text
bottleneck is not a weakness — it is exactly the seam where grounding, validation, and zh-TW support
get inserted. For "have a natural spoken conversation in English with sub-100 ms responsiveness,"
LFM2.5-Audio's fused design wins outright.

## The two architectures

### LFM2.5-Audio-1.5B (Liquid AI)
A single end-to-end multimodal speech+text model — "does not require separate ASR and TTS components."

| Component | Detail |
|---|---|
| Backbone | LFM2.5 hybrid LM, **1.2B** (Liquid's conv+attention "Liquid" architecture) |
| Audio **in** | **FastConformer** encoder, **115M** (NVIDIA Canary `canary-180m-flash`); raw waveform chunked to **~80 ms** segments → projected to **continuous** embeddings |
| Audio **out** | **RQ-transformer** emits **discrete** audio codes → **Mimi** codec (Kyutai), **8 codebooks @ 24 kHz**, up to 8 tokens/step |
| Total | **~1.5B** params |
| Modes | **Interleaved** (real-time speech↔speech, early audio emission) · **Sequential** (ASR / TTS / modality switch) |
| Languages | **English only** (separate `-JP` variant exists) |
| Reported | VoiceBench **54.92** (LFM2.5) / 56.78 (LFM2-Audio); ASR WER avg **7.53** / 7.24; **<100 ms** time-to-first-audio after a 4 s prompt |
| Tool use | **Not documented** — it's a conversational S2S model, no function-calling / retrieval |
| License | LFM Open License v1.0 |

Signal flow: `waveform → 80ms chunks → FastConformer → continuous embeds → LFM2.5 backbone
(text⊕audio token stream) → RQ-transformer → Mimi codes → waveform`. Audio is a *first-class token
type* on both ends; the model can interleave text and audio tokens and start speaking before it has
finished "thinking."

### Ours: Qwen3-ASR-0.6B-Agent + resolver + PrimeTTS
Three independently-trained, independently-swappable stages joined by a **text** interface.

| Stage | Detail |
|---|---|
| **Perceive + decide** | `Qwen3-ASR-0.6B-Agent` — frozen **AuT** audio encoder (186M, ~20M-hr ASR pretraining) + LoRA-re-elicited **Qwen3 decoder** (596M; ~0.8B total "thinker"). Speech in → **text**: a `<tool_call>` or a reply. |
| **Ground** | `resolver.py` — RapidFuzz + pinyin + double-metaphone over a **closed** directory. Deterministic; the extension comes from the DB, not the model. |
| **Speak** | `PrimeTTS v1b_16k` — **~5M-param** FastSpeech + Snake-HiFiGAN, ONNX, text → **16 kHz**. Entity-aware text normalizer (phone numbers, emails, dates read correctly). |
| Languages | **zh-TW + English + code-mix** |
| Measured | single-turn agent **94.3%** acc / 1.27% misroute, zh **99.2%**; free-running multi-turn **93.8%**; demo turn ~8–10 s on a free 2-vCPU CPU (two 0.6B text passes + retrieval + TTS at RTF ~0.1) |

Signal flow: `waveform → AuT encoder → Qwen3 decoder → <tool_call> (text) → resolver (DB) →
<tool_response> (text) → Qwen3 decoder → reply (text) → PrimeTTS → waveform`. **Text is the explicit
bus** between every stage.

## The one structural difference everything follows from

> **LFM2.5-Audio keeps the entire task — perception, knowledge, dialogue, and speech generation —
> inside one neural network. The cascade externalizes knowledge and decisions to a verifiable tool,
> and makes text the interface between perception and speech.**

Every implication below is a consequence of *where the seams are*.

## Deeper implications

### 1. End-to-end latency — perceived vs. total compute
- **LFM2.5 optimizes *perceived* latency.** Interleaved generation emits the first Mimi codes while
  still consuming input → **<100 ms time-to-first-sound**, plus natural barge-in / full-duplex. This
  is the headline win for conversational feel.
- **But sustained audio generation is heavy.** To *say* an N-second reply, a 1.5B model must
  autoregressively decode ~12.5 frames/s × (up to 8 codebooks) of Mimi tokens through the full
  backbone. On a GPU/strong NPU that streams comfortably; on a constrained **CPU** the *sustained*
  real-time factor — not the first token — becomes the bottleneck.
- **The cascade is the opposite shape.** It is **non-streaming by construction**: the tool call must
  finish before retrieval before the reply — so **time-to-first-sound is high** (you wait for the
  whole reply text). But the *total compute* is small: two short **text** generations (~48–80 tokens)
  through a 0.6B model, then a **5M-param** vocoder at **RTF ~0.1**. Generating short text + cheaply
  vocoding it costs far less than autoregressively generating *audio tokens* for the same utterance.

  **Net:** e2e wins responsiveness (streaming, barge-in) and loses on total compute per second of
  speech; the cascade wins footprint/throughput and loses on turn-based, non-streaming latency. On a
  GPU, pick e2e for feel. On a weak edge CPU, a tiny dedicated vocoder is cheaper — but you give up
  streaming. *(Our 8–10 s wall-clock is mostly the two 0.6B text passes on 2 free vCPUs, not the TTS.)*

### 2. Speech quality — naturalness vs. determinism
- **LFM2.5 (Mimi neural codec)** has the higher *naturalness* ceiling: prosody, emotion,
  continuous expressive audio. Cost: it is a *generative* audio process — it can drift, slur, or
  mis-voice content over long spans, and the voice is whatever it learned.
- **PrimeTTS (FastSpeech + vocoder)** has a lower naturalness ceiling — one fixed voice, 16 kHz — but
  is **deterministic and controllable**, with an **entity-aware normalizer** that guarantees
  "分機 9765" / "extension 9765" is read digit-correct. A codec-LM has no such guarantee.
- **Implication:** neural-codec audio-LMs trade controllability for expressiveness. For an attendant
  whose payload is *entities* (names, extensions, emails), correctness ≫ expressiveness, and a
  separate text stage you can normalize and unit-test is a feature, not a bug.

### 3. Agentic ability — the decisive axis
- **LFM2.5 has no documented tool calling / retrieval.** It is conversational. Asked for an extension
  it would **answer from parameters → i.e. hallucinate** a plausible number, with confident prosody
  and no way to verify.
- **The cascade is agentic by construction.** It emits `search_contacts`, a deterministic resolver
  grounds it against the closed DB, and it runs multi-turn disambiguation ("which department?"). The
  extension is **retrieved, never generated** → structurally zero hallucinated extensions.
- **Implication (the deep one):** a closed-world factual task needs a *seam to insert a database*. An
  end-to-end S2S model has none — knowledge lives in weights. The cascade's text bottleneck **is** that
  seam: grounding, business rules, logging, and compliance all bolt on there. This is why "put the
  whole thing in one audio LM" is the wrong shape for a directory agent, regardless of model quality.

### 4. Languages
- LFM2.5-Audio: **English only** (a separate JP model exists). The Taiwan-office task is out of scope.
- Cascade: **zh-TW + English + code-mix**, because we can pick a multilingual encoder (Qwen3-ASR,
  ~20M-hr) and a zh-TW TTS (PrimeTTS) **independently**. Modularity directly buys language coverage.

### 5. Footprint & edge fit
- LFM2.5 1.5B (GGUF available) is genuinely small for a foundation model, but **sustained audio-token
  generation** through 1.5B is the edge bottleneck on CPU-class silicon.
- Cascade: 0.8B thinker + **5M** TTS; PrimeTTS is RTF **0.35 on a Jetson Nano** (their number). The
  heavy part is the two ASR-agent passes, which you can quantize (INT8 confirmed) or shrink further.

### 6. Failure modes & debuggability
- **E2E:** errors are *implicit and compounded inside the net* — mishear → directly speak a wrong name
  with confident prosody. Hard to gate, hard to inspect, fixable only by retraining.
- **Cascade:** errors are *localized and inspectable* — an ASR slip shows up as a visible `query`
  string; the resolver can **reject or disambiguate** with a confidence gate; the TTS only reads text.
  The classic cascade risk (ASR error → wrong query) is real, but here it's blunted because the fuzzy
  resolver only has to land **phonetically near** a name in the **closed set** — perception doesn't
  need to be perfect, just in-set.

### 7. What the text bottleneck costs you
Forcing everything through discrete text **discards paralinguistics** end-to-end — emotion, speaker
identity, tone — that LFM2.5 preserves natively. For an attendant that's irrelevant; for an empathetic
or affect-aware voice agent it's a real loss, and there the e2e model is the right base.

## When to use which
| If you need… | Pick |
|---|---|
| Grounded, factual, transactional voice agent (lookup, booking, directory) | **Cascade** |
| Never-hallucinate guarantees, audit logs, business rules, compliance | **Cascade** |
| zh-TW / multilingual / code-switch | **Cascade** (swap encoder + TTS) |
| Tiniest edge footprint with deterministic output | **Cascade** |
| Open-ended natural English conversation | **LFM2.5-Audio** |
| Lowest perceived latency, barge-in, full-duplex turn-taking | **LFM2.5-Audio** |
| Expressive prosody / emotion / paralinguistics | **LFM2.5-Audio** |
| Single-model simplicity, minimal glue code | **LFM2.5-Audio** |

## Outlook — the hybrid
These converge. The strongest near-term design is an **audio-native front-end that also calls tools**:
keep the e2e model's streaming perception + dialogue + expressive speech, but give it a function-calling
seam so factual answers are *retrieved, not generated* (Qwen3-Omni already ships tool use; LFM-class
models adding it is the natural next step). At that point the distinction collapses to: *does the
audio LM emit a verifiable structured call before it speaks?* Our cascade is the explicit, debuggable
version of exactly that pattern today — and a useful reference for what an agentic audio-LM must
preserve: a groundable interface between hearing and speaking.

## Sources
- [LiquidAI/LFM2.5-Audio-1.5B — model card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)
- [LFM2-Audio: An End-to-End Audio Foundation Model — Liquid AI blog](https://www.liquid.ai/blog/lfm2-audio-an-end-to-end-audio-foundation-model)
- [Liquid AI's LFM2.5 beats a 7.7B voice model at 1.5B — AlphaSignal](https://alphasignal.ai/news/liquid-ai-s-lfm2-5-beats-a-7-7b-voice-model-at-just-1-5b-parameters)
- [Liquid AI released LFM2-Audio-1.5B with sub-100 ms latency — MarkTechPost](https://www.marktechpost.com/2025/10/01/liquid-ai-released-lfm2-audio-1-5b-an-end-to-end-audio-foundation-model-with-sub-100-ms-response-latency/)
- Ours: [Luigi/Qwen3-ASR-0.6B-Agent](https://huggingface.co/Luigi/Qwen3-ASR-0.6B-Agent) · [Luigi/PrimeTTS](https://huggingface.co/Luigi/PrimeTTS) · [demo](https://huggingface.co/spaces/Luigi/contact-attendant-omni)
