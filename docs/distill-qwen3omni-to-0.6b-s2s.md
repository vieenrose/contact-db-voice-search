# Plan: distill **Qwen3-Omni-30B-A3B-Instruct** → a **0.6B zh/en agentic S2S** model (Jetson Nano gen1)

*Plan — 2026-06-30. Teacher: [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
(Apache-2.0). Student: a 0.6B speech-in → (text tool-call + speech-out) agent, real-time on Nano gen1,
**zh/en agent use**. Builds on the feasibility study (`feasibility-0.6b-agentic-audio-lm.md`); supersedes
the VoxCPM2-teacher plan — a native S2S teacher carries the conversational "soul" a canned TTS can't.*

## Why Qwen3-Omni is the *ideal* teacher (three structural gifts)
1. **Same audio encoder family.** Qwen3-Omni's input encoder is **AuT** (20M-hr); our student already uses
   the AuT from Qwen3-ASR-0.6B. Encoder space is shared → **no input-side alignment to learn.**
2. **Same text family → white-box text KD.** The teacher's **Thinker** is a Qwen3 LM with the **same
   tokenizer** as our Qwen3-0.6B backbone. The agentic/text half can be distilled with **logit-level KD**
   (KL on token distributions), the gold standard — not just imitation.
3. **Native, agentic, bilingual S2S.** Qwen3-Omni is the **first native S2S with function calling**
   (`<tool_call>`, the format our agent already emits), speech-gen in zh+en, and its **Talker** emits
   speech *conditioned on the dialogue* — so the student can learn context-appropriate, expressive speech,
   not canned TTS. This is the part the VoxCPM2 plan couldn't give.

## Architecture mapping: teacher Thinker–Talker → student "mini Thinker–Talker" at 0.6B
Teacher (30B-A3B MoE):
```
 audio ─▶ AuT enc ─▶ [Thinker: MoE LLM] ─▶ text (incl. <tool_call>)
                              └─▶ [Talker] ─▶ multi-codebook codec tokens (MTP) ─▶ Code2Wav (causal ConvNet) ─▶ speech
```
Student (0.6B dense, warm-started from `Qwen3-ASR-0.6B-Agent`):
```
 audio ─▶ AuT enc (shared, frozen) ─▶ [Qwen3-0.6B backbone] ─▶ text (incl. <tool_call>) ─▶ resolver/tools
                                              └─▶ [depth/MTP audio head] ─▶ codec tokens ─▶ Code2Wav ─▶ speech
```
We are literally building a **distilled miniature of the teacher's own architecture** — Thinker→backbone,
Talker→audio head, and (ideally) **the teacher's own codec + Code2Wav reused on the edge**.

### Student architecture — concrete spec
A dense, streaming **mini Thinker–Talker**. The "0.6B" is the backbone; the encoder and vocoder are
frozen bolt-ons (as in Mini-Omni2). Four blocks:

| # | Block | Role | Params | Trained? |
|---|---|---|---|---|
| 1 | **AuT encoder** (shared w/ teacher) | speech in → continuous frame embeds; block-wise window attention (streaming) | ~186M | frozen |
| 2 | **Qwen3-0.6B backbone** ("Thinker") | temporal LM over audio-embeds⊕text tokens → text stream (`<tool_call>` + reply); 28 layers, d=1024, Qwen3 tokenizer (151k) | ~596M | LoRA/FT (warm-start = `Qwen3-ASR-0.6B-Agent`) |
| 3 | **MTP / depth audio head** ("Talker") | per backbone step, predicts the **N codec codebooks** for that audio frame (inter-codebook AR), **conditioned on the emitted text token** | ~20–60M | **new, trained** |
| 4 | **Code2Wav / Mimi decoder** | codec tokens → waveform (lightweight causal ConvNet, streaming) | ~tens of M | frozen |

**Total ≈ 0.8–0.9B** (the trainable "brain" is the 0.6B backbone + the tiny audio head). Codec-token
**embeddings + heads** add a few M (N codebooks × 2048).

**Two output streams, one backbone (text-audio parallel decode, Mini-Omni delay pattern):**
- The backbone emits a **text** token each step; the depth head emits that step's **audio frame** (N
  codebooks) for the *delayed* text token (1–2 frame delay) → audio always renders already-decided,
  tool-grounded text. **Cannot speak a number it didn't write.**
- **Phase split within a turn:** `<tool_call>` and `<tool_response>` are **text-only** (no audio); audio
  generation switches **on** for the user-facing **reply** tokens. So agency stays pure-text and grounded;
  only the reply is spoken.

**Streaming end-to-end:** AuT (windowed) → causal backbone → causal depth head → causal Code2Wav ⇒ first
waveform packet right after the first reply frame (the teacher's "stream from first codec frame", at 0.6B).

**Turn flow:**
`speech →[AuT]→ embeds →[backbone]→ <tool_call> →[resolver/DB]→ <tool_response> →[backbone]→ reply text
⊕[depth head]→ codec frames →[Code2Wav]→ spoken reply`.

## Two distillation channels
**(A) Text / agentic — white-box logit KD (high confidence).** Teacher and student share the Qwen3
tokenizer, so distill the Thinker's next-token distribution over `<tool_call>` + reply text via **KL
divergence** (+ CE on hard targets). Warm-start from `Qwen3-ASR-0.6B-Agent` (already agentic) → this
mostly *sharpens* an existing skill. Transfers the teacher's tool-use reasoning into 0.6B efficiently.

**(B) Speech — codec-token KD (match the teacher's codec).** The Talker autoregressively predicts
**multi-codebook** tokens via **MTP**. If the student emits the **same codec**, we get **token-level KD on
the audio codebooks** (per-codebook KL, MTP-style) *and* **reuse Code2Wav** (a *lightweight causal
ConvNet* — already built for low-latency streaming, the right shape for Nano). Audio is generated
**text-conditioned** (Mini-Omni parallel decode) so spoken extensions stay grounded. *Fallback:* if the
teacher's codec/Code2Wav won't fit Nano, transcode teacher speech → **Mimi** (already gated: zh-TW
F0-corr 0.994) and do sequence-level distill instead — we lose audio-logit KD but keep a proven edge codec.

## Data: a zh/en **agentic** speech corpus + teacher harvesting
Scope = zh/en voice **agent**, so the corpus is agentic, not open chat:
- **Seed (have it):** our 13.5k contact-attendant agentic dialogs (tool calls + multi-turn disambiguation).
- **Broaden:** zh/en assistant/function-calling query sets, rendered to speech with diverse voices
  (VoxCPM2 + edge-tts) for input-acoustic variety; include code-switch (English names in zh sentences).
- **Harvest from teacher (the heavy step):** run Qwen3-Omni on every prompt and log **(a)** Thinker text
  tokens **+ logits**, **(b)** Talker **codec tokens**, **(c)** decoded **audio**. These are the distill
  targets. Use **one fixed Talker voice** so the student learns a single consistent identity.

## Curriculum (phases)
- **P0 — Harvest.** Build the prompt corpus; run the 30B-A3B teacher to collect text-logits + codec-tokens
  + audio. *(Cloud GPU — see compute.)* **Gate G1:** extract the teacher codec's **frame-rate × codebooks**
  and re-run the Nano RTF budget (the Mimi budget assumed 12.5 Hz; a faster codec tightens it).
- **P1 — Text/agentic KD.** Warm-start from `Qwen3-ASR-0.6B-Agent`; logit-KD the Thinker's text + tool
  calls. Eval: tool-call accuracy ≥ our cascade's 94%/99% zh on the held-out set.
- **P2 — Speech distill.** Add the MTP/depth audio head over the teacher's codec; distill Talker →
  student, text-conditioned. Eval: student-spoken-reply CER (ASR round-trip) + MOS vs teacher-through-codec.
- **P3 — Joint SFT.** Parallel text+audio decode with the delay pattern; mix ASR + agentic replay to stop
  forgetting; balance text/audio/tool losses.
- **P4 — Edge.** INT8 quantize (decoder INT8 already proven); export backbone (GGUF/llama.cpp sm_53) +
  AuT enc + codec + Code2Wav (ONNX); build the streaming inference loop; benchmark RTF on the real Nano.

## Compute plan
- **Teacher harvest:** 30B-A3B = ~60 GB bf16 (A3B ⇒ ~3B-active *speed*, 30B *memory*). Needs an **80 GB
  GPU** (A100/H100) or 2× 48 GB via vLLM — **rented/cloud, one-time**. The GTX-1070 cannot host the teacher;
  it only needs to see the *harvested targets*. (DashScope API is insufficient — we need raw codec tokens
  + logits, so run the open weights ourselves.)
- **Student training:** 0.6B + small audio head — single modern GPU; the agentic LoRA already trained on
  the **GTX-1070**, so P1–P3 are within reach locally or on one rented card.
- **Edge:** Nano gen1, INT8 — budget already says RTF ≈0.5 (empirically anchored: 0.6B Q8_0 = 44.6 tok/s
  on i5 ⇒ ~26 tok/s bandwidth-scaled to Nano).

## Evaluation
- **Agentic:** `eval_qwen3asr*.py` tool-call accuracy + multi-turn (zh/en); target ≈ teacher on our set.
- **Speech:** ASR-CER of the student's *spoken* output (content correctness, esp. extensions/tones) +
  MOS/PESQ vs teacher-through-codec; F0-corr for tones.
- **Real-time:** RTF on Nano gen1 (CPU + Maxwell-GPU paths); first-packet latency.

## Risks & open gates
1. **50× compression (30B→0.6B).** Can't capture general omni; **scope to zh/en agent** (narrow, templated,
   tool-grounded) — where small models stay good (Mini-Omni2 0.5B precedent for the audio mechanics).
2. **Teacher codec frame-rate (Gate G1).** If Qwen3-Omni's codec runs faster than Mimi's 12.5 Hz, the Nano
   RTF tightens — measure it in P0 before committing; Mimi is the validated 12.5 Hz fallback.
3. **Code2Wav on Nano.** "Lightweight causal ConvNet" is promising but unbenchmarked on Maxwell — verify in
   P0; fall back to Mimi's decoder if needed.
4. **Teacher-inference cost.** One-time 80 GB-GPU harvesting run is the main $ outlay.
5. **Forgetting / capacity contention.** Text+audio+agency in 0.6B — mitigate with replay + loss balancing.
6. **License:** teacher **Apache-2.0** → student is freely distributable.

## What we already have vs. what's new
**Have:** AuT encoder (= teacher's family), `Qwen3-ASR-0.6B-Agent` warm-start (agentic), the agentic dialog
corpus + data pipeline, the Mimi-codec gate (zh tones ✓), the Nano real-time budget (✓ INT8), INT8 quant.
**New:** teacher harvesting (cloud GPU), the MTP/depth audio head, codec/Code2Wav integration, the two-channel
KD training, and the Nano streaming engine.

## Concrete next step
**P0, step 1 — Gate G1, cheaply, before any harvesting:** load the teacher's config/Talker on a rented
80 GB GPU (or read the released code) to read off **codec frame-rate × codebook count + Code2Wav size**,
then re-run `scripts/bench_nano_0p6b.sh`-style numbers for that codec. That single check decides
*teacher-codec vs Mimi* and confirms the Nano RTF — the one unknown between here and a green build plan.

## Sources
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765) · [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni) · [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- Ours: [`feasibility-0.6b-agentic-audio-lm.md`](./feasibility-0.6b-agentic-audio-lm.md) · [`lfm2audio-vs-cascade.md`](./lfm2audio-vs-cascade.md) · [Qwen3-ASR-0.6B-Agent](https://huggingface.co/Luigi/Qwen3-ASR-0.6B-Agent)
