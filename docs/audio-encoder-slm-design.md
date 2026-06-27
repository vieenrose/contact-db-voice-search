# Design verdict: Qwen3-ASR encoder as the audio frontend for an agentic SLM

*Workflow: 4 research agents + 2 adversarial verdicts. Conclusion is decisive.*

## TL;DR
Your idea (Qwen3-ASR **encoder → new projector → separate SLM**) is the right architecture
*family* (Ultravox/SLAM-ASR/SALMONN) but the **wrong cut** — it re-buys the cross-modal
projector alignment that killed your v1 at 0.2%. **Qwen3-ASR already IS that stack, aligned on
~40M hours.** Recommended path: **LoRA Qwen3-ASR's own decoder to emit `search_contacts` tool
calls** (projector reuse = free), with one open risk to test cheaply.

## Qwen3-ASR architecture (confirmed)
AuT attention enc-dec **audio encoder** (0.6B: 180M, hidden 896; 1.7B: 300M) → **2-layer MLP
projector** (enc-dim → LLM-hidden) → a **full 28-layer Qwen3 LLM decoder** (ChatML, 152k vocab,
65k ctx; "posttrained from Qwen3-Omni"). Audio injected by replacing `<|audio_pad|>` tokens in a
standard system/user/assistant prompt. **Context biasing is already prompt-based** (glossary /
homophone-correction demos) — inject directory surnames per call. It's a genuine chat
transformer, not a thin transcription head — which is *why* agentic fine-tune is plausible.

## Why (A) "encoder + NEW projector + separate SLM" is **refuted**
From-scratch projector alignment is paid in **hundreds-to-thousands of hours of paired speech,
not task-format samples**:
- SLAM-ASR: **960 h**, paper says "still not enough." Ultravox: **~2,500 h** (+KD). SALMONN:
  dedicated alignment-pretrain stage. Soundwave/Qwen2-Audio: 10k–520k h.
- Low-resource study (arXiv 2508.05149): scratch projector @10 h → **14% WER**; need **~100–200 h
  to match a plain ASR baseline**; the *only* 10–15 h rescue is a **pretrained** projector
  (14.0→8.6% WER) — i.e. your v2 fix. Narrow data → diverges to ∞ WER (hallucinate).
- **Your ~5–50k utterances ≈ 10–60 h → the documented failure band.** v1's 0.2% was the median
  outcome, not bad luck. **Verdict: refuted.**

## Why (B) "LoRA Qwen3-ASR's own decoder" is the recommendation
- **No new projector** (40M-hr alignment reused); LoRA teaches output *format* only
  (transcript → `<tool_call>` JSON), AuT encoder **frozen**. ~2–4 GPU-hrs, ~5k paired samples.
- Decoder-repurposing precedents at low cost: **WhisperNER** (ASR decoder → tagged spans, F1 81),
  ASR→SLU (90% intent / 82 SLU-F1), **Step-Audio** (speech→function-call, zh P/R 88/96).
- **Best edge profile:** 0.6B → ~1–3 s CPU perception (vs Omni-3B's 84–129 s); Apache-2.0.
- **The honest catch (verdict `mixed`):** Qwen3-ASR's decoder was *deliberately* de-instruction-
  tuned ("ASR-only … does not follow natural-language instructions" — anti-injection). So
  tool-calling is a **suppressed, recoverable** capability, not free reuse. Single-turn very
  likely recovers; **multi-turn is the open question.** Don't heavily fine-tune the AuT encoder
  (drifts features text-ward, erodes acoustic grounding).

## Four-way comparison
| Approach | Train risk | Edge latency (2 vCPU) | Homophones | Agentic | Verdict |
|---|---|---|---|---|---|
| A. enc + new projector + SLM | 🔴 re-buys 0.2% failure | ~1–3 s | collapses if unaligned | depends | **refuted** |
| **B. LoRA Qwen3-ASR decoder** ⭐ | 🟡 lowest; no projector | **~1–3 s** | best preserved + bias | re-elicit; multi-turn TBD | **recommended** |
| C. Qwen2.5-Omni-3B (v2) | done | 🔴 84–129 s | strong (proven) | **98.2 / 92.6** | accuracy ceiling |
| D. ASR→text cascade | trivial | fast | 🔴 text bottleneck | high | homophone floor |

## Decisive probe (~a day, 2–4 GPU-hrs on the 0.6B)
1. **Zero-shot** raw `Qwen3ASRForConditionalGeneration` (not `.transcribe()`) with the
   `search_contacts` schema + a few surnames on ~200 held-out names → expect ~0% valid tool-calls
   (confirms suppression, sets floor).
2. **LoRA re-elicit:** r16 on decoder q/k/v/o+MLP, **frozen AuT**, target = tool-call JSON;
   curriculum text-first then ~5k paired audio late.
3. **Metrics vs Omni-3B v2:** single-turn exact-name (post-resolver) + **homophone-pair
   disambiguation** (hardest zh-TW set) + a 50-dialog multi-turn slice.
4. **Gate:** single-turn ≥ ~90% of Omni-3B *and* homophones hold → **green-light B** (Omni-class
   perception at ~1–3 s). Multi-turn collapses → route dialogue to a tiny text controller.
   Single-turn won't recover → fall back to a *pretrained-projector* Ultravox/Omni build —
   **never** from-scratch projector (A).

*Sources: [Qwen3-ASR report 2601.21337](https://arxiv.org/html/2601.21337v1),
[Qwen3-ASR-0.6B config](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json),
[antirez/qwen-asr MODEL.md](https://github.com/antirez/qwen-asr/blob/main/MODEL.md),
[SLAM-ASR 2402.08846](https://arxiv.org/html/2402.08846v1), [low-resource projector 2508.05149](https://arxiv.org/html/2508.05149),
[WhisperNER 2409.08107](https://arxiv.org/html/2409.08107v1), [Step-Audio-2 2507.16632](https://arxiv.org/html/2507.16632v3).*
