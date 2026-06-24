# Results

Benchmark = voice-disjoint test set (held-out edge-tts voices: zh-TW HsiaoYu, en Aria),
8 kHz telephony. Metric = routing OUTCOME (eval.py), plus exact-name accuracy on resolve cases.

## Model comparison

| Run | Model / recipe | Task success | Misroute | Exact-name (overall / en / zh) |
|---|---|---|---|---|
| v1 | Ultravox-1B, whisper-base, **projector from scratch** | 7% | 14% | 0.2% / — / — |
| v2 | Ultravox-1B, **stock enc + pretrained projector + LoRA** | 36% | 58% | 32% / 58% / 3% |
| v3 | v2 + more zh data + 3 epochs | 24% | 67% | 23% / **68%** / 0.6% |
| **v4** | **Qwen2.5-Omni-3B QLoRA (4-bit)** | **81%** | **8.2%** | **87% / 65% / 96.7%** |

## Key findings
1. **Can't train the projector from scratch** on a small dataset (v1). Must start from a model whose
   audio→LLM alignment is pretrained (v2+).
2. **Ultravox-1B (Llama) cannot hear Mandarin names** — 0.6–3%, and *more data/epochs does not fix it*
   (v2→v3). It's a base-model capability limit, not a data problem.
3. **Qwen2.5-Omni-3B solves Mandarin** — 0.6% → **96.7%** exact-name on zh. Stock Qwen already perceives
   Taiwan names near-perfectly (蔡孟儒, 周宜蓁 — exact pronunciation); QLoRA maps audio→English directory name.
4. **bitsandbytes 4-bit works on the Pascal GTX 1070**, so QLoRA of a 3B fits the 8 GB card.

## v4 remaining weakness — OOD rejection
not_found cases: 42% success, **38% misroute** (model guesses a real name for out-of-directory callers).
**99 of 123 total misroutes come from this single case**; resolve cases are already at 2.1% misroute.
Fix in progress: confidence gate (low avg token log-prob → not_found) and/or more rejection training.

## Deployment implication
- **Ultravox-1B** = gen1-deployable, English-first (68% en), Mandarin-weak.
- **Qwen-Omni-3B** = genuine bilingual (97% zh), but 3B → **Orin-class**, not gen1 real-time.
