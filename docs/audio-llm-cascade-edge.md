# Round-2: more models + the biased-ASR cascade (edge path)

*Follow-up to `audio-llm-jetson-research.md`. 10 agents, 158 lookups, 3 adversarial verdicts +
critique. Key load-bearing fact independently verified (see below).*

## Headline
Exploring more models did **not** dethrone Qwen2.5-Omni-3B as the *server-side end-to-end*
choice, but it **changed the edge story decisively**: the right edge architecture is a
**cascade** — a context-/list-biased ASR (perceive a name from the *closed* directory) → your
existing fuzzy resolver → a tiny text tool-caller (the agent). It fits in 4 GB and is ~40-100×
faster on perception than the 3B audio encoder.

## Verified fact (was the critique's #1 risk)
**Qwen3-ASR ships open Apache-2.0 small-dense weights** — `Qwen/Qwen3-ASR-0.6B`,
`Qwen/Qwen3-ASR-1.7B` (52 languages; an int4-QAD quant exists `vrfai/Qwen3-ASR-0.6B-int4-QAD`).
Its LLM decoder resolves homophones *in context* and it takes a context/hotword bias list —
exactly the perception front-end the cascade needs. (Confirmed via HF; tech report paper 2601.21337.)
*Caveat: the exact "CV Taiwan-Mandarin CER" number must be confirmed on YOUR audio in the bake-off.*

## New audio-LLMs surveyed — none clears all 4 axes (audio + tools + zh-TW names + ≤4 GB)
| Model | Size | Why it's out for edge |
|---|---|---|
| Step-Audio-2-mini | 8B | **native tool-calling + strong Chinese ASR**, but 8B, no zh-TW claim, no llama.cpp audio → *server bake-off candidate only* |
| Ultravox v0.5-0.7 | projector + 1-70B | small variants have **no tools**; tool-capable ones are huge; zh weak |
| LFM2.5-Audio-1.5B | 1.5B | real llama.cpp runner, fits edge — but **English-only, no tools** |
| Gemma-4 E2B/E4B | 2.3/4.5B | only ≤4B with audio+tools, but **empirically fails zh-TW names ("李明")** |
| Kimi-Audio / GLM-4-Voice / LLaMA-Omni2 / MERaLiON-2 | 7-10B / small | too big, or no tools, or non-commercial license, or wrong region |

→ **Verdict (adversarially confirmed): no ≤4B audio-LLM beats Qwen2.5-Omni-3B on all four axes.**

## The cascade — edge-feasible (confirmed), homophone-robust (plausible, must test)
**Single best edge stack:**
`Qwen3-ASR-0.6B (list-biased) → RapidFuzz+pinyin+double-metaphone resolver (retriever+corrector) → small text tool-caller`

- **Memory:** ASR-0.6B (~0.5-0.8 GB Q8/int4) + 1.5B tool-caller Q4 (~1.0 GB) + OS/KV → **< 4 GB** ✓
- **Latency:** ASR at multiple-× realtime → **~1-3 s perception** (vs the Omni encoder's 84-129 s on 2 vCPU); the short tool-call decode is not the bottleneck.
- **Why it suits homophones:** the directory is **closed**, and the resolver already absorbs near-misses, so the ASR only needs to land *phonetically close* to an in-set name — it doesn't need perfect hanzi.

### The two upgrades the critique added (do these, not naive prompt biasing)
1. **List-constrained decoding / n-best rescoring against the actual directory** — stronger and more robust than *soft* prompt biasing ("may or may not care"). With a closed 20k list, rescore the ASR n-best against the name set (or constrain decoding via an FST/trie). This is the **most under-exploited method** given the closed set.
2. **GBNF grammar-constrained decoding** for the small tool-caller — forces valid `search_contacts` JSON from even a weak-FC 1.5B model, which weakens the "must fine-tune the tool-caller" conclusion. (Tool-caller options: **Hammer2.1-1.5b** BFCL 73 but **CC-BY-NC**; commercial-permissive fallback **Arch-Function-1.5B** / a zh-TW FC LoRA on **Qwen2.5-1.5B-Instruct** Apache-2.0.)

### Honest gaps (the **mixed** verdict)
- **zh-TW / Traditional-hanzi homophone surnames are UNTESTED in the biasing literature** (all measured biasing = mainland Simplified or English). This is the make-or-break unknown.
- Naive hotword lists **degrade at scale** (recall 55%@231 → ~28%@4000) → the retriever/resolver is *mandatory* for 20k, not optional.
- Also worth including for a Taiwan deployment: **Breeze-ASR (MediaTek) and zh-TW Whisper checkpoints** as front-end candidates (critique-flagged miss).

## Updated recommendation
- **Server / has-GPU / tolerant of latency →** Qwen2.5-Omni-3B (your validated end-to-end model). Keep it.
- **Edge (≤4 GB / ARM) or just want fastest+cheapest →** the **cascade** (most deployments). It exploits the closed list + your resolver directly and sidesteps every audio-LLM's homophone collapse.

## Bake-off plan (the decisive experiment)
**First, define the gate:** the resolver recovers iff perception NE-CER < X% *and* the true name is in the top-K phonetic neighbors — pick X, K up front so the result is pass/fail, not vibes.

Then test on **real Taiwan-accent audio** (not TTS) of your homophone-cluster names:
1. **Cascade A:** Qwen3-ASR-0.6B + **list-constrained decoding** → resolver → tool-caller.
2. **Cascade B (ultra-light):** SenseVoice-Small / Fun-ASR-Nano + sherpa-onnx hotwords → resolver → tool-caller. *(Does a non-LLM-decoder ASR + resolver suffice?)*
3. **Baseline:** Qwen2.5-Omni-3B end-to-end. *(Optional: Step-Audio-2-mini 8B server.)*

**Primary metric:** top-1 **name-resolution accuracy** (routed to the correct person?), split by
**clean vs homophone-cluster**, **list size 200/2k/20k**, and **code-switch vs pure-zh**.
**Secondary:** perception-only NE-CER on the surname span (isolates ASR vs resolver), p50/p95
latency on the target CPU, peak RAM (<4 GB), agentic success (correct tool call + disambiguation).

**The make-or-break cell:** *perception-only NE-CER on homophone surnames at 20k, with hard
list-constrained decoding* — the exact zh-TW/Traditional/at-scale combination the literature
hasn't measured. Run it first; everything downstream is low-risk.

*Sources: [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [int4-QAD](https://huggingface.co/vrfai/Qwen3-ASR-0.6B-int4-QAD), [Qwen3-ASR report](https://huggingface.co/papers/2601.21337), [Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini), [Hammer2.1](https://huggingface.co/MadeAgents/Hammer2.1-1.5b), [Arch-Function-1.5B](https://huggingface.co/katanemo/Arch-Function-1.5B).*
