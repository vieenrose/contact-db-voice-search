# P0 — Teacher-harvest spec (Qwen3-Omni-30B-A3B → distillation targets)

*Spec — 2026-06-30. The first build phase: run the teacher offline to manufacture the distillation
corpus for the 0.6–0.8B zh/en agentic S2S student. Grounded in the real modeling code
(`Qwen3OmniMoeForConditionalGeneration.generate`). Companion to `student-arch-design.md` /
`distill-qwen3omni-to-0.6b-s2s.md`.*

## Where the codes & logits actually come from (grounded capture points)
From `modeling_qwen3_omni_moe.py`, the teacher's `generate`:
```
 thinker → text token ids (+ logits if output_scores)           # Thinker, MoE
 talker_result = self.talker.generate(...)                       # Talker
 talker_codes  = stack(talker_result.hidden_states)  → [B,16,T]  # 16-codebook codes / frame  (line ~4021)
 talker_wavs   = self.code2wav.chunked_decode(talker_codes, chunk_size=300, left_context_size=25)  # 24 kHz
```
So the three harvest tensors are: **Thinker token ids + top-k logits**, **`talker_codes` [16,T]**, and the
**decoded `talker_wavs`**. A `speaker` arg fixes the output voice (set it constant for one student identity).

## 1. Prompt corpus (input side) — zh/en **agent** scope
| Bucket | Source | Purpose |
|---|---|---|
| Attendant-agentic (seed) | our **13.5k** dialogs (have audio+text) | core tool-calling + multi-turn disambiguation |
| Broader zh/en tool-use | synth queries (weather/calendar/lookup/FAQ) → speech | agentic robustness beyond one tool |
| Code-switch | English names in zh sentences (our name bank) | the hard zh/en case |
| Negatives / OOD | not-found, chit-chat, barge-in cues | rejection + conversational edges |

**Input speech = voice-diverse** (reuse our request audio + re-synthesize with VoxCPM2 multi-voice + edge-tts)
so the student's *perception* is robust. **Output (teacher) voice = single fixed speaker** so the student
learns one consistent identity. Start **~30k turns**; scale with GPU budget.

## 2. Teacher run mechanics
- **HW:** one **80 GB** GPU (A100/H100) bf16 (~60 GB), or AWQ-INT4 (~20 GB) on a 24–48 GB card; vLLM or
  transformers. A3B ⇒ ~3B-active *speed*. (GTX-1070 cannot host the teacher — cloud/rented, one-time.)
- **Mode:** Thinker text **+** Talker audio enabled; `output_scores=True, return_dict_in_generate=True` to get
  logits; capture `talker_codes` before `code2wav`. Fixed `speaker`.
- **Agentic multi-turn loop** (the teacher must see tool results to produce the spoken answer):
  ```
  input_speech → teacher → <tool_call>            # capture text ids+logits, (no audio yet — tool_call is text-only)
               → run OUR resolver(query[,dept])   # grounded execution
               → feed <tool_response> back        # as a tool turn
  teacher      → reply (text) + talker_codes + wav # capture all three  (this is the spoken answer)
  ```
  Disambiguation turns ("which department?") are spoken too — capture their codes as well.

## 3. Per-turn capture schema (what to log)
```jsonc
{ "id": "...", "lang": "zh|en|mix", "voice_in": "...", "turn_type": "tool_call|ask_dept|reply|not_found",
  "input_wav": "audio/in/<id>.wav",                 // 16 kHz mono (perception input)
  "thinker": { "token_ids": [...],                   // text tokens (tool_call and/or reply span)
               "topk": {"k": 32, "ids": [[...]], "logp": [[...]]} },  // top-32 logprobs/pos → logit-KD
  "talker_codes": "codes/<id>.npy",                  // int16 [16, T_frames] @12.5 Hz  (audio token-KD targets)
  "reply_wav": "audio/out/<id>.wav",                 // 24 kHz Code2Wav output (reference / eval)
  "text2frame": [ /* per audio frame: index of the text token it realizes */ ],  // MCTP / parallel-decode alignment
  "trace": { "tool_call": {...}, "tool_response": [...], "reply_text": "..." },   // agentic SFT
  "meta": { "reply_dur_s": ..., "n_frames": ..., "teacher_rev": "..." } }
```
Notes: store **top-k** logprobs (k≈32), not full 151,936-way (size). Keep `talker_codes` int16 (16×T×2 B —
tiny). `text2frame` alignment comes from the interleaved decode positions (needed to train the zero-delay MCTP).

## 4. Embedded gate — parallel-vs-AR predictability (runnable on harvested codes, no teacher)
The open quality question (does **parallel 16-heads** lose vs the teacher's **AR** MTP?) is answerable *from
the harvested `talker_codes` alone* — an information-theoretic probe:
- For each codebook *i* (1…15), train two tiny probes to predict `code_i[t]`: **(P) parallel** = from frame
  context only (the backbone-hidden proxy / text-token + lower-frame features **excluding** codes 0..i−1);
  **(A) autoregressive** = same **plus** codes 0..i−1. **Accuracy gap (A−P) = the AR benefit for codebook i.**
- **Decision:** small gaps on the upper codebooks ⇒ **parallel-16-heads is safe** (ship RTF 0.53 design);
  large gaps ⇒ keep partial AR or fall back to **Mimi-8**. Also a perceptual spot-check: Code2Wav-decode the
  real codes vs codes with upper layers **independently resampled** (parallel proxy) → F0-corr + CER delta.
- This converts the deferred §"top gate" into a concrete measurement the moment the harvest exists.

## 5. Storage & compute (≈30k turns)
- **Per turn:** logits top-32 ≈ 12 KB · codes ≈ 1.6 KB · in-wav ≈ 96 KB · out-wav ≈ 192 KB → ~**0.3 MB**.
- **Total ≈ 10 GB** for 30k turns. (Logits dominate the "small" tier; audio dominates bytes.)
- **Compute:** teacher emits ~50 audio frames + ~30 text tokens/turn; batched on one A100 ≈ 0.5–2 s/turn ⇒
  **~8–30 GPU-h** (under a day). One-time.

## 6. How each field feeds training (P1/P2)
| Harvested | Loss / use | Phase |
|---|---|---|
| `thinker.topk` logprobs | **text/agentic logit-KD** (KL) over shared 151,936 vocab | P1 |
| `trace` (tool_call/response/reply) | agentic SFT + resolver integration | P1 |
| `talker_codes` [16,T] | **audio token-KD** — per-codebook CE/KL for the parallel 16-heads | P2 |
| `text2frame` | **MCTP zero-delay** alignment supervision | P2 |
| `reply_wav` | eval reference (MOS / CER / F0-corr) | P2/eval |
| `input_wav` | ASR/agentic **replay** to prevent forgetting | P1–P3 |

## 7. Risks / notes
- **Custom hooks:** default `generate` returns audio but not the intermediate `talker_codes` + per-pos logits;
  the harvester calls the components (thinker generate w/ scores → resolver → talker.generate capturing
  `talker_codes` → code2wav) rather than the one-shot path. Grounded by the code above.
- **Fixed output voice** is essential (consistent student identity) — set `speaker` constant.
- **Teacher `<tool_call>` format** already matches our resolver/agent — no schema translation.
- **One-time cloud cost**; the 30B never touches the edge model.

## 8. Runnable now (no GPU) vs needs the teacher
- **Now (local):** build the **prompt corpus + input-speech manifest** (reuse the 13.5k dialogs; synth the
  broader/code-switch buckets with VoxCPM2/edge-tts) — produces the harvest *input* list and the resolver
  wiring. *(I can scaffold `data/build_harvest_corpus.py` next.)*
- **Needs the 80 GB GPU:** the harvest harness (the capture loop above) + the parallel-vs-AR probe on the
  resulting `talker_codes`.

## Next step
Scaffold `data/build_harvest_corpus.py` (local) to assemble the input manifest, so the moment a GPU is
available the harvest harness runs against a ready corpus. Then provision the 80 GB GPU for the harvest +
the parallel-vs-AR gate.

## Sources
- `modeling_qwen3_omni_moe.py` (`Qwen3OmniMoeForConditionalGeneration.generate`, `talker_codes`, `code2wav.chunked_decode`)
- ours: [`student-arch-design.md`](./student-arch-design.md) · [`distill-qwen3omni-to-0.6b-s2s.md`](./distill-qwen3omni-to-0.6b-s2s.md)
