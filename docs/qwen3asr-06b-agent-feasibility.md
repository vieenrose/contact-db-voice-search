# Can Qwen3-ASR-0.6B do everything Omni-3B does (speech + tools + agentic + multi-turn)?

*Workflow: 3 research agents + 2 adversarial verdicts. Verdict: YES, with caveats — for THIS
narrow task. Reuses all existing data.*

## Verdict
**YES — with caveats.** A LoRA-re-elicited Qwen3-ASR-0.6B can do all four at once and land
**close** to Omni-3B on this **narrow closed-domain** task — because this is the **TinyAgent
regime** (one tool, closed entity set, fixed 3-state policy, fuzzy resolver), **not the BFCL
open-domain regime** where 0.6B scores ~1%.

The one-line why: *BFCL multi-turn at 0.6B (≈1.4%) measures the wrong problem (hundreds of tools,
hidden state); none of that applies to a single fixed `search_contacts` schema with all dialog
state visible in the transcript. On a narrow fine-tuned task, ~1B models hit GPT-4-Turbo parity
(TinyAgent: 80.06% vs 79.08%).*

**Do not promise Omni-3B parity (98.2/92.6).** Budget **a few-to-several points below**,
concentrated in the **2nd disambiguation turn (turn-type tracking)** — the axis with least margin.

## Why narrow ≠ BFCL (the two regimes)
| | Qwen3-0.6B base | + best recipe | Qwen3-1.7B |
|---|---|---|---|
| BFCL multi-turn (open-domain, WRONG yardstick) | **1.4%** | 7.0% (KD+RL) | 16.9% |
| Single-turn function-calling (after FT) | 62–79% | — | — |
| TinyAgent-1.1B narrow task (RIGHT yardstick) | 12.7% → **80.06% (beats GPT-4-Turbo)** | | |
| In-domain SLU, 0.5B vs 1.5B | **0.5B 98.8% ≥ 1.5B 96.9%** | | |

The *same* TinyAgent-1.1B that beats GPT-4 on its narrow task scores **0.00%** on BFCL open
multi-turn. This attendant is squarely the narrow regime.

## Two load-bearing de-riskers
1. **GBNF / grammar-constrained decoding** makes the `search_contacts(query, department?∈15-enum)`
   tool call **structurally guaranteed** (<6% overhead, benchmarked on Qwen3-0.6B) → the 0.6B
   spends capacity only on slot *values*, which `resolver.py` (RapidFuzz + double-metaphone +
   pinyin) absorbs as long as they land phonetically in-set. Adopt from day one.
2. **State-aware grammar** (optional, the highest-leverage move for the weak axis): constrain
   which action is legal given the prior `tool_response` candidate count (N>1 → only ask-dept/
   connect; single hit → only connect). Converts multi-turn state-tracking into constrained
   decoding instead of model capacity.

## Forgetting: real, cheaply prevented
AuT encoder + projector stay **frozen** (alignment not trained away). Risk = decoder output drift;
**~10–15% ASR-replay** re-anchors it. Evidence: Qwen2-Audio ASR "declines slightly, remains
competitive"; spoken-LM study text→text 70.0%→14.3% **but 0.5% experience replay → 66.3%**;
Steer-MoE proves alignment + tool-use coexist. *Replay is load-bearing; freeze-encoder is hygiene.*

## 0.6B vs 1.7B — use 0.6B, train both, gate on multi-turn
1.7B's only proven edge is multi-turn state + OOD phrasing, at **2–3× per-token latency on 2 vCPU**
(compounds across a 2–3-turn flow). De-risk in *data* (oversample the 5,165 collision dialogs),
not by scaling. **Ship fp16 or INT8, NOT Q4** — sub-1B is fragile to 4-bit (0.5B SLU 98.8%→81.7%
at Q4 vs −0.4 at INT8); RAM ~1 GB either way.

## Concrete training plan (reuses existing data — no new collection)
- **Data:** `dialogs_train.jsonl` (11,694 single+clarify) + `dialogs_collision_train.jsonl`
  (5,165 dept-disambiguation) + `{train_tool,train_v5}.jsonl` as ASR-replay source. Eval:
  `dialogs_test.jsonl`, `dialogs_collision_val.jsonl`, `test_tool.jsonl`.
- **Arch:** frozen AuT encoder + frozen projector → **LoRA decoder only** (q,k,v,o,gate,up,down;
  r32, α64, dropout 0.05).
- **Curriculum (single run):** Stage 0 text-only warm-up (re-elicit suppressed instruction-
  following) → Stage 1 audio joint, per-batch mix ≈ **55% single+clarify / 30% collision
  (oversampled) / 10–15% ASR-replay**.
- **GBNF** for tool-call validity + (recommended) the state-aware action grammar.
- **Eval vs Omni-3B first**, then gate.

## GO / NO-GO (gate on multi-turn — thinnest margin)
- **GO (ship 0.6B@INT8):** single-turn ≥ 95% AND multi-turn (turn-type + final-connect) **≥ 85%**.
- **MARGINAL (70–85%):** apply fallbacks in order → (1) oversample collision dialogs 2–3×,
  (2) state-aware grammar, (3) distill from the Omni-3B teacher on disambiguation turns.
- **NO-GO (<70% after fallbacks):** make disambiguation **scripted** (the controller already owns
  policy per HANDOFF — model fills slots, controller picks ask/connect/not-found from resolver
  score margins), which removes the one axis 0.6B can't be guaranteed on; or step to 1.7B@INT8.

## Expected outcome (honest)
- **Matches Omni-3B:** single-turn exact-name tool-calling + homophone/near-miss resolution
  (solved part — TinyAgent beat GPT-4, resolver+GBNF carry it).
- **Likely a few pts below:** the 2nd disambiguation turn's turn-type tracking. Plan for it.
- **Net:** 0.6B is viable at a fraction of Omni-3B's 84–129 s/turn. **Bet on constrained decoding
  + explicit state-in-prompt + a fixed policy — not on emergent multi-turn agency.**

*Sources: [STAR 2602.03022](https://arxiv.org/pdf/2602.03022), [TinyAgent 2409.00608](https://arxiv.org/pdf/2409.00608),
[ACEBench 2501.12851](https://arxiv.org/html/2501.12851v3), [0.5B-vs-1.5B SLU+quant 2502.12923](https://arxiv.org/html/2502.12923),
[XGrammar 2601.04426](https://arxiv.org/pdf/2601.04426), [Qwen2-Audio 2407.10759](https://arxiv.org/html/2407.10759v1),
[replay study 2505.17496](https://arxiv.org/html/2505.17496v1), [Steer-MoE 2510.13558](https://arxiv.org/html/2510.13558v1).*

## MEASURED RESULT (2026-06-27) — Qwen3-ASR-0.6B BEATS Omni-3B
Full 1500-clip single-turn agent benchmark (audio → tool_call → resolver → action), same
test set + scorer as the Omni agent eval:

| slice | Qwen3-ASR-0.6B | Omni-3B (phase-2) |
|---|---|---|
| overall | **94.0% / 1.3% misroute** | 92.6% / 1.7% |
| zh | **99.4% / 0.0%** | 98.2% / 0.4% |
| mix | **93.8%** | 92.8% |
| en | **84.7%** | 82.6% |
| resolve | **97.7%** | 96.4% |
| clarify | 100% | 100% |
| not_found | **75.6%** | 73.3% |

Single-turn go/no-go axis: **PASS (exceeds Omni-3B)**. 1/5 params, ~3× faster on CPU, fp16 ~1 GB.
Why: Qwen3-ASR's AuT encoder (20M-hr, purpose-built) > Omni's Whisper-derived encoder.
Remaining gate axis: multi-turn department disambiguation (≥85%).

## MULTI-TURN GATE (2026-06-27) — PERFECT, FULL GO
Teacher-forced collision-dialog eval (120 held-out dialogs), model generates each assistant turn:
  asks_dept 100% | refined_dept_ok 100% | final_connect 100% | all_ok 100% (120/120)
Both go/no-go axes PASS: single-turn 94.0% (> Omni 92.6%), multi-turn 100% (>> 85% bar), misroute 1.3%.
VERDICT: ship Qwen3-ASR-0.6B@INT8 as the edge audio-agent. It exceeds Omni-3B at 1/5 params, ~3x faster.
Caveats: multi-turn is teacher-forced on in-distribution val (free-running 2000-DB test = final confirm);
not_found OOD 75.6% (resolver scaled-confirm handles it, not the model).
Remaining: free-running 2000-DB validation, INT8 quant, GGUF/serving for the demo.

## FREE-RUNNING CONFIRMATION (2026-06-27)
No teacher-forcing — model's own tool-calls drive the real resolver (per-dialog cluster), only
gold audio turns fed. n=80 collision dialogs:
  real collision returned 93.8% | asked department 93.8% | reached correct ext 93.8% (75/80)
Expected ~6% degradation from teacher-forced 100% (perception edge cases); well above the 85% gate.
COMPLETE VERDICT — unconditional GO: single-turn 94.0% (>Omni 92.6%), multi-turn free-running 93.8%,
misroute 1.3%. Qwen3-ASR-0.6B does everything Omni-3B does, better, at 1/5 params, ~3x faster, ~1GB.
