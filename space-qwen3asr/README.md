---
title: Qwen3-ASR-0.6B-Agent Voice Attendant
emoji: ☎️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: 0.6B speech agent that beats Omni-3B; ~5s/turn on CPU
---

# ☎️ Qwen3-ASR-0.6B-Agent — voice attendant (audio, on CPU)

Speak a request (zh-TW or English). **[Qwen3-ASR-0.6B-Agent](https://huggingface.co/Luigi/Qwen3-ASR-0.6B-Agent)**
— our fine-tune of Qwen3-ASR-0.6B — runs **in plain transformers on CPU**: it *hears the name* →
emits a `search_contacts` **tool call** → `tools.py`/resolver grounds it to a real extension →
connects / clarifies / rejects.

A **0.6B that beats Qwen2.5-Omni-3B** on this task (single-turn **94.0%** vs 92.6%, Mandarin
**99.4%** vs 98.2%) at **1/5 the params** and **~5–10 s/turn on CPU** (vs the 3B's ~90 s) — the
edge-feasible "Qwen3-Omni-0.6B" that doesn't otherwise exist.

## How it works
1. Browser records audio → `POST /listen`
2. Qwen3-ASR-0.6B-Agent (frozen AuT encoder + LoRA decoder) → `<tool_call>{"name":"search_contacts",...}</tool_call>`
3. `tools.py` parses + dispatches the call against the directory → ranked matches
4. The reply (connect / clarify / not-found) is shown with the tool-call trace

Models pulled at startup: `Qwen/Qwen3-ASR-0.6B` (base) + `Luigi/Qwen3-ASR-0.6B-Agent` (our LoRA).
Apache-2.0. Not affiliated with Alibaba/Qwen.
