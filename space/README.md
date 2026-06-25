---
title: Taiwan Office Attendant — DB Query View (v5)
emoji: ☎️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: Live contact-DB query view for the zh-TW/en voice attendant
---

# ☎️ Taiwan Office Attendant — live DB-query view (v5)

Demonstrates the **winning design** from the project (92.8% task success / 1.8% misroute,
**Mandarin 98.4%**): a heard name → the **live contact directory** is searched with phonetic
**distance scores** → the person is **located**, the system **asks to clarify**, or it **rejects**
an unknown name.

## What you see
- **Type a name** (English, 中文, or an unknown one) → the *exact* component-aware resolver from
  the benchmark ranks all 200 contacts by distance and decides resolve / clarify / not-found.
- **Or speak** → a small CPU Whisper transcribes first (rougher than the GPU Qwen-Omni-3B used in
  the benchmark, but the DB-query view is identical).
- The ranked candidate table + score bars show **how the DB is queried** and **which person is located**.

## The key architecture point
The directory lives in **`directory.csv`** — *not in the model weights*. Edit a contact, the
search updates instantly, **no retraining**. Unknown names get low scores → rejected (no misroute).
On the gen1 the resolver runs trivially on CPU; only the perception model needs a GPU/Orin.

Try: `蔡孟儒` (resolve), `Tseng` (surname only → clarify), `David Miller` (unknown → not found).
