---
title: Taiwan Office Attendant (Ultravox baseline)
emoji: ☎️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: zh-TW/en phone attendant — stock Ultravox baseline
---

# ☎️ Taiwan Office Attendant — Ultravox baseline (untrained)

Real-time voice demo of a **zh-TW/en telephone auto-attendant** that finds a
colleague's **extension** from a spoken name (English first name + Chinese surname,
often code-switched). Speak into the mic: *"Could you put me through to Kevin Chen?"*
or *"我要找陳凱文"*.

## How it works
- **Ultravox** (`fixie-ai/ultravox-v0_5-llama-3_2-1b`) does **perception** — it hears
  the requested name.
- **`resolver.py`** does **policy** — fuzzy-matches the heard name to a real row in a
  closed 200-person directory and returns the extension (resolve / clarify / not-found).
- **Push-to-talk** (record → submit); Kokoro TTS speaks the confirmation.

> Runs on a **free CPU** Space, so a 1B model is **slow** (tens of seconds per request).
> The Ultravox base (Llama-3.2-1B) is gated — the Space uses an `HF_TOKEN` secret with access.
> This is the untrained baseline, not the optimized on-device build.

## This is a baseline
The model is **not fine-tuned**. It shows how *stock* Ultravox handles Taiwan
code-switched names over voice — i.e. the gap that domain fine-tuning + a
telephony-adapted whisper-base encoder are meant to close for on-device (Jetson Nano)
deployment.
