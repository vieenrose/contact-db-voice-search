"""Taiwan-office phone attendant — STOCK Ultravox baseline (free CPU, push-to-talk).

Architecture mirrors the real design: Ultravox does PERCEPTION (hear the requested
name); resolver.py does POLICY (ground to a real extension over the closed 200-set).
This is the *untrained* baseline — it shows how stock Ultravox copes with zh-TW/en
code-switched names, i.e. the gap fine-tuning is meant to close.

Record a request, submit, wait (a 1B on free CPU is slow). The Ultravox base LLM
(Llama-3.2-1B) is gated, so the Space needs an HF_TOKEN secret with access.
"""
import json
import time
from fractions import Fraction

import numpy as np
import gradio as gr

# Work around a gradio_client schema bug that crashes /api/info:
# "argument of type 'bool' is not iterable" when a JSON schema value is a bool.
import gradio_client.utils as _gcu
_orig_jstp = _gcu._json_schema_to_python_type
_orig_get_type = _gcu.get_type
_gcu._json_schema_to_python_type = lambda s, d=None: "Any" if isinstance(s, bool) else _orig_jstp(s, d)
_gcu.get_type = lambda s: "Any" if not isinstance(s, dict) else _orig_get_type(s)

import torch
import transformers
from scipy.signal import resample_poly

from resolver import Resolver

# spaces.GPU only exists on ZeroGPU Spaces; no-op on CPU so import never crashes.
try:
    import spaces
    GPU = spaces.GPU
except Exception:
    def GPU(fn):
        return fn

MODEL_ID = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
ASR_SR = 16000

R = Resolver("directory.csv")

# Optional spoken reply via Kokoro (numpy out, no ffmpeg). Degrades to text-only.
try:
    from fastrtc import get_tts_model
    _tts = get_tts_model()
except Exception:
    _tts = None

SYS = (
    "You are a telephone operator at a Taiwan company. The caller asks to be connected "
    "to a colleague. The name is usually an English first name plus a Chinese surname "
    "(e.g. 'Kevin Chen'), and may be spoken in English, Mandarin, or a mix. "
    "Reply with ONLY the person's name as you heard it — no greetings, no other words."
)

pipe = transformers.pipeline(model=MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
_on_gpu = False


@GPU
def hear_name(audio_16k: np.ndarray) -> str:
    global _on_gpu
    if not _on_gpu and torch.cuda.is_available():
        pipe.model.to("cuda")
        pipe.device = torch.device("cuda")
        _on_gpu = True
    turns = [{"role": "system", "content": SYS}]
    out = pipe({"audio": audio_16k, "turns": turns, "sampling_rate": ASR_SR}, max_new_tokens=24)
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return str(out[0].get("generated_text", "")).strip()
    return str(out).strip()


def to_16k_mono(sr: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    x = x.reshape(-1)
    if np.max(np.abs(x)) > 1.5:          # int16-range -> normalize
        x = x / 32768.0
    if sr != ASR_SR:
        fr = Fraction(ASR_SR, sr).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    return x


def reply_text(action: dict) -> str:
    a = action.get("action")
    if a == "resolve":
        return f"Connecting you to {action['name']} in {action['dept']}, extension {action['ext']}."
    if a == "clarify":
        names = ", ".join(c["name"] for c in action.get("candidates", [])[:3])
        return f"I found a few possible matches: {names}. Which one did you mean?"
    return "Sorry, I couldn't find that name in the directory. Could you repeat it?"


def attend(audio):
    if audio is None:
        return "🎙️ Please record a request first.", None
    sr, arr = audio
    t0 = time.time()
    x16 = to_16k_mono(sr, arr)
    heard = hear_name(x16)
    action = R.resolve(heard)
    reply = reply_text(action)
    dt = time.time() - t0
    md = (f"🗣️ **heard:** {heard}\n\n"
          f"🔎 **resolver:** `{json.dumps(action, ensure_ascii=False)}`\n\n"
          f"📞 **reply:** {reply}\n\n"
          f"<sub>⏱️ {dt:.1f}s on CPU — untrained baseline</sub>")
    audio_out = None
    if _tts is not None:
        try:
            audio_out = _tts.tts(reply)
        except Exception:
            audio_out = None
    return md, audio_out


with gr.Blocks(title="Taiwan Office Attendant — Ultravox baseline") as demo:
    gr.Markdown(
        "# ☎️ Taiwan Office Attendant — Ultravox baseline (untrained)\n"
        "Record a request like *“Could you put me through to Kevin Chen?”* or "
        "*“我要找陳凱文”*, then press **Find extension**. Ultravox hears the name; the "
        "resolver grounds it to an extension in a closed 200-person directory.\n\n"
        "**Note:** free CPU + a 1B model → expect this to be *slow* (tens of seconds). "
        "It is the untrained baseline that shows the gap fine-tuning closes."
    )
    inp = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your request")
    btn = gr.Button("Find extension", variant="primary")
    out_md = gr.Markdown()
    out_audio = gr.Audio(label="Attendant reply", autoplay=True)
    btn.click(attend, inputs=inp, outputs=[out_md, out_audio])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssr_mode=False, show_api=False)
