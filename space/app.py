"""Taiwan Office Attendant — v5 agentic perception→retrieval demo.

Shows the WINNING design: a heard name → the live contact DB is searched with
distance scores → the person is located (or clarify / not-found). The resolver here
is the exact component-aware one from the 92.8%-success / 1.8%-misroute benchmark.

Type a name (English, 中文, or an unknown one) to see the DB query faithfully, or
speak (a small CPU Whisper transcribes — rougher than the GPU Qwen-Omni-3B used in
the benchmark, but the DB-query view is identical).
"""
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import gradio as gr
from scipy.signal import resample_poly

from resolver import Resolver, _norm_en, _to_pinyin

R = Resolver("directory.csv")
N = len(R.contacts)

# Optional CPU transcription. Lazy + guarded so text input always works.
_asr = None
def transcribe(sr, wav):
    global _asr
    try:
        if _asr is None:
            from faster_whisper import WhisperModel
            _asr = WhisperModel("small", device="cpu", compute_type="int8")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(-1)
        if np.max(np.abs(wav)) > 1.5:
            wav = wav / 32768.0
        if sr != 16000:
            fr = Fraction(16000, sr).limit_denominator(1000)
            wav = resample_poly(wav, fr.numerator, fr.denominator).astype(np.float32)
        segs, _ = _asr.transcribe(wav, language=None, beam_size=1)
        return "".join(s.text for s in segs).strip()
    except Exception as e:
        return f"[transcription unavailable: {type(e).__name__}]"

# strip common request filler so the residual is the name (the FT model does this implicitly)
FILLER = ["我要找", "請幫我轉接", "麻煩幫我接", "可以幫我接", "請問", "的分機是多少", "的分機幾號",
          "幫我接", "幫我轉", "麻煩轉", "找", "在嗎", "嗎", "您好", "你好", "謝謝", "先生", "小姐",
          "could you put me through to", "i'd like to speak to", "can i get", "transfer me to",
          "what's the extension for", "connect me with", "please", "extension", "'s", "hi", "hello",
          "i'm looking for", "is", "there", "the", "?", ",", ".", "之分機"]
def extract_name(text):
    t = text
    for f in sorted(FILLER, key=len, reverse=True):
        t = t.replace(f, " ")
    return " ".join(t.split()).strip() or text


def search(audio, typed):
    # 1) get the heard query
    if typed and typed.strip():
        heard, query = typed.strip(), typed.strip()
    elif audio is not None:
        sr, wav = audio
        heard = transcribe(sr, wav)
        query = extract_name(heard)
    else:
        yield "Type a name or record a request.", "", pd.DataFrame(), pd.DataFrame(), ""
        return

    yield (f"### 🗣️ heard: **{heard}**\n**query →** `{query}`",
           f"🔍 searching **{N}** contacts…", pd.DataFrame(), pd.DataFrame(), "")

    # 2) the actual v5 resolver: rank the live DB by distance score
    ranked = R.rank(query, k=6)
    action = R.resolve(query)
    rows = [{"rank": i + 1, "name": c.name, "中文名": c.zh, "dept": c.dept,
             "ext": c.ext, "score": round(s, 1)} for i, (s, c) in enumerate(ranked)]
    df = pd.DataFrame(rows)
    plot_df = pd.DataFrame({"contact": [r["name"] for r in rows],
                            "score": [r["score"] for r in rows]})

    # 3) the decision card
    a = action["action"]
    if a == "resolve":
        card = (f"## ✅ Located\n# {action['name']}　{ranked[0][1].zh}\n"
                f"### 📞 extension **{action['ext']}**　·　{action['dept']}\n"
                f"<sub>top match score {action['score']} — clear winner → connect</sub>")
    elif a == "clarify":
        opts = " / ".join(f"**{c['name']}**" for c in action["candidates"][:3])
        card = (f"## 🤔 Ambiguous — needs clarification\n"
                f"Several strong matches: {opts}\n\n<sub>→ ask the caller which one</sub>")
    else:
        card = (f"## 🚫 Not found\n`{query}` isn't in the directory "
                f"(best score {ranked[0][0]:.0f} < threshold)\n\n<sub>→ reject, don't misroute</sub>")
    yield (f"### 🗣️ heard: **{heard}**\n**query →** `{query}`",
           f"🔎 ranked **{N}** contacts by phonetic distance:", df, plot_df, card)


EXAMPLES = [[None, "蔡孟儒"], [None, "Coco Kuo"], [None, "周宜蓁"],
            [None, "Tseng"], [None, "Carol Hsieh"], [None, "David Miller"]]

with gr.Blocks(title="Taiwan Attendant — DB query view") as demo:
    gr.Markdown(
        "# ☎️ Taiwan Office Attendant — live DB-query view (v5)\n"
        "The model hears a name → the **live contact DB** is searched with distance scores → "
        "the person is **located**, or it asks to **clarify**, or **rejects** an unknown name.\n\n"
        "**Type a name** (English, 中文, or an unknown one) for the exact v5 resolver, or **speak**. "
        "Try a real name (`蔡孟儒`), a surname only (`Tseng` → clarify), or an unknown one "
        "(`Carol Hsieh` → not found). The directory lives in a CSV — *edit it, no retraining*.")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="🎙️ Speak a request")
            text_in = gr.Textbox(label="…or type a name", placeholder="蔡孟儒 / Kevin Chen / Tseng / Carol Hsieh")
            btn = gr.Button("🔍 Find extension", variant="primary")
            gr.Examples(EXAMPLES, inputs=[audio_in, text_in], label="Try these")
        with gr.Column(scale=2):
            heard_md = gr.Markdown()
            status_md = gr.Markdown()
            cand_df = gr.Dataframe(label="DB candidates (ranked by distance score)", interactive=False)
            score_plot = gr.BarPlot(x="contact", y="score", title="match score", y_lim=[0, 100], height=200)
            result_md = gr.Markdown()
    btn.click(search, [audio_in, text_in], [heard_md, status_md, cand_df, score_plot, result_md])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssr_mode=False, show_api=False)
