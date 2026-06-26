"""Taiwan Office Attendant — LiveKit-style tool-calling demo (no LiveKit).

Shows the agent loop that powers the attendant: a heard name → the model emits a
`search_contacts` **tool call** → our `tools.py` registry **dispatches** it against the
live directory → the ranked matches are fed back as the **tool response** → the model
**replies** (connect / clarify / not-found).

The tool layer (registry + parser + dispatch) is the *exact same* `tools.py` used by the
fine-tuned Qwen-Omni agent — registered just like LiveKit's `@function_tool`, but with no
LiveKit dependency. On this free CPU Space the model's turn is simulated (the 3B needs a
GPU); the tool *protocol and dispatch are real*. Swapping in the GPU model later changes
only who emits the tool call — the loop, tools, and DB are unchanged.
"""
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import gradio as gr
from scipy.signal import resample_poly

from resolver import Resolver
from tools import registry, parse_tool_calls

# Gradio 5.9.x get_api_info() (run at launch even with show_api=False) crashes on boolean
# JSON schemas -> "TypeError: argument of type 'bool' is not iterable", which cascades into
# a "localhost not accessible" launch failure. Guard the recursion against bool schemas.
import gradio_client.utils as _gcu
_orig_js2pt = _gcu._json_schema_to_python_type
def _safe_js2pt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_js2pt(schema, defs)
_gcu._json_schema_to_python_type = _safe_js2pt

R = Resolver("directory.csv")
N = len(R.contacts)
TOOL_SCHEMA_JSON = json.dumps(registry.schemas(), indent=2, ensure_ascii=False)

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


def compose_reply(matches):
    """The model's spoken turn, reasoned purely from the tool response (as the real
    model does): one match → connect, several → clarify, none → reject. No misroute."""
    if len(matches) == 1:
        m = matches[0]
        return "resolve", (f"## ✅ Located\n# {m['name']}\n"
                           f"### 📞 extension **{m['ext']}**　·　{m.get('dept','')}\n"
                           f"> “Connecting you to {m['name']}, extension {m['ext']}.”")
    if len(matches) >= 2:
        names = " / ".join(f"**{m['name']}**" for m in matches[:3])
        return "clarify", (f"## 🤔 Needs clarification\nSeveral strong matches: {names}\n\n"
                           f"> “I found several — {', '.join(m['name'] for m in matches[:3])}. "
                           f"Which one would you like?”")
    return "not_found", ("## 🚫 Not found\nNo confident match in the directory.\n\n"
                         "> “Sorry, I couldn't find that name in the directory.” *(rejected, not misrouted)*")


def trace_md(query, tool_call_obj, matches):
    """Render the LiveKit-style tool-call protocol exactly as it flows through the loop."""
    tc = json.dumps(tool_call_obj, ensure_ascii=False)
    tr = json.dumps(matches, ensure_ascii=False)
    return (
        "#### 🔁 Tool-calling trace (LiveKit-style)\n"
        "**1 · 🤖 assistant → tool call**\n"
        f"```json\n{tc}\n```\n"
        "**2 · 🔧 `search_contacts` → tool response** *(live directory, distance-scored)*\n"
        f"```json\n{tr}\n```\n"
        "**3 · 🤖 assistant → reply** *(reasons from the tool response below)*")


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
           f"🔍 calling `search_contacts` over **{N}** contacts…", pd.DataFrame(), pd.DataFrame(), "")

    # 2) the LiveKit-style loop — REAL tool layer: the model emits a Hermes tool call,
    #    tools.py parses it and dispatches it against the live directory.
    tool_call_obj = {"name": "search_contacts", "arguments": {"query": query}}
    model_out = f"<tool_call>{json.dumps(tool_call_obj, ensure_ascii=False)}</tool_call>"
    call = parse_tool_calls(model_out)[0]                  # parse (handles Hermes + OpenAI)
    matches = registry.dispatch(call["name"], call["arguments"])   # dispatch via the registry

    # 3) viz: full ranked DB + the model's reply reasoned from the tool response
    ranked = R.rank(query, k=6)
    rows = [{"rank": i + 1, "name": c.name, "中文名": c.zh, "dept": c.dept,
             "ext": c.ext, "score": round(s, 1)} for i, (s, c) in enumerate(ranked)]
    df = pd.DataFrame(rows)
    plot_df = pd.DataFrame({"contact": [r["name"] for r in rows], "score": [r["score"] for r in rows]})
    _, card = compose_reply(matches)

    yield (f"### 🗣️ heard: **{heard}**\n**query →** `{query}`",
           trace_md(query, tool_call_obj, matches), df, plot_df, card)


EXAMPLES = [[None, "蔡孟儒"], [None, "Coco Kuo"], [None, "周宜蓁"],
            [None, "Tseng"], [None, "Carol Hsieh"], [None, "David Miller"]]

with gr.Blocks(title="Taiwan Attendant — LiveKit-style tool calling") as demo:
    gr.Markdown(
        "# ☎️ Taiwan Office Attendant — LiveKit-style tool calling *(no LiveKit)*\n"
        "A heard name → the model emits a **`search_contacts` tool call** → our `tools.py` registry "
        "**dispatches** it against the **live directory** → the ranked matches come back as the "
        "**tool response** → the model **connects**, **clarifies**, or **rejects** an unknown name.\n\n"
        "The registry + parser + dispatch loop is the *same* code the fine-tuned Qwen-Omni agent uses — "
        "registered just like LiveKit's `@function_tool`, with **zero LiveKit dependency**. "
        "*(On free CPU the model's turn is simulated; the tool protocol & dispatch are real.)*\n\n"
        "Try a real name (`蔡孟儒`), a surname only (`Tseng` → clarify), or an unknown one "
        "(`David Miller` → not found). The directory is a CSV — *edit it, no retraining*.")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="🎙️ Speak a request")
            text_in = gr.Textbox(label="…or type a name", placeholder="蔡孟儒 / Kevin Chen / Tseng / David Miller")
            btn = gr.Button("🔍 Run agent turn", variant="primary")
            gr.Examples(EXAMPLES, inputs=[audio_in, text_in], label="Try these")
            with gr.Accordion("🔧 Registered tools (LiveKit-style schema)", open=False):
                gr.Markdown("Auto-derived from each Python function's signature — the same OpenAI "
                            "tool schema LiveKit builds for `@function_tool`:")
                gr.Code(TOOL_SCHEMA_JSON, language="json")
        with gr.Column(scale=2):
            heard_md = gr.Markdown()
            trace = gr.Markdown()
            cand_df = gr.Dataframe(label="DB candidates (ranked by distance score)", interactive=False)
            score_plot = gr.BarPlot(x="contact", y="score", title="match score", y_lim=[0, 100], height=200)
            result_md = gr.Markdown()
    btn.click(search, [audio_in, text_in], [heard_md, trace, cand_df, score_plot, result_md])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssr_mode=False, show_api=False)
