"""Taiwan Office Attendant — LiveKit-style tool-calling demo (no LiveKit).

A typed name → the model emits a `search_contacts` tool call → our `tools.py` registry
dispatches it against the live directory → the ranked matches come back as the tool
response → the model connects / clarifies / rejects. The registry + parser + dispatch
loop is the SAME code the fine-tuned Qwen-Omni agent uses (registered like LiveKit's
@function_tool), with zero LiveKit dependency.

Text-only on this free CPU Space: the 3B audio model needs a GPU, and the tool protocol
is what this demo is about. Voice input returns with the GPU-served model.
"""
import json

import pandas as pd
import gradio as gr

from resolver import Resolver
from tools import registry, parse_tool_calls

# Defensive guard: Gradio 5.9.x get_api_info() crashes on boolean JSON schemas
# ("TypeError: argument of type 'bool' is not iterable"). Harmless once the offending
# component (gr.Audio) is gone, but kept so a future component can't reintroduce it.
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
DIRECTORY_DF = pd.DataFrame([{"name": c.name, "中文名": c.zh, "dept": c.dept, "ext": c.ext}
                             for c in R.contacts])


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


def search(typed):
    query = (typed or "").strip()
    if not query:
        return "Type a name above.", "", pd.DataFrame(), pd.DataFrame(), ""

    # the LiveKit-style loop — REAL tool layer: the model emits a Hermes tool call,
    # tools.py parses it and dispatches it against the live directory.
    tool_call_obj = {"name": "search_contacts", "arguments": {"query": query}}
    model_out = f"<tool_call>{json.dumps(tool_call_obj, ensure_ascii=False)}</tool_call>"
    call = parse_tool_calls(model_out)[0]
    matches = registry.dispatch(call["name"], call["arguments"])

    ranked = R.rank(query, k=6)
    rows = [{"rank": i + 1, "name": c.name, "中文名": c.zh, "dept": c.dept,
             "ext": c.ext, "score": round(s, 1)} for i, (s, c) in enumerate(ranked)]
    df = pd.DataFrame(rows)
    plot_df = pd.DataFrame({"contact": [r["name"] for r in rows], "score": [r["score"] for r in rows]})
    _, card = compose_reply(matches)
    return (f"### 🗣️ query: **{query}**", trace_md(query, tool_call_obj, matches), df, plot_df, card)


EXAMPLES = [["蔡孟儒"], ["Coco Kuo"], ["周宜蓁"], ["Tseng"], ["Carol Hsieh"], ["David Miller"]]

with gr.Blocks(title="Taiwan Attendant — LiveKit-style tool calling") as demo:
    gr.Markdown(
        "# ☎️ Taiwan Office Attendant — LiveKit-style tool calling *(no LiveKit)*\n"
        "Type a name → the model emits a **`search_contacts` tool call** → our `tools.py` registry "
        "**dispatches** it against the **live directory** → the ranked matches come back as the "
        "**tool response** → the model **connects**, **clarifies**, or **rejects** an unknown name.\n\n"
        "The registry + parser + dispatch loop is the *same* code the fine-tuned Qwen-Omni agent uses — "
        "registered just like LiveKit's `@function_tool`, with **zero LiveKit dependency**. "
        "*(Text-only on free CPU; voice input returns with the GPU-served model.)*\n\n"
        "Try a real name (`蔡孟儒`), a surname only (`Tseng` → clarify), or an unknown one "
        "(`David Miller` → not found). The directory is a CSV — *edit it, no retraining*.")
    with gr.Row():
        with gr.Column(scale=1):
            text_in = gr.Textbox(value="蔡孟儒", label="Type a name",
                                 placeholder="蔡孟儒 / Kevin Chen / Tseng / David Miller")
            btn = gr.Button("🔍 Run agent turn", variant="primary")
            gr.Examples(EXAMPLES, inputs=[text_in], label="Try these")
            with gr.Accordion(f"📇 The live directory — {N} contacts (edit the CSV, no retraining)", open=False):
                gr.Dataframe(DIRECTORY_DF, interactive=False, max_height=320)
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
    outputs = [heard_md, trace, cand_df, score_plot, result_md]
    btn.click(search, [text_in], outputs, api_name="search")
    text_in.submit(search, [text_in], outputs)
    demo.load(search, [text_in], outputs)   # populate a result on page load (DB is live)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name="0.0.0.0", ssr_mode=False, show_error=True)
