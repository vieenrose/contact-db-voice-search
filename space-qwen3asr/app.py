"""Qwen3-ASR-0.6B-Agent voice attendant — audio + LiveKit-style tool calling, in-process CPU.

The browser records a request → Qwen3-ASR-0.6B-Agent (our fine-tune; frozen AuT encoder + LoRA
decoder) HEARS the name and emits a search_contacts tool call → tools.py/resolver grounds it →
reply. Runs in plain transformers on CPU at ~4-8 s/turn (vs the 3B Omni's 84-129 s) — and is more
accurate (94.0% vs 92.6%, zh 99.4%). The model does perception; the tool layer does retrieval.
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from resolver import Resolver
from tools import registry, parse_tool_calls

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
ROOT = Path(__file__).resolve().parent
BASE_ID = "Qwen/Qwen3-ASR-0.6B"
ADAPTER = os.environ.get("ADAPTER", "Luigi/Qwen3-ASR-0.6B-Agent")
SYS = ('You are a phone attendant for a Taiwan office. To find a colleague, call the directory '
       'tool by writing exactly <tool_call>{"name":"search_contacts","arguments":{"query":"<name as '
       'heard>"}}</tool_call>. If several people share that name, ask which department, then call the '
       'tool again adding "department":"<dept>". After a unique result, connect the caller; if none '
       'match, say it was not found. Ignore the caller\'s own name.')

R = Resolver(str(ROOT / "directory.csv"))
N = len(R.contacts)
M = {}                       # filled at startup: proc, model
app = FastAPI(title="Qwen3-ASR-0.6B-Agent voice attendant")


@app.on_event("startup")
def _load():
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    from peft import PeftModel
    M["proc"] = Qwen3ASRProcessor.from_pretrained(BASE_ID)
    base = Qwen3ASRForConditionalGeneration.from_pretrained(BASE_ID, dtype=torch.float32)
    M["model"] = PeftModel.from_pretrained(base.thinker.eval(), ADAPTER).eval()
    print("Qwen3-ASR-0.6B-Agent ready", flush=True)


def load_wav_16k(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32")
    if getattr(w, "ndim", 1) > 1:
        w = w.mean(-1)
    if sr != 16000:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    return w


def to_wav(data: bytes, suffix: str) -> str:
    src = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    src.write(data); src.close()
    wav = src.name + ".wav"
    subprocess.run(["ffmpeg", "-y", "-i", src.name, "-ar", "16000", "-ac", "1", wav],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return wav


def model_toolcall(wav):
    proc, model = M["proc"], M["model"]
    text = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n<|audio_pad|><|im_end|>\n"
            f"<|im_start|>assistant\n")
    enc = proc(text=text, audio=[load_wav_16k(wav)], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=48, do_sample=False, eos_token_id=151645)
    return proc.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def compose(matches):
    if len(matches) == 1:
        m = matches[0]
        return {"kind": "resolve", "title": "✅ Located",
                "detail": f"{m['name']} · extension {m['ext']} · {m.get('dept','')}",
                "say": f"Connecting you to {m['name']}, extension {m['ext']}."}
    if len(matches) >= 2:
        ns = ", ".join(m["name"] for m in matches[:3])
        return {"kind": "clarify", "title": "🤔 Needs clarification",
                "detail": f"Several matches: {ns}", "say": f"I found several — {ns}. Which one?"}
    return {"kind": "not_found", "title": "🚫 Not found", "detail": "No confident match.",
            "say": "Sorry, I couldn't find that name in the directory."}


@app.post("/listen")
async def listen(audio: UploadFile = File(...)):
    t0 = time.time()
    data = await audio.read()
    ext = "." + (audio.filename.rsplit(".", 1)[-1] if "." in (audio.filename or "") else "webm")
    raw = model_toolcall(to_wav(data, ext))
    calls = parse_tool_calls(raw)
    secs = round(time.time() - t0, 1)
    if not calls:
        return {"empty": True, "raw": raw[:300], "secs": secs}
    query = (calls[0].get("arguments") or {}).get("query", "")
    matches = registry.dispatch("search_contacts", {"query": query})
    ranked = R.rank(query, k=6)
    cands = [{"rank": i + 1, "name": c.name, "zh": c.zh, "dept": c.dept, "ext": c.ext, "score": round(s, 1)}
             for i, (s, c) in enumerate(ranked)]
    return {"empty": False, "query": query, "tool_call": calls[0], "tool_response": matches,
            "candidates": cands, "decision": compose(matches), "secs": secs}


@app.get("/health")
def health():
    return {"ok": bool(M), "contacts": N}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("__N__", str(N))


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-ASR-0.6B-Agent — voice attendant</title>
<style>
 :root{color-scheme:light dark} *{box-sizing:border-box}
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:0 auto;padding:24px;background:#fafafa;color:#1c1c1f}
 h1{font-size:1.3rem;margin:0 0 4px} .sub{color:#555;font-size:.92rem;margin-bottom:16px}
 button{padding:12px 20px;font-size:1rem;border:0;border-radius:10px;background:#3b5bdb;color:#fff;cursor:pointer}
 button.rec{background:#b42318} button:disabled{opacity:.5}
 .status{margin:12px 0;font-size:.95rem;min-height:1.4em}
 .card{background:#fff;border:1px solid #e6e6e6;border-radius:10px;padding:14px;margin-top:14px}
 .card h3{margin:0 0 8px;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:#777}
 pre{background:#0d1117;color:#c9d1d9;padding:10px;border-radius:8px;overflow-x:auto;font-family:ui-monospace,Menlo,monospace;font-size:.82rem;margin:4px 0}
 .step{font-weight:600;font-size:.82rem;color:#444;margin-top:8px}
 table{width:100%;border-collapse:collapse;font-size:.88rem} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #eee} th{color:#777}
 tr.top td{background:#eef2ff;font-weight:600}
 .decision{font-size:1.05rem} .decision.resolve{color:#18794e}.decision.clarify{color:#9a6700}.decision.not_found{color:#b42318}
 .say{font-style:italic;color:#333;border-left:3px solid #c7c7c7;padding-left:10px;margin-top:6px}
 .spinner{display:inline-block;width:14px;height:14px;border:2px solid #ccc;border-top-color:#3b5bdb;border-radius:50%;animation:spin 1s linear infinite;vertical-align:middle;margin-right:6px}
 @keyframes spin{to{transform:rotate(360deg)}}
 @media(prefers-color-scheme:dark){body{background:#0f0f11;color:#e7e7ea}.sub,th{color:#a0a0a8}.card{background:#17171a;border-color:#2a2a2e}tr.top td{background:#1e2547}th,td{border-color:#26262b}.say{color:#cfcfd4}}
</style></head><body>
<h1>☎️ Qwen3-ASR-0.6B-Agent — voice attendant</h1>
<div class="sub">Speak a request (zh-TW or English). <b>Qwen3-ASR-0.6B-Agent</b> (our fine-tune — a
 0.6B that beats Omni-3B) runs <b>in transformers on CPU</b>: it <b>hears the name</b> → emits a
 <code>search_contacts</code> tool call → <code>tools.py</code> grounds it against the live directory
 (<b>__N__ contacts</b>) → connects / clarifies / rejects. <b>~5-10 s/turn</b> (vs the 3B's ~90 s).</div>
<button id="rec" onclick="toggle()">🎙️ Start recording</button>
<div class="status" id="status">Tip: “可以幫我接蔡孟儒嗎” or “I'd like to reach Coco Kuo”.</div>
<div id="out" style="display:none">
 <div class="card"><h3>🔁 Tool-calling trace</h3>
  <div class="step">1 · 🎙️ you → 🤖 Qwen3-ASR-0.6B-Agent → tool call</div><pre id="tc"></pre>
  <div class="step">2 · 🔧 search_contacts → tool response</div><pre id="tr"></pre>
  <div class="step">3 · 🤖 reply</div><div id="decision" class="decision"></div><div id="say" class="say"></div></div>
 <div class="card"><h3>DB candidates (ranked)</h3>
  <table><thead><tr><th>#</th><th>name</th><th>中文名</th><th>dept</th><th>ext</th><th>score</th></tr></thead>
  <tbody id="cands"></tbody></table></div>
</div>
<script>
let rec,chunks=[],recording=false,t0=0,timer=null;
async function toggle(){
 const b=document.getElementById('rec');
 if(!recording){let s;try{s=await navigator.mediaDevices.getUserMedia({audio:true})}catch(e){st('🎙️ mic permission denied');return}
  rec=new MediaRecorder(s);chunks=[];rec.ondataavailable=e=>chunks.push(e.data);rec.onstop=submit;rec.start();recording=true;
  b.textContent='⏹️ Stop & send';b.classList.add('rec');st('🔴 recording… speak, then Stop');}
 else{rec.stop();rec.stream.getTracks().forEach(t=>t.stop());recording=false;b.textContent='🎙️ Start recording';b.classList.remove('rec');b.disabled=true;}}
function st(h){document.getElementById('status').innerHTML=h}
async function submit(){
 const fd=new FormData();fd.append('audio',new Blob(chunks,{type:'audio/webm'}),'rec.webm');
 t0=Date.now();timer=setInterval(()=>st('<span class="spinner"></span> Qwen3-ASR-0.6B-Agent listening… '+Math.round((Date.now()-t0)/1000)+'s'),500);
 let d;try{const r=await fetch('listen',{method:'POST',body:fd});d=await r.json();}catch(e){clearInterval(timer);st('failed: '+e);rs();return;}
 clearInterval(timer);rs();
 if(d.empty){st('🤖 no tool call: '+(d.raw||''));return;}
 st('✅ done in '+d.secs+'s');document.getElementById('out').style.display='block';
 document.getElementById('tc').textContent=JSON.stringify(d.tool_call);
 document.getElementById('tr').textContent=JSON.stringify(d.tool_response);
 const dc=document.getElementById('decision');dc.className='decision '+d.decision.kind;dc.innerHTML='<b>'+d.decision.title+'</b> — '+d.decision.detail;
 document.getElementById('say').textContent='“'+d.decision.say+'”';
 document.getElementById('cands').innerHTML=d.candidates.map(c=>'<tr class="'+(c.rank===1&&d.decision.kind!=='not_found'?'top':'')+'"><td>'+c.rank+'</td><td>'+c.name+'</td><td>'+c.zh+'</td><td>'+c.dept+'</td><td>'+c.ext+'</td><td>'+c.score+'</td></tr>').join('');}
function rs(){const b=document.getElementById('rec');b.disabled=false;b.textContent='🎙️ Start recording';}
</script></body></html>"""
