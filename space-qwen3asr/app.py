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
    try:                                     # give the attendant a voice (non-fatal if it fails)
        _load_tts()
    except Exception as e:
        print("PrimeTTS load failed (text-only fallback):", e, flush=True)


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


# ──────────────────────────────────────────────────────────────────────────────
# PrimeTTS (Luigi/PrimeTTS) — gives the attendant a VOICE. The model writes the reply
# (in the caller's language); PrimeTTS speaks it. Tiny FastSpeech+Snake-HiFiGAN, ONNX,
# CPU-only, zh-TW + English single voice. Pipeline: text -> bopomofo/arpabet frontend ->
# encoder.onnx -> numpy length-regulate -> decoder.onnx -> vocoder.onnx -> wav.
# ──────────────────────────────────────────────────────────────────────────────
import re as _re

TTS_REPO = os.environ.get("PRIMETTS_REPO", "Luigi/PrimeTTS")
TTS_VARIANT = os.environ.get("PRIMETTS_VARIANT", "v1b_16k")   # flagship 16 kHz (clearest 0-8 kHz band)
TTS = {}                                      # filled lazily / at startup
_TTS_BN = ["frames", "frame_meta", "local_ctx_raw", "abs_pos", "pitch_frame", "frame_mask"]
_TTS_SPLIT = _re.compile(r'(?<=[。！？；;!?\n,，、])')   # split AFTER punctuation, keep delimiter


def _tts_regulate(cond, dur, pitch, abs_bins, max_frames):
    import numpy as np
    c = cond[0]; d = dur[0].astype(np.int64); d[d < 0] = 0
    T, H = c.shape
    frames = np.repeat(c, d, axis=0); Fn = frames.shape[0]
    tok = np.repeat(np.arange(T), d); starts = np.cumsum(d) - d
    within = np.arange(Fn) - starts[tok]; dpf = d[tok].astype(np.float32)
    rel = (within / np.maximum(dpf - 1, 1)).astype(np.float32)
    tc = max(1, int((d > 0).sum())); token_pos = (tok / max(1, tc - 1)).astype(np.float32)
    ld = (np.log1p(dpf) / 6.0).astype(np.float32); center = 1.0 - np.abs(rel * 2 - 1)
    fm = np.stack([rel, 1 - rel, center, np.sin(rel*np.pi), np.cos(rel*np.pi), token_pos, ld, dpf/40.0], -1).astype(np.float32)
    prev = np.concatenate([c[:1], c[:-1]], 0); nxt = np.concatenate([c[1:], c[-1:]], 0)
    lc = np.repeat(np.concatenate([prev, c, nxt], -1), d, axis=0).astype(np.float32)
    pos = np.arange(Fn); ap = np.minimum(pos*abs_bins//max(1, max_frames), abs_bins-1).astype(np.int64)
    pf = np.repeat(pitch[0], d, axis=0).astype(np.float32)
    return {"frames": frames[None].astype(np.float32), "frame_meta": fm[None], "local_ctx_raw": lc[None],
            "abs_pos": ap[None].astype(np.int64), "pitch_frame": pf[None], "frame_mask": np.ones((1, Fn), bool)}


def _load_tts():
    if TTS:
        return TTS
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    import frontend_bopomofo as F
    w = lambda fn: hf_hub_download(TTS_REPO, f"{TTS_VARIANT}/{fn}")
    meta = json.load(open(w("meta.json")))
    nth = int(os.environ.get("TORCH_THREADS", "2"))

    def _sess(p):
        so = ort.SessionOptions(); so.intra_op_num_threads = nth; so.inter_op_num_threads = 1
        return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

    F.text_to_ids("您好")                      # warm the frontend (pulls the g2pw model once)
    TTS.update(F=F, sr=meta["sample_rate"], abs_bins=meta["abs_frame_bins"], max_frames=meta["max_frames"],
               enc=_sess(w("acoustic_encoder.onnx")), dec=_sess(w("acoustic_decoder.onnx")), voc=_sess(w("vocoder.onnx")))
    print(f"PrimeTTS ready ({TTS_VARIANT}, {meta['sample_rate']} Hz)", flush=True)
    return TTS


def synth_reply(text):
    """Reply text -> (float32 wav, sr). Chunk at punctuation under a frame budget (the acoustic
    model's absolute positional code saturates past max_frames, garbling long single passes)."""
    import numpy as np
    eng = _load_tts()
    spk = np.array([0], np.int64)

    def enc(t):
        o = eng["F"].text_to_ids(t)
        if not o["phone_ids"]:
            return None
        ph = np.array([o["phone_ids"]], np.int64); tn = np.array([o["tone_ids"]], np.int64)
        lg = np.array([o["lang_ids"]], np.int64)
        return eng["enc"].run(None, {"phone": ph, "tone": tn, "lang": lg, "speaker": spk})

    budget = int(eng["max_frames"] * 0.8)
    clauses = [c for c in _TTS_SPLIT.split(text) if c.strip()] or [text]
    chunks, cur, cur_f = [], "", 0
    for cl in clauses:
        e = enc(cl); f = int(e[1].sum()) if e else 0
        if cur and cur_f + f > budget:
            chunks.append(cur); cur, cur_f = "", 0
        cur += cl; cur_f += f
        if cur_f > budget and cur == cl:
            chunks.append(cur); cur, cur_f = "", 0
    if cur:
        chunks.append(cur)
    gap = np.zeros(int(eng["sr"] * 0.07), np.float32)
    wavs = []
    for i, c in enumerate(chunks):
        e = enc(c)
        if e is None:
            continue
        cond, dur, pitch = e
        feeds = _tts_regulate(cond, dur, pitch, eng["abs_bins"], eng["max_frames"])
        mel = eng["dec"].run(None, {n: feeds[n] for n in _TTS_BN})[0]
        wavs.append(eng["voc"].run(None, {"mel": mel.astype(np.float32)})[0].reshape(-1))
        if i < len(chunks) - 1:
            wavs.append(gap)
    if not wavs:
        return None, eng["sr"]
    wav = np.concatenate(wavs)
    peak = float(np.max(np.abs(wav)))
    if peak > 1e-6:
        wav = wav * (0.97 / peak)
    return wav.astype(np.float32), eng["sr"]


def wav_to_b64(wav, sr):
    import io, base64, soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gen(prompt, wavs, n):
    proc, model = M["proc"], M["model"]
    enc = proc(text=prompt, audio=wavs, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n, do_sample=False, eos_token_id=151645)
    return proc.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_agent(wav):
    """Two-turn agent loop, all done by the 0.6B model: hear -> tool_call -> (resolver) ->
    tool_response -> the model SPEAKS the reply back in the caller's own language (zh-TW / en)."""
    w = load_wav_16k(wav)
    p1 = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n<|audio_pad|><|im_end|>\n"
          f"<|im_start|>assistant\n")
    tc_raw = _gen(p1, [w], 48)                       # turn 1: audio -> tool call
    calls = parse_tool_calls(tc_raw)
    if not calls:
        return {"calls": [], "raw": tc_raw}
    args = calls[0].get("arguments") or {}
    query, dept = args.get("query", ""), args.get("department", "")
    td = {"query": query, "department": dept} if dept else {"query": query}
    matches = registry.dispatch("search_contacts", td)
    tr = json.dumps(matches, ensure_ascii=False)
    p2 = (p1 + tc_raw + "<|im_end|>\n<|im_start|>tool\n<tool_response>" + tr +
          "</tool_response><|im_end|>\n<|im_start|>assistant\n")
    say = _gen(p2, [w], 80)                          # turn 2: tool response -> spoken reply (model)
    kind = "resolve" if len(matches) == 1 else ("clarify" if matches else "not_found")
    return {"calls": calls, "query": query, "dept": dept, "matches": matches, "say": say, "kind": kind}


@app.post("/listen")
async def listen(audio: UploadFile = File(...)):
    t0 = time.time()
    data = await audio.read()
    ext = "." + (audio.filename.rsplit(".", 1)[-1] if "." in (audio.filename or "") else "webm")
    res = run_agent(to_wav(data, ext))
    secs = round(time.time() - t0, 1)
    if not res["calls"]:
        return {"empty": True, "raw": res.get("raw", "")[:300], "secs": secs}
    query = res["query"]
    ranked = R.rank(query, k=6)
    cands = [{"rank": i + 1, "name": c.name, "zh": c.zh, "dept": c.dept, "ext": c.ext, "score": round(s, 1)}
             for i, (s, c) in enumerate(ranked)]
    title = {"resolve": "✅ Located", "clarify": "🤔 Needs clarification",
             "not_found": "🚫 Not found"}[res["kind"]]
    tts_b64 = tts_sr = None                          # speak the reply with PrimeTTS (non-fatal)
    try:
        wav, sr = synth_reply(res["say"])
        if wav is not None and len(wav):
            tts_b64, tts_sr = wav_to_b64(wav, sr), sr
    except Exception as e:
        print("TTS synth failed:", e, flush=True)
    secs = round(time.time() - t0, 1)               # include synth time in the reported latency
    return {"empty": False, "query": query, "tool_call": res["calls"][0], "tool_response": res["matches"],
            "candidates": cands,
            "decision": {"kind": res["kind"], "title": title, "say": res["say"]},
            "tts_audio": tts_b64, "tts_sr": tts_sr, "secs": secs}


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
 h1{font-size:1.3rem;margin:0 0 4px} .sub{color:#555;font-size:.95rem;margin-bottom:10px}
 .flow{margin:6px 0 12px;padding-left:22px;font-size:.92rem;color:#444;line-height:1.75}
 .flow li{margin:3px 0} .muted{color:#888;font-size:.86rem} .tagline{color:#555;font-size:.9rem;margin-bottom:16px}
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
 @media(prefers-color-scheme:dark){body{background:#0f0f11;color:#e7e7ea}.sub,.flow,.tagline,th{color:#b8b8c0}.muted{color:#888}.card{background:#17171a;border-color:#2a2a2e}tr.top td{background:#1e2547}th,td{border-color:#26262b}.say{color:#cfcfd4}}
</style></head><body>
<h1>☎️ Qwen3-ASR-0.6B-Agent — voice attendant</h1>
<div class="sub">Say a colleague's name — in <b>Chinese or English</b> — and the attendant finds their extension.</div>
<ol class="flow">
 <li>🎙️ <b>Qwen3-ASR-0.6B-Agent</b> hears you and calls a directory tool <span class="muted">— a 0.6B that beats Omni-3B</span></li>
 <li>🔍 the tool looks up the name in the live directory <span class="muted">— __N__ contacts</span></li>
 <li>🗣️ the model replies <b>in your language</b>, and <a href="https://huggingface.co/Luigi/PrimeTTS" target="_blank">PrimeTTS</a> <b>speaks it aloud</b></li>
</ol>
<div class="tagline">Speech in → tool use → speech out — three tiny on-device models, CPU only, ~10–20 s per turn.</div>
<button id="rec" onclick="toggle()">🎙️ Start recording</button>
<div class="status" id="status">Tip: “可以幫我接蔡孟儒嗎” or “I'd like to reach Coco Kuo”.</div>
<div id="out" style="display:none">
 <div class="card"><h3>🔁 Tool-calling trace</h3>
  <div class="step">1 · 🎙️ you → 🤖 Qwen3-ASR-0.6B-Agent → tool call</div><pre id="tc"></pre>
  <div class="step">2 · 🔧 search_contacts → tool response</div><pre id="tr"></pre>
  <div class="step">3 · 🗣️ model speaks back — <i>in the caller's own language</i></div><div id="decision" class="decision"></div><div id="say" class="say"></div>
  <audio id="player" controls style="width:100%;margin-top:10px;display:none"></audio>
  <div class="step" id="ttslbl" style="display:none">🔊 voice by <a href="https://huggingface.co/Luigi/PrimeTTS" target="_blank">PrimeTTS</a> (~5M-param zh-TW/en flagship, 16 kHz, on-device)</div></div>
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
 const dc=document.getElementById('decision');dc.className='decision '+d.decision.kind;dc.innerHTML='<b>'+d.decision.title+'</b>';
 document.getElementById('say').textContent='“'+d.decision.say+'”';
 const pl=document.getElementById('player'),tl=document.getElementById('ttslbl');
 if(d.tts_audio){pl.src='data:audio/wav;base64,'+d.tts_audio;pl.style.display='block';tl.style.display='block';pl.play().catch(()=>{});}
 else{pl.style.display='none';tl.style.display='none';}
 document.getElementById('cands').innerHTML=d.candidates.map(c=>'<tr class="'+(c.rank===1&&d.decision.kind!=='not_found'?'top':'')+'"><td>'+c.rank+'</td><td>'+c.name+'</td><td>'+c.zh+'</td><td>'+c.dept+'</td><td>'+c.ext+'</td><td>'+c.score+'</td></tr>').join('');}
function rs(){const b=document.getElementById('rec');b.disabled=false;b.textContent='🎙️ Start recording';}
</script></body></html>"""
