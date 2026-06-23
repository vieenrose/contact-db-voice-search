#!/usr/bin/env python3
"""Step 3a: synthesize a base 8 kHz clip per distinct request text via PrimeTTS.

Reuses PrimeTTS's exported ONNX pipeline (encoder -> host length-regulate ->
decoder -> vocoder) and its zh-TW/en bopomofo+arpabet frontend. The upstream
synth_from_text.py hard-codes the frontend path; here we add the downloaded repo's
scripts/ dir to sys.path instead, so it is self-contained.

Run with the project venv (has g2pw/g2p_en):
    .venv/bin/python data/synthesize.py [--limit N]

Inputs : data/requests.jsonl, models/PrimeTTS/{*.onnx,meta.json}
Outputs: data/audio/base/<id>.wav  +  data/audio/base/manifest.jsonl
         (manifest carries the supervised target so augment.py can label clips)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
PTTS = ROOT / "models" / "PrimeTTS"
sys.path.insert(0, str(PTTS / "scripts"))   # frontend_bopomofo + text_norm live here
import frontend_bopomofo as F               # noqa: E402  (needs g2pw + g2p_en)


def host_regulate(cond, dur, pitch, abs_bins, max_frames):
    """Verbatim from PrimeTTS synth_from_text.py — expands phone-level conditioning
    to frame level with the positional/meta features the decoder expects."""
    c = cond[0]; d = dur[0].astype(np.int64); d[d < 0] = 0
    T, H = c.shape
    frames = np.repeat(c, d, axis=0); Fn = frames.shape[0]
    tok = np.repeat(np.arange(T), d); starts = np.cumsum(d) - d
    within = np.arange(Fn) - starts[tok]; dpf = d[tok].astype(np.float32)
    rel = (within / np.maximum(dpf - 1, 1)).astype(np.float32)
    tc = max(1, int((d > 0).sum())); token_pos = (tok / max(1, tc - 1)).astype(np.float32)
    ld = (np.log1p(dpf) / 6.0).astype(np.float32); center = 1.0 - np.abs(rel * 2 - 1)
    fm = np.stack([rel, 1 - rel, center, np.sin(rel*np.pi), np.cos(rel*np.pi),
                   token_pos, ld, dpf/40.0], -1).astype(np.float32)
    prev = np.concatenate([c[:1], c[:-1]], 0); nxt = np.concatenate([c[1:], c[-1:]], 0)
    lc = np.repeat(np.concatenate([prev, c, nxt], -1), d, axis=0).astype(np.float32)
    pos = np.arange(Fn); ap = np.minimum(pos*abs_bins//max(1, max_frames), abs_bins-1).astype(np.int64)
    pf = np.repeat(pitch[0], d, axis=0).astype(np.float32)
    return {"frames": frames[None].astype(np.float32), "frame_meta": fm[None],
            "local_ctx_raw": lc[None], "abs_pos": ap[None],
            "pitch_frame": pf[None], "frame_mask": np.ones((1, Fn), bool)}


def load_distinct(requests_path):
    """One entry per distinct text (texts uniquely resolve, so target is well-defined)."""
    seen = {}
    for line in open(requests_path, encoding="utf-8"):
        s = json.loads(line)
        if s["text"] not in seen:
            seen[s["text"]] = s
    items = []
    for i, (text, s) in enumerate(seen.items()):
        items.append({"id": f"u{i:05d}", "text": text, "target": s["target"],
                      "lang": s["lang"], "split": s["split"], "style": s["style"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="synthesize only first N (smoke test)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    meta = json.load(open(PTTS / "meta.json"))
    sr = meta["sample_rate"]
    so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
    sA = ort.InferenceSession(str(PTTS / "acoustic_encoder.onnx"), so, providers=["CPUExecutionProvider"])
    sB = ort.InferenceSession(str(PTTS / "acoustic_decoder.onnx"), so, providers=["CPUExecutionProvider"])
    sV = ort.InferenceSession(str(PTTS / "vocoder.onnx"), so, providers=["CPUExecutionProvider"])
    bn = ["frames", "frame_meta", "local_ctx_raw", "abs_pos", "pitch_frame", "frame_mask"]

    out_dir = ROOT / "data" / "audio" / "base"; out_dir.mkdir(parents=True, exist_ok=True)
    items = load_distinct(ROOT / "data" / "requests.jsonl")
    if args.limit:
        items = items[:args.limit]

    man = open(out_dir / "manifest.jsonl", "w", encoding="utf-8")
    for k, it in enumerate(items):
        o = F.text_to_ids(it["text"])
        phone = np.array([o["phone_ids"]], np.int64)
        tone = np.array([o["tone_ids"]], np.int64)
        lang = np.array([o["lang_ids"]], np.int64)
        spk = np.zeros(1, np.int64)
        cond, dur, pitch = sA.run(None, {"phone": phone, "tone": tone, "lang": lang, "speaker": spk})
        reg = host_regulate(cond, dur, pitch, meta["abs_frame_bins"], meta["max_frames"])
        feeds = {n: (reg[n].astype(np.float32) if reg[n].dtype != bool else reg[n]) for n in bn}
        feeds["abs_pos"] = reg["abs_pos"].astype(np.int64)
        mel = sB.run(None, feeds)[0]
        wav = sV.run(None, {"mel": mel.astype(np.float32)})[0].reshape(-1)
        wp = out_dir / f"{it['id']}.wav"
        sf.write(wp, wav, sr)
        man.write(json.dumps({"id": it["id"], "text": it["text"], "wav": str(wp),
                              "dur": round(len(wav)/sr, 2), "target": it["target"],
                              "lang": it["lang"], "style": it["style"],
                              "voice": "primetts", "source": "primetts"},
                             ensure_ascii=False) + "\n")
        if k % 50 == 0 or args.limit:
            print(f"  [{k+1}/{len(items)}] {it['id']} {len(wav)/sr:.1f}s  {it['text'][:32]}")
    man.close()
    print(f"DONE {len(items)} clips -> {out_dir}/manifest.jsonl (sr={sr})")


if __name__ == "__main__":
    main()
