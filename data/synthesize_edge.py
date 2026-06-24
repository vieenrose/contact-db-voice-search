#!/usr/bin/env python3
"""Step 3a (alt): synthesize request clips with edge-tts multi-voice.

PrimeTTS gives one telephony-matched female voice; edge-tts adds real speaker
variety — crucially a zh-TW MALE voice — for training robustness and for a
VOICE-DISJOINT test set (held-out voices the model never trained on). Output
mirrors synthesize.py's base manifest so augment.py degrades all sources to the
same 8 kHz phone channel identically.

edge-tts emits 24 kHz MP3 (needs internet + ffmpeg); we decode to mono WAV and
let augment.py do the telephony downsampling.

Run: .venv/bin/python data/synthesize_edge.py --voices zh-TW-HsiaoChenNeural,zh-TW-YunJheNeural
In : data/requests.jsonl
Out: data/audio/base/<id>__<voice>.wav + data/audio/base/manifest_edge.jsonl
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, subprocess, sys, tempfile
from pathlib import Path

import edge_tts
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent

# TRAIN voice pool (edge-tts is the only source now; PrimeTTS dropped — too experimental).
# zh-TW for accent authenticity + a couple zh-CN and en for speaker variety. The two
# held-out TEST voices (augment.TEST_VOICES) are deliberately NOT in this list.
DEFAULT_VOICES = [
    "zh-TW-HsiaoChenNeural", "zh-TW-YunJheNeural",     # Taiwan F + M
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",       # Mainland F + M (speaker variety)
    "en-US-GuyNeural", "en-US-EmmaNeural",             # English M + F (code-switch parts)
]


def load_distinct(requests_path):
    seen = {}
    for line in open(requests_path, encoding="utf-8"):
        s = json.loads(line)
        seen.setdefault(s["text"], s)
    out = []
    for i, (text, s) in enumerate(seen.items()):
        out.append({"idx": i, "text": text, "target": s["target"],
                    "lang": s["lang"], "split": s["split"], "style": s["style"]})
    return out


async def synth_mp3(text, voice, path, timeout=25):
    # hard timeout so a hung edge-tts network call becomes a (resumable) skip, not a freeze
    await asyncio.wait_for(edge_tts.Communicate(text, voice).save(str(path)), timeout=timeout)


def mp3_to_wav(mp3, wav, sr=24000):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ac", "1", "-ar", str(sr), str(wav)], check=True)


def vkey(voice: str) -> str:
    # zh-TW-HsiaoChenNeural -> zhtw_hsiaochen
    parts = voice.replace("Neural", "").split("-")
    return (parts[0] + parts[1]).lower() + "_" + "".join(parts[2:]).lower() if len(parts) >= 3 \
        else voice.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES),
                    help="comma list of edge-tts voices")
    ap.add_argument("--assign", choices=["random", "all"], default="random",
                    help="random: 1 voice/text (spread); all: every voice per text")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-manifest", default="manifest_edge.jsonl",
                    help="manifest filename under data/audio/base (augment globs manifest*.jsonl)")
    args = ap.parse_args()
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    # Language pools: an English voice can't speak a Chinese-only text (edge-tts returns
    # NoAudioReceived), so route by the text's language. zh/mix -> zh voices (they read
    # embedded English names fine); en -> en voices.
    zh_pool = [v for v in voices if v.startswith("zh-")]
    en_pool = [v for v in voices if v.startswith("en-")]

    def pick_voices(lang, idx):
        pool = en_pool if lang == "en" else (zh_pool or en_pool)
        if not pool:
            pool = voices
        return pool if args.assign == "all" else [pool[idx % len(pool)]]

    items = load_distinct(ROOT / "data" / "requests.jsonl")
    if args.limit:
        items = items[:args.limit]
    out_dir = ROOT / "data" / "audio" / "base"; out_dir.mkdir(parents=True, exist_ok=True)
    man = open(out_dir / args.out_manifest, "w", encoding="utf-8")

    n = 0
    with tempfile.TemporaryDirectory() as td:
        for it in items:
            # language-aware voice pick (zh/mix -> zh voices, en -> en voices)
            picks = pick_voices(it["lang"], it["idx"])
            for v in picks:
                # stable id from text+voice so resume stays correct across request changes
                th = hashlib.md5(it["text"].encode("utf-8")).hexdigest()[:10]
                cid = f"e{th}_{vkey(v)}"
                mp3 = Path(td) / f"{cid}.mp3"
                wav = out_dir / f"{cid}.wav"
                if not (wav.exists() and wav.stat().st_size > 0):   # resume: skip done clips
                    try:
                        asyncio.run(synth_mp3(it["text"], v, mp3))
                        mp3_to_wav(mp3, wav)
                    except Exception as e:
                        print(f"  SKIP {cid}: {type(e).__name__} {e}", file=sys.stderr)
                        continue
                dur = round(sf.info(str(wav)).duration, 2)
                man.write(json.dumps({"id": cid, "text": it["text"], "wav": str(wav),
                                      "dur": dur, "target": it["target"], "lang": it["lang"],
                                      "split": it["split"], "style": it["style"],
                                      "voice": v, "source": "edge"}, ensure_ascii=False) + "\n")
                n += 1
                if n % 50 == 0 or args.limit:
                    print(f"  [{n}] {cid} {dur}s  {it['text'][:30]}")
    man.close()
    print(f"DONE {n} edge clips -> {out_dir}/manifest_edge.jsonl")


if __name__ == "__main__":
    main()
