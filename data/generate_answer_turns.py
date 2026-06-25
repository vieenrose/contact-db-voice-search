#!/usr/bin/env python3
"""Synthesize clarify-ANSWER audio for multi-turn dialogs (phase-2).

When the agent asks "which Tseng — Henry or Wesley?" the caller answers with a first
name ("Henry"), a department ("the one in Sales / 業務那位"), or a full name. We need
short audio clips of those answers. This synthesizes them with edge-tts (held-out
test voices kept separate), tagging each with the answer it carries.

Out: data/audio/answers/<id>.wav  +  data/audio/answers/manifest.jsonl
     fields: {wav, answer_kind, answer_value, lang, voice}
"""
import argparse, asyncio, csv, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import edge_tts, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
TRAIN_VOICES = ["zh-TW-HsiaoChenNeural", "zh-TW-YunJheNeural", "en-US-GuyNeural", "en-US-EmmaNeural"]
DEPT_ZH = {"Sales": "業務", "Marketing": "行銷", "Finance": "財務", "HR": "人資",
           "Engineering": "工程", "R&D": "研發", "IT": "資訊", "Procurement": "採購",
           "Logistics": "物流", "Legal": "法務", "Customer Service": "客服", "Admin": "行政",
           "Quality": "品保", "Production": "生產", "Accounting": "會計"}


def answer_texts(firsts, depts):
    items = []
    for f in firsts:                                   # first-name answers
        items += [("first", f, "en", f), ("first", f, "en", f"It's {f}."),
                  ("first", f, "mix", f"{f} 那位")]
    for d in depts:                                    # department answers
        items += [("dept", d, "en", f"the one in {d}"),
                  ("dept", d, "zh", f"{DEPT_ZH[d]}那位"),
                  ("dept", d, "zh", f"在{DEPT_ZH[d]}的")]
    return items


async def synth(text, voice, path, timeout=25):
    await asyncio.wait_for(edge_tts.Communicate(text, voice).save(str(path)), timeout=timeout)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "data" / "directory.csv", encoding="utf-8")))
    firsts = sorted({r["english_first"] for r in rows})
    depts = sorted({r["department"] for r in rows})
    items = answer_texts(firsts, depts)
    if args.limit:
        items = items[:args.limit]
    out_dir = ROOT / "data" / "audio" / "answers"; out_dir.mkdir(parents=True, exist_ok=True)
    man = open(out_dir / "manifest.jsonl", "w", encoding="utf-8")
    n = 0
    with tempfile.TemporaryDirectory() as td:
        for i, (kind, value, lang, text) in enumerate(items):
            voice = TRAIN_VOICES[2 + (i % 2)] if lang == "en" else TRAIN_VOICES[i % 2]
            cid = "a" + hashlib.md5(f"{text}{voice}".encode()).hexdigest()[:10]
            wav = out_dir / f"{cid}.wav"
            if not (wav.exists() and wav.stat().st_size > 0):
                mp3 = Path(td) / f"{cid}.mp3"
                try:
                    asyncio.run(synth(text, voice, mp3))
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                                    "-ac", "1", "-ar", "24000", str(wav)], check=True)
                except Exception as e:
                    print(f"  SKIP {cid}: {type(e).__name__}", file=sys.stderr); continue
            man.write(json.dumps({"wav": str(wav), "text": text, "answer_kind": kind,
                                  "answer_value": value, "lang": lang, "voice": voice},
                                 ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  [{n}/{len(items)}] {text[:24]}")
    man.close()
    print(f"DONE {n} answer-turn clips -> {out_dir}/manifest.jsonl")


if __name__ == "__main__":
    main()
