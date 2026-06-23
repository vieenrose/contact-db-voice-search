#!/usr/bin/env python3
"""Step 3b: telephony + speaker augmentation of PrimeTTS base clips.

PrimeTTS gives one young-female voice at 8 kHz. To narrow the single-voice /
sim-to-real gap we manufacture variety per clip:
  - speed perturbation (0.9-1.12x): fakes speaker rate + pitch/formant shift,
    a cheap stand-in for different speakers (incl. roughly lower/male timbre).
  - phone band-pass 300-3400 Hz: the classic telephony channel.
  - G.711 mu-law round-trip: real PSTN codec quantization grit.
  - additive noise at random SNR, gain变化, occasional dropouts: line grime.
The whisper encoder (frozen in step 4) supplies the real-speaker robustness this
augmentation can only approximate.

Run: .venv/bin/python data/augment.py [--variants K]
In : data/audio/base/manifest.jsonl
Out: data/audio/clips/<id>_aN.wav  +  data/audio/train.jsonl / val.jsonl
     each line: {audio, target_text, name, ext, lang, split, augs}
"""
from __future__ import annotations
import argparse, audioop, json, random
from fractions import Fraction
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

ROOT = Path(__file__).resolve().parent.parent


def to_i16(x): return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
def to_f32(x): return x.astype(np.float32) / 32768.0


def speed_perturb(x, factor):
    fr = Fraction(factor).limit_denominator(100)
    # factor>1 == faster == fewer samples: up=denominator, down=numerator
    return resample_poly(x, fr.denominator, fr.numerator).astype(np.float32)


def bandpass(x, sr, lo=300, hi=3400):
    sos = butter(4, [lo, min(hi, sr / 2 - 1)], btype="band", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def mulaw_roundtrip(x):
    b = to_i16(x).tobytes()
    return to_f32(np.frombuffer(audioop.ulaw2lin(audioop.lin2ulaw(b, 2), 2), np.int16))


def add_noise(x, snr_db, rng):
    p = float(np.mean(x ** 2)) + 1e-9
    npow = p / (10 ** (snr_db / 10))
    return (x + rng.normal(0, np.sqrt(npow), len(x))).astype(np.float32)


def apply_gain(x, db): return (x * (10 ** (db / 20))).astype(np.float32)


def dropouts(x, sr, rng, n=2, ms=40):
    x = x.copy(); w = int(sr * ms / 1000)
    for _ in range(n):
        if len(x) > w:
            s = rng.randint(0, len(x) - w); x[s:s + w] = 0.0
    return x


def augment_one(x, sr, rng, light=False):
    augs = []
    if not light:
        f = rng.uniform(0.9, 1.12); x = speed_perturb(x, f); augs.append(f"speed{f:.2f}")
    x = bandpass(x, sr); augs.append("bp300-3400")
    x = mulaw_roundtrip(x); augs.append("mulaw")
    if not light:
        snr = rng.uniform(12, 30); x = add_noise(x, snr, rng); augs.append(f"snr{snr:.0f}")
        g = rng.uniform(-6, 4); x = apply_gain(x, g); augs.append(f"gain{g:+.0f}")
        if rng.random() < 0.15:
            x = dropouts(x, sr, rng); augs.append("dropout")
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.99:
        x = x / peak * 0.97
    return x.astype(np.float32), augs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=3, help="augmented clips per base text")
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()
    nprng = np.random.default_rng(args.seed)
    R = rng_np(nprng)

    base_dir = ROOT / "data" / "audio" / "base"
    clips_dir = ROOT / "data" / "audio" / "clips"; clips_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(base_dir / "manifest.jsonl", encoding="utf-8")]

    out = {"train": open(ROOT / "data" / "audio" / "train.jsonl", "w", encoding="utf-8"),
           "val": open(ROOT / "data" / "audio" / "val.jsonl", "w", encoding="utf-8")}
    n = 0
    for r in rows:
        x, sr = sf.read(r["wav"]); x = x.astype(np.float32)
        for j in range(args.variants):
            # first variant is "light" (clean channel) so easy cases stay learnable
            y, augs = augment_one(x, sr, R, light=(j == 0))
            wp = clips_dir / f"{r['id']}_a{j}.wav"
            sf.write(wp, y, sr)
            tgt = r["target"]
            out[r["split"]].write(json.dumps({
                "audio": str(wp), "target_text": json.dumps(tgt, ensure_ascii=False),
                "action": tgt.get("action"), "name": tgt.get("name"), "ext": tgt.get("ext"),
                "lang": r["lang"], "split": r["split"], "style": r["style"], "augs": augs,
            }, ensure_ascii=False) + "\n")
            n += 1
    for f in out.values():
        f.close()
    print(f"DONE {n} clips from {len(rows)} base texts -> data/audio/{{train,val}}.jsonl")


class rng_np:
    """Adapter so augment_one can use numpy-style .uniform/.normal/.random/.randint
    off the stdlib Random seed (keeps everything deterministic on one seed)."""
    def __init__(self, _np): self._np = _np
    def uniform(self, a, b): return float(self._np.uniform(a, b))
    def normal(self, m, s, n): return self._np.normal(m, s, n)
    def random(self): return float(self._np.random())
    def randint(self, a, b): return int(self._np.integers(a, b + 1))


if __name__ == "__main__":
    main()
