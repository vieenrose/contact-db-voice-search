#!/usr/bin/env python3
"""Probe STOCK Qwen2.5-Omni-3B on our test clips BEFORE any fine-tuning.

The key question: does Qwen-Omni (SOTA Mandarin ASR) actually hear the Taiwan names
over our 8 kHz telephony audio? If stock already does well on zh, fine-tuning will
work; if even stock fails, the Qwen direction is wrong. Zero training cost.
"""
import json, sys
from fractions import Fraction
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import (Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor,
                          BitsAndBytesConfig)

MODEL = "Qwen/Qwen2.5-Omni-3B"
SYS = ("You are a phone attendant at a Taiwan office. The caller asks for a colleague "
       "by name (an English first name + a Chinese surname, spoken in English, Mandarin, "
       "or a mix). Reply with ONLY the person's name as you heard it.")


def wav16(path):
    x, sr = sf.read(path); x = np.asarray(x, np.float32)
    if x.ndim > 1: x = x.mean(-1)
    if sr != 16000:
        fr = Fraction(16000, sr).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    return x


def main():
    print("loading Qwen2.5-Omni-3B in 4-bit (thinker only)...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=bnb, attn_implementation="sdpa", device_map="cuda",
        enable_audio_output=False)
    model.eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)

    gold = {json.loads(l)["audio"]: (json.loads(json.loads(l)["target_text"]), json.loads(l).get("lang"))
            for l in open("data/audio/test.jsonl")}
    items = list(gold.items())
    zh = [it for it in items if it[1][1] == "zh"][:4]
    en = [it for it in items if it[1][1] == "en"][:3]
    print("\n===== STOCK Qwen2.5-Omni-3B on test clips =====")
    for audio, (g, lang) in zh + en:
        wav = wav16(audio)
        conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "audio", "audio": wav}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[wav], sampling_rate=16000, return_tensors="pt")
        inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        dec = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        print(f"  [{lang}] gold={g.get('name') or g['action']:14}  qwen={dec.strip()[:50]!r}")


if __name__ == "__main__":
    main()
