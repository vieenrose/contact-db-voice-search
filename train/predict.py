#!/usr/bin/env python3
"""Run a trained Ultravox attendant on test audio -> predictions.jsonl for eval.py.

Loads the base whisper-base build + the trained LoRA/projector adapter, greedily
decodes each test clip into the model's text output, and writes {audio, prediction}.
Then: eval.py --gold data/audio/test.jsonl --pred runs/v1/preds.jsonl

  .venv/bin/python train/predict.py --model runs/v1 --test data/audio/test.jsonl
"""
import argparse, json, sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import build_model, build_processor, AUDIO_MODEL
from train import SYS, load_wav_16k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v1")
    ap.add_argument("--test", default="data/audio/test.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stock", action="store_true", help="match a --stock trained model")
    args = ap.parse_args()
    out_path = args.out or str(Path(args.model) / "preds.jsonl")

    audio_model = None if args.stock else AUDIO_MODEL
    processor = build_processor(audio_model)
    base, _ = build_model(audio_model)
    model = PeftModel.from_pretrained(base, args.model).cuda().eval()
    tok = processor.tokenizer

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": "<|audio|>"}],
        tokenize=False, add_generation_prompt=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            wav = load_wav_16k(r["audio"])
            inp = processor(text=prompt, audio=wav, sampling_rate=16000, return_tensors="pt")
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=32, do_sample=False)
            pred = tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"audio": r["audio"], "prediction": pred}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  [{i+1}/{len(rows)}] {pred[:48]}")
    print(f"wrote {len(rows)} preds -> {out_path}")


if __name__ == "__main__":
    main()
