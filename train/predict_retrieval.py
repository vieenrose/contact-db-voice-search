#!/usr/bin/env python3
"""Perception -> retrieval pipeline: STOCK Qwen transcribes the heard name, then
resolver.rank() retrieves the closest N from the LIVE directory with distance scores.
NO name-mapping fine-tuning, DB never in the weights.

  .venv/bin/python train/predict_retrieval.py --test data/audio/test.jsonl
Then: eval.py --gold data/audio/test.jsonl --pred runs/retrieval/preds.jsonl
"""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_qwen import build_model, build_processor
from train_qwen import load_wav_16k, ASR_SR
from resolver import Resolver

SYS_T = ("You are a phone operator at a Taiwan office. The caller wants to reach a colleague. "
         "Output ONLY that colleague's name exactly as spoken — Chinese characters or English — "
         "and nothing else. Ignore the caller's own name.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/audio/test.jsonl")
    ap.add_argument("--out", default="runs/retrieval/preds.jsonl")
    ap.add_argument("--directory", default="data/directory.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    proc = build_processor()
    model = build_model().eval()          # STOCK Qwen thinker, no adapter
    R = Resolver(args.directory)
    prompt = proc.apply_chat_template(
        [{"role": "system", "content": [{"type": "text", "text": SYS_T}]},
         {"role": "user", "content": [{"type": "audio", "audio": None}]}],
        add_generation_prompt=True, tokenize=False)

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    with open(args.out, "w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            wav = load_wav_16k(r["audio"])
            inp = proc(text=prompt, audio=[wav], sampling_rate=ASR_SR, return_tensors="pt")
            inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=16, do_sample=False)
            heard = proc.tokenizer.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            action = R.resolve(heard)     # top-N retrieval + margin -> resolve/clarify/not_found
            out.write(json.dumps({"audio": r["audio"], "prediction": json.dumps(action, ensure_ascii=False),
                                  "heard": heard}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  [{i+1}/{len(rows)}] heard={heard[:20]!r} -> {action['action']}")
    print(f"wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
