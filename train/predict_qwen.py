#!/usr/bin/env python3
"""Run a QLoRA-tuned Qwen2.5-Omni-3B on test audio -> predictions.jsonl for eval.py.

  .venv/bin/python train/predict_qwen.py --model runs/qwen --test data/audio/test.jsonl
"""
import argparse, json, sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_qwen import build_model, build_processor
from train_qwen import convo, load_wav_16k, ASR_SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/qwen")
    ap.add_argument("--test", default="data/audio/test.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_path = args.out or str(Path(args.model) / "preds.jsonl")

    proc = build_processor()
    base = build_model()
    model = PeftModel.from_pretrained(base, args.model).eval()
    tok = proc.tokenizer
    prompt = proc.apply_chat_template(convo(), add_generation_prompt=True, tokenize=False)

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    with open(out_path, "w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            wav = load_wav_16k(r["audio"])
            inp = proc(text=prompt, audio=[wav], sampling_rate=ASR_SR, return_tensors="pt")
            inp = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=24, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True)
            seq = gen.sequences[0][inp["input_ids"].shape[1]:]
            pred = tok.decode(seq, skip_special_tokens=True).strip()
            # avg token log-prob of the generated text = confidence (closer to 0 = more sure)
            lps = [torch.log_softmax(s[0].float(), -1)[t].item() for t, s in zip(seq, gen.scores)]
            conf = sum(lps) / len(lps) if lps else -99.0
            out.write(json.dumps({"audio": r["audio"], "prediction": pred,
                                  "conf": round(conf, 4)}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  [{i+1}/{len(rows)}] {pred[:48]}")
    print(f"wrote {len(rows)} preds -> {out_path}")


if __name__ == "__main__":
    main()
