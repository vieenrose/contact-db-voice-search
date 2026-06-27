#!/usr/bin/env python3
"""Eval the fine-tuned Qwen3-ASR-0.6B audio-agent: audio -> tool_call -> resolver -> action.

Single-generation-per-clip (the request turn): the model hears the name and emits the
search_contacts tool call; we parse the query and resolve it against the directory, then
score the outcome with eval.py (same as the Omni agent eval). Run in .venv-qa.

  .venv-qa/bin/python train/eval_qwen3asr.py --adapter runs/qwen3asr-agent --limit 300
  then: .venv/bin/python eval.py --gold data/audio/test.jsonl --pred runs/qwen3asr-agent/preds.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_qwen3asr_agent import build_processor, SYS, load_wav_16k, MODEL_ID
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from resolver import Resolver
from tools import parse_tool_calls

IM_END = 151645


class Qwen3ASRAgent:
    def __init__(self, adapter="runs/qwen3asr-agent", directory="data/directory.csv"):
        self.proc = build_processor()
        full = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16)
        self.m = PeftModel.from_pretrained(full.thinker.to("cuda").eval(), adapter).eval()
        self.R = Resolver(directory)

    def route(self, clip):
        text = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
                f"<|im_start|>user\n<|audio_pad|><|im_end|>\n<|im_start|>assistant\n")
        enc = self.proc(text=text, audio=[load_wav_16k(clip)], sampling_rate=16000, return_tensors="pt")
        enc = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in enc.items()}
        if "input_features" in enc:
            enc["input_features"] = enc["input_features"].half()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = self.m.generate(**enc, max_new_tokens=48, do_sample=False, eos_token_id=IM_END)
        gen = self.proc.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        calls = parse_tool_calls(gen)
        if not calls:
            return gen, {"action": "not_found"}
        query = (calls[0].get("arguments") or {}).get("query", "")
        return query, self.R.resolve(query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/qwen3asr-agent")
    ap.add_argument("--test", default="data/audio/test.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_path = args.out or str(Path(args.adapter) / "preds.jsonl")

    ag = Qwen3ASRAgent(args.adapter)
    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    n_ok_call = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            query, action = ag.route(r["audio"])
            n_ok_call += action.get("action") != "not_found" or "search_contacts" in str(query)
            out.write(json.dumps({"audio": r["audio"], "prediction": json.dumps(action, ensure_ascii=False),
                                  "query": query}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  [{i+1}/{len(rows)}] q={query!r:24} -> {action.get('action')}", flush=True)
    print(f"wrote {len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
