#!/usr/bin/env python3
"""FREE-RUNNING multi-turn eval (no teacher-forcing): the model's OWN tool-calls drive the REAL
resolver against a real per-dialog directory (the collision cluster), with only the gold audio
turns fed. This is the final confirmation that the model's errors don't compound across the
full agentic loop. Score = reaches the correct final extension.

  .venv-qa/bin/python train/eval_qwen3asr_freerun.py --limit 60
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_qwen3asr_agent import build_processor, SYS, load_wav_16k, MODEL_ID
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from resolver import Resolver
from tools import parse_tool_calls, format_matches

IM_END = 151645
ROOT = Path(__file__).resolve().parent.parent

# name -> full_chinese, from the 200-person directory (collision dialogs draw from these names)
NAME2ZH = {}
for r in csv.DictReader(open(ROOT / "data" / "directory.csv", encoding="utf-8")):
    NAME2ZH[r["display_en"]] = r["full_chinese"]


def mini_csv(cluster):
    """Write a tiny real directory from a dialog's fabricated collision cluster."""
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["ext", "display_en", "full_chinese", "department"])
    w.writeheader()
    for c in cluster:
        w.writerow({"ext": c["ext"], "display_en": c["name"],
                    "full_chinese": NAME2ZH.get(c["name"], ""), "department": c["dept"]})
    f.close()
    return f.name


def fmt(turns):
    parts = [f"<|im_start|>system\n{SYS}<|im_end|>\n"]
    audios = []
    for t in turns:
        if t["role"] == "user":
            audios.append(load_wav_16k(t["audio"])); parts.append("<|im_start|>user\n<|audio_pad|><|im_end|>\n")
        elif t["role"] == "assistant" and "tool_call" in t:
            tc = json.dumps(t["tool_call"], ensure_ascii=False); parts.append(f"<|im_start|>assistant\n<tool_call>{tc}</tool_call><|im_end|>\n")
        elif t["role"] == "assistant":
            parts.append(f"<|im_start|>assistant\n{t['text']}<|im_end|>\n")
        elif t["role"] == "tool":
            tr = json.dumps(t["content"], ensure_ascii=False); parts.append(f"<|im_start|>tool\n<tool_response>{tr}</tool_response><|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts), audios


class FreeRun:
    def __init__(self, adapter="runs/qwen3asr-agent"):
        self.proc = build_processor()
        full = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16)
        self.m = PeftModel.from_pretrained(full.thinker.to("cuda").eval(), adapter).eval()

    def gen(self, turns):
        text, audios = fmt(turns)
        enc = self.proc(text=text, audio=audios or None, sampling_rate=16000, return_tensors="pt")
        enc = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in enc.items()}
        if "input_features" in enc:
            enc["input_features"] = enc["input_features"].half()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = self.m.generate(**enc, max_new_tokens=64, do_sample=False, eos_token_id=IM_END)
        return self.proc.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/qwen3asr-agent")
    ap.add_argument("--val", default="data/audio/dialogs_collision_val.jsonl")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    fr = FreeRun(args.adapter)
    rows = [json.loads(l) for l in open(args.val, encoding="utf-8")][:args.limit]

    n = ok = collided = asked = 0
    for i, r in enumerate(rows):
        t = r["turns"]
        if len(t) < 8 or not NAME2ZH.get(t[2]["content"][0]["name"]):
            continue
        n += 1
        R = Resolver(mini_csv(t[2]["content"]))            # real directory = the cluster
        gold_ext = t[6]["content"][0]["ext"]
        req_audio, ans_audio = t[0], t[4]
        # turn 1: model -> tool_call -> REAL resolve
        conv = [req_audio]
        g1 = fr.gen(conv); c1 = parse_tool_calls(g1)
        if not c1:
            continue
        q1 = (c1[0].get("arguments") or {}).get("query", "")
        res1 = R.resolve(q1); m1 = format_matches(res1)
        collided += int(len(m1) > 1)
        conv += [{"role": "assistant", "tool_call": c1[0]}, {"role": "tool", "content": m1}]
        # turn 2: model -> ask department
        g2 = fr.gen(conv); asked += int(not parse_tool_calls(g2) and ("department" in g2.lower() or "which" in g2.lower()))
        conv += [{"role": "assistant", "text": g2}, ans_audio]
        # turn 3: model -> refined tool_call -> REAL resolve with dept filter
        g3 = fr.gen(conv); c3 = parse_tool_calls(g3)
        if not c3:
            continue
        a3 = c3[0].get("arguments", {})
        res2 = R.resolve(a3.get("query", ""), filters={"department": a3.get("department", "")})
        ok += int(res2.get("action") == "resolve" and res2.get("ext") == gold_ext)
        if i % 15 == 0:
            print(f"  [{i+1}/{len(rows)}] collide={len(m1)>1} ask={asked} final_ext_ok={res2.get('ext')==gold_ext}", flush=True)
    n = max(n, 1)
    print(f"\n=== FREE-RUNNING multi-turn (n={n}) ===")
    print(f"  real collision returned : {collided/n*100:5.1f}%")
    print(f"  asked department        : {asked/n*100:5.1f}%")
    print(f"  reached correct ext     : {ok/n*100:5.1f}%  ({ok}/{n})")


if __name__ == "__main__":
    main()
