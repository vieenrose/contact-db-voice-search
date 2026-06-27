#!/usr/bin/env python3
"""Multi-turn gate for the Qwen3-ASR-0.6B agent: department-disambiguation turn-type tracking.

Teacher-forced replay of the collision dialogs: at each assistant position we let the MODEL
generate (feeding the gold tool_responses + gold audio turns in between) and check it took the
right turn-type. The crux axes:
  - turn 1 (audio request)            -> emits a tool_call
  - after a multi-match tool_response -> ASKS which department (a clarify, not a connect)
  - after the dept-answer audio       -> emits a refined tool_call WITH the correct department
  - after the unique tool_response    -> connects

  .venv-qa/bin/python train/eval_qwen3asr_mt.py --limit 120
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
from tools import parse_tool_calls

IM_END = 151645


def format_prefix(turns):
    parts = [f"<|im_start|>system\n{SYS}<|im_end|>\n"]
    audios = []
    for t in turns:
        if t["role"] == "user":
            audios.append(load_wav_16k(t["audio"])); parts.append("<|im_start|>user\n<|audio_pad|><|im_end|>\n")
        elif t["role"] == "assistant" and "tool_call" in t:
            tc = json.dumps({"name": t["tool_call"]["name"], "arguments": t["tool_call"]["arguments"]}, ensure_ascii=False)
            parts.append(f"<|im_start|>assistant\n<tool_call>{tc}</tool_call><|im_end|>\n")
        elif t["role"] == "assistant":
            parts.append(f"<|im_start|>assistant\n{t['text']}<|im_end|>\n")
        elif t["role"] == "tool":
            tr = json.dumps(t["content"], ensure_ascii=False)
            parts.append(f"<|im_start|>tool\n<tool_response>{tr}</tool_response><|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts), audios


class MT:
    def __init__(self, adapter="runs/qwen3asr-agent"):
        self.proc = build_processor()
        full = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16)
        self.m = PeftModel.from_pretrained(full.thinker.to("cuda").eval(), adapter).eval()

    def gen(self, prefix_turns, n=64):
        text, audios = format_prefix(prefix_turns)
        enc = self.proc(text=text, audio=audios or None, sampling_rate=16000, return_tensors="pt")
        enc = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in enc.items()}
        if "input_features" in enc:
            enc["input_features"] = enc["input_features"].half()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = self.m.generate(**enc, max_new_tokens=n, do_sample=False, eos_token_id=IM_END)
        return self.proc.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/qwen3asr-agent")
    ap.add_argument("--val", default="data/audio/dialogs_collision_val.jsonl")
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()
    mt = MT(args.adapter)
    rows = [json.loads(l) for l in open(args.val, encoding="utf-8")][:args.limit]

    m = {"t1_toolcall": 0, "asks_dept": 0, "refined_dept_ok": 0, "final_connect": 0, "all_ok": 0, "n": 0}
    for i, r in enumerate(rows):
        t = r["turns"]
        if len(t) < 8:
            continue
        m["n"] += 1
        gold_dept = (t[5]["tool_call"]["arguments"].get("department") or "")
        gold_ext = next((x["ext"] for x in t[6]["content"]), "")
        # turn 1: request audio -> tool_call?
        g1 = mt.gen(t[:1]); a1 = bool(parse_tool_calls(g1))
        # after multi-match: ASK department (clarify, not a tool_call)
        g2 = mt.gen(t[:3]); a2 = (not parse_tool_calls(g2)) and ("department" in g2.lower() or "which" in g2.lower())
        # after dept-answer audio: refined tool_call WITH correct department
        g3 = mt.gen(t[:5]); c3 = parse_tool_calls(g3)
        a3 = bool(c3) and (c3[0].get("arguments", {}).get("department", "").lower() == gold_dept.lower())
        # after unique: connect with the right ext
        g4 = mt.gen(t[:7]); a4 = (gold_ext in g4) and ("not found" not in g4.lower())
        for k, v in (("t1_toolcall", a1), ("asks_dept", a2), ("refined_dept_ok", a3), ("final_connect", a4)):
            m[k] += int(v)
        m["all_ok"] += int(a1 and a2 and a3 and a4)
        if i % 20 == 0:
            print(f"  [{i+1}/{len(rows)}] tc={a1} ask_dept={a2} refine={a3} connect={a4}", flush=True)
    n = max(m["n"], 1)
    print("\n=== Qwen3-ASR-0.6B multi-turn disambiguation ({} dialogs) ===".format(m["n"]))
    for k in ("t1_toolcall", "asks_dept", "refined_dept_ok", "final_connect", "all_ok"):
        print(f"  {k:16} {m[k]/n*100:5.1f}%  ({m[k]}/{n})")


if __name__ == "__main__":
    main()
