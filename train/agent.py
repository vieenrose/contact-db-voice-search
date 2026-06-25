#!/usr/bin/env python3
"""Phase-2 agent runtime: the multi-turn tool-calling loop around the trained model.

The model emits <tool_call>{"name":"search_contacts","arguments":{"query":..}}</tool_call>;
we run the resolver against the LIVE directory, feed the result back as
<tool_response>...</tool_response>, and let the model continue — until it produces a
plain reply (connect / not-found) or a clarify question (then it waits for the next
audio turn). DB is external; nothing is baked into the model.

Used both interactively (an audio turn at a time) and by eval (run a test clip ->
final routed action) — reusing the same loop.
"""
import json, re, sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_qwen import build_model, build_processor
from train_qwen_agent import SYS, build_messages   # reuse the exact training format
from train_qwen import load_wav_16k, ASR_SR
from resolver import Resolver

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def topn(res):
    return ([{"name": res["name"], "ext": res["ext"], "dept": res.get("dept"), "score": res.get("score")}]
            if res["action"] == "resolve" else res.get("candidates", []))


class Agent:
    def __init__(self, model_dir="runs/agent", directory="data/directory.csv", max_tool_hops=3):
        self.proc = build_processor()
        base = build_model()
        self.model = PeftModel.from_pretrained(base, model_dir).eval()
        self.R = Resolver(directory)
        self.tok = self.proc.tokenizer
        self.max_tool_hops = max_tool_hops

    def _gen(self, msgs, audios):
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.proc(text=text, audio=audios or None, sampling_rate=ASR_SR, return_tensors="pt")
        enc = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=64, do_sample=False)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def turn(self, msgs, audios):
        """Run one user turn: generate, auto-resolve any tool-calls, return (reply, action, msgs)."""
        last_action = None
        for _ in range(self.max_tool_hops):
            gen = self._gen(msgs, audios)
            m = TOOL_RE.search(gen)
            if not m:                                   # plain reply -> turn complete
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": gen}]})
                return gen, last_action, msgs
            # tool-call: run the resolver against the live DB, feed the response back
            try:
                query = json.loads(m.group(1)).get("arguments", {}).get("query", "")
            except Exception:
                query = ""
            res = self.R.resolve(query)
            last_action = res
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"<tool_call>{m.group(1)}</tool_call>"}]})
            msgs.append({"role": "tool", "content": [{"type": "text",
                         "text": f"<tool_response>{json.dumps(topn(res), ensure_ascii=False)}</tool_response>"}]})
        return "(too many tool hops)", last_action, msgs


def route_clip(agent, audio_path):
    """Eval helper: one request clip -> the agent's resolved action (after its tool-calls)."""
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "audio", "audio": None}]}]
    reply, action, _ = agent.turn(msgs, [load_wav_16k(audio_path)])
    return reply, action


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/agent")
    ap.add_argument("--audio", help="a request clip to route")
    args = ap.parse_args()
    ag = Agent(args.model)
    if args.audio:
        reply, action = route_clip(ag, args.audio)
        print("reply:", reply)
        print("action:", action)
