#!/usr/bin/env python3
"""Benchmark the phase-2 multi-turn tool-calling agent on the test set.

Runs the agent runtime (model emits <tool_call> -> resolver -> <tool_response> -> reply)
on each request clip, takes the agent's resolved action, and scores it with eval.py's
outcome logic against the original gold. Single-turn-per-clip (the request turn); the
agent's own tool-hops happen inside route_clip.

  .venv/bin/python train/eval_agent.py --model runs/agent --limit 300   # quick read
  then: eval.py --gold data/audio/test.jsonl --pred runs/agent/preds.jsonl
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import Agent, route_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/agent")
    ap.add_argument("--test", default="data/audio/test.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_path = args.out or str(Path(args.model) / "preds.jsonl")

    ag = Agent(args.model, directory="data/directory.csv")
    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    with open(out_path, "w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            reply, action = route_clip(ag, r["audio"])
            # action is the resolver result the agent acted on (resolve/clarify/not_found)
            pred = json.dumps(action, ensure_ascii=False) if action else json.dumps({"action": "not_found"})
            out.write(json.dumps({"audio": r["audio"], "prediction": pred,
                                  "reply": reply}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  [{i+1}/{len(rows)}] {reply[:48]}")
    print(f"wrote {len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
