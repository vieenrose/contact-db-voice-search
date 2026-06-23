#!/usr/bin/env python3
"""Benchmark scorer for the attendant — outcome classes, not raw accuracy.

Scores model predictions against gold by ROUTING OUTCOME, the only thing the
application cares about. Predicted names are grounded through resolver.py exactly
like the runtime controller, so the score reflects the real system, not the model
in isolation.

Outcome classes (cost-ordered):
  correct   right extension (or correct rejection of an unknown name)
  clarify   asked to disambiguate, and the gold person is among the candidates
  miss      safe failure: said not-found / asked when it should have resolved cleanly
  MISROUTE  confident WRONG extension — the critical error to drive toward zero

Top-line: task-success rate (correct + recoverable clarify) under a misroute ceiling.
Sliced by language, voice, caller-style and gold action.

Usage:
  eval.py --gold data/audio/test.jsonl --pred preds.jsonl   # preds: {audio, prediction}
  eval.py                                                    # self-test
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

from resolver import Resolver

ROOT = Path(__file__).resolve().parent


def load_dir_ext(directory_csv):
    return {r["display_en"]: r["ext"] for r in csv.DictReader(open(directory_csv, encoding="utf-8"))}


def parse_pred(prediction: str) -> dict:
    """Model output -> {action, name?}. Accepts JSON or a bare name string."""
    prediction = (prediction or "").strip()
    try:
        d = json.loads(prediction)
        if isinstance(d, dict) and "action" in d:
            return d
    except Exception:
        pass
    return {"action": "resolve", "name": prediction}  # bare name


def ground(pred: dict, R: Resolver) -> dict:
    """Mirror the runtime controller: a 'resolve' is re-grounded through the resolver
    (which may downgrade to clarify/not_found); clarify/not_found pass through."""
    if pred.get("action") == "resolve":
        return R.resolve(pred.get("name", ""))
    return pred


def outcome(gold: dict, grounded: dict, ext_of: dict) -> str:
    ga = gold.get("action")
    pa = grounded.get("action")
    if ga == "resolve":
        gold_ext = ext_of.get(gold["name"])
        if pa == "resolve":
            return "correct" if grounded.get("ext") == gold_ext else "MISROUTE"
        if pa == "clarify":
            cands = [c["name"] for c in grounded.get("candidates", [])]
            return "clarify" if gold["name"] in cands else "miss"
        return "miss"                                   # not_found on a real person
    if ga == "not_found":
        if pa == "not_found":
            return "correct"                            # correct rejection
        if pa == "clarify":
            return "miss"                               # didn't misroute, but didn't reject
        return "MISROUTE"                               # false-accept: routed an unknown caller
    # ga == clarify (underspecified)
    if pa == "clarify":
        return "correct"
    if pa == "resolve":
        return "MISROUTE"                               # guessed instead of asking
    return "miss"


def score(items, R, ext_of):
    """items: list of (gold_target, prediction, slice_dict)."""
    buckets = defaultdict(lambda: defaultdict(int))      # slice_key -> outcome -> n
    def add(key, oc):
        buckets[key][oc] += 1; buckets[key]["_n"] += 1
    for gold, prediction, sl in items:
        oc = outcome(gold, ground(parse_pred(prediction), R), ext_of)
        add(("overall", ""), oc)
        for dim in ("lang", "voice", "style"):
            add((dim, sl.get(dim, "?")), oc)
        add(("gold_action", gold.get("action", "?")), oc)
    return buckets


def report(buckets):
    def line(key, b):
        n = b["_n"] or 1
        succ = (b["correct"] + b["clarify"]) / n * 100
        mis = b["MISROUTE"] / n * 100
        return (f"  {key[0]:12} {str(key[1]):20} n={n:4d}  "
                f"success={succ:5.1f}%  MISROUTE={mis:4.1f}%  "
                f"(ok={b['correct']} clar={b['clarify']} miss={b['miss']} mis={b['MISROUTE']})")
    order = ["overall", "gold_action", "lang", "voice", "style"]
    print("\n=== BENCHMARK SCORECARD ===")
    for dim in order:
        keys = sorted(k for k in buckets if k[0] == dim)
        for k in keys:
            print(line(k, buckets[k]))
    ov = buckets[("overall", "")]
    n = ov["_n"] or 1
    print(f"\nTASK SUCCESS {(ov['correct']+ov['clarify'])/n*100:.1f}%  |  "
          f"MISROUTE {ov['MISROUTE']/n*100:.2f}% (target <0.5%)  |  n={n}")


def load_items(gold_path, pred_path):
    preds = {json.loads(l)["audio"]: json.loads(l)["prediction"]
             for l in open(pred_path, encoding="utf-8")}
    items = []
    for l in open(gold_path, encoding="utf-8"):
        g = json.loads(l)
        if g["audio"] in preds:
            items.append((json.loads(g["target_text"]), preds[g["audio"]],
                          {"lang": g.get("lang"), "voice": g.get("voice"), "style": g.get("style")}))
    return items


def self_test(R, ext_of):
    real = next(iter(ext_of))                            # a real person
    fake = "David Miller"
    items = [
        ({"action": "resolve", "name": real}, json.dumps({"action": "resolve", "name": real}),
         {"lang": "en", "voice": "primetts", "style": "request"}),                 # correct
        ({"action": "resolve", "name": real}, json.dumps({"action": "resolve", "name": "Totally Wrong"}),
         {"lang": "en", "voice": "x", "style": "request"}),                        # MISROUTE
        ({"action": "not_found"}, json.dumps({"action": "not_found"}),
         {"lang": "en", "voice": "x", "style": "negative"}),                       # correct reject
        ({"action": "not_found"}, json.dumps({"action": "resolve", "name": real}),
         {"lang": "en", "voice": "x", "style": "negative"}),                       # MISROUTE (false accept)
        ({"action": "clarify", "field": "surname"}, json.dumps({"action": "clarify"}),
         {"lang": "mix", "voice": "x", "style": "clarify"}),                       # correct
    ]
    print(f"self-test on {len(items)} fabricated cases (real='{real}', fake='{fake}')")
    report(score(items, R, ext_of))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "data" / "audio" / "test.jsonl"))
    ap.add_argument("--pred")
    ap.add_argument("--directory", default=str(ROOT / "data" / "directory.csv"))
    args = ap.parse_args()
    R = Resolver(args.directory)
    ext_of = load_dir_ext(args.directory)
    if not args.pred:
        self_test(R, ext_of)
        return
    items = load_items(args.gold, args.pred)
    print(f"scored {len(items)} predictions from {args.pred}")
    report(score(items, R, ext_of))


if __name__ == "__main__":
    main()
