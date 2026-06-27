#!/usr/bin/env python3
"""Phase-3 (scaling) training data: department-disambiguation dialogs.

Teaches the agent the POLICY for a large directory where names collide: when
search_contacts returns several same-name people, ASK which department, then issue a
REFINED search with the department filter. This policy is DB-size-independent — the
directory stays an external CSV; the model only learns the behaviour.

Built from EXISTING request clips + SYNTHETIC collisions in the tool response + the
department-answer audio (data/audio/answers). No new request audio needed. Each dialog:

  [audio request "find X"] -> call(X) -> [3-4 same-name X across depts] ->
  "Which department — A, B, C?" -> [audio "the one in B"] ->
  call(X, department=B) -> [unique] -> "Connecting you to X in B, extension N."

Out: data/audio/dialogs_collision_<split>.jsonl
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260626
DEPARTMENTS = ["Sales", "Marketing", "Finance", "HR", "Engineering", "R&D", "IT",
               "Procurement", "Logistics", "Legal", "Customer Service", "Admin",
               "Quality", "Production", "Accounting"]


def call(query, department=None):
    args = {"query": query}
    if department:
        args["department"] = department
    return {"role": "assistant", "tool_call": {"name": "search_contacts", "arguments": args}}


def tool(matches):
    return {"role": "tool", "name": "search_contacts", "content": matches}


def say(text):
    return {"role": "assistant", "text": text}


def main():
    rng = random.Random(SEED)
    # department-answer clips, indexed by (lang, department)
    ans = defaultdict(list)
    for l in open(ROOT / "data" / "audio" / "answers" / "manifest.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d.get("answer_kind") == "dept":
            ans[(d["lang"], d["answer_value"])].append(d["wav"])

    for split, frac in (("train", 0.55), ("val", 0.5)):
        src = ROOT / "data" / "audio" / f"{split}.jsonl"
        if not src.exists():
            continue
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        resolves = [r for r in rows if r.get("action") == "resolve"]
        rng.shuffle(resolves)
        out = open(ROOT / "data" / "audio" / f"dialogs_collision_{split}.jsonl", "w", encoding="utf-8")
        n = 0
        for r in resolves:
            if rng.random() > frac:
                continue
            name = r["name"]
            lang = r.get("lang", "en")
            alang = "zh" if lang in ("zh", "mix") else "en"
            # choose 3-4 distinct departments; the chosen one must have an answer clip
            k = rng.choice([3, 3, 4])
            depts = rng.sample(DEPARTMENTS, k)
            chosen = rng.choice([d for d in depts if ans.get((alang, d))] or depts)
            if not ans.get((alang, chosen)):
                continue
            exts = rng.sample(range(1000, 9999), k)
            cluster = [{"name": name, "ext": str(e), "dept": d, "score": 100.0}
                       for e, d in zip(exts, depts)]
            chosen_row = next(c for c in cluster if c["dept"] == chosen)
            ans_wav = rng.choice(ans[(alang, chosen)])

            names_str = ", ".join(depts[:-1]) + f", or {depts[-1]}"
            turns = [
                {"role": "user", "audio": r["audio"]},
                call(name),
                tool(cluster),
                say(f"I found several people named {name} — in {names_str}. Which department?"),
                {"role": "user", "audio": ans_wav},
                call(name, department=chosen),
                tool([chosen_row]),
                say(f"Connecting you to {name} in {chosen}, extension {chosen_row['ext']}."),
            ]
            out.write(json.dumps({"turns": turns, "gold_action": "disambiguate",
                                  "lang": lang}, ensure_ascii=False) + "\n")
            n += 1
        out.close()
        print(f"  {split}: {n} collision dialogs -> dialogs_collision_{split}.jsonl")


if __name__ == "__main__":
    main()
