#!/usr/bin/env python3
"""Phase-2: multi-turn tool-calling dialogs for the agentic Qwen-Omni-3B.

Each conversation interleaves audio turns, the model's search_contacts tool-calls,
the tool results (from the live resolver), and the model's replies:

  resolve   : [audio req] -> call(query) -> [unique] -> "Connecting you to X, ext N."
  not_found : [audio req] -> call(query) -> [no match] -> "Sorry, not found."
  clarify   : [audio req] -> call(surname) -> [several] -> "Which? Henry/Joyce/Helen?"
              -> [audio answer "Henry"] -> call("Henry Tseng") -> [unique] -> "Connecting you..."

Built from existing request clips (data/audio/<split>.jsonl) + answer clips
(data/audio/answers). Out: data/audio/dialogs_<split>.jsonl (list of turns per line).
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resolver import Resolver
from data.generate_toolcall import base_id, load_manifest_text, spoken_name  # reuse helpers

ROOT = Path(__file__).resolve().parent.parent


def call(query):
    return {"role": "assistant", "tool_call": {"name": "search_contacts", "arguments": {"query": query}}}

def tool(result):
    return {"role": "tool", "name": "search_contacts", "content": result}

def say(text):
    return {"role": "assistant", "text": text}

def reply_resolve(a):
    return f"Connecting you to {a['name']}, extension {a['ext']}."

def clarify_q(cands):
    names = ", ".join(c["name"] for c in cands[:3])
    return f"I found several: {names}. Which one would you like?"


def main():
    R = Resolver(ROOT / "data" / "directory.csv")
    dir_zh = {r["display_en"]: r["full_chinese"]
              for r in csv.DictReader(open(ROOT / "data" / "directory.csv", encoding="utf-8"))}
    texts = load_manifest_text()
    # answer clips indexed by the first name they say
    ans_by_first = defaultdict(list)
    amf = ROOT / "data" / "audio" / "answers" / "manifest.jsonl"
    if amf.exists():
        for l in open(amf, encoding="utf-8"):
            d = json.loads(l)
            if d["answer_kind"] == "first":
                ans_by_first[d["answer_value"]].append(d["wav"])

    def topn(res):
        return ([{"name": res["name"], "ext": res["ext"], "dept": res.get("dept"),
                  "score": res.get("score")}] if res["action"] == "resolve"
                else res.get("candidates", []))

    for split in ("train", "val", "test"):
        src = ROOT / "data" / "audio" / f"{split}.jsonl"
        if not src.exists():
            continue
        out = open(ROOT / "data" / "audio" / f"dialogs_{split}.jsonl", "w", encoding="utf-8")
        n = mt = 0
        for l in open(src, encoding="utf-8"):
            r = json.loads(l)
            gold = json.loads(r["target_text"])
            query = spoken_name(texts.get(base_id(r["audio"]), ""), gold, r.get("lang"), dir_zh)
            res = R.resolve(query)
            turns = [{"role": "user", "audio": r["audio"]}, call(query), tool(topn(res))]

            if res["action"] == "clarify":
                cands = res["candidates"]
                # pick a target whose first name we have an answer clip for -> full multi-turn
                target = next((c for c in cands if c["name"].split()[0] in ans_by_first), None)
                if target:
                    turns.append(say(clarify_q(cands)))
                    turns.append({"role": "user", "audio": ans_by_first[target["name"].split()[0]][0]})
                    res2 = R.resolve(target["name"])           # refined query = first+surname
                    turns.append(call(target["name"]))
                    turns.append(tool(topn(res2)))
                    turns.append(say(reply_resolve(res2) if res2["action"] == "resolve"
                                     else clarify_q(cands)))
                    mt += 1
                else:
                    turns.append(say(clarify_q(cands)))
            elif res["action"] == "resolve":
                turns.append(say(reply_resolve(res)))
            else:
                turns.append(say("Sorry, I couldn't find that name in the directory."))

            out.write(json.dumps({"turns": turns, "gold_action": gold.get("action"),
                                  "lang": r.get("lang"), "style": r.get("style")},
                                 ensure_ascii=False) + "\n")
            n += 1
        out.close()
        print(f"  {split}: {n} dialogs ({mt} multi-turn) -> dialogs_{split}.jsonl")


if __name__ == "__main__":
    main()
