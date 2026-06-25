#!/usr/bin/env python3
"""Build agentic tool-use training data from EXISTING audio clips (no re-synthesis).

Each example: audio request -> assistant emits search_contacts(query=<heard name>)
-> tool returns the resolver's top-N from the LIVE directory -> assistant replies
(resolve / clarify / not_found). The DB lives in the CSV (a tool), never in the weights.

Per split (train/val/test) reads data/audio/<split>.jsonl, recovers the spoken name
from the base manifest, runs the resolver for a realistic tool result, and writes
data/audio/<split>_tool.jsonl with fields {audio, query, tool_result, reply, action, lang, style}.
"""
import csv, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resolver import Resolver

ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile(r"[一-鿿]+")
EN_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")


def base_id(audio_path):                      # clips/<baseid>_aN.wav -> <baseid>
    stem = Path(audio_path).stem
    return stem.rsplit("_a", 1)[0]


def load_manifest_text():
    text = {}
    for mf in (ROOT / "data" / "audio" / "base").glob("manifest*.jsonl"):
        for l in open(mf, encoding="utf-8"):
            d = json.loads(l); text[d["id"]] = d["text"]
    return text


def spoken_name(text, gold, lang, dir_zh):
    """The name as the caller said it -> the tool-call query target."""
    if gold.get("action") == "resolve":
        nm = gold["name"]
        return dir_zh.get(nm, nm) if lang == "zh" else nm        # zh -> 中文名, else English
    if gold.get("action") == "clarify":
        h = gold.get("heard", {})
        return h.get("surname") or h.get("first_name") or ""
    # not_found: pull the OOD name out of the spoken text
    cj = CJK.findall(text)
    if cj:
        return max(cj, key=len)                                   # longest CJK span ~ the name
    m = EN_NAME.findall(text)
    return m[-1] if m else text


def reply_for(action):
    a = action.get("action")
    if a == "resolve":
        return f"Connecting you to {action['name']}, extension {action['ext']}."
    if a == "clarify":
        names = ", ".join(c["name"] for c in action.get("candidates", [])[:3])
        return f"I found a few matches: {names}. Which one did you mean?"
    return "Sorry, I couldn't find that name in the directory."


def main():
    R = Resolver(ROOT / "data" / "directory.csv")
    dir_zh = {r["display_en"]: r["full_chinese"]
              for r in csv.DictReader(open(ROOT / "data" / "directory.csv", encoding="utf-8"))}
    texts = load_manifest_text()
    for split in ("train", "val", "test"):
        src = ROOT / "data" / "audio" / f"{split}.jsonl"
        if not src.exists():
            continue
        out = open(ROOT / "data" / "audio" / f"{split}_tool.jsonl", "w", encoding="utf-8")
        n = 0
        for l in open(src, encoding="utf-8"):
            r = json.loads(l)
            gold = json.loads(r["target_text"])
            text = texts.get(base_id(r["audio"]), "")
            query = spoken_name(text, gold, r.get("lang"), dir_zh)
            res = R.resolve(query)                                # realistic tool result + action
            tool_result = ([{"name": res["name"], "ext": res["ext"], "dept": res.get("dept"),
                             "score": res.get("score")}] if res["action"] == "resolve"
                           else res.get("candidates", []))
            out.write(json.dumps({
                "audio": r["audio"], "query": query,
                "tool_result": tool_result, "reply": reply_for(res),
                "action": res["action"], "gold_action": gold.get("action"),
                "lang": r.get("lang"), "style": r.get("style"),
            }, ensure_ascii=False) + "\n")
            n += 1
        out.close()
        print(f"  {split}: {n} tool-use examples -> {split}_tool.jsonl")


if __name__ == "__main__":
    main()
