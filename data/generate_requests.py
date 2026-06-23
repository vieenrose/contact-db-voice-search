#!/usr/bin/env python3
"""Generate synthetic caller requests paired with action-based targets.

Reads data/directory.csv and emits the text a caller might speak (step 3 voices it
with TTS), paired with the structured action the Ultravox model must produce. The
schema fits the hybrid design (model = perception, controller = policy):

  resolve   {action:"resolve", name, ext}                 # uniquely identified
  clarify   {action:"clarify", field, heard:{...}}        # underspecified -> ask
  not_found {action:"not_found"}                          # not in directory

Plus "distractor" resolves: a caller self-introduction precedes the request, so the
model must extract the REQUESTED person, not the caller.

Clarify targets do NOT enumerate candidates — the controller looks those up in the
directory. This keeps outputs short and labels consistent (a given underspecified
text always maps to the same action).

Deterministic. Usage: .venv/bin/python data/generate_requests.py [PER_CONTACT]
Writes data/requests.jsonl (fields: text, target, lang, style, split).
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260623
PER_CONTACT_DEFAULT = 12
DISTRACTOR_PER_CONTACT = 3
CLARIFY_PER_GROUP = 4
NEG_FRACTION = 0.15
VAL_FRACTION = 0.10

DEPT_ZH = {
    "Sales": "業務", "Marketing": "行銷", "Finance": "財務", "HR": "人資",
    "Engineering": "工程", "R&D": "研發", "IT": "資訊", "Procurement": "採購",
    "Logistics": "物流", "Legal": "法務", "Customer Service": "客服",
    "Admin": "行政", "Quality": "品保", "Production": "生產", "Accounting": "會計",
}
FEMALE = {
    "Amy", "Vivian", "Cindy", "Grace", "Joyce", "Sharon", "Tina", "Wendy", "Iris",
    "Emily", "Stella", "Fiona", "Karen", "Rita", "Nina", "Daisy", "Coco", "Ruby",
    "Helen", "Angel", "Yuki", "Annie", "Carol", "Mia",
}
EN_FILLERS = ["um, ", "uh, ", "hi, ", "hello, ", "sorry, ", "yeah, ", "ok so, "]
ZH_FILLERS = ["喂你好，", "那個，", "嗯，", "請問一下，", "麻煩一下，", "你好，", "欸，"]

# Caller self-introductions (the distractor the model must ignore).
EN_INTROS = ["hi, this is {c} from {org}, ", "hello, {c} here, ", "yeah this is {c}, ",
             "good morning, {c} speaking, "]
ZH_INTROS = ["你好我是{c}，", "喂您好，我這邊是{c}，", "我是{c}，", "您好敝姓{c}，"]
CALLER_EN = ["John", "Mike", "Sarah", "David", "Linda", "Paul", "Mary", "Tom"]
CALLER_ZH = ["王", "李", "張", "林", "周", "趙"]
ORGS = ["ABC Company", "the bank", "DHL", "head office", "Acme", "the supplier"]

# resolve templates: (template, lang, reveal-fields) — reveal must be unique.
RESOLVE = [
    ("could you put me through to {en}?", "en", ("display_en",)),
    ("i'd like to speak to {en} please.", "en", ("display_en",)),
    ("can i get {en}'s extension?", "en", ("display_en",)),
    ("transfer me to {en}.", "en", ("display_en",)),
    ("hi, {en} please.", "en", ("display_en",)),
    ("what's the extension for {en}?", "en", ("display_en",)),
    ("{en} in {dept}, please.", "en", ("display_en",)),
    ("i'm looking for {hon_en} {sur_en}, {first}.", "en", ("english_first", "surname_pinyin")),
    ("connect me with {first} {sur_en} from {dept}.", "en", ("english_first", "surname_pinyin")),
    ("我要找{zh}", "zh", ("full_chinese",)),
    ("請幫我轉接{zh}", "zh", ("full_chinese",)),
    ("麻煩幫我接{zh}{hon_zh}", "zh", ("full_chinese",)),
    ("請問{zh}的分機是多少", "zh", ("full_chinese",)),
    ("我找{dept_zh}的{zh}", "zh", ("full_chinese",)),
    ("可以幫我接{zh}嗎", "zh", ("full_chinese",)),
    ("幫我轉 {first} {sur_en}", "mix", ("english_first", "surname_pinyin")),
    ("請接 {first}，{sur_zh}{hon_zh}", "mix", ("english_first", "surname_hanzi")),
    ("麻煩接一下 {en}", "mix", ("display_en",)),
    ("請問 {first} {sur_en} 的分機", "mix", ("english_first", "surname_pinyin")),
    ("找 {sur_zh} 的 {first}", "mix", ("english_first", "surname_hanzi")),
]
# first-name-only -> clarify by surname
FIRST_ONLY = [
    ("i need {first}.", "en"), ("is {first} there?", "en"), ("can i talk to {first}?", "en"),
    ("找 {first}", "mix"), ("{first} 在嗎", "mix"), ("我要找 {first}", "mix"),
]
# surname-only -> clarify by first name (gender-neutral phrasings)
SURNAME_ONLY = [
    ("i'm looking for someone called {sur_en}.", "en"), ("do you have a {sur_en} there?", "en"),
    ("我要找姓{sur_zh}的", "zh"), ("請問你們有姓{sur_zh}的嗎", "zh"), ("麻煩找一下姓{sur_zh}的", "zh"),
]


def hon_en(f): return "Ms." if f in FEMALE else "Mr."
def hon_zh(f): return "小姐" if f in FEMALE else "先生"


def fill(t, c, rng):
    g = c.get
    first = g("english_first", "")
    return t.format(
        en=g("display_en", ""), first=first, sur_en=g("surname_pinyin", ""),
        sur_zh=g("surname_hanzi", ""), zh=g("full_chinese", ""), dept=g("department", ""),
        dept_zh=DEPT_ZH.get(g("department", ""), ""), hon_en=hon_en(first), hon_zh=hon_zh(first),
        ext_word=rng.choice(["extension", "ext", "extension number"]))


def filler(text, lang, rng):
    if rng.random() < 0.45:
        return rng.choice(ZH_FILLERS if lang == "zh" else EN_FILLERS) + text
    return text


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else PER_CONTACT_DEFAULT
    rng = random.Random(SEED)
    here = Path(__file__).parent
    contacts = list(csv.DictReader(open(here / "directory.csv", encoding="utf-8")))

    reveal_sets = {t[2] for t in RESOLVE}
    counts = {fs: Counter(tuple(c[f] for f in fs) for c in contacts) for fs in reveal_sets}
    usable = lambda c, fs: counts[fs][tuple(c[f] for f in fs)] == 1
    LANG_W = {"en": 1, "zh": 2, "mix": 2}

    by_first = defaultdict(list)
    by_sur = defaultdict(list)
    for c in contacts:
        by_first[c["english_first"]].append(c)
        by_sur[c["surname_pinyin"]].append(c)
    valid_display = {c["display_en"] for c in contacts}
    valid_zh = {c["full_chinese"] for c in contacts}
    givens = sorted({c["given_hanzi"] for c in contacts})
    firsts = sorted(by_first)
    surs = sorted({(c["surname_pinyin"], c["surname_hanzi"]) for c in contacts})

    S = []

    def emit(text, lang, style, target):
        S.append({"text": text, "lang": lang, "style": style, "target": target})

    # 1) resolve
    for c in contacts:
        pool = [t for t in RESOLVE if usable(c, t[2])]
        weighted = [t for t in pool for _ in range(LANG_W[t[1]])]
        tgt = {"action": "resolve", "name": c["display_en"], "ext": c["ext"]}
        for _ in range(per):
            tmpl, lang, _ = rng.choice(weighted)
            emit(filler(fill(tmpl, c, rng), lang, rng), lang, "request", tgt)

    # 2) distractor resolves (caller self-intro prefix)
    for c in contacts:
        pool = [t for t in RESOLVE if usable(c, t[2])]
        tgt = {"action": "resolve", "name": c["display_en"], "ext": c["ext"]}
        for _ in range(DISTRACTOR_PER_CONTACT):
            tmpl, lang, _ = rng.choice(pool)
            body = fill(tmpl, c, rng)
            if lang == "zh":
                intro = rng.choice(ZH_INTROS).format(c=rng.choice(CALLER_ZH))
            else:
                intro = rng.choice(EN_INTROS).format(c=rng.choice(CALLER_EN), org=rng.choice(ORGS))
            emit(intro + body, lang, "distractor", tgt)

    # 3) clarify — first-name-only (>=2 share the first name) -> ask surname
    for first, members in by_first.items():
        if len(members) < 2:
            continue
        tgt = {"action": "clarify", "field": "surname", "heard": {"first_name": first}}
        for _ in range(CLARIFY_PER_GROUP):
            tmpl, lang = rng.choice(FIRST_ONLY)
            emit(filler(fill(tmpl, {"english_first": first}, rng), lang, rng), lang, "clarify", tgt)

    # 4) clarify — surname-only (>=2 share the surname) -> ask first name
    for (sur_en, members) in [(k, v) for k, v in by_sur.items() if len(v) >= 2]:
        sur_zh = members[0]["surname_hanzi"]
        tgt = {"action": "clarify", "field": "first_name", "heard": {"surname": sur_en}}
        for _ in range(CLARIFY_PER_GROUP):
            tmpl, lang = rng.choice(SURNAME_ONLY)
            emit(filler(fill(tmpl, {"surname_pinyin": sur_en, "surname_hanzi": sur_zh}, rng), lang, rng),
                 lang, "clarify", tgt)

    # 5) not_found — full name absent from the directory
    n_neg = int(len(S) * NEG_FRACTION)
    en_full = [t for t in RESOLVE if t[2] == ("display_en",)]
    zh_full = [t for t in RESOLVE if t[2] == ("full_chinese",)]
    made = 0
    while made < n_neg:
        first = rng.choice(firsts); sur_en, sur_zh = rng.choice(surs); given = rng.choice(givens)
        if rng.random() < 0.5:
            if f"{first} {sur_en}" in valid_display:
                continue
            nc = {"display_en": f"{first} {sur_en}", "english_first": first, "surname_pinyin": sur_en,
                  "department": rng.choice(list(DEPT_ZH))}
            tmpl, lang, _ = rng.choice(en_full)
        else:
            if f"{sur_zh}{given}" in valid_zh:
                continue
            nc = {"full_chinese": f"{sur_zh}{given}", "english_first": first,
                  "department": rng.choice(list(DEPT_ZH))}
            tmpl, lang, _ = rng.choice(zh_full)
        emit(filler(fill(tmpl, nc, rng), lang, rng), lang, "negative", {"action": "not_found"})
        made += 1

    rng.shuffle(S)
    for s in S:
        s["split"] = "val" if rng.random() < VAL_FRACTION else "train"

    out = here / "requests.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in S:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    by_style = Counter(s["style"] for s in S)
    by_action = Counter(s["target"]["action"] for s in S)
    n_val = sum(1 for s in S if s["split"] == "val")
    print(f"wrote {len(S)} samples -> {out}", file=sys.stderr)
    print(f"  by style: {dict(by_style)}", file=sys.stderr)
    print(f"  by action: {dict(by_action)}  val={n_val}", file=sys.stderr)


if __name__ == "__main__":
    main()
