#!/usr/bin/env python3
"""Closed-set fuzzy name resolver — grounds a (possibly mis-heard) spoken name to a
real directory row + extension. Used by the dialog controller, the eval scorer, and
the demo Space.

The model emits a name string; ASR/telephony makes it imperfect ("Calvin Cheng" for
"Kevin Chen", a hanzi name, a romanization variant). We match against the CLOSED
200-person directory using a blend of:
  - normalized string similarity (RapidFuzz) over English + Chinese + pinyin keys,
  - romanization alias folding (Hsu/Xu/Syu, Chang/Zhang, Lee/Li, ...),
  - cross-script pinyin (so a hanzi query matches a romanized record and vice-versa).
Score margins drive the action: resolve / clarify / not_found.

CPU, sub-ms over 200 rows. Deps: rapidfuzz, pypinyin (optional — degrades gracefully).
"""
from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

try:
    from pypinyin import lazy_pinyin
    _HAS_PINYIN = True
except Exception:  # pragma: no cover
    _HAS_PINYIN = False

# Decision thresholds (0-100). Tunable on the benchmark.
HIGH = 82      # top-1 above this and clearly ahead -> resolve
MARGIN = 8     # top-1 must beat top-2 by this to be unambiguous
LOW = 60       # top-1 below this -> not_found

# Romanization variants seen on Taiwan business cards / from ASR -> canonical pinyin.
ALIAS = {
    "xu": "hsu", "syu": "hsu", "shyu": "hsu",
    "zhang": "chang", "jhang": "chang",
    "li": "lee", "lyi": "lee",
    "cai": "tsai", "tsay": "tsai",
    "zhou": "chou", "jhou": "chou",
    "huang": "huang", "hwang": "huang", "wong": "huang",
    "chen": "chen", "chern": "chen",
    "wu": "wu", "woo": "wu",
    "zheng": "cheng", "jheng": "cheng",
    "xie": "hsieh", "shieh": "hsieh",
    "guo": "kuo", "kwo": "kuo",
    "hong": "hung",
    "qiu": "chiu", "chiou": "chiu",
    "zeng": "tseng",
    "lyu": "lu", "lv": "lu",
    "zhong": "chung", "jhong": "chung",
}
HONORIFICS = re.compile(r"\b(mr|mrs|ms|miss|mister|sir)\b\.?", re.I)
ZH_HON = re.compile(r"(先生|小姐|女士|經理|你好|您好|請問)")
CJK = re.compile(r"[一-鿿]")


def _norm_en(s: str) -> str:
    s = HONORIFICS.sub(" ", s.lower())
    s = re.sub(r"[^a-z\s]", " ", s)
    toks = [ALIAS.get(t, t) for t in s.split()]
    return " ".join(toks).strip()


def _to_pinyin(s: str) -> str:
    if not _HAS_PINYIN:
        return ""
    return " ".join(lazy_pinyin(s))


@dataclass
class Contact:
    name: str          # display_en, the canonical key
    ext: str
    zh: str
    dept: str
    en_key: str        # normalized english "first surname"
    en_first: str
    en_sur: str
    zh_sur: str        # surname hanzi (first char)
    zh_given: str      # given hanzi (rest)
    zh_pin: str        # pinyin of full chinese name


def _pair(q, c):
    """Both-components score: a FULL name (2 tokens) must match first AND surname;
    a single token matches either (underspecified -> will tie multiple -> clarify)."""
    qf = q.split()
    cf, cs = (c[0], c[-1]) if len(c) >= 2 else (c[0], c[0])
    if len(qf) >= 2:
        f, s = fuzz.ratio(qf[0], cf), fuzz.ratio(qf[-1], cs)
        return (f + s) / 2 if f >= 75 and s >= 75 else min(f, s) * 0.5   # both required
    return max(fuzz.ratio(qf[0], cf), fuzz.ratio(qf[0], cs)) if qf else 0.0


class Resolver:
    def __init__(self, directory_csv: str | Path):
        self.contacts: list[Contact] = []
        with open(directory_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                en = _norm_en(r["display_en"]); parts = en.split()
                zh = r["full_chinese"]
                self.contacts.append(Contact(
                    name=r["display_en"], ext=r["ext"], zh=zh, dept=r["department"],
                    en_key=en, en_first=parts[0], en_sur=parts[-1],
                    zh_sur=zh[:1], zh_given=zh[1:], zh_pin=_to_pinyin(zh)))

    def _score(self, q_en: str, q_zh: str, q_pin: str, c: Contact) -> float:
        if q_zh:                              # Chinese query -> precise hanzi path (surname char + given)
            if len(q_zh) >= 2:
                fs = 100 if q_zh[0] == c.zh_sur else 0
                gs = fuzz.ratio(q_zh[1:], c.zh_given)
                return (fs + gs) / 2 if fs and gs >= 60 else min(fs, gs) * 0.5
            return 100.0 if q_zh[0] == c.zh_sur else 0.0     # surname-only -> ties many -> clarify
        if q_en:                              # romanized / English query
            s = _pair(q_en, [c.en_first, c.en_sur])          # English name (component-aware)
            if c.zh_pin:                      # cross-script: romanized -> chinese pinyin (order-free)
                s = max(s, fuzz.token_sort_ratio(q_en, c.zh_pin))
            return s
        return 0.0

    def rank(self, query: str, k: int = 5):
        q = query.strip()
        q_zh = "".join(CJK.findall(q))
        q_zh = ZH_HON.sub("", q) if not q_zh else q_zh
        q_zh = "".join(CJK.findall(q_zh))
        q_en = _norm_en(q) if re.search(r"[A-Za-z]", q) else ""
        q_pin = _to_pinyin(q_zh) if q_zh else ""
        scored = [(self._score(q_en, q_zh, q_pin, c), c) for c in self.contacts]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def resolve(self, query: str) -> dict:
        """Return an action dict the controller can act on."""
        ranked = self.rank(query)
        if not ranked:
            return {"action": "not_found", "query": query}
        top, c = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        cand = [{"name": x.name, "ext": x.ext, "dept": x.dept, "score": round(s, 1)}
                for s, x in ranked if s >= LOW]
        if top < LOW:
            return {"action": "not_found", "query": query, "best": c.name, "score": round(top, 1)}
        n_strong = sum(1 for s, _ in ranked if s >= HIGH)
        # resolve when there's ONE clearly-best strong match; clarify only when two strong
        # candidates are genuinely close (underspecified), or the best match is weak.
        if top >= HIGH and (n_strong < 2 or (top - second) >= MARGIN):
            return {"action": "resolve", "name": c.name, "ext": c.ext, "dept": c.dept,
                    "score": round(top, 1)}
        return {"action": "clarify", "query": query, "candidates": cand[:4]}


if __name__ == "__main__":
    import sys
    here = Path(__file__).parent
    R = Resolver(here / "data" / "directory.csv")
    if len(sys.argv) > 1:
        import json
        print(json.dumps(R.resolve(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))
    else:
        print("loaded", len(R.contacts), "contacts; running self-test\n")
        # pick a few real contacts and corrupt them
        import random
        random.seed(1)
        ok = 0
        samples = random.sample(R.contacts, 8)
        for c in samples:
            tests = [c.name, c.zh]
            # phonetic corruption of the english name
            corrupt = c.name.replace("Chen", "Cheng").replace("Lin", "Ling").replace("Kevin", "Calvin")
            tests.append(corrupt)
            for t in tests:
                res = R.resolve(t)
                hit = res.get("name") == c.name or any(
                    x["name"] == c.name for x in res.get("candidates", []))
                ok += hit
                print(f"  {t!r:32} -> {res.get('action'):9} "
                      f"{res.get('name') or [x['name'] for x in res.get('candidates',[])]}  {'OK' if hit else 'MISS'}")
        print(f"\n{ok} hits across corrupted queries")
