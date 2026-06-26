#!/usr/bin/env python3
"""Scaling benchmark: resolver/retrieval metrics across the 200/2000/20000 collision tiers.

Pure-text retrieval eval (no audio, no GPU): builds query cases from each scaled
directory and measures how the perception->retrieval split holds up as the directory
outgrows the spoken-name namespace. The cardinal metric is MISROUTE (caller connected to
the WRONG person); we also track disambiguation success (collisions resolved after a
department refine), over-clarify, not-found rejection, turns-to-resolve, and latency@N.

  .venv/bin/python eval_scaling.py
"""
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from resolver import Resolver

sys.path.insert(0, str(Path(__file__).parent / "data"))
from generate_directory_scaled import ENGLISH_FIRST, SURNAMES, GIVEN_HANZI

# phonetic / romanization neighbours an ASR or caller might land on (the resolver's job
# is to absorb these without misrouting).
SUR_NB = {"Chen": "Cheng", "Cheng": "Chen", "Lin": "Ling", "Wang": "Wong", "Lee": "Li",
          "Wu": "Woo", "Chang": "Chiang", "Huang": "Hwang", "Tsai": "Tsay", "Hsu": "Shu",
          "Liu": "Liou", "Yang": "Yan", "Chou": "Chow", "Su": "Soo", "Kuo": "Kwo"}
FST_NB = {"Kevin": "Calvin", "Jason": "Jayson", "Eric": "Erik", "Tony": "Toni",
          "Henry": "Henri", "Gary": "Garry", "Allen": "Alan", "Sean": "Shawn",
          "Karen": "Karin", "Vivian": "Vivien", "Steve": "Steven", "Andy": "Andi"}


def corrupt_en(name):
    """Mild phonetic corruption of 'First Surname'. Returns a changed string."""
    parts = name.split()
    if len(parts) < 2:
        return name + "n"
    f, s = parts[0], parts[-1]
    if s in SUR_NB:
        s = SUR_NB[s]
    elif f in FST_NB:
        f = FST_NB[f]
    else:
        s = s + "g" if not s.endswith("g") else s[:-1]   # last-resort perturbation
    return f"{f} {s}"


def build_cases(R, rng, n_each):
    by_en = defaultdict(list)
    for c in R.contacts:
        by_en[c.name].append(c)
    uniques = [c for c in R.contacts if len(by_en[c.name]) == 1]
    clusters = [v for v in by_en.values() if len(v) > 1]
    names_present = set(by_en)

    cases = []
    # 1) unique name -> should resolve to that ext
    for c in rng.sample(uniques, min(n_each, len(uniques))):
        cases.append(("unique", c.name, None, c.ext))
    # 2) collision -> name alone must clarify; dept refine must resolve the target
    for v in rng.sample(clusters, min(n_each, len(clusters))):
        tgt = rng.choice(v)
        same_dept = sum(1 for x in v if x.dept == tgt.dept)
        cases.append(("collision", tgt.name, tgt.dept, tgt.ext if same_dept == 1 else "AMBIG"))
    # 3) misheard unique name -> resolver must still find source, never a different ext
    for c in rng.sample(uniques, min(n_each, len(uniques))):
        cases.append(("misheard", corrupt_en(c.name), None, c.ext))
    # 4) not-found -> a plausible name absent from the directory
    tries = 0
    nf = 0
    while nf < n_each and tries < n_each * 40:
        tries += 1
        nm = f"{rng.choice(ENGLISH_FIRST)} {rng.choice(SURNAMES)[0]}"
        if nm not in names_present:
            cases.append(("not_found", nm, None, None))
            nf += 1
    return cases


def evaluate(tier, path, scaled=False, n_each=200, seed=11):
    """Caller oracle: on a `confirm`, the caller says YES iff the offered ext is the
    intended person — so confirm turns silent misroutes into safe one-turn checks and
    lets absent names be rejected. clarify is followed by a department refine."""
    rng = random.Random(seed)
    R = Resolver(path)
    cases = build_cases(R, rng, n_each)

    m = defaultdict(int)
    bt = defaultdict(lambda: defaultdict(int))
    turns_sum = turns_n = 0
    t0 = time.perf_counter()
    n_resolves = 0

    def confirm_yes(res, intended):
        return res.get("ext") == intended           # caller confirms iff it's the right person

    for kind, query, dept, exp in cases:
        bt[kind]["n"] += 1
        res = R.resolve(query, scaled=scaled); n_resolves += 1
        a, got = res["action"], res.get("ext")

        if kind == "unique":
            if a == "resolve" and got == exp:
                bt[kind]["ok"] += 1; turns_sum += 1; turns_n += 1
            elif a == "resolve":
                m["misroute"] += 1; bt[kind]["misroute"] += 1
            elif a == "confirm":
                if confirm_yes(res, exp): bt[kind]["ok"] += 1; turns_sum += 2; turns_n += 1
                else: bt[kind]["miss"] += 1          # safe (caller declines) but didn't connect
            elif a == "clarify":
                bt[kind]["overclarify"] += 1
            else:
                bt[kind]["miss"] += 1

        elif kind == "collision":
            if a == "resolve":                        # picked one ext from many same-name -> misroute
                m["misroute"] += 1; bt[kind]["misroute"] += 1
            elif a in ("clarify", "confirm"):         # refine by department
                r2 = R.resolve(query, filters={"department": dept}, scaled=scaled); n_resolves += 1
                if r2["action"] == "resolve" and r2.get("ext") == exp:
                    bt[kind]["disambig_ok"] += 1; turns_sum += 2; turns_n += 1
                elif r2["action"] == "resolve" and exp == "AMBIG":
                    m["misroute"] += 1; bt[kind]["misroute"] += 1
                elif exp == "AMBIG":
                    bt[kind]["needs_finer_key"] += 1
                else:
                    bt[kind]["refine_miss"] += 1
            else:
                bt[kind]["miss"] += 1

        elif kind == "misheard":
            cands = {x["ext"] for x in res.get("candidates", [])} | ({got} if got else set())
            if a == "resolve" and got == exp:
                bt[kind]["ok"] += 1; turns_sum += 1; turns_n += 1
            elif a == "resolve":
                m["misroute"] += 1; bt[kind]["misroute"] += 1
            elif a == "confirm":
                if confirm_yes(res, exp): bt[kind]["ok"] += 1; turns_sum += 2; turns_n += 1
                else: bt[kind]["safe_decline"] += 1   # offered wrong person, caller declines -> no misroute
            elif exp in cands:
                bt[kind]["clarify_has_src"] += 1
            else:
                bt[kind]["lost"] += 1

        elif kind == "not_found":
            if a == "not_found":
                bt[kind]["ok"] += 1
            elif a == "resolve":
                m["misroute"] += 1; bt[kind]["false_accept"] += 1
            elif a == "confirm":                      # absent -> caller declines -> correctly rejected
                bt[kind]["ok"] += 1
            else:
                bt[kind]["overclarify"] += 1

    latency = (time.perf_counter() - t0) / n_resolves * 1000
    total = len(cases)
    return {
        "tier": tier, "mode": "scaled" if scaled else "legacy", "n": total,
        "misroute_%": m["misroute"] / total * 100,
        "uniq_resolve_%": bt["unique"]["ok"] / max(bt["unique"]["n"], 1) * 100,
        "overclarify_%": bt["unique"]["overclarify"] / max(bt["unique"]["n"], 1) * 100,
        "disambig_%": bt["collision"]["disambig_ok"] / max(bt["collision"]["n"], 1) * 100,
        "notfound_rej_%": bt["not_found"]["ok"] / max(bt["not_found"]["n"], 1) * 100,
        "misheard_ok_%": (bt["misheard"]["ok"] + bt["misheard"]["clarify_has_src"]) / max(bt["misheard"]["n"], 1) * 100,
        "turns": turns_sum / max(turns_n, 1),
        "ms": latency,
    }


def main():
    tiers = [("200", "data/directory_200.csv"),
             ("2000", "data/directory_2000.csv"),
             ("20000", "data/directory_20000.csv")]
    cols = ["tier", "mode", "misroute_%", "uniq_resolve_%", "overclarify_%",
            "disambig_%", "notfound_rej_%", "misheard_ok_%", "turns", "ms"]
    w = {c: max(len(c), 6) for c in cols}
    print(" | ".join(c.rjust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for t, p in tiers:
        for scaled in (False, True):
            r = evaluate(t, p, scaled=scaled)
            cells = [(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c])).rjust(w[c]) for c in cols]
            print(" | ".join(cells))
        print("-+-".join("-" * w[c] for c in cols))
    print("\nMISROUTE = caller connected to the WRONG person (cardinal safety metric).")
    print("scaled mode adds: unique-exact -> resolve (no over-clarify); homonyms -> clarify;")
    print("strong non-exact -> CONFIRM (silent misroute -> safe 1-turn check; absent name -> rejectable).")


if __name__ == "__main__":
    main()
