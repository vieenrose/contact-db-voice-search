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


def evaluate(tier, path, n_each=200, seed=11):
    rng = random.Random(seed)
    R = Resolver(path)
    by_ext = {c.ext: c for c in R.contacts}
    cases = build_cases(R, rng, n_each)

    m = defaultdict(int)
    by_type = defaultdict(lambda: defaultdict(int))
    turns_sum = turns_n = 0
    t0 = time.perf_counter()
    n_resolves = 0

    for kind, query, dept, exp in cases:
        by_type[kind]["n"] += 1
        res = R.resolve(query)
        n_resolves += 1
        a = res["action"]
        got = res.get("ext")

        if kind == "unique":
            if a == "resolve" and got == exp:
                by_type[kind]["ok"] += 1; turns_sum += 1; turns_n += 1
            elif a == "resolve":
                m["misroute"] += 1; by_type[kind]["misroute"] += 1
            elif a == "clarify":
                by_type[kind]["overclarify"] += 1
            else:
                by_type[kind]["miss"] += 1

        elif kind == "collision":
            if a == "resolve":                      # picked one ext from many same-name -> misroute
                m["misroute"] += 1; by_type[kind]["misroute"] += 1
            elif a == "clarify":                    # correct: now refine by department
                r2 = R.resolve(query, filters={"department": dept}); n_resolves += 1
                if r2["action"] == "resolve" and r2.get("ext") == exp:
                    by_type[kind]["disambig_ok"] += 1; turns_sum += 2; turns_n += 1
                elif r2["action"] == "resolve" and exp == "AMBIG":
                    by_type[kind]["misroute"] += 1; m["misroute"] += 1   # dept couldn't split -> wrong pick
                elif exp == "AMBIG":
                    by_type[kind]["needs_finer_key"] += 1
                else:
                    by_type[kind]["refine_miss"] += 1
            else:
                by_type[kind]["miss"] += 1

        elif kind == "misheard":
            cands = {x["ext"] for x in res.get("candidates", [])} | ({got} if got else set())
            if a == "resolve" and got == exp:
                by_type[kind]["ok"] += 1; turns_sum += 1; turns_n += 1
            elif a == "resolve":                    # resolved to a DIFFERENT person
                m["misroute"] += 1; by_type[kind]["misroute"] += 1
            elif exp in cands:
                by_type[kind]["clarify_has_src"] += 1
            else:
                by_type[kind]["lost"] += 1

        elif kind == "not_found":
            if a == "not_found":
                by_type[kind]["ok"] += 1
            elif a == "resolve":
                m["misroute"] += 1; by_type[kind]["false_accept"] += 1
            else:
                by_type[kind]["overclarify"] += 1

    latency = (time.perf_counter() - t0) / n_resolves * 1000
    total = len(cases)
    return {
        "tier": tier, "n": total,
        "misroute_%": m["misroute"] / total * 100,
        "uniq_resolve_%": by_type["unique"]["ok"] / max(by_type["unique"]["n"], 1) * 100,
        "overclarify_%": by_type["unique"]["overclarify"] / max(by_type["unique"]["n"], 1) * 100,
        "disambig_%": by_type["collision"]["disambig_ok"] / max(by_type["collision"]["n"], 1) * 100,
        "needs_finer_%": by_type["collision"]["needs_finer_key"] / max(by_type["collision"]["n"], 1) * 100,
        "notfound_rej_%": by_type["not_found"]["ok"] / max(by_type["not_found"]["n"], 1) * 100,
        "misheard_ok_%": (by_type["misheard"]["ok"] + by_type["misheard"]["clarify_has_src"]) / max(by_type["misheard"]["n"], 1) * 100,
        "turns": turns_sum / max(turns_n, 1),
        "ms": latency,
    }


def main():
    tiers = [("200", "data/directory_200.csv"),
             ("2000", "data/directory_2000.csv"),
             ("20000", "data/directory_20000.csv")]
    rows = [evaluate(t, p) for t, p in tiers]
    cols = ["tier", "n", "misroute_%", "uniq_resolve_%", "overclarify_%",
            "disambig_%", "needs_finer_%", "notfound_rej_%", "misheard_ok_%", "turns", "ms"]
    w = {c: max(len(c), 7) for c in cols}
    print(" | ".join(c.rjust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append((f"{v:.1f}" if isinstance(v, float) else str(v)).rjust(w[c]))
        print(" | ".join(cells))
    print("\nMISROUTE = caller connected to the WRONG person (cardinal safety metric).")
    print("disambig_% = collisions resolved after a department refine; needs_finer_% = "
          "department couldn't split the cluster (needs floor/team — the 20k problem).")


if __name__ == "__main__":
    main()
