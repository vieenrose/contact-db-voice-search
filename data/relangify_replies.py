#!/usr/bin/env python3
"""Mirror the caller's language in assistant TEXT replies (the spoken-reply surface only).

zh / mix dialogs -> Traditional-Chinese replies; en dialogs -> English (unchanged). Audio turns and
`<tool_call>` turns are left byte-for-byte identical, so tool-calling supervision is untouched — this
ONLY rewrites the natural-language `assistant.text` turns. A Chinese caller now hears a zh-TW answer,
an English caller an English one. Run after the dialog files exist:

  .venv/bin/python data/relangify_replies.py \
      --in data/audio/dialogs_phase3_train.jsonl --out data/audio/dialogs_phase3_train_ml.jsonl
"""
import argparse
import json
import re
import sys

# English department label -> Traditional-Chinese (Taiwan office usage).
DEPT_ZH = {
    "Admin": "行政部", "Sales": "業務部", "HR": "人資部", "Marketing": "行銷部", "R&D": "研發部",
    "IT": "資訊部", "Finance": "財務部", "Procurement": "採購部", "Legal": "法務部",
    "Accounting": "會計部", "Logistics": "物流部", "Production": "生產部",
    "Customer Service": "客服部", "Engineering": "工程部", "Quality": "品保部",
}

R_RESOLVE_DEPT = re.compile(r"^Connecting you to (.+?) in (.+?), extension (\d+)\.$")
R_RESOLVE = re.compile(r"^Connecting you to (.+?), extension (\d+)\.$")
R_NOTFOUND = re.compile(r"^Sorry, I couldn't find that name in the directory\.$")
R_CLAR_DEPT = re.compile(r"^I found several people named (.+?) — in (.+?)\. Which department\?$")
R_CLAR_SIMPLE = re.compile(r"^I found several: (.+?)\. Which one would you like\?$")


def parse_dept_list(s):
    """'R&D, Procurement, or Quality' / 'A or B' / 'A, B, C, or D' -> ['R&D','Procurement','Quality']."""
    s = s.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:                                   # two-item "A or B"
        parts = [p.strip() for p in s.split(" or ")]
    return [re.sub(r"^or ", "", p).strip() for p in parts if p.strip()]


def zh_dept(d):
    return DEPT_ZH[d]                       # KeyError surfaces an unmapped dept loudly


def to_zh(text):
    """English reply -> zh-TW, or None if no known pattern matches (caller must treat as an error)."""
    m = R_RESOLVE_DEPT.match(text)
    if m:
        name, dept, ext = m.groups()
        return f"為您轉接{zh_dept(dept)}的{name}，分機{ext}。"
    m = R_RESOLVE.match(text)
    if m:
        name, ext = m.groups()
        return f"為您轉接{name}，分機{ext}。"
    if R_NOTFOUND.match(text):
        return "抱歉，通訊錄裡找不到這個名字。"
    m = R_CLAR_DEPT.match(text)
    if m:
        name, depts = m.groups()
        zl = "、".join(zh_dept(d) for d in parse_dept_list(depts))
        return f"找到多位{name}，分別在{zl}，請問您要找哪個部門?"
    m = R_CLAR_SIMPLE.match(text)
    if m:
        names = [n.strip() for n in m.group(1).split(",")]
        return f"找到幾位：{'、'.join(names)}，請問您要哪一位?"
    return None


def reply_in_zh(lang):
    """Mirror the caller: Chinese (zh) and code-switched (mix) callers get zh-TW; English stays English."""
    return lang in ("zh", "mix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    n_dialogs = n_zh = n_en = n_conv = 0
    misses = []
    with open(args.inp, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            r = json.loads(line)
            n_dialogs += 1
            want_zh = reply_in_zh(r.get("lang", "en"))
            for t in r["turns"]:
                if t["role"] == "assistant" and "text" in t:
                    if want_zh:
                        z = to_zh(t["text"])
                        if z is None:
                            misses.append(t["text"])
                        else:
                            t["text"] = z
                            n_conv += 1
            n_zh += want_zh
            n_en += (not want_zh)
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"dialogs={n_dialogs}  zh/mix(mirrored)={n_zh}  en(kept)={n_en}  text-turns-converted={n_conv}")
    if misses:
        print(f"ERROR: {len(misses)} unmatched reply patterns (would leak English). e.g.:", file=sys.stderr)
        for s in misses[:5]:
            print("   ", repr(s), file=sys.stderr)
        sys.exit(1)
    print("OK: 100% of zh/mix replies converted, no English leaks.")


if __name__ == "__main__":
    main()
