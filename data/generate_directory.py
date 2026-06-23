#!/usr/bin/env python3
"""Generate a realistic mock zh-TW/en office directory.

Models a Taiwan office: each person has an English first name used at work plus a
Chinese surname + given name (漢字). People are referred to either as
"<English first> <Surname-pinyin>" (e.g. "Kevin Chen") or by full Chinese name
(e.g. "陳凱文"). Extensions are unique 4-digit numbers.

Deterministic (fixed seed) so the directory is stable across runs.
Usage: python3 generate_directory.py [N] > /dev/null   # writes data/directory.csv
"""
import csv
import random
import sys
from pathlib import Path

SEED = 20260623
N_DEFAULT = 200

# Common English given names actually used in Taiwan offices.
ENGLISH_FIRST = [
    "Kevin", "Jason", "Vincent", "Eric", "Allen", "Jacky", "Sam", "Leo", "Ray",
    "Andy", "Roger", "Steve", "Frank", "Albert", "Brian", "Howard", "Wesley",
    "Jerry", "Tony", "Ivan", "Oscar", "Victor", "Henry", "Gary", "Peter", "Sean",
    "Amy", "Vivian", "Cindy", "Grace", "Joyce", "Sharon", "Tina", "Wendy", "Iris",
    "Emily", "Stella", "Fiona", "Karen", "Rita", "Nina", "Daisy", "Coco", "Ruby",
    "Helen", "Angel", "Yuki", "Annie", "Carol", "Mia",
]

# Top Taiwanese surnames: (pinyin, hanzi). Pinyin spelling reflects common romanized
# forms seen on Taiwan business cards (Wade-Giles-ish), which is what callers/ASR hit.
SURNAMES = [
    ("Chen", "陳"), ("Lin", "林"), ("Huang", "黃"), ("Chang", "張"), ("Lee", "李"),
    ("Wang", "王"), ("Wu", "吳"), ("Liu", "劉"), ("Tsai", "蔡"), ("Yang", "楊"),
    ("Hsu", "許"), ("Cheng", "鄭"), ("Hsieh", "謝"), ("Kuo", "郭"), ("Hung", "洪"),
    ("Chiu", "邱"), ("Tseng", "曾"), ("Liao", "廖"), ("Lai", "賴"), ("Hsu2", "徐"),
    ("Chou", "周"), ("Yeh", "葉"), ("Su", "蘇"), ("Chiang", "江"), ("Lu", "呂"),
    ("Chung", "鍾"), ("Tu", "杜"), ("Fu", "傅"), ("Tsao", "曹"), ("Peng", "彭"),
]
# normalize the disambiguation hack above
SURNAMES = [(p.rstrip("2"), h) for p, h in SURNAMES]

GIVEN_HANZI = [
    "凱文", "建宏", "志明", "家豪", "俊傑", "冠宇", "雅婷", "怡君", "欣怡", "美玲",
    "淑芬", "宗翰", "承恩", "詠晴", "品妍", "柏翰", "彥廷", "宥廷", "思妤", "子涵",
    "宜蓁", "婉婷", "佳穎", "建良", "文傑", "信宏", "志豪", "孟儒", "郁婷", "庭瑋",
]

DEPARTMENTS = [
    "Sales", "Marketing", "Finance", "HR", "Engineering", "R&D", "IT",
    "Procurement", "Logistics", "Legal", "Customer Service", "Admin",
    "Quality", "Production", "Accounting",
]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    rng = random.Random(SEED)

    extensions = rng.sample(range(1000, 9999), n)
    rows = []
    seen_en = set()
    seen_zh = set()
    while len(rows) < n:
        first = rng.choice(ENGLISH_FIRST)
        pinyin, surname_hz = rng.choice(SURNAMES)
        given_hz = rng.choice(GIVEN_HANZI)
        display_en = f"{first} {pinyin}"          # "Kevin Chen"
        full_zh = f"{surname_hz}{given_hz}"        # "陳凱文"
        # Both reference forms must be globally unique, else a spoken name maps to
        # two extensions and the training labels contradict each other.
        if display_en in seen_en or full_zh in seen_zh:
            continue
        seen_en.add(display_en)
        seen_zh.add(full_zh)
        rows.append({
            "ext": extensions[len(rows)],
            "english_first": first,
            "surname_pinyin": pinyin,
            "surname_hanzi": surname_hz,
            "given_hanzi": given_hz,
            "full_chinese": full_zh,
            "display_en": display_en,
            "department": rng.choice(DEPARTMENTS),
        })

    out = Path(__file__).parent / "directory.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} contacts -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
