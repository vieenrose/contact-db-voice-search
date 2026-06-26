#!/usr/bin/env python3
"""Scaled directory generator for the 200 -> 2000 -> 20000 collision challenge.

Unlike generate_directory.py (which enforces globally-unique spoken names — fine at
N=200), this models what happens as the directory outgrows the *spoken-name namespace*.
The identifier "<English-first> <Surname-pinyin>" lives in a small space — here
|ENGLISH_FIRST| x |SURNAMES| ~ 15k combos — so as N grows, collisions appear NATURALLY:
two+ people who sound identical and differ only by department / extension.

We do NOT inject collisions artificially: surnames are drawn Zipfian (Chen/Lin/Wang
dominate, as in Taiwan), names are otherwise uniform, departments are random, and the
emergent collision structure is measured and printed. Extensions stay globally unique
(they are the key the attendant returns). Romanization collisions are intentional and
real (Hsu = 許 and 徐; Yu = 游 and 余; Lu = 呂/盧/陸; ...), adding cross-script ambiguity.

Writes data/directory_<N>.csv. Never overwrites data/directory.csv (the frozen 200-row
production directory the trained model depends on).

Usage: python3 generate_directory_scaled.py 2000
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
import random

SEED = 20260626

# ~150 English given names used in Taiwan offices (uniform draw).
ENGLISH_FIRST = [
    "Kevin", "Jason", "Vincent", "Eric", "Allen", "Jacky", "Sam", "Leo", "Ray", "Andy",
    "Roger", "Steve", "Frank", "Albert", "Brian", "Howard", "Wesley", "Jerry", "Tony",
    "Ivan", "Oscar", "Victor", "Henry", "Gary", "Peter", "Sean", "David", "Michael",
    "James", "John", "Robert", "William", "Richard", "Charles", "Thomas", "Daniel",
    "Matthew", "Anthony", "Mark", "Paul", "George", "Kenneth", "Edward", "Jeffrey",
    "Ryan", "Jacob", "Nicholas", "Jonathan", "Justin", "Adam", "Aaron", "Nathan",
    "Patrick", "Benjamin", "Samuel", "Jack", "Dennis", "Jordan", "Alan", "Bruce",
    "Derek", "Elvis", "Felix", "Gordon", "Harry", "Isaac", "Jimmy", "Johnny", "Kenny",
    "Lawrence", "Marco", "Morris", "Nelson", "Owen", "Phil", "Randy", "Simon", "Terry",
    "Wallace", "Willie", "Zack",
    "Amy", "Vivian", "Cindy", "Grace", "Joyce", "Sharon", "Tina", "Wendy", "Iris",
    "Emily", "Stella", "Fiona", "Karen", "Rita", "Nina", "Daisy", "Coco", "Ruby",
    "Helen", "Angel", "Yuki", "Annie", "Carol", "Mia", "Bella", "Catherine",
    "Christine", "Claire", "Connie", "Debby", "Doris", "Elsa", "Emma", "Evelyn", "Gina",
    "Hannah", "Ivy", "Jane", "Jenny", "Jessica", "Judy", "Julia", "June", "Kelly",
    "Linda", "Lisa", "Lucy", "Maggie", "Mandy", "Maria", "Michelle", "Molly", "Nancy",
    "Olivia", "Pearl", "Penny", "Phoebe", "Rachel", "Rebecca", "Sandy", "Sarah",
    "Selina", "Shirley", "Sophia", "Sophie", "Susan", "Tiffany", "Vicky", "Wanda",
    "Yvonne", "Zoe",
]

# ~75 Taiwan surnames (pinyin business-card form, hanzi), roughly frequency-ordered.
# Intentional romanization collisions across different hanzi are kept (they are real).
SURNAMES = [
    ("Chen", "陳"), ("Lin", "林"), ("Huang", "黃"), ("Chang", "張"), ("Lee", "李"),
    ("Wang", "王"), ("Wu", "吳"), ("Liu", "劉"), ("Tsai", "蔡"), ("Yang", "楊"),
    ("Hsu", "許"), ("Cheng", "鄭"), ("Hsieh", "謝"), ("Kuo", "郭"), ("Hung", "洪"),
    ("Chiu", "邱"), ("Tseng", "曾"), ("Liao", "廖"), ("Lai", "賴"), ("Hsu", "徐"),
    ("Chou", "周"), ("Yeh", "葉"), ("Su", "蘇"), ("Chiang", "江"), ("Lu", "呂"),
    ("Chung", "鍾"), ("Tu", "杜"), ("Fu", "傅"), ("Tsao", "曹"), ("Peng", "彭"),
    ("Yu", "游"), ("Chan", "詹"), ("Hu", "胡"), ("Shih", "施"), ("Shen", "沈"),
    ("Yu", "余"), ("Chao", "趙"), ("Lu", "盧"), ("Liang", "梁"), ("Yen", "顏"),
    ("Ko", "柯"), ("Sun", "孫"), ("Wei", "魏"), ("Weng", "翁"), ("Tai", "戴"),
    ("Fan", "范"), ("Fang", "方"), ("Sung", "宋"), ("Teng", "鄧"), ("Hou", "侯"),
    ("Wen", "溫"), ("Hsueh", "薛"), ("Ting", "丁"), ("Cho", "卓"), ("Ma", "馬"),
    ("Tung", "董"), ("Tang", "唐"), ("Lan", "藍"), ("Shih", "石"), ("Chiang", "蔣"),
    ("Chi", "紀"), ("Chien", "簡"), ("Pan", "潘"), ("Chu", "朱"), ("Hsiao", "蕭"),
    ("Kao", "高"), ("Lo", "羅"), ("Ho", "何"), ("Chuang", "莊"), ("Chien", "錢"),
    ("Lu", "陸"), ("Kung", "孔"), ("Mao", "毛"), ("Pai", "白"), ("Tai", "邰"),
]

# ~95 common given names (漢字). Some near-homophone pairs (志偉/志瑋, 雅雯/雅文) are kept.
GIVEN_HANZI = [
    "凱文", "建宏", "志明", "家豪", "俊傑", "冠宇", "雅婷", "怡君", "欣怡", "美玲",
    "淑芬", "宗翰", "承恩", "詠晴", "品妍", "柏翰", "彥廷", "宥廷", "思妤", "子涵",
    "宜蓁", "婉婷", "佳穎", "建良", "文傑", "信宏", "志豪", "孟儒", "郁婷", "庭瑋",
    "志偉", "志瑋", "俊宏", "國華", "文彬", "建中", "世昌", "宏偉", "嘉文", "智凱",
    "偉倫", "冠廷", "柏宇", "宇軒", "子瑜", "佳蓉", "心怡", "雅雯", "雅文", "淑惠",
    "麗華", "玉芬", "秀英", "美惠", "雅琪", "怡如", "詩涵", "家瑜", "思齊", "育誠",
    "政翰", "哲瑋", "俊賢", "建勳", "明哲", "志成", "永昌", "文山", "大為", "國棟",
    "振宇", "偉哲", "冠霖", "彥宏", "宛真", "佩珊", "雅惠", "淑娟", "鈺婷", "曉萍",
    "慧君", "嘉玲", "麗君", "淑貞", "美如", "雅芳", "怡靜", "子晴", "欣妤", "詩婷",
    "文君", "惠雯", "佳燕", "玉婷", "靜宜",
]

DEPARTMENTS = [
    "Sales", "Marketing", "Finance", "HR", "Engineering", "R&D", "IT",
    "Procurement", "Logistics", "Legal", "Customer Service", "Admin",
    "Quality", "Production", "Accounting",
]


def generate(n: int, seed: int = SEED):
    rng = random.Random(seed)
    # Zipfian surname weights (s=1): rank-1 ~ Chen is most common.
    weights = [1.0 / (i + 1) for i in range(len(SURNAMES))]
    # Unique extensions: width scales so capacity >> n.
    lo, hi = (1000, 9999) if n <= 8000 else (10000, 99999)
    extensions = rng.sample(range(lo, hi + 1), n)

    rows = []
    for i in range(n):
        first = rng.choice(ENGLISH_FIRST)
        pinyin, surname_hz = rng.choices(SURNAMES, weights=weights, k=1)[0]
        given_hz = rng.choice(GIVEN_HANZI)
        rows.append({
            "ext": extensions[i],
            "english_first": first,
            "surname_pinyin": pinyin,
            "surname_hanzi": surname_hz,
            "given_hanzi": given_hz,
            "full_chinese": f"{surname_hz}{given_hz}",
            "display_en": f"{first} {pinyin}",
            "department": rng.choice(DEPARTMENTS),
        })
    return rows


def collision_report(rows):
    by_en = defaultdict(list)
    by_zh = defaultdict(list)
    for r in rows:
        by_en[r["display_en"]].append(r)
        by_zh[r["full_chinese"]].append(r)

    def stats(buckets, label):
        clusters = {k: v for k, v in buckets.items() if len(v) > 1}
        involved = sum(len(v) for v in clusters.values())
        sizes = Counter(len(v) for v in clusters.values())
        # how many collision clusters are resolvable by department alone?
        dept_ok = sum(1 for v in clusters.values()
                      if len({r["department"] for r in v}) == len(v))
        biggest = max((len(v) for v in clusters.values()), default=1)
        print(f"  [{label}] {len(buckets)} distinct / {len(rows)} rows | "
              f"collision clusters: {len(clusters)} | people in a collision: "
              f"{involved} ({involved/len(rows)*100:.1f}%) | biggest cluster: {biggest}")
        print(f"           cluster sizes: {dict(sorted(sizes.items()))} | "
              f"resolvable by department alone: {dept_ok}/{len(clusters)} "
              f"({(dept_ok/len(clusters)*100) if clusters else 0:.0f}%)")

    print(f"namespace ceiling: {len(ENGLISH_FIRST)} firsts x {len(SURNAMES)} surnames "
          f"= {len(ENGLISH_FIRST)*len(SURNAMES)} spoken-name combos")
    stats(by_en, "EN  display_en")
    stats(by_zh, "ZH  full_chinese")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rows = generate(n)
    out = Path(__file__).parent / f"directory_{n}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} contacts -> {out}")
    collision_report(rows)


if __name__ == "__main__":
    main()
