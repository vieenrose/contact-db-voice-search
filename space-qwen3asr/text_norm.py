"""Phone-attendant text normalizer (zh-TW + English) — the single source of truth for how entities are
read, used BOTH to prep teacher text (so VoxCPM2 audio matches) and at inference (frontend). Handles:
email, phone/extension, serial number, price, percent, temperature(°C), date, person-count, address —
digit-by-digit vs cardinal vs ordinal chosen by entity + language context. Plain numbers are left for
cn2an (zh) / g2p_en (en) downstream. Idempotent-ish: safe to run once on raw text."""
import re
try:
    import inflect; _P = inflect.engine()
except Exception:
    _P = None
import cn2an

_ZH = {"0":"零","1":"一","2":"二","3":"三","4":"四","5":"五","6":"六","7":"七","8":"八","9":"九"}
_EN = {"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine"}
_ADDR = "號樓段巷弄室坪"
_zh = lambda c: '一' <= c <= '鿿'

def _en_ctx(text, s, e):
    L = next((c for c in reversed(text[:s]) if _zh(c) or re.match(r'[A-Za-z]', c)), None)
    R = next((c for c in text[e:] if _zh(c) or re.match(r'[A-Za-z]', c)), None)
    for c in (L, R):
        if c is None: continue
        if re.match(r'[A-Za-z]', c): return True
        if _zh(c): return False
    return False

def _dd(d, en): return (" ".join(_EN[c] for c in d)) if en else ("".join(_ZH[c] for c in d))
def _card_zh(d):
    try: return cn2an.an2cn(int(d), "low")
    except Exception: return _dd(d, False)
def _card_en(n):
    return _P.number_to_words(int(n), andword="").replace("-", " ").replace(",", "") if _P else _dd(str(n), True)
def _ord_en(n):
    return _P.number_to_words(_P.ordinal(int(n))).replace("-", " ") if _P else str(n)
def _ord_zh(n): return _card_zh(n)

_MONTH = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",
          9:"September",10:"October",11:"November",12:"December"}
def _year_en(y):
    y = int(y)
    if 2000 <= y <= 2009: return "two thousand" + ("" if y == 2000 else " " + _card_en(y % 100))
    if 1000 <= y <= 2099:
        return f"{_card_en(y//100)} {('oh ' + _card_en(y%100)) if 0 < y%100 < 10 else (_card_en(y%100) if y%100 else 'hundred')}"
    return _card_en(y)

def normalize(text: str) -> str:
    if not text: return text
    # 1) EMAIL  -> spell (at / dot, en digits)
    def email(m):
        s = m.group(0).replace("@", " at ").replace(".", " dot ")
        s = re.sub(r"\d", lambda d: " " + _EN[d.group(0)] + " ", s)
        return " " + re.sub(r"\s+", " ", s).strip() + " "
    text = re.sub(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", email, text)

    # 1.5) TIME of day  9:30 / 3:15 PM / 14:30 (needs :MM so ratios like 3:2 are untouched)
    def time_repl(m):
        h, mm, ap = int(m.group(1)), m.group(2), m.group(3)
        mi = int(mm); en = _en_ctx(text, m.start(), m.end())
        if en:
            hh = _card_en(h)
            t = f"{hh} o'clock" if mi == 0 else (f"{hh} oh {_card_en(mm)}" if mi < 10 else f"{hh} {_card_en(mm)}")
            if ap: t += " " + ("a m" if 'a' in ap.lower() else "p m")
            return " " + t + " "
        pre = ("上午" if 'a' in ap.lower() else "下午") if ap else ""
        t = f"{pre}{_card_zh(str(h))}點" + ("整" if mi == 0 else ("半" if mi == 30 else _card_zh(mm) + "分"))
        return " " + t + " "
    text = re.sub(r"(?<![\d.])(\d{1,2}):(\d{2})(?::\d{2})?\s*([AaPp][.]?[Mm][.]?)?(?![\d.])", time_repl, text)

    # 2) DATE  zh 2024年3月15日 / en mixes
    def date_zh(m):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        out = _dd(y, False) + "年"
        if mo: out += _card_zh(mo) + "月"
        if d: out += _card_zh(d) + "日"
        return out
    text = re.sub(r"(\d{4})\s*年\s*(?:(\d{1,2})\s*月)?\s*(?:(\d{1,2})\s*[日號])?", date_zh, text)
    def month_day(m):  # en "March 15" / "March 15, 2024"
        mon, d, y = m.group(1), m.group(2), m.group(3)
        out = f"{mon} {_ord_en(d)}"
        if y: out += " " + _year_en(y)
        return out
    text = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b", month_day, text)

    # 3) TEMPERATURE  28°C / 攝氏28度 / -5°C
    def temp(m):
        sign, n = m.group(1), m.group(2)
        en = _en_ctx(text, m.start(), m.end())
        neg = sign in ("-", "－")
        if en:
            return f" {'minus ' if neg else ''}{_card_en(n)} degrees Celsius "
        return f" 攝氏{'零下' if neg else ''}{_card_zh(n)}度 "
    text = re.sub(r"(?:攝氏)?\s*([-－]?)(\d+)\s*(?:°\s*[Cc]|度[Cc]?|℃|°|\s*degrees?(?:\s+[Cc]elsius)?)", temp, text)

    # 4) PERCENT  70%
    def pct(m):
        n = m.group(1)
        return f" {_card_en(n)} percent " if _en_ctx(text, m.start(), m.end()) else f" 百分之{_card_zh(n)} "
    text = re.sub(r"(\d+)\s*[%％]", pct, text)

    # 5) PRICE  $1,299 / NT$500 / 1299元 / USD 49.99  (context-aware; USD/US$ force English)
    def price_cur(m):
        cur = m.group(0); whole = m.group(1).replace(",", ""); cents = m.group(2)
        en = _en_ctx(text, m.start(), m.end()) or ("USD" in cur) or ("US$" in cur)
        if en:
            out = _card_en(whole) + " dollar" + ("" if whole == "1" else "s")
            if cents: out += " and " + _card_en(cents) + " cent" + ("" if cents == "01" else "s")
        else:
            out = _card_zh(whole) + "元"
            if cents: out += _card_zh(cents) + "分"
        return " " + out + " "
    text = re.sub(r"(?:NT\$|US\$|USD|\$)\s*([\d,]+)(?:\.(\d{2}))?", price_cur, text)
    def price_zh(m):
        return " " + _card_zh(m.group(1).replace(",", "")) + (m.group(2) or "元") + " "
    text = re.sub(r"([\d,]+)\s*(元|塊錢|塊|台幣|新台幣)", price_zh, text)

    # 6) PHONE / extension groups -> digit-by-digit
    def phone(m):
        en = _en_ctx(text, m.start(), m.end())
        groups = [re.sub(r"\D", "", g) for g in re.split(r"[-\s]+", m.group(0).strip("()"))]
        return " " + " ".join(_dd(g, en) for g in groups if g) + " "
    text = re.sub(r"\(?\d{2,4}\)?(?:[-\s]\d{2,4}){1,4}", phone, text)
    text = re.sub(r"(分機|內線|ext\.?|extension)\s*(\d{2,6})",
                  lambda m: m.group(1) + " " + _dd(m.group(2), _en_ctx(text, m.start(), m.end())), text, flags=re.I)

    # 7) SERIAL / order code  (序號/型號/SN context OR alnum code with both letters+digits)
    def serial(m):
        s = m.group(0)
        return " " + " ".join((_EN[c] if c.isdigit() else c.upper()) for c in s if c.isalnum()) + " "
    text = re.sub(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]{2,}(?:-[A-Za-z0-9]+)*(?![A-Za-z0-9])", serial, text)

    # 7.5) DECIMALS  12.5 -> 十二點五 / twelve point five (avoid IPs/versions a.b.c)
    def dec_repl(m):
        whole, frac = m.group(1), m.group(2)
        if _en_ctx(text, m.start(), m.end()):
            return f" {_card_en(whole)} point {' '.join(_EN[c] for c in frac)} "
        return f" {_card_zh(whole)}點{''.join(_ZH[c] for c in frac)} "
    text = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", dec_repl, text)

    # 8) COUNTS / remaining standalone digit runs
    def num(m):
        d = m.group(0); en = _en_ctx(text, m.start(), m.end()); after = text[m.end():m.end()+1]
        if len(d) >= 5:                       # long id/order -> digit-by-digit
            return _dd(d, en)
        if en:
            return _dd(d, True) if len(d) >= 4 else d  # 4-digit en standalone (ext) digits; else let g2p_en handle
        return _dd(d, False) if (len(d) >= 4 and after not in _ADDR) else _card_zh(d)
    text = re.sub(r"\d+", num, text)

    text = text.replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)        # no space before punctuation
    return text.strip()
