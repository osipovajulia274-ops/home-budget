"""Turn a clean product name into a "cash-register" style receipt line.

Rules live in data/abbreviations.yaml. Everything is driven by an explicit random.Random so the
dataset is reproducible (seed 42).
"""
from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "data" / "abbreviations.yaml"


@lru_cache(maxsize=1)
def load_rules(path: str = str(RULES_PATH)) -> dict:
    with open(path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    # longest phrase first, so "высший сорт" wins over "сорт"
    rules["_abbr_order"] = sorted(rules["word_abbr"], key=len, reverse=True)
    return rules


def _match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def abbreviate(text: str, rng: random.Random, rules: dict, p_word: float = 0.7) -> str:
    for phrase in rules["_abbr_order"]:
        pattern = re.compile(r"(?<![а-яёa-z])" + re.escape(phrase) + r"(?![а-яёa-z])", re.IGNORECASE)
        if not pattern.search(text):
            continue
        if rng.random() > p_word:
            continue
        options = rules["word_abbr"][phrase]
        text = pattern.sub(lambda m: _match_case(m.group(0), rng.choice(options)), text, count=1)
    return text


def _fill_suffix(template: str, rng: random.Random, rules: dict) -> str:
    pools = rules["suffixes"]["numbers"]
    out = template
    for key, pool in pools.items():
        tag = "{" + key + "}"
        while tag in out:
            out = out.replace(tag, str(rng.choice(pool)), 1)
    while "{d}" in out:
        out = out.replace("{d}", str(rng.randint(1, 9)), 1)
    return out


def unit_kind_for(subcategory: Optional[str], rules: Optional[dict] = None) -> str:
    rules = rules or load_rules()
    return rules.get("unit_by_subcategory", {}).get(subcategory or "", "none")


def add_suffix(text: str, rng: random.Random, rules: dict, kind: str = "any") -> str:
    """Append a unit that fits `kind` (mass | volume | count | any); 'none' leaves the text alone."""
    if kind == "none":
        return text
    if kind == "any":
        w = rules["suffixes"]["any_weights"]
        kind = rng.choices(list(w), weights=list(w.values()))[0]
    return f"{text} {_fill_suffix(rng.choice(rules['suffixes']['kinds'][kind]), rng, rules)}"


def add_prefix(text: str, rng: random.Random, rules: dict, p_article: Optional[float] = None,
               p_position: Optional[float] = None, p_token: Optional[float] = None) -> Tuple[str, bool]:
    """Prepend what cash registers print before the name: an article code ("8361 "), a position number
    ("1 ") and/or a marker token ("[М+]"). Returns (text, digits_added).

    The probabilities come from the noise profile (`article`, `position`), the defaults live in the
    `prefixes` section of abbreviations.yaml.
    """
    pre = rules["prefixes"]
    p_article = pre["article_digits"]["p"] if p_article is None else p_article
    p_position = pre["position_number"]["p"] if p_position is None else p_position
    p_token = pre["tokens_p"] if p_token is None else p_token
    parts: List[str] = []
    digits = False
    if rng.random() < p_position:
        parts.append(str(rng.randint(1, 9)))
        digits = True
    if rng.random() < p_article:
        lens = pre["article_digits"]["digits"]
        n = rng.randint(lens[0], lens[-1])
        if rng.random() < pre["article_digits"].get("barcode_p", 0.0):
            n = 13                                   # EAN-13 printed instead of a short article code
        parts.append("".join(rng.choice("0123456789") for _ in range(n)))
        digits = True
    if rng.random() < p_token:
        parts.append(rng.choice(pre["tokens"]))
    if not parts:
        return text, False
    # only marker tokens are ever glued to the name ("ШТ.[М+]ГУРМЭ"); an article code or position number
    # is always followed by a space
    sep = " " if digits else rng.choice([" ", " ", ""])
    return (sep.join(parts + [text]) if sep else "".join(parts) + text), digits


def add_decor(text: str, rng: random.Random, rules: dict) -> str:
    """Append a letters-only register marker ("шт.", "<СП>", "[М]"). It gives short names distinct,
    realistic variants WITHOUT random digits, so thin classes do not end up with 'digit first => this class'."""
    return f"{text} {rng.choice(rules['decor_tokens'])}"


def clip_words(text: str, rng: random.Random) -> str:
    """Cut 1-2 long words to 3-5 letters plus a dot ("Сметана" -> "Смет.", "порошок" -> "пор."), the way
    cash registers shorten names. Gives many distinct, realistic lines without random digits."""
    words = text.split(" ")
    if len(words) < 2:                      # a one-word name would lose everything that identifies it
        return text
    idx = [i for i, w in enumerate(words) if len(re.sub(r"[^А-Яа-яЁёA-Za-z]", "", w)) >= 7]
    if not idx:
        return text
    take = 1 if rng.random() < 0.7 else 2
    for i in rng.sample(idx, min(take, len(idx), len(words) - 1)):   # always keep at least one word whole
        words[i] = words[i][: rng.randint(3, 5)] + "."
    return " ".join(words)


def drop_word(text: str, rng: random.Random) -> str:
    """Drop one non-leading word. Needs >= 4 words so the discriminating head noun survives
    ("Батон на молоке" must never become "на молоке")."""
    words = text.split()
    if len(words) < 4:
        return text
    del words[rng.randrange(1, len(words))]
    out = " ".join(words)
    return out if len(out) >= 8 else text


def truncate(text: str, rng: random.Random, rules: dict, width_range) -> str:
    """Cut like a fixed-width register: D6 has spikes at 22 and 35 characters, otherwise uniform."""
    r = rng.random()
    acc = 0.0
    for width, p in rules.get("truncate_widths", []):
        acc += p
        if r < acc:
            return text[:width]
    lo, hi = width_range
    return text[: rng.randint(lo, hi)]


def bank_decor(text: str, rng: random.Random, rules: dict) -> str:
    cfg = rules["bank_style"]
    both = rng.random() < cfg["both_p"]
    use_prefix = both or rng.random() < 0.5
    use_suffix = both or not use_prefix
    if use_prefix:
        text = rng.choice(cfg["prefixes"]) + text
    if use_suffix:
        suffix = rng.choice(cfg["suffixes"])
        while "{d}" in suffix:
            suffix = suffix.replace("{d}", str(rng.randint(0, 9)), 1)
        text = text + suffix
    return text


def ocr_typo(text: str, rng: random.Random, rules: dict) -> str:
    chars = list(text)
    candidates = list(range(len(chars)))
    rng.shuffle(candidates)
    for i in candidates:
        for a, b in rules["ocr_confusions"]:
            if chars[i] == a:
                chars[i] = b
                return "".join(chars)
            if chars[i] == b:
                chars[i] = a
                return "".join(chars)
    if len(chars) > 4:  # fallback: drop one letter
        del chars[rng.randrange(1, len(chars) - 1)]
    return "".join(chars)


def latin_doubles(text: str, rng: random.Random, rules: dict, p: float = 0.35) -> str:
    table: Dict[str, str] = rules["latin_doubles"]
    return "".join(table[ch] if ch in table and rng.random() < p else ch for ch in text)


def noisy_variant_ex(clean: str, rng: random.Random, profile: str = "goods", unit_kind: str = "any",
                     boost: float = 1.0) -> Tuple[str, bool]:
    """One random 'dirty' version of `clean` for the given noise profile -> (text, high_entropy).

    `high_entropy` is True when the line got random digits (article code, position number, unit suffix) or
    a typo; those lines are almost never duplicates of each other, which is why `noisy_variants` has to
    treat them as a separate stratum (see there).

    `unit_kind` says which unit suffix (if any) is plausible for this product, see unit_by_subcategory.
    `boost` (>= 1) multiplies only the probabilities of the cosmetic changes (abbreviation, clipping, dropping
    a word, upper case, truncation, decor marker); digits and typos are never boosted. It is used to top up
    thin classes whose short names cannot give enough distinct lines at the calibrated rates.
    """
    rules = load_rules()
    cfg = rules["profiles"][profile]

    def p(key: str) -> float:                 # cosmetic change, may be boosted
        return min(0.9, cfg.get(key, 0.0) * boost)

    hi = False
    s = clean
    if rng.random() < p("abbreviate"):
        s = abbreviate(s, rng, rules)
    if rng.random() < p("clip"):
        s = clip_words(s, rng)
    if rng.random() < p("drop_word"):
        s = drop_word(s, rng)
    if (unit_kind != "none" and rng.random() < cfg.get("suffix", 0.0)
            and not re.search(r"\d\s?(г|кг|мл|л|шт)\b", s, re.IGNORECASE)):
        s = add_suffix(s, rng, rules, unit_kind)
        hi = True
    if rng.random() < p("decor"):
        s = add_decor(s, rng, rules)
    s, digits = add_prefix(s, rng, rules, cfg.get("article", 0.0), cfg.get("position", 0.0), cfg.get("token", 0.0))
    hi = hi or digits
    if rng.random() < cfg.get("bank_style", 0.0):
        s = bank_decor(s, rng, rules)
    if rng.random() < p("upper"):
        s = s.upper()
    if rng.random() < p("truncate"):
        s = truncate(s, rng, rules, cfg["truncate_range"])
    if rng.random() < cfg["typo"]:
        s = ocr_typo(s, rng, rules)
        hi = True
    if rng.random() < cfg["latin_double"]:
        s = latin_doubles(s, rng, rules)
        hi = True
    s = re.sub(r"\s+", " ", s).strip()
    s = s or clean
    hi = hi or len(re.findall(r"\d", s)) > len(re.findall(r"\d", clean))   # e.g. bank-style "*1234"
    return s, hi


def noisy_variant(clean: str, rng: random.Random, profile: str = "goods", unit_kind: str = "any",
                  boost: float = 1.0) -> str:
    return noisy_variant_ex(clean, rng, profile, unit_kind, boost)[0]


def noisy_variants(clean: str, k: int, rng: random.Random, profile: str = "goods", max_tries: int = 0,
                   unit_kind: str = "any", oversample: float = 3.0, boost: float = 1.0) -> List[str]:
    """Up to `k` distinct noisy variants (all different from `clean` and from each other), with the same
    share of 'random-digit' lines as the noise model itself produces.

    Why not simply "draw and remove duplicates": a lower-case or upper-case copy of a short name is
    produced again and again and is removed, a line with a random article code is never a duplicate. The
    survivors would be mostly digit lines, and the more variants a class needs the more so (found in the
    third Colab run: share of lines starting with a digit 20%..63% depending on the class). So the draws
    are split into two strata - lines with random digits / a typo (`hi`) and all the others (`plain`) - and
    the output keeps the stratum proportions of the DRAWS, counting the clean row as a plain line. If a
    name has few plain variants it simply gets fewer rows; more clean names, not more digits, fix that.

    `max_tries`, when given, overrides the number of draws.
    """
    draws = max_tries if max_tries else max(k, int(round(k * oversample)))
    seen = {clean}
    plain: List[str] = []
    hi_lines: List[str] = []
    n_hi = 0
    for _ in range(draws):
        v, hi = noisy_variant_ex(clean, rng, profile, unit_kind, boost)
        n_hi += hi
        if v in seen:
            continue
        seen.add(v)
        (hi_lines if hi else plain).append(v)
    share = n_hi / max(1, draws)
    for n in range(k, 0, -1):
        a = int(round(share * (n + 1)))            # random-digit lines among n noisy rows + the clean row
        b = n - a
        if 0 <= b <= len(plain) and a <= len(hi_lines):
            out = rng.sample(plain, b) + rng.sample(hi_lines, a)
            rng.shuffle(out)
            return out
    return []
