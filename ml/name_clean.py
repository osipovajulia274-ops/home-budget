"""Cleaning and sanity filters for product names that come from Open*Facts.

Why this exists (found in the first real Colab run, see PROGRESS.md):
  * names contained HTML entities (`&quot;`, `&QUOT;`) that never appear on a real receipt;
  * the "Russian name" test (>= 3 Cyrillic letters) also accepted Bulgarian and Ukrainian names;
  * the tag->subcategory map put eggs under alcohol, kefir under drinks, yoghurt under sweets and
    buckwheat under pet food.

Everything here is pure Python, deterministic and unit-tested (tests/test_name_clean.py).
The rules that move or drop rows live in data/name_guards.yaml so they can be reviewed and edited
without touching code. Every action is counted and sampled into ml/reports/off_guard_report.json.
"""
from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
GUARDS_PATH = ROOT / "data" / "name_guards.yaml"

CYR_RE = re.compile(r"[А-Яа-яЁё]")
# letters that exist in Ukrainian / Belarusian / Serbian / Macedonian / Kazakh / old Russian orthography but not
# in modern Russian names
FOREIGN_CYR_RE = re.compile(r"[іїєґІЇЄҐўЎјљњћђџЈЉЊЋЂЏѓќѕЃЌЅѣѝѫѢәөүұқғңһӘӨҮҰҚҒҢҺ]")
# hard sign that is NOT followed by е ё ю я (объём, съёмный are Russian; Пълно, хлъб are Bulgarian)
BG_HARD_SIGN_RE = re.compile(r"ъ(?![еёюя])")
BG_WORDS = frozenset(
    "със към които като още мляко хляб сирене бира месо ориз зеле гъби сладолед краве козе овче прясно "
    "прясна кисело пълнозърнест пълнозърнеста ягоди ядки кленов пуешки пуешко хапки домати краставици картофи "
    "захар бонбони бисквити кашкавал наденица луканка пилешко свинско телешко оцет олио брашно сладко "
    "шоколадов какаов доматен зеленчуци плодове".split()
)
# Bulgarian / Serbian masculine adjectives end in -ски ("цейлонски чай"); Russian has -ский, except adverbs
# "по-деревенски", which are excluded by the look-behind
BG_ADJ_RE = re.compile(r"(?<!по-)(?<![а-яё])[а-яё]{5,}ски(?![а-яё])")
BG_PHRASE_RE = re.compile(r"(?<![а-яё])вкус на(?![а-яё])")
# Ukrainian function words and adjective endings that never occur in Russian ("суп з телятиною", "рисовий",
# "гречаний"); the letters і ї є ґ alone miss names written without them
UK_TOKENS = frozenset("з зі із та від або чи що як сир сиру кава кави мелена мелений яйця олія риба пшоно "
                      "кукурудза морозиво картопля морква цибуля часник яблука варення печиво".split())
UK_STEMS = ("ковбас", "цукор", "борошн", "кукуруд", "морозив", "картопл", "солодощ", "згущ")
UK_ENDING_RE = re.compile(r"(?:овий|евий|аний|ський|цький|ичний|очний|ений|ований)\b")
# language tags of the picked Open*Facts name that are certainly not Russian
FOREIGN_LANGS = frozenset("bg uk sr mk be bs hr sh sl mn kk ky".split())
# Countries whose products carry Cyrillic names that are NOT Russian. The first real run showed that
# requiring a Russian-speaking country wrongly dropped 5 286 plainly Russian names (imported goods are
# filed under other countries), so the country test is now a BLOCK list, used only as a last resort.
BLOCK_COUNTRIES = frozenset("en:bulgaria en:serbia en:north-macedonia en:montenegro en:bosnia-and-herzegovina "
                            "en:mongolia".split())
# countries whose products may reasonably carry a Russian-language name
RU_COUNTRIES = frozenset(
    "en:russia en:belarus en:kazakhstan en:ukraine en:kyrgyzstan en:uzbekistan en:armenia en:georgia "
    "en:azerbaijan en:moldova en:latvia en:estonia en:lithuania en:tajikistan en:turkmenistan".split()
)


def clean_name(text: object) -> str:
    """Unescape HTML entities (twice: `&amp;quot;`), drop control chars, collapse spaces."""
    if text is None:
        return ""
    s = str(text)
    for _ in range(2):
        un = html.unescape(s)
        if un == s:
            break
        s = un
    # entities written in capitals survive a plain unescape only if lowercase ones exist, handle the rest
    s = re.sub(r"&QUOT;?", '"', s, flags=re.IGNORECASE)
    s = re.sub(r"&AMP;?", "&", s, flags=re.IGNORECASE)
    s = s.replace(" ", " ")
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> List[str]:
    return re.findall(r"[а-яёіїєґ]+", s.lower())


def lexicon_tokens(name: str) -> List[str]:
    """Words of >= 4 letters, lower case, ё -> е: the units of the learned foreign vocabulary.
    Ukrainian letters are kept inside the word ("арахіс" is one word, not "арах" + "с")."""
    return re.findall(r"[а-яіїєґ]{4,}", name.lower().replace("ё", "е"))


def reject_reason(name: str, countries: Optional[str] = None, min_cyr: int = 3, max_len: int = 100,
                  lang: Optional[str] = None, lexicon: Optional[Iterable[str]] = None) -> Optional[str]:
    """None if the name looks like a Russian product name, otherwise a short reason code.

    `lang` is the language tag of the name entry that was picked in the Open*Facts dump, `lexicon` the
    foreign vocabulary learned from the data itself (learn_foreign_lexicon).
    """
    if not name or len(name) < 3:
        return "short"
    if len(name) > max_len:
        return "too_long"
    if len(CYR_RE.findall(name)) < min_cyr:
        return "few_cyrillic"
    if lang and str(lang).strip().lower() in FOREIGN_LANGS:
        return "language_tag"
    if FOREIGN_CYR_RE.search(name):
        return "ukrainian_or_other"
    low = name.lower()
    if BG_HARD_SIGN_RE.search(low):
        return "bulgarian"
    toks = _tokens(name)
    if any(t in BG_WORDS for t in toks) or BG_ADJ_RE.search(low) or BG_PHRASE_RE.search(low):
        return "bulgarian"
    if (any(t in UK_TOKENS or t.startswith(UK_STEMS) for t in toks) or UK_ENDING_RE.search(low)):
        return "ukrainian_or_other"
    if lexicon and any(t in lexicon for t in lexicon_tokens(name)):
        return "foreign_lexicon"
    if countries:
        tags = _country_tags(countries)
        if tags & BLOCK_COUNTRIES and not (tags & RU_COUNTRIES):
            return "country"
    return None


def _country_tags(countries: object) -> set:
    return {c.strip().lower() for c in str(countries or "").split(",") if c.strip()}


_STATIC_FOREIGN = {"bulgarian", "ukrainian_or_other", "language_tag", "country"}


def learn_foreign_lexicon(rows: Iterable[Tuple[str, Optional[str], Optional[str]]], min_foreign: int = 5,
                          max_ru_ratio: float = 0.0) -> Dict[str, Tuple[int, int]]:
    """Words that occur (almost) only in names that are certainly NOT Russian.

    The hand-written word lists above cannot know every Bulgarian or Ukrainian word ("ягоди", "сир", "кава").
    The dump itself tells us: a name is certainly foreign if a static rule rejects it (letters і ї ъ,
    Bulgarian-only country, ...), and certainly Russian if its only country is Russia and no rule rejects it.
    A word seen in >= `min_foreign` foreign names and in at most `max_ru_ratio` of that many Russian ones is
    treated as foreign vocabulary. The default is zero Russian names: even 1-3 purely Russian names mean a
    word is also Russian ("чипс", "салата", "кафе", "хумус", "печени" slipped through at 10% in the fourth run). Nothing is downloaded or pretrained: the counts come from the same extract.

    rows: (name, countries, lang). Returns {word: (n_foreign_names, n_russian_names)} for the flagged words.
    """
    foreign: Counter = Counter()
    russian: Counter = Counter()
    for name, countries, lang in rows:
        name = clean_name(name)
        why = reject_reason(name, countries, lang=lang)
        words = set(lexicon_tokens(name))
        if why in _STATIC_FOREIGN:
            foreign.update(words)
        elif why is None and _country_tags(countries) == {"en:russia"}:
            russian.update(words)
    return {w: (n, russian.get(w, 0)) for w, n in foreign.items()
            if n >= min_foreign and russian.get(w, 0) <= max_ru_ratio * n}


# --------------------------------------------------------------------------- guards

@lru_cache(maxsize=1)
def load_guards(path: str = str(GUARDS_PATH)) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for rule in raw.get("reassign", []) + raw.get("drop_if", []):
        rule["_pat"] = _compile(rule.get("stems", []), rule.get("words", []))
        rule["_unless"] = _compile(rule.get("unless_stems", []), rule.get("unless_words", []))
        if rule["_pat"] is None:
            raise ValueError(f"guard rule {rule.get('id')!r} has no stems/words")
    return raw


_NOT_CYR = r"(?<![а-яёa-z])"


def _compile(stems: Sequence[str], words: Sequence[str]) -> Optional["re.Pattern[str]"]:
    """`stems` match at the start of a word, `words` only as a whole word (case-insensitive)."""
    parts = [re.escape(x.lower()) for x in stems]
    parts += [re.escape(x.lower()) + r"(?![а-яёa-z])" for x in words]
    if not parts:
        return None
    return re.compile(_NOT_CYR + "(?:" + "|".join(parts) + ")", re.IGNORECASE)


def _from_matches(sub: str, patterns: Sequence[str]) -> bool:
    return any(p == sub or (p.endswith(".*") and sub.startswith(p[:-1])) for p in patterns)


def _apply_once(name: str, sub: str, guards: dict) -> Tuple[Optional[str], Optional[str]]:
    low = name.lower().replace("ё", "е")
    for rule in guards.get("drop_if", []):
        if not _from_matches(sub, rule["subcategories"]):
            continue
        if rule["_unless"] is not None and rule["_unless"].search(low):
            continue
        if rule["_pat"].search(low):
            return None, rule["id"]
    for rule in guards.get("reassign", []):
        if sub == rule["to"] or not _from_matches(sub, rule["from"]):
            continue
        if rule["_unless"] is not None and rule["_unless"].search(low):
            continue
        if rule["_pat"].search(low):
            return rule["to"], rule["id"]
    return sub, None


def apply_guards(name: str, sub: str, guards: Optional[dict] = None,
                 max_steps: int = 6) -> Tuple[Optional[str], Optional[str]]:
    """Return (new_subcategory_or_None_if_dropped, id_of_the_first_rule_that_fired).

    Rules are applied until the label stops changing (first matching rule wins within a step), so the
    result is a fixed point: applying the guards to their own output changes nothing. That is what the
    Stage 1 check "name guards find nothing to fix" relies on. A cycle (A -> B -> A) would mean two rules
    contradict each other; it is cut at the first repeated label and tests/test_name_clean.py makes sure
    the shipped rules have none.
    """
    guards = guards or load_guards()
    cur, first_id = sub, None
    visited = [sub]
    for _ in range(max_steps):
        new, rule_id = _apply_once(name, cur, guards)
        if new is None:
            return None, rule_id
        if rule_id is None or new == cur or new in visited:
            break
        first_id = first_id or rule_id
        visited.append(new)
        cur = new
    return cur, first_id


class GuardReport:
    """Counts what the cleaning did and keeps a few examples of each action for human review."""

    def __init__(self, max_examples: int = 6):
        self.counts: Counter = Counter()
        self.examples: Dict[str, List[str]] = defaultdict(list)
        self.max_examples = max_examples

    def add(self, key: str, example: str) -> None:
        self.counts[key] += 1
        if len(self.examples[key]) < self.max_examples:
            self.examples[key].append(example)

    def as_dict(self) -> dict:
        return {k: {"count": n, "examples": self.examples[k]} for k, n in sorted(self.counts.items())}


def process_rows(rows: Iterable[tuple], report: Optional[GuardReport] = None,
                 lexicon: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
    """rows: (raw_name, subcategory, countries, source_label[, lang]) -> kept (clean_name, subcategory)."""
    report = report or GuardReport()
    guards = load_guards()
    lexicon = frozenset(lexicon or ())
    out: List[Tuple[str, str]] = []
    for row in rows:
        raw, sub, countries, label = row[:4]
        lang = row[4] if len(row) > 4 else None
        name = clean_name(raw)
        if name != str(raw).strip():
            report.counts["html_or_whitespace_cleaned"] += 1
        why = reject_reason(name, countries, lang=lang, lexicon=lexicon)
        if why:
            report.add(f"{label}:reject:{why}", name)
            continue
        new_sub, rule_id = apply_guards(name, sub, guards)
        if new_sub is None:
            report.add(f"{label}:drop:{rule_id}", f"{name}  [{sub}]")
            continue
        if rule_id:
            report.add(f"{label}:reassign:{rule_id}", f"{name}  [{sub} -> {new_sub}]")
        out.append((name, new_sub))
    return out
