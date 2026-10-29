"""Style "fingerprint" of receipt lines: shares of upper-case, leading digits, units, truncation...

Used by 03_augment_receipt_style --compare-d6 to compare our noisy lines with real D6 lines in numbers,
not only by eye. Only aggregate shares are compared - no D6 text is ever copied into training data.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable

import pandas as pd

# feature -> (kind, tolerance): "pp" = percentage points, "rel" = relative difference
FEATURES: Dict[str, tuple] = {
    "all_upper %": ("pp", 10),
    "starts_digit %": ("pp", 10),
    "bracket_token %": ("pp", 4),
    "has_digit %": ("pp", 12),
    "unit %": ("pp", 10),
    "mixed_cyr_lat_in_word %": ("pp", 3),
    "digit_letter_swap %": ("pp", 3),
    "len_mean": ("rel", 0.25),
    "len_p90": ("rel", 0.30),
}


def style_stats(texts: Iterable[str]) -> Dict[str, float]:
    t = pd.Series([str(x) for x in texts])
    t = t[t.str.strip() != ""]
    if t.empty:
        return {}
    L = t.str.len()
    letters = t[t.str.contains(r"[A-Za-zА-Яа-яЁё]")]
    return {
        "n": float(len(t)),
        "all_upper %": 100 * letters.map(lambda s: s == s.upper()).mean(),
        "starts_digit %": 100 * t.str.match(r"^\d").mean(),
        "bracket_token %": 100 * t.str.contains(r"[\[\(][МТ+]+[\]\)]|^ШТ").mean(),
        "has_digit %": 100 * t.str.contains(r"\d").mean(),
        "unit %": 100 * t.str.contains(r"\d\s?(?:г|кг|мл|л|шт)\b", case=False).mean(),
        "mixed_cyr_lat_in_word %": 100 * t.str.contains(r"\b(?=\w*[A-Za-z])(?=\w*[А-Яа-я])\w+\b").mean(),
        "digit_letter_swap %": 100 * t.str.contains(r"[А-Яа-я][038][А-Яа-я]").mean(),
        "len_mean": float(L.mean()),
        "len_p90": float(L.quantile(0.9)),
        "len<=28 %": 100 * float((L <= 28).mean()),
    }


# Features that are NOT compared for the eating-out group. 47% of the real café lines in D6 start with a
# position number ("1 Сырник") because they come from one point-of-sale format. Matching that would feed the
# quirk of the test set to the model, so for the café group these two shares are shown but not judged.
SKIP_BY_GROUP: Dict[str, tuple] = {"meal": ("starts_digit %", "has_digit %")}


def compare(real: Dict[str, float], ours: Dict[str, float], skip: tuple = ()) -> pd.DataFrame:
    """One row per feature; `ok` is True / False, or "skip" for features that are shown but not judged."""
    rows = []
    for feat, (kind, tol) in FEATURES.items():
        r, o = real.get(feat, float("nan")), ours.get(feat, float("nan"))
        diff = abs(o - r) if kind == "pp" else abs(o - r) / max(r, 1e-9)
        rows.append({"feature": feat, "real_D6": round(r, 1), "ours": round(o, 1),
                     "tolerance": f"{tol}{'pp' if kind == 'pp' else ' rel'}",
                     "ok": "skip" if feat in skip else bool(diff <= tol)})
    return pd.DataFrame(rows)


def judged(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["ok"] != "skip"]
