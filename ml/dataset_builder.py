"""Build the category-classification dataset (guide, sections 6.2 and 6.5).

Input : clean names with a subcategory (Open*Facts extracts + seed dictionary).
Output: DataFrame with columns
        text, text_clean, subcategory, category, source, parent_source, origin, profile, group, split

Leakage rules
  * the split is made by `group` (all fills / noisy variants of one product stay together);
  * afterwards any val/test row whose *normalised* clean name also exists in train (or test in val)
    is dropped, so no normalised clean name is shared between parts;
  * clean names that map to several subcategories are resolved by majority (>= 2/3) or dropped.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.services.normalize import normalize
from ml.receipt_noise import noisy_variants, unit_kind_for

SPLITS = ("train", "val", "test_synth")


def resolve_label_conflicts(clean: pd.DataFrame, min_share: float = 2 / 3) -> pd.DataFrame:
    """Same normalised name under different subcategories: keep majority label, else drop."""
    clean = clean.copy()
    clean["norm"] = clean["text_clean"].map(normalize)
    keep = []
    for norm, g in clean.groupby("norm", sort=False):
        counts = Counter(g["subcategory"])
        if len(counts) == 1:
            keep.append(g)
            continue
        label, n = counts.most_common(1)[0]
        if n / len(g) >= min_share:
            keep.append(g[g["subcategory"] == label])
    out = pd.concat(keep) if keep else clean.iloc[0:0]
    return out.drop_duplicates(subset=["norm", "subcategory"]).reset_index(drop=True)


def assign_splits(clean: pd.DataFrame, rng: random.Random, val_frac: float = 0.1, test_frac: float = 0.1) -> pd.DataFrame:
    """Stratified (per subcategory) split of *groups*."""
    split_of_group: Dict[Tuple[str, str], str] = {}
    for sub, g in clean.groupby("subcategory"):
        groups = sorted(g["group"].unique())
        rng.shuffle(groups)
        n = len(groups)
        n_val = max(1, round(n * val_frac)) if n >= 5 else 0
        n_test = max(1, round(n * test_frac)) if n >= 5 else 0
        for i, grp in enumerate(groups):
            split_of_group[(sub, grp)] = "val" if i < n_val else ("test_synth" if i < n_val + n_test else "train")
    clean = clean.copy()
    clean["split"] = [split_of_group[(s, g)] for s, g in zip(clean["subcategory"], clean["group"])]
    # remove cross-part overlaps of normalised clean names
    train_norms = set(clean.loc[clean.split == "train", "norm"])
    clean = clean[~((clean.split != "train") & clean["norm"].isin(train_norms))]
    val_norms = set(clean.loc[clean.split == "val", "norm"])
    clean = clean[~((clean.split == "test_synth") & clean["norm"].isin(val_norms))]
    return clean.reset_index(drop=True)


def variants_per_sub(clean: pd.DataFrame, target_train: int, default_k: int, max_k: int) -> Dict[str, int]:
    """More noisy variants for thin subcategories so each one reaches `target_train` train rows."""
    ks: Dict[str, int] = {}
    train_counts = clean[clean.split == "train"].groupby("subcategory").size().to_dict()
    for sub in clean["subcategory"].unique():
        n = max(1, train_counts.get(sub, 1))
        need = math.ceil(target_train / n) - 1  # minus the clean row itself
        ks[sub] = int(min(max_k, max(default_k, need)))
    return ks


def build_dataset(
    clean: pd.DataFrame,
    taxonomy: Dict[str, dict],
    seed: int = 42,
    default_variants: int = 5,
    max_variants: int = 12,
    target_train_per_sub: int = 320,
    max_clean_per_sub: Optional[int] = 4000,
) -> pd.DataFrame:
    """`clean` columns: text_clean, subcategory, source, profile, group."""
    rng = random.Random(seed)
    clean = clean[clean["subcategory"].isin(taxonomy)].copy()
    clean = resolve_label_conflicts(clean)
    if max_clean_per_sub:
        parts = []
        for _, g in clean.groupby("subcategory"):
            parts.append(g.sample(n=min(len(g), max_clean_per_sub), random_state=seed))
        clean = pd.concat(parts).reset_index(drop=True)
    clean = assign_splits(clean, rng)
    ks = variants_per_sub(clean, target_train_per_sub, default_variants, max_variants)

    rows: List[dict] = []
    for r in clean.itertuples(index=False):
        base = dict(
            text_clean=r.text_clean, subcategory=r.subcategory, category=taxonomy[r.subcategory]["category"],
            parent_source=r.source, profile=r.profile, group=r.group, split=r.split,
        )
        rows.append({**base, "text": r.text_clean, "source": r.source, "origin": "clean"})
        for v in noisy_variants(r.text_clean, ks[r.subcategory], rng, r.profile,
                                unit_kind=unit_kind_for(r.subcategory)):
            rows.append({**base, "text": v, "source": "aug", "origin": "noisy"})
    rows = _top_up_thin_classes(rows, clean, taxonomy, rng, target_train_per_sub)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df[["text", "text_clean", "subcategory", "category", "source", "parent_source",
               "origin", "profile", "group", "split"]]


def build_dataset_min_rows(clean: pd.DataFrame, taxonomy: Dict[str, dict], min_rows: int = 50_000,
                           default_variants: int = 5, max_default_variants: int = 9, **kwargs):
    """build_dataset(), raising `default_variants` one step at a time until the table has `min_rows` rows.

    The guide asks for >= 50 000 rows. More noisy lines per clean name is ordinary augmentation (they are
    distinct lines, duplicates are removed), so the minimum number that reaches the target is used.
    Returns (DataFrame, variants_used).
    """
    k = default_variants
    max_variants = kwargs.pop("max_variants", 12)
    while True:
        df = build_dataset(clean, taxonomy, default_variants=k, max_variants=max(max_variants, k), **kwargs)
        if len(df) >= min_rows or k >= max_default_variants:
            return df, k
        print(f"[02] {len(df)} rows with {k} noisy lines per clean name < {min_rows}; trying {k + 1}")
        k += 1


def _top_up_thin_classes(rows: List[dict], clean: pd.DataFrame, taxonomy: Dict[str, dict], rng: random.Random,
                         target_train: int, max_rounds: int = 24) -> List[dict]:
    """Independent extra noisy draws for classes still below `target_train` train rows.

    Short names cannot always give k distinct variants at the calibrated rates. Extra independent draws
    are made round by round, with the cosmetic noise rates multiplied by 1, 1.2, 1.4, 1.6 (`boost`, capped), so
    only classes that really lack distinct lines get stronger noise. Every split of the class grows.
    """
    for rnd in range(max_rounds):
        boost = min(1.6, 1.0 + 0.2 * rnd)   # round 0 is the calibrated noise; later rounds a bit stronger, capped
        counts: Counter = Counter(r["subcategory"] for r in rows if r["split"] == "train")
        thin = {s for s in clean["subcategory"].unique() if counts.get(s, 0) < target_train}
        if not thin:
            break
        n_clean = clean.groupby("subcategory").size().to_dict()
        have: Dict[Tuple[str, str], set] = defaultdict(set)
        for r in rows:
            have[(r["subcategory"], r["text_clean"])].add(r["text"])
        for r in clean[clean["subcategory"].isin(thin)].itertuples(index=False):
            need = max(2, 2 * math.ceil((target_train - counts.get(r.subcategory, 0)) / max(1, n_clean[r.subcategory] * 0.8)))
            key = (r.subcategory, r.text_clean)
            for v in noisy_variants(r.text_clean, need, rng, r.profile, unit_kind=unit_kind_for(r.subcategory),
                                    boost=boost):
                if v in have[key]:
                    continue
                have[key].add(v)
                rows.append(dict(
                    text_clean=r.text_clean, subcategory=r.subcategory, category=taxonomy[r.subcategory]["category"],
                    parent_source=r.source, profile=r.profile, group=r.group, split=r.split,
                    text=v, source="aug", origin="noisy"))
    return rows


def class_distribution(df: pd.DataFrame) -> dict:
    out: dict = {"rows": int(len(df)), "by_split": df["split"].value_counts().to_dict(),
                 "by_source": df["source"].value_counts().to_dict(), "by_subcategory": {}}
    for sub, g in df.groupby("subcategory"):
        out["by_subcategory"][sub] = {
            **{s: int((g.split == s).sum()) for s in SPLITS},
            "total": int(len(g)),
        }
    return out


def check_no_overlap(df: pd.DataFrame) -> Dict[str, int]:
    """Number of normalised clean names shared between each pair of parts (must all be 0)."""
    norms = {s: set(df.loc[df.split == s, "text_clean"].map(normalize)) for s in SPLITS}
    return {
        "train&val": len(norms["train"] & norms["val"]),
        "train&test_synth": len(norms["train"] & norms["test_synth"]),
        "val&test_synth": len(norms["val"] & norms["test_synth"]),
    }
