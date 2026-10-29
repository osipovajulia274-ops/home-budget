"""Preview of the receipt-style augmentation (the real work happens inside 02_build...).

  python -m ml.scripts.03_augment_receipt_style --n 20
  python -m ml.scripts.03_augment_receipt_style --compare-d6   # print real D6 lines next to ours

Use it for the visual check "do our noisy lines look like real receipt lines?" (check 6.6).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ml.seed_generator import generate_seed_clean
from ml.receipt_noise import noisy_variants, unit_kind_for
from ml.style_stats import SKIP_BY_GROUP, compare, judged, style_stats
from ml.tableio import read_table

ROOT = Path(__file__).resolve().parents[2]


def d6_sample(n: int, seed: int) -> list:
    files = sorted((ROOT / "ml" / "data" / "raw" / "cdek_receipts").rglob("*.parquet"))
    if not files:
        return []
    from ml.scripts_helpers import iter_d6_items  # lazy: needs pandas+pyarrow only
    items = [t for _, t in iter_d6_items(files)]
    random.Random(seed).shuffle(items)
    return items[:n]


def _real_groups(seed: int) -> dict:
    """Real D6 lines split like our profiles: eating-out (meal) vs everything else (goods).

    Needs the labelled real test (ml/data/processed/real_test_labeled.csv). Without it all D6 lines are
    compared with all of our goods + meal lines, as in the first version of this check."""
    labeled = ROOT / "ml" / "data" / "processed" / "real_test_labeled.csv"
    if labeled.exists():
        real = pd.read_csv(labeled, keep_default_na=False)
        meal = real["subcategory"].astype(str).str.startswith("eating_out")
        return {"goods": real.loc[~meal, "text"].tolist(), "meal": real.loc[meal, "text"].tolist()}
    return {"all": d6_sample(20000, seed)}


def print_fingerprint(seed: int) -> None:
    """Numeric comparison of receipt-style shares with the real D6 lines, group by group.

    goods: our goods-profile rows (clean + noisy) vs real lines that are not eating-out.
    meal : our eating-out rows vs real eating-out lines (leading digits / digits are shown, not judged).
    """
    path = ROOT / "ml" / "data" / "processed" / "category_dataset.parquet"
    try:
        df = read_table(str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"\n[03] no dataset to fingerprint yet ({exc}); run step 02 first")
        return
    real_groups = _real_groups(seed)
    ours_groups = ({"goods": df[df["profile"] == "goods"]["text"].tolist(),
                    "meal": df[df["profile"] == "meal"]["text"].tolist()}
                   if "all" not in real_groups else
                   {"all": df[df["profile"].isin(["goods", "meal"])]["text"].tolist()})
    result = {}
    ok_total = n_total = 0
    for name, real_lines in real_groups.items():
        ours = ours_groups[name]
        if not real_lines or not ours:
            print(f"\n[03] fingerprint of '{name}' skipped: no real lines or no augmented rows")
            continue
        table = compare(style_stats(real_lines), style_stats(ours), skip=SKIP_BY_GROUP.get(name, ()))
        print(f"\nStyle fingerprint, group '{name}' (real D6 lines: {len(real_lines)}, our lines, clean + noisy: {len(ours)})")
        print(table.to_string(index=False))
        j = judged(table)
        ok_total += int(j["ok"].sum())
        n_total += len(j)
        result[name] = table.to_dict("records")
    # shortcut guard: the classifier must not be able to tell a class by its noise style alone
    noisy = df[(df["source"] == "aug") & (df["profile"].isin(["goods", "meal"]))]
    per_class = noisy.assign(dig=noisy["text"].str.match(r"^\d")).groupby("subcategory")["dig"].mean() * 100
    print(f"\nShare of noisy lines that start with a digit, per class: min {per_class.min():.0f}%  "
          f"median {per_class.median():.0f}%  max {per_class.max():.0f}%  "
          f"(a wide spread lets the model learn 'digit first => this class')")
    out = ROOT / "ml" / "reports" / "style_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"groups": result, "judged_ok": ok_total, "judged_total": n_total,
                               "digit_start_per_class": {"min": float(per_class.min()),
                                                         "median": float(per_class.median()),
                                                         "max": float(per_class.max())}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[03] {ok_total}/{n_total} judged features within tolerance -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--compare-d6", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = generate_seed_clean(rng, 1)
    rng.shuffle(rows)
    print(f"{'subcategory':26s} | clean  ->  noisy")
    for clean, sub, profile, _ in rows[: args.n]:
        v = noisy_variants(clean, 3, rng, profile, unit_kind=unit_kind_for(sub))   # k=3, show one at random
        print(f"{sub:26s} | {clean}  ->  {rng.choice(v) if v else '(no distinct variant)'}")
    if args.compare_d6:
        print("\nReal D6 lines:")
        for t in d6_sample(args.n, args.seed):
            print("  ", t)
        print_fingerprint(args.seed)


if __name__ == "__main__":
    main()
