"""Stage 1: prepare / finalize the hand-labelled REAL test set built from D6 (CDEK receipts).

  python -m ml.scripts.04_label_real_test export
        -> ml/data/processed/real_items_to_label.csv   (fill the `subcategory` column by hand)
  python -m ml.scripts.04_label_real_test finalize [--labeled path.csv]
        -> ml/data/processed/real_test_labeled.csv     (text, subcategory, receipt_id)
        -> ml/data/processed/real_test_review_sample.csv  (10% of rows for a second human look)

Rules (guide 6.4): minimum 500 labelled items (goal 1000+); this set is NEVER used for training;
items of one receipt never leave the same part (we keep receipt_id on every row).
Valid values for `subcategory` are the 48 ids from data/taxonomy.yaml (e.g. food.dairy).
Leave `subcategory` empty to skip an item. Real receipts stay out of git (ml/data is ignored).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from ml.scripts_helpers import as_list
from ml.seed_generator import load_taxonomy

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml" / "data" / "raw" / "cdek_receipts"
PROCESSED = ROOT / "ml" / "data" / "processed"

JUNK = re.compile(r"не\s*распознан|групп[аы]\s+товар|скидк|итог|сдача|налич|безнал|ндс|округлен", re.IGNORECASE)
LONG_NUMBER = re.compile(r"\d{9,}")


def is_junk(text: str) -> bool:
    t = text.strip()
    return len(re.findall(r"[А-Яа-яA-Za-z]", t)) < 3 or bool(JUNK.search(t))


def export(max_receipts: int | None) -> None:
    files = sorted(RAW.rglob("*.parquet"))
    if not files:
        sys.exit(f"no parquet files in {RAW} - run 01_download.py --only d6 first")
    rows = []
    for f in files:
        split = f.stem.split("-")[0]
        df = pd.read_parquet(f)
        if "item_texts" not in df.columns:
            sys.exit(f"{f.name}: no item_texts column; columns = {list(df.columns)}")
        for i, rec in enumerate(df.to_dict("records")):
            if max_receipts and len({r['receipt_id'] for r in rows}) >= max_receipts:
                break
            seller = LONG_NUMBER.sub("", str(rec.get("seller_text", "") or ""))[:60]  # no INN in files
            for text in as_list(rec.get("item_texts")):
                if not is_junk(text):
                    rows.append({"receipt_id": f"{split}_{i}", "d6_split": split, "seller": seller,
                                 "text": re.sub(r"\s+", " ", text).strip(), "subcategory": ""})
    out = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "real_items_to_label.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig: opens correctly in Excel
    print(f"[04] {len(out)} items from {out['receipt_id'].nunique()} receipts -> {path}")
    print("     Fill `subcategory` (ids from data/taxonomy.yaml) for at least 500 rows, then run `finalize`.")


def finalize(labeled: Path, min_rows: int) -> None:
    tax = load_taxonomy()
    df = pd.read_csv(labeled, keep_default_na=False, encoding="utf-8-sig")
    df["subcategory"] = df["subcategory"].str.strip()
    df = df[df["subcategory"] != ""]
    bad = sorted(set(df["subcategory"]) - set(tax))
    if bad:
        sys.exit(f"unknown subcategory ids: {bad}")
    out = df[["text", "subcategory", "receipt_id"]].reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "real_test_labeled.csv", index=False, encoding="utf-8")
    review = out.sample(frac=0.10, random_state=42) if len(out) else out
    review.to_csv(PROCESSED / "real_test_review_sample.csv", index=False, encoding="utf-8-sig")
    print(f"[04] labelled: {len(out)} items from {out['receipt_id'].nunique()} receipts "
          f"({'OK' if len(out) >= min_rows else f'need >= {min_rows}'})")
    print(out["subcategory"].str.split(".").str[0].value_counts().to_string())
    print(f"[04] review sample (10%): {len(review)} rows -> real_test_review_sample.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--max-receipts", type=int, default=None)
    f = sub.add_parser("finalize")
    f.add_argument("--labeled", default=str(PROCESSED / "real_items_to_label.csv"))
    f.add_argument("--min-rows", type=int, default=500)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.max_receipts)
    else:
        finalize(Path(args.labeled), args.min_rows)


if __name__ == "__main__":
    main()
