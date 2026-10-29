"""Stage 1: build ml/data/processed/category_dataset.parquet.

  python -m ml.scripts.02_build_category_dataset              # seed + Open*Facts extracts if present
  python -m ml.scripts.02_build_category_dataset --seed-only  # smoke run, no downloads needed

Open*Facts extracts (produced by 01_download.py) are expected as
  ml/data/processed/off_food.csv, obf.csv, opf.csv, opff.csv   (columns: code,name,subcategory,tags)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from ml.dataset_builder import build_dataset, build_dataset_min_rows, check_no_overlap, class_distribution
from ml.name_clean import GuardReport, learn_foreign_lexicon, process_rows
from ml.off_extract import load_map, map_tags
from ml.seed_generator import generate_seed_clean, load_taxonomy
from ml.tableio import write_table
from app.services.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"

# file -> (source label, section of off_category_map.yaml)
OFF_FILES = {"off_food.csv": ("off", "food"), "obf.csv": ("obf", "beauty"),
             "opf.csv": ("opf", "products"), "opff.csv": ("opff", "pet_food")}


def load_off_clean(report: GuardReport) -> pd.DataFrame:
    """Read the Open*Facts extracts and clean them.

    1. the subcategory is recomputed from the stored tags with the CURRENT off_category_map.yaml, so a
       corrected map does not need a new multi-GB extraction;
    2. names are HTML-unescaped and Russian-only: static rules (letters, word lists, language tag, block-listed
       countries) plus a foreign vocabulary LEARNED from the extract itself (ml/name_clean.learn_foreign_lexicon);
    3. name guards (data/name_guards.yaml) move or drop rows whose name contradicts the tag label.
    """
    rules_all = load_map()
    loaded = {}
    for fname, (source, kind) in OFF_FILES.items():
        path = PROCESSED / fname
        if not path.exists():
            print(f"[02] missing {path.name} - skipped")
            continue
        df = pd.read_csv(path, keep_default_na=False)
        subs = df["subcategory"].tolist()
        if "tags" in df.columns:
            remapped = [map_tags(str(t).split(","), rules_all[kind]) for t in df["tags"]]
            report.counts[f"{source}:remap_changed_label"] += sum(1 for o, n in zip(subs, remapped) if n and n != o)
            report.counts[f"{source}:remap_unmapped_dropped"] += sum(1 for n in remapped if not n)
            subs = remapped
        n_in = len(df)
        countries = df["countries"].tolist() if "countries" in df.columns else [""] * n_in
        langs = df["lang"].tolist() if "lang" in df.columns else [""] * n_in
        rows = [(n, sub, c, source, lg) for n, sub, c, lg in zip(df["name"], subs, countries, langs) if sub]
        loaded[fname] = (source, df, rows)

    # one vocabulary for all files: the big food dump gives the statistics, the small ones profit from it
    lexicon = learn_foreign_lexicon((r[0], r[2], r[4]) for _, _, rows in loaded.values() for r in rows)
    report.learned_lexicon = lexicon

    frames = []
    stats = {}
    for fname, (source, df, rows) in loaded.items():
        kept = process_rows(rows, report, lexicon=lexicon)
        out = pd.DataFrame(kept, columns=["text_clean", "subcategory"])
        out["source"] = source
        out["profile"] = "goods"
        out["group"] = "off:" + out["text_clean"].map(normalize)
        frames.append(out)
        has_c = "countries" in df.columns
        stats[fname] = {
            "rows_in": len(df), "rows_kept": len(out),
            "has_countries": bool(has_c and (df["countries"] != "").any()),
            "top_country_values": [[c, n] for c, n in Counter(df["countries"]).most_common(12)] if has_c else [],
            "top_lang_values": [[c, n] for c, n in Counter(df["lang"]).most_common(10)] if "lang" in df.columns else [],
        }
        print(f"[02] {fname}: {len(df)} -> {len(out)} rows after cleaning")
    report.stats = stats
    print(f"[02] learned foreign vocabulary: {len(lexicon)} words (see off_guard_report.json -> learned_foreign_words)")
    if not frames:
        return pd.DataFrame(columns=["text_clean", "subcategory", "source", "profile", "group"])
    return pd.concat(frames, ignore_index=True)


def write_off_samples(off: pd.DataFrame, path: Path, n: int = 8) -> None:
    """Human review aid for the draft tag map: random names per subcategory."""
    lines = ["# OFF map samples (review and fix data/off_category_map.yaml)\n"]
    for sub, g in off.groupby("subcategory"):
        lines.append(f"\n## {sub}  ({len(g)} rows)\n")
        for t in g["text_clean"].sample(n=min(n, len(g)), random_state=1):
            lines.append(f"- {t}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fills", type=int, default=2, help="distinct placeholder fills per seed template")
    ap.add_argument("--default-variants", type=int, default=5, help="noisy lines per clean name (before duplicates)")
    ap.add_argument("--max-variants", type=int, default=12)
    ap.add_argument("--target-train", type=int, default=320, help="min train rows per subcategory to aim for")
    ap.add_argument("--max-clean-per-sub", type=int, default=4000)
    ap.add_argument("--min-rows", type=int, default=50_000,
                    help="guide, 6.6: at least this many rows; if the build is smaller, --default-variants is raised "
                         "one step at a time (up to --max-default-variants)")
    ap.add_argument("--max-default-variants", type=int, default=9)
    ap.add_argument("--out", default=str(PROCESSED / "category_dataset.parquet"))
    args = ap.parse_args()

    taxonomy = load_taxonomy()
    seed_rows = pd.DataFrame(generate_seed_clean(random.Random(args.seed), args.fills),
                             columns=["text_clean", "subcategory", "profile", "group"])
    seed_rows["source"] = "seed"
    parts = [seed_rows]
    if not args.seed_only:
        report = GuardReport()
        off = load_off_clean(report)
        REPORTS.mkdir(parents=True, exist_ok=True)
        lex = getattr(report, "learned_lexicon", {})
        top = sorted(lex.items(), key=lambda kv: -kv[1][0])
        (REPORTS / "off_guard_report.json").write_text(json.dumps(
            {"files": getattr(report, "stats", {}), "actions": report.as_dict(),
             "learned_foreign_words": [[w, f, r] for w, (f, r) in top[:300]]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (REPORTS / "foreign_lexicon.json").write_text(json.dumps(sorted(lex), ensure_ascii=False), encoding="utf-8")
        if len(off):
            write_off_samples(off, REPORTS / "off_map_samples.md")
            parts.append(off)
    clean = pd.concat(parts, ignore_index=True)
    print(f"[02] clean names before dedup: {len(clean)}")

    if args.seed_only:
        df = build_dataset(clean, taxonomy, seed=args.seed, default_variants=args.default_variants,
                           max_variants=args.max_variants, target_train_per_sub=args.target_train,
                           max_clean_per_sub=args.max_clean_per_sub)
        k = args.default_variants
    else:
        df, k = build_dataset_min_rows(clean, taxonomy, min_rows=args.min_rows, default_variants=args.default_variants,
                                       max_default_variants=args.max_default_variants, seed=args.seed,
                                       max_variants=args.max_variants, target_train_per_sub=args.target_train,
                                       max_clean_per_sub=args.max_clean_per_sub)
    print(f"[02] noisy lines per clean name (before duplicate removal): {k}")
    out = write_table(df, args.out)
    dist = class_distribution(df)
    dist["overlap"] = check_no_overlap(df)
    dist["seed_only"] = bool(args.seed_only)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "class_distribution.json").write_text(json.dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[02] wrote {len(df)} rows -> {out}")
    print(f"[02] splits: {dist['by_split']}  overlap: {dist['overlap']}")


if __name__ == "__main__":
    main()
