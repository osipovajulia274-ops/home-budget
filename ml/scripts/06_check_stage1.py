"""Stage 1 checks (guide, section 6.6). Prints a table, writes ml/reports/stage1_check.json.

  python -m ml.scripts.06_check_stage1 [--show 20]
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ml.dataset_builder import SPLITS, check_no_overlap
from ml.name_clean import apply_guards, clean_name, reject_reason
from ml.tableio import read_table

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=20, help="random rows to print per part")
    ap.add_argument("--visual-ok", action="store_true",
                    help="tick the manual check 'noisy lines look like real D6 lines' after you compared them")
    args = ap.parse_args()

    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    try:
        df = read_table(PROCESSED / "category_dataset.parquet")
    except (FileNotFoundError, ValueError) as exc:
        print(f"category_dataset not found: {exc}")
        sys.exit(1)

    check("dataset >= 50000 rows", len(df) >= 50_000, f"{len(df)} rows")
    train_counts = df[df.split == "train"].groupby("subcategory").size()
    thin = train_counts[train_counts < 300]
    check("every subcategory >= 300 train rows", thin.empty and len(train_counts) == 48,
          f"{len(train_counts)} subcategories, min={int(train_counts.min())}"
          + (f", thin: {thin.to_dict()}" if not thin.empty else ""))
    overlap = check_no_overlap(df)
    check("no clean-name overlap between parts", sum(overlap.values()) == 0, str(overlap))

    real = PROCESSED / "real_test_labeled.csv"
    n_real = len(pd.read_csv(real)) if real.exists() else 0
    check("real test >= 500 labelled items", n_real >= 500, f"{n_real} rows" if real.exists() else "file missing")
    check("class_distribution.json exists", (REPORTS / "class_distribution.json").exists(), "")

    # data quality of the clean names (found in the first real run): no HTML entities, no Ukrainian /
    # Bulgarian names, and the name guards have nothing left to move or drop.
    clean_names = df.drop_duplicates("text_clean")[["text_clean", "subcategory", "parent_source"]]
    # seed names are written by hand (brands like "ChatGPT Plus" are fine), so the language test is for OFF only
    ext = clean_names[clean_names["parent_source"] != "seed"]
    lex_path = REPORTS / "foreign_lexicon.json"
    lexicon = frozenset(json.loads(lex_path.read_text(encoding="utf-8"))) if lex_path.exists() else frozenset()
    bad_text = [t for t in ext["text_clean"] if clean_name(t) != t or reject_reason(clean_name(t), lexicon=lexicon)]
    bad_html = [t for t in clean_names["text_clean"] if clean_name(t) != t]
    bad_text = sorted(set(bad_text) | set(bad_html))
    check("clean names: no HTML entities, OFF names are Russian", not bad_text,
          f"{len(bad_text)} bad, e.g. {bad_text[:3]}" if bad_text else f"{len(clean_names)} names ok ({len(ext)} from Open*Facts)")
    off_names = clean_names[clean_names["parent_source"].isin(["off", "obf", "opf", "opff"])]
    drift = [(t, s) for t, s in zip(off_names["text_clean"], off_names["subcategory"])
             if apply_guards(t, s)[0] != s]
    check("OFF labels: name guards find nothing to fix", not drift,
          f"{len(drift)} rows still contradict their label, e.g. {drift[:3]}" if drift else
          f"{len(off_names)} OFF names checked")
    # visual check is manual: we only make sure the samples can be printed
    for split in SPLITS:
        part = df[df.split == split]
        print(f"\n--- {split}: {len(part)} rows, {args.show} random ---")
        for t in part["text"].sample(n=min(args.show, len(part)), random_state=1):
            print("  ", t)
    style = REPORTS / "style_compare.json"
    style_note = ""
    if style.exists():
        data = json.loads(style.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "groups" in data:
            bad = [f"{g}: {r['feature']}" for g, rows in data["groups"].items() for r in rows if r["ok"] is False]
            style_note = (f" Numeric style fingerprint: {data['judged_ok']}/{data['judged_total']} within tolerance"
                          + (f", off: {bad}" if bad else "") + ".")
        else:   # older format: a flat list of rows
            bad = [r["feature"] for r in data if not r["ok"]]
            style_note = (f" Numeric style fingerprint: {len(data) - len(bad)}/{len(data)} within tolerance"
                          + (f", off: {bad}" if bad else "") + ".")
    check("visual comparison with D6 (manual)", args.visual_ok,
          ("ticked by hand (--visual-ok)." if args.visual_ok else
           "compare printed rows with real D6 lines (03_augment_receipt_style --compare-d6), then re-run with --visual-ok.")
          + style_note)

    print("\n=== Stage 1 checks ===")
    for r in results:
        print(f"[{'x' if r['ok'] else ' '}] {r['check']}  -- {r['detail']}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "stage1_check.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(r["ok"] for r in results)
    print(f"passed {passed}/{len(results)}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
