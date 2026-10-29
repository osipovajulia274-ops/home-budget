"""Stage 1: download raw datasets (D1, D3-D6) and extract Russian-named rows.

  pip install -r requirements-train.txt
  python -m ml.scripts.01_download                 # everything
  python -m ml.scripts.01_download --only d6      # one source (d1 d3 d4 d5 d6)
  python -m ml.scripts.01_download --no-extract   # download only

Rules (guide, section 0.8): official URLs only. If a source is unreachable the script reports it
and continues; do NOT look for mirrors - record the gap in PROGRESS.md and rely on the seed
dictionary instead. D7 (Data Fusion 2021) is optional and not downloaded here.

The CDEK Donut model that lives next to the D6 dataset is a pretrained model and is forbidden:
only dataset files (parquet/json/md) are fetched, never model weights.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.downloader import download
from ml.off_extract import extract_names, write_schema_report

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml" / "data" / "raw"
PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"

SOURCES = {
    "d1": {
        "url": "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet?download=true",
        "dest": RAW / "off_food" / "food.parquet",
        "kind": "food", "out": PROCESSED / "off_food.csv",
        "note": "~4 GB; DuckDB reads it directly, nothing is loaded into RAM at once",
    },
    "d3": {
        "url": "https://static.openproductsfacts.org/data/en.openproductsfacts.org.products.csv.gz",
        "dest": RAW / "opf" / "products.csv.gz", "kind": "products", "out": PROCESSED / "opf.csv",
        "note": "URL built from the verified D5 pattern - confirm on https://world.openproductsfacts.org/data",
    },
    "d4": {
        "url": "https://static.openbeautyfacts.org/data/en.openbeautyfacts.org.products.csv.gz",
        "dest": RAW / "obf" / "products.csv.gz", "kind": "beauty", "out": PROCESSED / "obf.csv",
        "note": "URL built from the verified D5 pattern - confirm on https://world.openbeautyfacts.org/data",
    },
    "d5": {
        "url": "https://static.openpetfoodfacts.org/data/en.openpetfoodfacts.org.products.csv.gz",
        "dest": RAW / "opff" / "products.csv.gz", "kind": "pet_food", "out": PROCESSED / "opff.csv",
        "note": "official link from https://world.openpetfoodfacts.org/data",
    },
}


def download_d6() -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[01] huggingface_hub is not installed (pip install huggingface_hub)", file=sys.stderr)
        return False
    try:
        path = snapshot_download(
            repo_id="cdek-ocr/receipt-ocr-ru", repo_type="dataset", local_dir=str(RAW / "cdek_receipts"),
            allow_patterns=["*.parquet", "*.json", "*.md", "*.csv"],  # dataset files only, never weights
        )
        print(f"[01] D6 downloaded to {path}")
        return True
    except Exception as exc:  # noqa: BLE001 - report any network/auth problem and move on
        print(f"[01] FAILED D6: {exc}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="d1,d3,d4,d5,d6")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max extracted rows per source (debug)")
    args = ap.parse_args()
    wanted = [s.strip().lower() for s in args.only.split(",") if s.strip()]

    ok = {}
    for key in ("d1", "d3", "d4", "d5"):
        if key not in wanted:
            continue
        spec = SOURCES[key]
        print(f"[01] {key.upper()}: {spec['note']}")
        ok[key] = download(spec["url"], spec["dest"])
    if "d6" in wanted:
        ok["d6"] = download_d6()

    if not args.no_extract:
        schema_paths = {}
        for key, spec in SOURCES.items():
            if ok.get(key) and spec["dest"].exists():
                schema_paths[key] = str(spec["dest"])
        write_schema_report(schema_paths, str(REPORTS / "raw_schemas.json"))
        for key, path in schema_paths.items():
            spec = SOURCES[key]
            extract_names(path, spec["kind"], str(spec["out"]), limit=args.limit)
    print("[01] summary:", {k: ("ok" if v else "FAILED") for k, v in ok.items()})


if __name__ == "__main__":
    main()
